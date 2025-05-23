#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import subprocess
from Bio.PDB import PDBParser

def get_residues_in_box(pdb_path, center, size):
    """Return residue IDs inside the grid box as chain:resnum (e.g. 'A:278')."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    xc, yc, zc = center
    sx, sy, sz = size
    hx, hy, hz = sx/2, sy/2, sz/2

    res_ids = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                # skip water/hetero
                if not residue.id[0] == ' ':
                    continue
                for atom in residue:
                    x, y, z = atom.coord
                    if (xc-hx) <= x <= (xc+hx) and (yc-hy) <= y <= (yc+hy) and (zc-hz) <= z <= (zc+hz):
                        res_ids.add(f"{chain.id}:{residue.id[1]}")
                        break
    return sorted(res_ids)

def main():
    parser = argparse.ArgumentParser(description="Batch Meeko flexible/rigid prep using CSV grid and automatic flexible-res extraction")
    parser.add_argument("--in-dir", required=True, help="Input dir of _H.pdb files")
    parser.add_argument("--out-dir", required=True, help="Output dir")
    parser.add_argument("--csv", required=True, help="CSV with columns: filename, center_x, center_y, center_z, size_x, size_y, size_z")
    parser.add_argument("--suffix", default="_H", help="Suffix for input PDBs (default: _H)")
    parser.add_argument("--meeko-script", default="mk_prepare_receptor.py", help="Meeko script (default: in $PATH)")
    parser.add_argument("--allow-bad-res", action='store_true', help="Pass --allow_bad_res to Meeko")
    parser.add_argument("--write-box", action='store_true', help="Write .box.txt Vina box file (-v)")
    parser.add_argument("--write-gpf", action='store_true', help="Write .gpf AutoGrid file (-g)")
    parser.add_argument("--write-json", action='store_true', help="Write receptor .json file (-j)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Map: base protein name (no _H) → grid info
    grid_dict = {row['filename']: row for _, row in df.iterrows()}

    pdb_files = sorted([f for f in os.listdir(args.in_dir) if f.endswith(f"{args.suffix}.pdb")])
    if not pdb_files:
        print(f"No *{args.suffix}.pdb files found in {args.in_dir}")
        sys.exit(1)

    failed = []
    for pdb in tqdm(pdb_files, desc="Batch Meeko FlexPrep"):
        base = os.path.splitext(pdb)[0]  # e.g. PSMA_1_H
        if not base.endswith(args.suffix):
            print(f"[WARN] Skipping {pdb}: does not end with {args.suffix}")
            continue
        prot_name = base[:-len(args.suffix)]  # Remove suffix
        if prot_name not in grid_dict:
            print(f"[WARN] No CSV grid for {prot_name}, skipping")
            continue

        grid = grid_dict[prot_name]
        center = (float(grid['center_x']), float(grid['center_y']), float(grid['center_z']))
        size = (float(grid['size_x']), float(grid['size_y']), float(grid['size_z']))

        pdb_path = os.path.join(args.in_dir, pdb)
        flex_res = get_residues_in_box(pdb_path, center, size)
        if not flex_res:
            print(f"[WARN] No residues in grid for {prot_name}, skipping")
            continue

        # Compose Meeko command
        outbase = os.path.abspath(os.path.join(args.out_dir, prot_name))
        cmd = [
            args.meeko_script,
            "--read_pdb", pdb_path,
            "-o", outbase,
            "-p",  # always output PDBQT
            "-f", ",".join(flex_res)
        ]
        if args.allow_bad_res:
            cmd.append("--allow_bad_res")
        if args.write_box:
            cmd.append("-v")
        if args.write_gpf:
            cmd.append("-g")
        if args.write_json:
            cmd.append("-j")

        # Box settings
        cmd += [
            "--box_center", str(center[0]), str(center[1]), str(center[2]),
            "--box_size", str(size[0]), str(size[1]), str(size[2])
        ]

        print(f"\n[INFO] Running: {' '.join(cmd)}")
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if ret.returncode != 0:
            print(f"[ERROR] Meeko failed for {pdb}")
            print("STDOUT:\n", ret.stdout)
            print("STDERR:\n", ret.stderr)
            failed.append(pdb)
        else:
            print(f"[OK] {pdb} → {prot_name}_rigid.pdbqt + {prot_name}_flex.pdbqt (and box/gpf/json if requested)")

    print("\nBatch Meeko prep complete.")
    if failed:
        print("Failed for these files:")
        for f in failed:
            print(" -", f)
    else:
        print("All files processed successfully.")

if __name__ == "__main__":
    main()
