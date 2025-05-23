#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import subprocess

def parse_pdbqt_atoms(pdbqt_file):
    """
    Yield (residue_name, residue_number, x, y, z) for every ATOM/HETATM line in the PDBQT.
    """
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                residue_name = line[17:20].strip()
                residue_number = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                yield (residue_name, residue_number, x, y, z)

def get_residues_in_box(pdbqt_file, center, size, max_residues=20):
    """
    Return a list of residue IDs (e.g., HIS112) with at least one atom within the specified box.
    Only the closest max_residues to the center are kept.
    """
    xc, yc, zc = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2

    res_dist = {}
    for residue_name, residue_number, x, y, z in parse_pdbqt_atoms(pdbqt_file):
        if (xc - hx <= x <= xc + hx and
            yc - hy <= y <= yc + hy and
            zc - hz <= z <= zc + hz):
            res_id = f"{residue_name}{residue_number}"
            dist = ((x-xc)**2 + (y-yc)**2 + (z-zc)**2)**0.5
            if res_id not in res_dist or dist < res_dist[res_id]:
                res_dist[res_id] = dist

    sorted_res = sorted(res_dist.items(), key=lambda x: x[1])
    top_res = [name for name, _ in sorted_res[:max_residues]]
    return top_res

def run_prepare_flexreceptor(prepare_flex_path, receptor_path, residues, rigid_out, flex_out, verbose=True):
    """
    Run prepare_flexreceptor4.py (with pythonsh shebang in script).
    Suppress all warnings and output from the subprocess.
    """
    residues_arg = "_".join(residues)
    cmd = [
        prepare_flex_path,
        "-r", receptor_path,
        "-s", residues_arg,
        "-g", rigid_out,
        "-x", flex_out
    ]
    if verbose:
        print(">> Running:", " ".join(cmd))
    # Suppress all output and warnings
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main(args):
    os.makedirs(args.out_folder, exist_ok=True)
    df = pd.read_csv(args.csv_file)

    for i, row in df.iterrows():
        pdbqt_filename = row['filename']
        center = (row['center_x'], row['center_y'], row['center_z'])
        size = (row['size_x'], row['size_y'], row['size_z'])
        pdbqt_file = os.path.join(args.pdbqt_folder, f"{pdbqt_filename}.pdbqt")
        if not os.path.isfile(pdbqt_file):
            print(f"[WARN] Missing PDBQT file: {pdbqt_file}")
            continue

        print(f"[INFO] Processing {pdbqt_file} | center={center}, size={size}")
        residues = get_residues_in_box(pdbqt_file, center, size, max_residues=args.max_residues)
        print(f"[INFO] Found {len(residues)} residues: {residues}")

        if not residues:
            print(f"[WARN] No residues found in box for {pdbqt_filename}, skipping.")
            continue

        rigid_out = os.path.join(args.out_folder, f"{pdbqt_filename}_rigid.pdbqt")
        flex_out = os.path.join(args.out_folder, f"{pdbqt_filename}_flex.pdbqt")

        if args.dry_run:
            print(f"[DRY RUN] Would run: {args.prepare_flex_path} -r {pdbqt_file} -s {'_'.join(residues)} -g {rigid_out} -x {flex_out}")
        else:
            run_prepare_flexreceptor(
                args.prepare_flex_path, pdbqt_file, residues,
                rigid_out, flex_out, verbose=True
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch flexible residue extraction and AutoDockTools flex receptor preparation."
    )
    parser.add_argument("--csv-file", required=True, help="CSV with columns: filename, center_x, center_y, center_z, size_x, size_y, size_z")
    parser.add_argument("--pdbqt-folder", required=True, help="Folder with PDBQT files")
    parser.add_argument("--out-folder", required=True, help="Output folder for generated flex/rigid files")
    parser.add_argument("--prepare-flex-path", required=True, help="Path to prepare_flexreceptor4.py (should be executable)")
    parser.add_argument("--max-residues", type=int, default=20, help="Maximum flexible residues (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions, do not run flex preparation")

    args = parser.parse_args()
    main(args)
