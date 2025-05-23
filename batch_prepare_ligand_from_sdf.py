#!/usr/bin/env python3
"""
batch_prepare_from_sdf.py

Takes every .sdf in --input-dir, runs Meeko to add hydrogens & atom types,
and exports back to SDF. No dangling PDBQT filenames, with progress bar.

Usage:
    python batch_prepare_from_sdf.py \
      --input-dir ./4AA_peptide_sequences_sdf \
      --output-dir ./4AA_peptide_sequences_prepared_sdf \
      [--scratch ./tmp] \
      [--jobs 8]
"""

import os
import glob
import argparse
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{cmd[0]} failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc.stdout

def process_sdf(sdf_path, out_dir, scratch_dir):
    base = os.path.splitext(os.path.basename(sdf_path))[0]
    ligand_scratch = os.path.join(scratch_dir, base)
    # clean slate
    shutil.rmtree(ligand_scratch, ignore_errors=True)
    os.makedirs(ligand_scratch, exist_ok=True)

    # 1) mk_prepare_ligand → PDBQT(s) with proper prefix
    cmd_prep = [
        "mk_prepare_ligand.py",
        "-i", sdf_path,
        "--multimol_outdir", ligand_scratch,
        "--multimol_prefix", base
    ]
    run(cmd_prep)

    # 2) find all produced PDBQTs
    pdbqts = glob.glob(os.path.join(ligand_scratch, f"{base}*.pdbqt"))
    if not pdbqts:
        return base, False, "no PDBQT"

    # 3) mk_export each → _prepared.sdf
    for pdbqt in pdbqts:
        name = os.path.splitext(os.path.basename(pdbqt))[0]
        out_sdf = os.path.join(out_dir, f"{name}_prepared.sdf")
        cmd_exp = ["mk_export.py", pdbqt, "-s", out_sdf]
        run(cmd_exp)

    # clean up intermediates
    shutil.rmtree(ligand_scratch, ignore_errors=True)
    return base, True, len(pdbqts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True, help="Folder of .sdf ligands")
    p.add_argument("--output-dir", required=True, help="Where to write *_prepared.sdf")
    p.add_argument("--scratch",    default="tmp", help="Scratch folder for PDBQTs")
    p.add_argument("--jobs",       type=int, default=1, help="Parallel jobs")
    args = p.parse_args()

    inp = os.path.abspath(args.input_dir)
    out = os.path.abspath(args.output_dir)
    scratch = os.path.abspath(args.scratch)
    os.makedirs(out, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)

    sdfs = sorted(glob.glob(os.path.join(inp, "*.sdf")))
    if not sdfs:
        print("No .sdf files found in", inp)
        return

    # Use ThreadPool to process with progress bar
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(process_sdf, sdf, out, scratch): sdf for sdf in sdfs}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Preparing ligands"):  
            fut = _
            sdf = os.path.basename(futures[fut])
            try:
                base, ok, info = fut.result()
                if ok:
                    print(f"✓ {base}: wrote {info} SDF(s)")
                else:
                    print(f"⚠ {base}: {info}")
            except Exception as e:
                print(f"⚠ {sdf}: ERROR: {e}")

    # final cleanup
    shutil.rmtree(scratch, ignore_errors=True)
    print("Done.")

if __name__ == "__main__":
    main()
