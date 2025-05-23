#!/usr/bin/env python3
"""
batch_prepare_receptors_meeko.py

Batch‐prepare receptor PDBs using Meeko’s mk_prepare_receptor.py,
auto‐handling alternate locations by removing bad residues and defaulting to altloc A.

Usage:
    python batch_prepare_receptors_meeko.py --input_dir ./run/receptors_pdb --output_dir ./prepared_receptors
"""

import os
import subprocess
import argparse


def prepare_receptors(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdb')]
    if not pdb_files:
        print(f"No PDB files found in {input_dir}")
        return

    for pdb in pdb_files:
        base = os.path.splitext(pdb)[0]
        out_base = os.path.join(output_dir, f"{base}_meeko")
        out_pdb = f"{out_base}.prep.pdb"

        print(f"Preparing {pdb} → {out_pdb}")
        cmd = [
            "mk_prepare_receptor.py",
            "--read_pdb", os.path.join(input_dir, pdb),
            "-o", out_base,
            "--write_pdb", out_pdb,
            "-p",                        # also write PDBQT
            "-a",                        # allow bad residues
            "--default_altloc", "A"    # pick alternate location A by default
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pdb}: {e}")

    print(f"Done. Prepared PDBs are in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch prepare receptors via Meeko with altloc handling"
    )
    parser.add_argument(
        "--input_dir", "-i", required=True,
        help="Directory containing input .pdb files"
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="Directory to write cleaned PDBs"
    )
    args = parser.parse_args()
    prepare_receptors(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
