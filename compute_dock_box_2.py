#!/usr/bin/env python3
"""
compute_dock_box.py

Compute docking box center & size from a protein:ligand PDB (chain A = protein, B = ligand).

Usage:
  ./compute_dock_box.py -i complex.pdb -c B -m 5.0
"""
import argparse
import sys

import numpy as np
from Bio.PDB import PDBParser

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute --center_x/y/z and --size_x/y/z for a docking box around chain B."
    )
    p.add_argument(
        "-i","--input", required=True,
        help="Input PDB file (protein+ligand complex)"
    )
    p.add_argument(
        "-c","--chain", default="B",
        help="Ligand chain ID (default: B)"
    )
    p.add_argument(
        "-m","--margin", type=float, default=5.0,
        help="Extra margin (Å) added to each box dimension (default: 5.0)"
    )
    return p.parse_args()

def main():
    args = parse_args()
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("complex", args.input)
    except Exception as e:
        sys.exit(f"ERROR reading PDB: {e}")

    # Collect all atom coords from the specified chain
    coords = []
    for model in struct:
        if args.chain not in model:
            continue
        for atom in model[args.chain].get_atoms():
            coords.append(atom.get_coord())
    if not coords:
        sys.exit(f"ERROR: no atoms found for chain '{args.chain}'")

    arr = np.array(coords)
    center = arr.mean(axis=0)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    size = (mx - mn) + args.margin

    print(f"--center_x {center[0]:.3f} --center_y {center[1]:.3f} --center_z {center[2]:.3f}")
    print(f"--size_x   {size[0]:.3f} --size_y   {size[1]:.3f} --size_z   {size[2]:.3f}")

if __name__ == "__main__":
    main()
