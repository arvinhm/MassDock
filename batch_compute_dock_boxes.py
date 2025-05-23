#!/usr/bin/env python3
"""
batch_compute_dock_boxes.py

Batch‐compute docking box center & size from protein:ligand complexes
(chain A = protein, B = ligand) in a directory. Outputs a CSV and shows a tqdm bar.
"""

import os
import sys
import argparse
import csv

import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm


def compute_box_coords(pdb_path, chain_id, margin):
    """Read pdb_path, extract atom coords for chain_id, then compute center & size."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    # assume first model
    model = struct[0]
    if chain_id not in model:
        raise ValueError(f"Chain '{chain_id}' not found")
    coords = np.array([atom.get_coord() for atom in model[chain_id].get_atoms()])
    if coords.size == 0:
        raise ValueError(f"No atoms in chain '{chain_id}'")
    center = coords.mean(axis=0)
    mn, mx = coords.min(axis=0), coords.max(axis=0)
    size = (mx - mn) + margin
    return center, size


def main():
    p = argparse.ArgumentParser(
        description="Batch compute docking box params for all PDBs in a folder"
    )
    p.add_argument(
        "--dir", "-d", required=True,
        help="Directory containing PDB files"
    )
    p.add_argument(
        "--chain", "-c", default="B",
        help="Ligand chain ID (default: B)"
    )
    p.add_argument(
        "--margin", "-m", type=float, default=5.0,
        help="Margin (Å) to add to each box dimension (default: 5.0)"
    )
    p.add_argument(
        "--output", "-o", default="dock_boxes.csv",
        help="Path to output CSV file (default: dock_boxes.csv)"
    )
    args = p.parse_args()

    # Gather all .pdb files
    pdb_files = sorted([
        f for f in os.listdir(args.dir)
        if f.lower().endswith(".pdb")
    ])
    if not pdb_files:
        print(f"No PDB files found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    # Open CSV for writing
    with open(args.output, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "filename",
            "center_x","center_y","center_z",
            "size_x","size_y","size_z"
        ])

        # Loop with progress bar
        for pdb_name in tqdm(pdb_files, desc="Processing complexes"):
            path = os.path.join(args.dir, pdb_name)
            try:
                center, size = compute_box_coords(path, args.chain, args.margin)
                writer.writerow([
                    pdb_name,
                    f"{center[0]:.3f}", f"{center[1]:.3f}", f"{center[2]:.3f}",
                    f"{size[0]:.3f}",   f"{size[1]:.3f}",   f"{size[2]:.3f}"
                ])
            except Exception as e:
                # log and continue
                print(f"Warning: {pdb_name}: {e}", file=sys.stderr)

    print(f"\n✅ Done! Results written to {args.output}")


if __name__ == "__main__":
    main()
