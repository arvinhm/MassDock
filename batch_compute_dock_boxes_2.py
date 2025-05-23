#!/usr/bin/env python3
"""
batch_compute_dock_boxes_2.py

Extract a docking box from a protein‑peptide complex (chain A = receptor,
chain B = docked peptide/ligand) and write:

protein,site_rank,score,center_x,center_y,center_z,
box_x,box_y,box_z,size_x,size_y,size_z,filepath

• box_*  : Å    = span(chain B) + 2*margin
• size_* : grid points = ceil(box_*/spacing) + 1    (AutoDock/Vina style)

Example
-------
python batch_compute_dock_boxes_2.py \
       --dir /mnt/e/Datasbe_protein_peptide/complex \
       --margin 6.0 \
       --spacing 0.7 \
       --output peptide_boxes.csv
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm


def compute_box(chain_atoms: np.ndarray, margin: float, spacing: float):
    """
    Return  (center_x/y/z, box_x/y/z, size_x/y/z)
    """
    mn = chain_atoms.min(axis=0)          # Å
    mx = chain_atoms.max(axis=0)

    # geometric centre of the bounding box
    center = (mn + mx) / 2.0

    # physical box size (Å) = span + both‑side margin
    box = (mx - mn) + 2.0 * margin

    # grid‑point counts  Nx = ceil(box_x/spacing)+1
    size = np.ceil(box / spacing).astype(int) + 1

    return center, box, size


def atoms_of_chain(pdb_path: Path, chain_id: str) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = struct[0]

    if chain_id not in model:
        raise ValueError(f"chain '{chain_id}' not found")

    coords = np.array([a.get_coord() for a in model[chain_id].get_atoms()
                       if a.element != 'H'])   # ignore hydrogens
    if coords.size == 0:
        raise ValueError(f"no atoms in chain '{chain_id}'")

    return coords


def main():
    ap = argparse.ArgumentParser(
        description="Compute docking grid boxes from chain‑B peptides"
    )
    ap.add_argument("-d", "--dir", required=True,
                    help="Folder with PDB (or PDB.gz/mmCIF) files")
    ap.add_argument("-c", "--chain", default="B",
                    help="Ligand chain ID (default: B)")
    ap.add_argument("-m", "--margin", type=float, default=6.0,
                    help="Å margin added around ligand (default: 6.0)")
    ap.add_argument("-s", "--spacing", type=float, default=0.7,
                    help="Grid spacing Å (default: 0.7)")
    ap.add_argument("-o", "--output", default="dock_boxes.csv",
                    help="Output CSV (default: dock_boxes.csv)")
    args = ap.parse_args()

    pdb_files = sorted(Path(args.dir).glob("*.[Pp][Dd][Bb]*"))   # .pdb / .pdb.gz
    if not pdb_files:
        sys.exit(f"No PDB files found under {args.dir}")

    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "protein", "site_rank", "score",
            "center_x", "center_y", "center_z",
            "box_x", "box_y", "box_z",
            "size_x", "size_y", "size_z",
            "filepath"
        ])

        for pdb_path in tqdm(pdb_files, desc="processing"):
            try:
                coords = atoms_of_chain(pdb_path, args.chain)
                center, box, size = compute_box(coords, args.margin,
                                                args.spacing)

                writer.writerow([
                    pdb_path.stem,           # protein
                    1,                       # site_rank (single ligand)
                    "",                      # score  (N/A)
                    *(f"{x:.3f}" for x in center),
                    *(f"{x:.3f}" for x in box),
                    *size,
                    str(pdb_path)
                ])

            except Exception as exc:
                print(f"⚠  {pdb_path.name}: {exc}", file=sys.stderr)

    print(f"✅  Finished. Results → {args.output}")


if __name__ == "__main__":
    main()
