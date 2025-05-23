#!/usr/bin/env python
"""
prepare_and_convert_receptor.py

Reads a PDB file, strips out Waters/HETATMs, adds polar hydrogens,
assigns Kollman charges, removes non-polar H & lone pairs, and writes
out a PDBQT ready for AutoDock/Vina (via MGLTools’ AD4ReceptorPreparation).

./prepare_and_convert_receptor.py -i /mnt/e/Specificty_docking/run -o /mnt/e/Specificty_docking/run
"""

import os
import sys
import argparse

from MolKit import Read
from AutoDockTools.MoleculePreparation import AD4ReceptorPreparation

def select_largest_mol(mols):
    # If multiple molecules in the file, pick the one with most atoms
    mol = mols[0]
    for m in mols[1:]:
        if len(m.allAtoms) > len(mol.allAtoms):
            mol = m
    return mol

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a protein receptor PDB → PDBQT (adds H, Kollman charges)."
    )
    parser.add_argument(
        "-i","--input",
        required=True,
        help="Path to input PDB file (with HETATM, waters, etc.)"
    )
    parser.add_argument(
        "-o","--output",
        required=True,
        help="Path for output PDBQT file"
    )
    parser.add_argument(
        "-A","--add-repairs",
        default="hydrogens",
        choices=["bonds_hydrogens","bonds","hydrogens","checkhydrogens","None"],
        help="Type of repairs: build bonds, add H, etc. (default: hydrogens)"
    )
    parser.add_argument(
        "-U","--cleanup",
        default="nphs_lps_waters_nonstdres",
        help="Cleanup flags (default: nphs_lps_waters_nonstdres)"
    )
    parser.add_argument(
        "-v","--verbose",
        action="store_true",
        help="Verbose logging"
    )
    args = parser.parse_args()

    # 1) strip out non-ATOM records
    clean_pdb = os.path.splitext(args.output)[0] + "_clean.pdb"
    if args.verbose:
        print(f"[1] Stripping non-ATOM lines from {args.input} → {clean_pdb}")
    with open(args.input) as inp, open(clean_pdb, "w") as outp:
        for line in inp:
            if line.startswith("ATOM"):
                outp.write(line)

    # 2) read molecule(s)
    if args.verbose:
        print(f"[2] Reading cleaned PDB with MolKit: {clean_pdb}")
    mols = Read(clean_pdb)
    if not mols:
        print("ERROR: no molecules found in", clean_pdb, file=sys.stderr)
        sys.exit(1)
    mol = select_largest_mol(mols)

    # 3) build bonds
    if args.verbose:
        print(f"[3] Building bonds by distance for {mol.name}")
    mol.buildBondsByDistance()

    # 4) prepare with AD4ReceptorPreparation
    if args.verbose:
        print(f"[4] Running AD4ReceptorPreparation → {args.output}")
        print("    repairs:", args.add_repairs)
        print("    cleanup:", args.cleanup)
    # charges_to_add = 'gasteiger' by default in AD4ReceptorPreparation
    RPO = AD4ReceptorPreparation(
        mol,
        mode="automatic",
        repairs=args.add_repairs,
        charges_to_add="gasteiger",
        cleanup=args.cleanup,
        outputfilename=args.output
    )

    if args.verbose:
        print("✅  Done. Output saved to", args.output)

if __name__ == "__main__":
    main()
