#!/usr/bin/env pythonsh
# -*- coding: utf-8 -*-
"""
batch_prepare_receptors.py

Batch-convert all PDB receptor files in a directory into PDBQT,
adding polar hydrogens, assigning Kollman+Gasteiger charges,
removing non-polar H, lone pairs, waters and non-standard residues.
If any file fails during prep (e.g. in addHydrogens), it will be skipped
with a warning rather than crashing the entire job.
"""

from __future__ import print_function
import os
import sys
import argparse
import tempfile

from MolKit import Read
from AutoDockTools.MoleculePreparation import AD4ReceptorPreparation

def select_largest_mol(mols):
    mol = mols[0]
    for m in mols[1:]:
        if len(m.allAtoms) > len(mol.allAtoms):
            mol = m
    return mol

def prepare_one(input_pdb, output_pdbqt, repairs, cleanup, verbose=False):
    # 1) Strip non-ATOM lines to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    clean_path = tmp.name
    if verbose:
        print("[clean] {} -> {}".format(input_pdb, clean_path))
    with open(input_pdb) as inp:
        for line in inp:
            if line.startswith("ATOM"):
                tmp.write(line)
    tmp.close()

    # 2) Read molecules (cast to list to avoid MolKit slicing bugs)
    mols = list(Read(clean_path))
    if not mols:
        os.unlink(clean_path)
        print("Warning: no molecules found in {}".format(clean_path),
              file=sys.stderr)
        return

    mol = select_largest_mol(mols)

    # 3) Build bonds
    if verbose:
        print("[bonds] Building bonds for {}".format(mol.name))
    mol.buildBondsByDistance()

    # 4) Prepare receptor, with error handling
    if verbose:
        print("[prep] -> {}  (repairs={}, cleanup={})"
              .format(output_pdbqt, repairs, cleanup))
    try:
        AD4ReceptorPreparation(
            mol,
            mode="automatic",
            repairs=repairs,
            charges_to_add="gasteiger",
            cleanup=cleanup,
            outputfilename=output_pdbqt
        )
    except Exception as e:
        # print full traceback if verbose, else just warning
        if verbose:
            import traceback
            traceback.print_exc()
        print("Warning: failed to prepare {}: {}"
              .format(input_pdb, e), file=sys.stderr)
        os.unlink(clean_path)
        # remove any partial output file
        if os.path.exists(output_pdbqt):
            try:
                os.remove(output_pdbqt)
            except:
                pass
        return

    # 5) Cleanup temp file
    os.unlink(clean_path)

def main():
    parser = argparse.ArgumentParser(
        description="Batch prepare receptors to PDBQT."
    )
    parser.add_argument(
        "-i", "--in-dir", required=True,
        help="Folder of .pdb receptor files"
    )
    parser.add_argument(
        "-o", "--out-dir", required=True,
        help="Folder to write .pdbqt files"
    )
    parser.add_argument(
        "--repairs", default="hydrogens",
        help="Repairs to make (default: hydrogens)"
    )
    parser.add_argument(
        "--cleanup", default="nphs_lps_waters_nonstdres",
        help="Cleanup flags (default: nphs_lps_waters_nonstdres)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging"
    )
    args = parser.parse_args()

    # Verify input directory
    if not os.path.isdir(args.in_dir):
        sys.exit("Error: input dir not found: {}".format(args.in_dir))
    # Create output directory if needed
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    pdbs = sorted(
        f for f in os.listdir(args.in_dir)
        if f.lower().endswith(".pdb")
    )
    if not pdbs:
        sys.exit("No .pdb files found in {}".format(args.in_dir))

    total = len(pdbs)
    for idx, fname in enumerate(pdbs, start=1):
        print("Processing {}/{}: {}".format(idx, total, fname))
        inp = os.path.join(args.in_dir, fname)
        base = os.path.splitext(fname)[0]
        out = os.path.join(args.out_dir, base + ".pdbqt")
        prepare_one(inp, out, args.repairs, args.cleanup, args.verbose)

    print("\nDone! Prepared {} receptors into {}/".format(total, args.out_dir))

if __name__ == "__main__":
    main()
