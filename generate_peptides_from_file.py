#!/usr/bin/env python3
"""
generate_peptides_from_file.py

Read peptide sequences from a text file, filter by atom/rot-bond limits,
and write capped 3D PDBs via pyPept in parallel.

./generate_peptides_from_file.py --input-file ./4AA.txt --max-atoms 150 --pdb-dir ./4AA_peptide_sequences

"""

import argparse
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from pyPept.sequence import Sequence, correct_pdb_atoms
from pyPept.molecule import Molecule
from pyPept.conformer import Conformer, SecStructPredictor


def read_sequences(input_file):
    """Read non-empty, stripped sequences (one per line) from a text file."""
    sequences = []
    with open(input_file, 'r') as f:
        for line in f:
            seq = line.strip()
            if seq:
                sequences.append(seq)
    return sequences


def filter_sequences(seq_list, max_atoms, max_rot_bonds):
    """
    From seq_list, keep only those whose RDKit Mol:
      - can be built from sequence
      - has ≤ max_atoms atoms
      - has ≤ max_rot_bonds rotatable bonds
    Returns two lists: (valid_seqs, invalid_info)
    where invalid_info is list of (seq, reason).
    """
    valid = []
    invalid = []
    for seq in tqdm(seq_list, desc="Filtering sequences", unit="seq"):
        mol = Chem.MolFromSequence(seq)
        if mol is None:
            invalid.append((seq, "invalid SMILES/sequence"))
            continue
        natoms = mol.GetNumAtoms()
        nrot  = Descriptors.NumRotatableBonds(mol)
        if natoms > max_atoms:
            invalid.append((seq, f"{natoms} atoms > {max_atoms}"))
            continue
        if nrot > max_rot_bonds:
            invalid.append((seq, f"{nrot} rot bonds > {max_rot_bonds}"))
            continue
        valid.append(seq)
    return valid, invalid


def build_and_write(seq, pdb_dir):
    """
    Worker: build capped peptide conformer and write PDB.
    Returns (seq, success, error_msg_or_None).
    """
    biln = "ac-" + "-".join(seq) + "-am"
    try:
        seq_obj = Sequence(biln)
        seq_obj = correct_pdb_atoms(seq_obj)
        mol_obj = Molecule(seq_obj)
        romol   = mol_obj.get_molecule(fmt='ROMol')
        fasta   = Conformer.get_peptide(biln)
        ss      = SecStructPredictor.predict_active_ss(fasta)
        romol_conf = Conformer.generate_conformer(romol, ss, generate_pdb=False)
        pdb_block  = Chem.MolToPDBBlock(romol_conf)
        out_path = Path(pdb_dir) / f"{seq}.pdb"
        with open(out_path, "w") as wf:
            wf.write(pdb_block)
        return seq, True, None
    except Exception as e:
        return seq, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Generate capped peptides from a list, filter by size, write 3D PDBs"
    )
    parser.add_argument(
        "--input-file", required=True,
        help="Path to text file with one peptide sequence per line"
    )
    parser.add_argument(
        "--max-atoms", type=int, default=150,
        help="Maximum atom count per peptide"
    )
    parser.add_argument(
        "--max-rot-bonds", type=int, default=43,
        help="Maximum number of rotatable bonds per peptide"
    )
    parser.add_argument(
        "--pdb-dir", type=str, default="peptide_pdbs",
        help="Output folder for .pdb files"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel workers"
    )
    args = parser.parse_args()

    # 1) Read and filter sequences
    seqs = read_sequences(args.input_file)
    print(f"Read {len(seqs)} sequences from '{args.input_file}'")
    valid_seqs, invalid_info = filter_sequences(
        seqs, args.max_atoms, args.max_rot_bonds
    )
    print(f"{len(valid_seqs)} valid; {len(invalid_info)} invalid/skipped")
    if invalid_info:
        print("First 10 invalid entries:")
        for seq, reason in invalid_info[:10]:
            print(f"  {seq} → {reason}")
        if len(invalid_info) > 10:
            print(f"  ... and {len(invalid_info)-10} more")

    # 2) Prepare output directory
    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(valid_seqs)} PDBs into '{pdb_dir}' using {args.workers} workers…")

    # 3) Build and write in parallel
    results_ok   = []
    results_fail = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_and_write, seq, pdb_dir): seq
                   for seq in valid_seqs}
        for fut in tqdm(as_completed(futures),
                        total=len(valid_seqs),
                        desc="Writing PDBs", unit="pep"):
            seq, success, err = fut.result()
            if success:
                results_ok.append(seq)
            else:
                results_fail.append((seq, err))

    # 4) Summary
    print(f"\n✅  Done!  PDBs written: {len(results_ok)}")
    if results_fail:
        print(f"⚠️  Failures: {len(results_fail)}")
        for seq, err in results_fail[:10]:
            print(f"   {seq} → {err}")
        if len(results_fail) > 10:
            print(f"   …and {len(results_fail)-10} more")


if __name__ == "__main__":
    main()
