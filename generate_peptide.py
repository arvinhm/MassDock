#!/usr/bin/env python3
"""
generate_peptide.py

Generate random capped peptides under atom/rot-bond limits and
write 3D PDBs via pyPept in parallel.
"""

import argparse
import random
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from pyPept.sequence import Sequence, correct_pdb_atoms
from pyPept.molecule  import Molecule
from pyPept.conformer import Conformer, SecStructPredictor

def random_peptides(residues, atom_counts, max_atoms, max_rot_bonds,
                    num_peptides, min_length=3):
    """
    Generate `num_peptides` unique random one-letter peptides whose
    atom count ≤ max_atoms, rotatable bonds ≤ max_rot_bonds,
    and length ≥ min_length.
    """
    min_atoms = min(atom_counts.values())
    max_len   = max_atoms // min_atoms
    if max_len < min_length:
        raise ValueError(f"max_atoms too small to build peptides of length {min_length}")

    peptides = set()
    pbar = tqdm(total=num_peptides, desc="Generating peptides", unit="pep")
    while len(peptides) < num_peptides:
        length = random.randint(min_length, max_len)
        seq = ''.join(random.choices(residues, k=length))
        mol = Chem.MolFromSequence(seq)
        if not mol:
            continue
        if (mol.GetNumAtoms() <= max_atoms and
            Descriptors.NumRotatableBonds(mol) <= max_rot_bonds):
            if seq not in peptides:
                peptides.add(seq)
                pbar.update(1)
    pbar.close()
    return list(peptides)

def build_and_write(seq, pdb_dir):
    """
    Worker function: build capped peptide conformer and write out PDB.
    Returns (seq, True, None) on success or (seq, False, error_msg) on failure.
    """
    biln = "ac-" + "-".join(seq) + "-am"
    try:
        seq_obj = Sequence(biln)
        seq_obj = correct_pdb_atoms(seq_obj)

        mol_obj = Molecule(seq_obj)
        romol   = mol_obj.get_molecule(fmt='ROMol')

        fasta = Conformer.get_peptide(biln)
        ss    = SecStructPredictor.predict_active_ss(fasta)

        romol_conf = Conformer.generate_conformer(romol, ss, generate_pdb=False)
        pdb_block  = Chem.MolToPDBBlock(romol_conf)

        out_path = Path(pdb_dir) / f"{seq}.pdb"
        with open(out_path, "w") as f:
            f.write(pdb_block)

        return seq, True, None

    except Exception as e:
        return seq, False, str(e)

def main():
    parser = argparse.ArgumentParser(
        description="Generate random capped peptides and write 3D PDBs in parallel"
    )
    parser.add_argument("--max-atoms",     type=int,   default=150,
                        help="Max atom count per peptide")
    parser.add_argument("--max-rot-bonds", type=int,   default=43,
                        help="Max rotatable bonds per peptide")
    parser.add_argument("--num-peptides",  type=int,   default=1000000,
                        help="How many random peptides to generate")
    parser.add_argument("--min-length",    type=int,   default=3,
                        help="Minimum peptide length")
    parser.add_argument("--pdb-dir",       type=str,   default="peptide_pdbs",
                        help="Output folder for .pdb files")
    parser.add_argument("--workers",       type=int,   default=os.cpu_count(),
                        help="Number of parallel workers")
    args = parser.parse_args()

    residues    = list("ARNDCEQGHILKMFPSTWYV")
    atom_counts = {aa: Chem.MolFromSequence(aa).GetNumAtoms()
                   for aa in residues}

    print(f"Generating {args.num_peptides} random peptides "
          f"(length ≥{args.min_length}, ≤{args.max_atoms} atoms, "
          f"≤{args.max_rot_bonds} rot bonds)…")
    peptides = random_peptides(
        residues, atom_counts,
        args.max_atoms, args.max_rot_bonds,
        args.num_peptides, args.min_length
    )

    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting capped 3D PDBs into '{pdb_dir}' using {args.workers} workers…")
    results = {"ok": [], "fail": []}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_seq = {
            pool.submit(build_and_write, seq, pdb_dir): seq
            for seq in peptides
        }
        for fut in tqdm(as_completed(future_to_seq),
                        total=len(peptides),
                        desc="Writing PDBs", unit="pep"):
            seq, ok, err = fut.result()
            if ok:
                results["ok"].append(seq)
            else:
                results["fail"].append((seq, err))

    print(f"\n✅  Done!  PDBs written: {len(results['ok'])}")
    if results["fail"]:
        print(f"⚠️  Failures: {len(results['fail'])}")
        for seq, err in results["fail"][:10]:
            print(f"   {seq} → {err}")
        if len(results["fail"]) > 10:
            print(f"   …and {len(results['fail'])-10} more")

if __name__ == "__main__":
    main()
