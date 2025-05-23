#!/usr/bin/env python3
"""
batch_convert_ligands.py

Batch-convert all .pdb → .pdbqt using MGLTools’ prepare_ligand4.py via pythonsh.
Ligands with >150 atoms or >43 rotatable bonds are skipped.
Parallelized for speed.

./batch_convert_ligands.py   -i  /mnt/e/Specificty_docking/4AA_peptide_sequences  -o /mnt/e/mnt/e/Specificty_docking_pdbqt/ -w 32

"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# external dependency: RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors

# hard-coded limits
MAX_ATOMS    = 150
MAX_ROTBONDS = 43

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a directory of PDBs → PDBQTs via pythonsh + prepare_ligand4.py"
    )
    p.add_argument("-i", "--in-dir",    required=True, help="Directory containing .pdb files")
    p.add_argument("-o", "--out-dir",   default="ligands_pdbqt", help="Where to write .pdbqt files")
    p.add_argument("-w", "--workers",   type=int, default=os.cpu_count(),
                   help="How many parallel workers to use")
    return p.parse_args()

def find_pythonsh():
    for name in ("pythonsh", "python2.7", "python2"):
        path = shutil.which(name)
        if path:
            return path
    return None

def count_atoms(path):
    # super-fast, just count lines
    with open(path, "r") as f:
        return sum(1 for l in f if l.startswith(("ATOM","HETATM")))

def count_rotatable(path):
    mol = Chem.MolFromPDBFile(str(path), removeHs=False)
    if mol is None:
        return 0
    return Descriptors.NumRotatableBonds(mol)

def needs_conversion(pdb_path):
    """Return (skip_reason or None)"""
    atoms = count_atoms(pdb_path)
    if atoms > MAX_ATOMS:
        return f"{atoms} atoms > {MAX_ATOMS}"
    rots = count_rotatable(pdb_path)
    if rots > MAX_ROTBONDS:
        return f"{rots} rot bonds > {MAX_ROTBONDS}"
    return None

def convert_one(args):
    pdb_path, out_dir, python2, prep = args
    stem = pdb_path.stem
    out_pdbqt = out_dir / f"{stem}.pdbqt"
    cmd = [
        python2, prep,
        "-l", str(pdb_path),
        "-o", str(out_pdbqt),
        "-v", "-A", "bonds_hydrogens"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_pdbqt.exists():
        err = (proc.stderr or proc.stdout).strip()
        return stem, False, err
    return stem, True, None

def main():
    args = parse_args()
    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    python2 = find_pythonsh()
    if not python2:
        sys.exit("❌ Could not find pythonsh/python2 on your PATH")

    here = Path(__file__).resolve().parent
    prep = shutil.which("prepare_ligand4.py") or str(here/"prepare_ligand4.py")
    if not os.path.isfile(prep):
        sys.exit("❌ prepare_ligand4.py not found on PATH or script folder")

    pdbs = sorted(in_dir.glob("*.pdb"))
    if not pdbs:
        sys.exit(f"❌ No .pdb files found in {in_dir!r}")

    # 1) Filter in parallel
    to_convert = []
    skipped = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = { pool.submit(needs_conversion, p): p for p in pdbs }
        for fut in tqdm(as_completed(futures), total=len(pdbs), desc="Filtering", unit="file"):
            reason = fut.result()
            p = futures[fut]
            if reason:
                skipped.append((p.name, reason))
            else:
                to_convert.append(p)

    # report skips
    for name, why in skipped:
        print(f"→ Skipped {name}: {why}", file=sys.stderr)

    # 2) Convert the rest in parallel
    results = {"ok": [], "fail": []}
    args_list = [
        (p, out_dir, python2, prep)
        for p in to_convert
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = { pool.submit(convert_one, a): a[0] for a in args_list }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Converting", unit="file"):
            stem = futures[fut].stem
            name, ok, err = fut.result()
            if ok:
                results["ok"].append(name)
            else:
                results["fail"].append((name, err))

    # 3) Summarize
    print()
    print(f"✅  Done.  Converted: {len(results['ok'])}")
    if results["fail"]:
        print(f"⚠️  Failed: {len(results['fail'])}")
        for name, err in results["fail"]:
            print(f"   {name} → {err}")
    if skipped:
        print(f"🚫  Skipped: {len(skipped)}")

if __name__ == "__main__":
    main()
