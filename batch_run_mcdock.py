#!/usr/bin/env python3
"""
batch_run_mcdock.py

Batch-run MultiConfDock (mcdock) across multiple receptors and ligands in chunks.

- Reads receptor PDBs named {protein}_meeko.prep.pdb from --receptor_dir
- Reads center/size parameters from a CSV (--receptor_csv)
- Reads all .sdf ligand files in --ligand_dir
- Splits ligand list into batches of N (--batch_size, default 5000)
- For each receptor, runs mcdock on each ligand batch sequentially
- Outputs results under --output_dir/{protein}/batch_{batch_index}

Usage:
    python batch_run_mcdock.py \
      --receptor_dir ./prepared_receptors \
      --receptor_csv receptor_params.csv \
      --ligand_dir ./ligands_sdf \
      --output_dir ./mcdock_results \
      --batch_size 5000 \
      --seed 181129
"""
import os
import csv
import argparse
import subprocess
from math import ceil
import glob


def read_receptor_params(csv_path):
    params = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['filename']
            params[name] = {
                'cx': row['center_x'], 'cy': row['center_y'], 'cz': row['center_z'],
                'sx': row['size_x'],   'sy': row['size_y'],   'sz': row['size_z']
            }
    return params


def batchify(items, batch_size):
    total = len(items)
    for i in range(0, total, batch_size):
        yield items[i:i+batch_size]


def main():
    parser = argparse.ArgumentParser(description="Batch run mcdock across receptors and ligand batches.")
    parser.add_argument('--receptor_dir', required=True,
                        help='Directory with {protein}_meeko.prep.pdb files')
    parser.add_argument('--receptor_csv', required=True,
                        help='CSV file with receptor parameters (filename, center_x, ... size_z)')
    parser.add_argument('--ligand_dir', required=True,
                        help='Directory containing ligand .sdf files')
    parser.add_argument('--output_dir', required=True,
                        help='Base directory to store mcdock results')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Number of ligands per batch')
    parser.add_argument('--seed', type=int, default=181129,
                        help='Random seed for mcdock')
    args = parser.parse_args()

    # Read receptors and parameters
    rec_params = read_receptor_params(args.receptor_csv)
    # List receptor pdb files
    pdb_files = glob.glob(os.path.join(args.receptor_dir, '*_meeko.prep.pdb'))
    # List ligand sdfs
    ligand_files = sorted(glob.glob(os.path.join(args.ligand_dir, '*.sdf')))
    if not ligand_files:
        print("No ligand .sdf files found in", args.ligand_dir)
        return

    for pdb_path in pdb_files:
        base = os.path.basename(pdb_path)
        protein = base.split('_meeko.prep.pdb')[0]
        if protein not in rec_params:
            print(f"Warning: {protein} not in CSV, skipping")
            continue
        params = rec_params[protein]
        # Output base dir for this receptor
        rec_out = os.path.join(args.output_dir, protein)
        os.makedirs(rec_out, exist_ok=True)

        # Split ligands into batches
        batches = list(batchify(ligand_files, args.batch_size))
        for idx, batch in enumerate(batches, start=1):
            batch_dir = os.path.join(rec_out, f'batch_{idx}')
            os.makedirs(batch_dir, exist_ok=True)

            ligands_arg = ','.join(batch)
            cmd = [
                'unidocktools', 'mcdock',
                '-r', pdb_path,
                '-l', ligands_arg,
                '-sd', batch_dir,
                '-cx', params['cx'], '-cy', params['cy'], '-cz', params['cz'],
                '-sx', params['sx'], '-sy', params['sy'], '-sz', params['sz'],
                '--seed', str(args.seed)
            ]
            print(f"Running {protein} batch {idx}/{len(batches)}: {len(batch)} ligands")
            subprocess.run(cmd, check=True)

    print("All batches completed.")

if __name__ == '__main__':
    main()
