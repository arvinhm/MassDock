#!/usr/bin/env python3
"""
batch_export_docking_scores_excel.py

Scan a directory of mcdock results, extract docking_score tags from each SDF,
and produce an Excel workbook with two sheets:

  - 'All Scores': a matrix of ligands (rows) vs. receptors (columns) with each docking score
  - 'Best Peptide': best score per peptide (prefix before first underscore) for each receptor

Usage:
    python batch_export_docking_scores_excel.py \
      --input_dir ./mcdock_results \
      --output_excel docking_summary.xlsx
"""
import os
import argparse
from rdkit import Chem
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Summarize mcdock docking scores in Excel"
    )
    parser.add_argument(
        '--input_dir', '-i', required=True,
        help='Directory containing receptor subfolders of SDF results'
    )
    parser.add_argument(
        '--output_excel', '-o', required=True,
        help='Path to write the Excel summary file'
    )
    args = parser.parse_args()

    records = []
    # Walk through each receptor directory
    for receptor in sorted(os.listdir(args.input_dir)):
        receptor_dir = os.path.join(args.input_dir, receptor)
        if not os.path.isdir(receptor_dir):
            continue
        # Find all .sdf under any subfolder
        for root, _, files in os.walk(receptor_dir):
            for fname in files:
                if not fname.lower().endswith('.sdf'):
                    continue
                path = os.path.join(root, fname)
                for mol in Chem.SDMolSupplier(path, removeHs=False):
                    if mol is None or not mol.HasProp('docking_score'):
                        continue
                    try:
                        score = float(mol.GetProp('docking_score'))
                    except ValueError:
                        continue
                    ligand = mol.GetProp('filename') if mol.HasProp('filename') else fname
                    peptide = ligand.split('_', 1)[0]
                    records.append({
                        'ligand': ligand,
                        'peptide': peptide,
                        'receptor': receptor,
                        'docking_score': score
                    })

    if not records:
        print(f"No docking scores found under {args.input_dir}")
        return

    df = pd.DataFrame(records)

    # 1) All Scores: ligand × receptor matrix
    df_all = df.pivot_table(
        index='ligand',
        columns='receptor',
        values='docking_score',
        aggfunc='min'
    )

    # 2) Best Peptide: for each peptide & receptor, the best score among its ligands
    df_pep = df.groupby(['peptide','receptor'])['docking_score'].min().reset_index()
    df_best_pep = df_pep.pivot_table(
        index='peptide',
        columns='receptor',
        values='docking_score'
    )

    # Write to Excel
    with pd.ExcelWriter(args.output_excel, engine='openpyxl') as writer:
        df_all.to_excel(writer, sheet_name='All Scores')
        df_best_pep.to_excel(writer, sheet_name='Best Peptide')

    print(f"Excel workbook saved to {args.output_excel} with sheets: All Scores, Best Peptide")

if __name__ == '__main__':
    main()
