#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter unique peptide sequences (no 'X') and copy corresponding PDBQT files."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to CSV file containing columns 'file_name' and 'peptide_sequence'"
    )
    parser.add_argument(
        "--src-dir", required=True,
        help="Directory where the original .pdbqt files are located"
    )
    parser.add_argument(
        "--dst-dir", required=True,
        help="Directory to copy unique .pdbqt files into"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.dst_dir, exist_ok=True)

    # Read the CSV
    df = pd.read_csv(args.csv, sep=None, engine='python')

    # 1) Remove any peptides containing 'X'
    df = df[~df['peptide_sequence'].str.contains('X', regex=False)]

    # 2) Keep only the first occurrence of each unique sequence
    unique_df = df.drop_duplicates(subset=["peptide_sequence"], keep="first")

    copied = 0
    for _, row in unique_df.iterrows():
        src_path = os.path.join(args.src_dir, row["file_name"])
        dst_path = os.path.join(args.dst_dir, row["file_name"])
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            print(f"Warning: source file not found: {src_path}")

    print(f"Copied {copied} unique PDBQT files (without 'X') to '{args.dst_dir}'")

if __name__ == "__main__":
    main()
