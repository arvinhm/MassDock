#!/usr/bin/env python3
import os
import shutil
import argparse
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(
        description="Copy only those .pdbqt files whose base names appear in a CSV into a new folder"
    )
    p.add_argument(
        "--csv", "-c", required=True,
        help="Path to your CSV file (must have a column named 'filename')"
    )
    p.add_argument(
        "--input-dir", "-i", required=True,
        help="Directory containing the original .pdbqt files"
    )
    p.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory where matching files will be copied (will be created if needed)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1. Load the list of names to keep
    df = pd.read_csv(args.csv)
    protein_ids = set(df['filename'].astype(str).tolist())

    # 2. Make sure the output folder exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Iterate through all files in the input directory
    copied = 0
    for fname in os.listdir(args.input_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() == ".pdbqt" and base in protein_ids:
            src = os.path.join(args.input_dir, fname)
            dst = os.path.join(args.output_dir, fname)
            shutil.copy2(src, dst)
            copied += 1

    print(f"Done. {copied} files copied to '{args.output_dir}'.")

if __name__ == "__main__":
    main()
