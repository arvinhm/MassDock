#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd

# Map three-letter AA codes to one-letter
three_to_one = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
    "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
    "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
}

def main():
    parser = argparse.ArgumentParser(
        description="Extract peptide sequences from PDBQT files and save to CSV."
    )
    parser.add_argument("pdbqt_dir", help="Directory containing .pdbqt files")
    parser.add_argument("output_csv", help="Path to output CSV file")
    args = parser.parse_args()

    records = []
    pattern = os.path.join(args.pdbqt_dir, "*.pdbqt")
    for filepath in glob.glob(pattern):
        seen = set()
        seq = []
        with open(filepath, "r") as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    resname = line[17:20].strip()
                    chain   = line[21].strip()
                    resnum  = line[22:26].strip()
                    key     = (chain, resnum)
                    if key not in seen:
                        seen.add(key)
                        seq.append( three_to_one.get(resname, "X") )
        peptide = "".join(seq)
        records.append({
            "file_name": os.path.basename(filepath),
            "peptide_sequence": peptide
        })

    df = pd.DataFrame(records, columns=["file_name", "peptide_sequence"])
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} records to {args.output_csv}")

if __name__ == "__main__":
    main()
