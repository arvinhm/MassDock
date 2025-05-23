#!/usr/bin/env python3
import os
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter out PDBQT files containing non-standard peptide atoms."
    )
    parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Directory containing your .pdbqt peptide files"
    )
    parser.add_argument(
        "--good-dir", "-g", required=True,
        help="Where to copy valid PDBQT files"
    )
    parser.add_argument(
        "--bad-dir", "-b", required=True,
        help="Where to move PDBQT files with non-peptide atoms"
    )
    parser.add_argument(
        "--allowed", "-a", default="C,H,N,O,S",
        help="Comma-separated list of allowed elements (default: C,H,N,O,S)"
    )
    return parser.parse_args()

def is_peptide_only(path, allowed_elements):
    """
    Returns True if every ATOM/HETATM line in `path` has an element
    in allowed_elements.  No early returns inside the loop.
    """
    valid = True
    with open(path, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                # PDBQT: element is in cols 77–78 (0-based 76:78)
                elem = line[76:78].strip().upper()
                if not elem:
                    # fallback: first letter(s) of atom name (cols 13–16)
                    name = line[12:16].strip()
                    # take first alpha character
                    for ch in name:
                        if ch.isalpha():
                            elem = ch.upper()
                            break
                if elem not in allowed_elements:
                    valid = False
                    break
    return valid

def main():
    args = parse_args()
    allowed = set(el.strip().upper() for el in args.allowed.split(","))

    os.makedirs(args.good_dir, exist_ok=True)
    os.makedirs(args.bad_dir, exist_ok=True)

    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(".pdbqt"):
            continue
        src = os.path.join(args.input_dir, fname)
        if is_peptide_only(src, allowed):
            dst = os.path.join(args.good_dir, fname)
            shutil.copy(src, dst)
            print(f"[OK]   {fname} → kept")
        else:
            bad = os.path.join(args.bad_dir, fname)
            shutil.move(src, bad)
            print(f"[DROP] {fname} → moved to bad-dir")

if __name__ == "__main__":
    main()
