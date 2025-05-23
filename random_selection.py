#!/usr/bin/env python3
r"""
random_selection.py

Instantly pick N .pdbqt files from a single folder and hard-link them
into a destination folder, with a live one-line progress bar.

python3 random_selection.py --source-dir ./ligands_pdbqt_collab --dest-dir ./random_database_peptide --num 5000 --seed 42
"""

import os
import sys
import random
from pathlib import Path

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Randomly select and hard-link .pdbqt files with minimal overhead"
    )
    p.add_argument("-s", "--source-dir", required=True,
                   help="Folder containing .pdbqt files")
    p.add_argument("-d", "--dest-dir",   required=True,
                   help="Folder to receive hard links")
    p.add_argument("-n", "--num", type=int, default=5000,
                   help="Number of files to pick (default: 5000)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (optional)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    src = Path(args.source_dir)
    dst = Path(args.dest_dir)
    if not src.is_dir():
        print(f"Error: source directory '{src}' not found.", file=sys.stderr)
        sys.exit(1)
    dst.mkdir(parents=True, exist_ok=True)

    # FAST listing by name only (no stat calls)
    print(f"Listing entries in '{src}' …", flush=True)
    all_names = os.listdir(src)
    pdbqt_names = [name for name in all_names if name.lower().endswith('.pdbqt')]
    total = len(pdbqt_names)
    print(f"Found {total} .pdbqt files.", flush=True)

    if total < args.num:
        print(f"Error: only {total} files available but {args.num} requested.",
              file=sys.stderr)
        sys.exit(1)

    selected = random.sample(pdbqt_names, args.num)
    print(f"Hard-linking {len(selected)} files into '{dst}' …", flush=True)

    # one-line progress bar
    bar_len = 40
    for i, name in enumerate(selected, start=1):
        src_path = src / name
        dst_path = dst / name
        try:
            os.link(src_path, dst_path)
        except OSError as e:
            print(f"\nWarning: could not hard-link {name}: {e}", file=sys.stderr)
        # redraw bar
        frac = i / args.num
        filled = int(bar_len * frac)
        bar = "=" * filled + " " * (bar_len - filled)
        print(f"\r[{bar}] {i}/{args.num}", end="", flush=True)

    print("\nDone.")

if __name__ == "__main__":
    main()
