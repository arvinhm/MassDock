#!/usr/bin/env python3
"""
batch_confgen_obabel_parallel.py

Parallel conformer generation for all .sdf ligands in a folder using Open Babel v3,
with a live tqdm progress bar.

Usage:
  export OBABEL_BIN=/usr/bin/obabel
  python batch_confgen_obabel_parallel.py \
    --input_dir ./4AA_peptide_sequences_sdf \
    --output_dir ./4AA_peptide_sequences_sdf_confgen \
    --nconf 200 --rmsd 0.3 --cores 32
"""
import os
import sys
import shutil
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# try to import tqdm; if unavailable, we’ll fall back to simple prints
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

def find_obabel():
    for candidate in (os.environ.get('OBABEL_BIN'),
                      '/usr/bin/obabel',
                      '/usr/local/bin/obabel',
                      shutil.which('obabel')):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    sys.stderr.write("ERROR: cannot find obabel v3+; install it or set $OBABEL_BIN\n")
    sys.exit(1)

def run_one(obabel, in_path, out_path, nconf, rmsd):
    base = os.path.basename(in_path)
    try:
        subprocess.run([
            obabel,
            '-isdf', in_path,
            '-osdf', '-O', out_path,
            '--conformer', '--nconf', str(nconf),
            '--writeconformers', '--unique',
            '--rmthresh', str(rmsd)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return base, True, None
    except subprocess.CalledProcessError as e:
        return base, False, str(e)

def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch conformer generation via Open Babel"
    )
    parser.add_argument('--input_dir', '-i', required=True,
                        help='Directory with input .sdf files')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Directory to write _conf.sdf outputs')
    parser.add_argument('--nconf', '-n', type=int, default=10,
                        help='Max conformers per molecule')
    parser.add_argument('--rmsd', '-r', type=float, default=0.3,
                        help='RMSD cutoff for pruning')
    parser.add_argument('--cores', '-c', type=int, default=multiprocessing.cpu_count(),
                        help='Number of parallel workers')
    args = parser.parse_args()

    obabel = find_obabel()
    print(f"Using Open Babel binary: {obabel}")
    os.makedirs(args.output_dir, exist_ok=True)

    sdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.sdf')]
    if not sdf_files:
        print("No .sdf files found in", args.input_dir, file=sys.stderr)
        sys.exit(1)

    tasks = []
    for sdf in sdf_files:
        in_path = os.path.join(args.input_dir, sdf)
        out_path = os.path.join(args.output_dir, f"{os.path.splitext(sdf)[0]}_conf.sdf")
        tasks.append((obabel, in_path, out_path, args.nconf, args.rmsd))

    total = len(tasks)
    print(f"Launching {total} jobs on {args.cores} cores...")

    successes = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        futures = [executor.submit(run_one, *t) for t in tasks]
        iterator = tqdm(as_completed(futures), total=total, unit="file") if USE_TQDM else as_completed(futures)
        for future in iterator:
            base, ok, err = future.result()
            if ok:
                successes += 1
            else:
                failures.append((base, err))

            if not USE_TQDM:
                # simple fallback progress
                done = successes + len(failures)
                print(f"[{done}/{total}] {'OK' if ok else 'FAIL'} {base}")

    print(f"\nDone: {successes}/{total} succeeded, {len(failures)} failed.")
    if failures:
        print("Failures:")
        for base, err in failures:
            print(f"  {base}: {err}")

if __name__ == '__main__':
    main()
