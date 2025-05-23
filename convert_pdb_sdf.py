
#!/usr/bin/env python3
"""
batch_confgen_obabel.py

Batch‐generate ligand conformers using Open Babel for all .sdf files in an input directory,
utilizing multiple CPU cores to parallelize the work.

Usage:
    python batch_confgen_obabel.py --input_dir ./ligands_sdf --output_dir ./confgen_results --cores 32
"""

import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def generate_conformers(sdf: str, input_dir: str, output_dir: str, nconf: int, rmsd: float):
    base = os.path.splitext(sdf)[0]
    in_path = os.path.join(input_dir, sdf)
    out_name = f"{base}_conf.sdf"
    out_path = os.path.join(output_dir, out_name)

    print(f"[CORE {os.getpid()}] Generating conformers for {sdf} → {out_name}")
    cmd = [
        'obabel',
        '-isdf', in_path,
        '-osdf',
        '-O', out_path,
        '--conformer',
        '--nconf', str(nconf),
        '--writeconformers',
        '--unique',
        '--rmthresh', str(rmsd)
    ]
    subprocess.run(cmd, check=True)
    return out_name


def main():
    parser = argparse.ArgumentParser(description="Parallel conformer generation via Open Babel")
    parser.add_argument(
        '--input_dir', '-i', required=True,
        help='Directory containing input .sdf files'
    )
    parser.add_argument(
        '--output_dir', '-o', required=True,
        help='Directory to write _conf.sdf output files'
    )
    parser.add_argument(
        '--nconf', '-n', type=int, default=200,
        help='Maximum number of conformers per molecule (default: 200)'
    )
    parser.add_argument(
        '--rmsd', '-r', type=float, default=0.3,
        help='RMSD threshold to prune conformers (default: 0.3 Å)'
    )
    parser.add_argument(
        '--cores', '-c', type=int, default=multiprocessing.cpu_count(),
        help='Number of parallel workers (default: all cores)'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.sdf')]
    if not sdf_files:
        print(f"No SDF files found in {args.input_dir}")
        return

    # Prepare parameters for each job
    tasks = [ (sdf, args.input_dir, args.output_dir, args.nconf, args.rmsd) for sdf in sdf_files ]

    print(f"Starting conformer generation on {len(tasks)} files using {args.cores} cores...")
    results = []
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        futures = [ executor.submit(generate_conformers, *task) for task in tasks ]
        for future in as_completed(futures):
            try:
                out_name = future.result()
                print(f"Completed {out_name}")
            except Exception as e:
                print(f"Error: {e}")

    print(f"All conformer jobs done. Outputs in {args.output_dir}")

if __name__ == '__main__':
    main()
