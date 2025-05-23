#!/usr/bin/env python3
"""
quickvina_cross_dock.py

Robust cross-docking script for QuickVina2-GPU-2.1 with batch GPU docking,
parallel execution, optional sampling for quick tests, and auto-detection
of output ligand filenames to avoid missing-file errors.
"""

import os
import sys
import csv
import glob
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-dock ligands to receptors with QuickVina2-GPU"
    )
    p.add_argument("--receptor-dir",    required=True,
                   help="Directory of receptor .pdbqt files")
    p.add_argument("--ligand-dir",      required=True,
                   help="Directory of ligand .pdbqt files")
    p.add_argument("--box-csv",         required=True,
                   help="CSV with columns: filename,center_x,center_y,center_z,size_x,size_y,size_z")
    p.add_argument("--output-dir",      required=True,
                   help="Base output directory for all docking results")
    p.add_argument("--binary",          default="quickvina2-gpu",
                   help="Path to QuickVina2-GPU binary")
    p.add_argument("--opencl-binary-path", default=None,
                   help="Path to OpenCL kernels (.bin) if needed")
    p.add_argument("--batch-size",      type=int, default=10000,
                   help="Max ligands per GPU batch")
    p.add_argument("--thread",          type=int, default=8000,
                   help="--thread (lanes) per QuickVina2 run")
    p.add_argument("--search-depth",    type=int, default=None,
                   help="--search_depth per QuickVina2 run")
    p.add_argument("--concurrency",     type=int, default=4,
                   help="Number of concurrent processes")
    p.add_argument("--max-receptors",   type=int, default=None,
                   help="If set, only use the first N receptors for testing")
    p.add_argument("--max-ligands",     type=int, default=None,
                   help="If set, only use the first N ligands for testing")
    return p.parse_args()


def read_box_csv(path):
    box = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            nm = os.path.splitext(row['filename'])[0]
            box[nm] = {
                'center_x': row['center_x'],
                'center_y': row['center_y'],
                'center_z': row['center_z'],
                'size_x':   row['size_x'],
                'size_y':   row['size_y'],
                'size_z':   row['size_z'],
            }
    return box


def list_pdbqt(dirpath):
    return sorted([
        fn for fn in os.listdir(dirpath)
        if fn.lower().endswith('.pdbqt')
    ])


def parse_score_from_pdbqt(pdbqt_path):
    """
    Extracts the first Vina result score from the PDBQT file.
    """
    try:
        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith('REMARK VINA RESULT:'):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        return float(parts[3])
    except FileNotFoundError:
        return None
    return None


def run_batch(receptor_path, ligand_batch, params, args, batch_id, receptor_name):
    out_subdir = os.path.join(
        args.output_dir, 'dock_outputs', receptor_name, f'batch_{batch_id}'
    )
    tmp_lig_dir = os.path.join(
        args.output_dir, 'tmp_ligands', receptor_name, f'batch_{batch_id}'
    )
    os.makedirs(out_subdir, exist_ok=True)
    os.makedirs(tmp_lig_dir, exist_ok=True)

    # prepare ligands
    for ligand in ligand_batch:
        src = os.path.join(args.ligand_dir, ligand)
        dst = os.path.join(tmp_lig_dir, ligand)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

    # build and run QuickVina2-GPU command
    cmd = [
        args.binary,
        '--receptor', receptor_path,
        '--ligand_directory', tmp_lig_dir,
        '--output_directory', out_subdir,
        '--thread', str(args.thread),
        '--center_x', params['center_x'],
        '--center_y', params['center_y'],
        '--center_z', params['center_z'],
        '--size_x',   params['size_x'],
        '--size_y',   params['size_y'],
        '--size_z',   params['size_z'],
    ]
    if args.search_depth:
        cmd += ['--search_depth', str(args.search_depth)]
    if args.opencl_binary_path:
        cmd += ['--opencl_binary_path', args.opencl_binary_path]

    log_path = os.path.join(out_subdir, 'dock.log')
    with open(log_path, 'w') as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

    # detect output pdbqt files and parse scores
    scores = {}
    for ligand in ligand_batch:
        base = os.path.splitext(ligand)[0]
        # first try exact filename
        pattern = os.path.join(out_subdir, ligand)
        candidates = glob.glob(pattern)
        if not candidates:
            # fallback to any file starting with the base name
            candidates = glob.glob(os.path.join(out_subdir, f'{base}*.pdbqt'))
        if not candidates:
            print(f'[!] Warning: no output for {ligand}')
            scores[ligand] = None
        else:
            out_file = candidates[0]
            scores[ligand] = parse_score_from_pdbqt(out_file)
    return scores


def main():
    args = parse_args()
    box = read_box_csv(args.box_csv)
    recs = list_pdbqt(args.receptor_dir)
    ligs = list_pdbqt(args.ligand_dir)

    # apply optional sampling
    if args.max_receptors is not None:
        recs = recs[:args.max_receptors]
    if args.max_ligands is not None:
        ligs = ligs[:args.max_ligands]

    if not recs or not ligs:
        sys.exit('ERROR: No receptors or ligands to process after sampling')

    results = {r: {l: None for l in ligs} for r in recs}
    tasks = []

    # build tasks
    for r in recs:
        prot = os.path.splitext(r)[0]
        if prot not in box:
            continue
        rpath = os.path.join(args.receptor_dir, r)
        params = box[prot]
        for i in range(0, len(ligs), args.batch_size):
            batch = ligs[i:i+args.batch_size]
            tasks.append((rpath, batch, params, args, len(tasks), prot))

    print(f'Scheduling {len(tasks)} receptor×batch runs')

    # execute tasks
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        fut2info = {exe.submit(run_batch, *t): t for t in tasks}
        for fut in as_completed(fut2info):
            _, batch, _, _, bid, prot = fut2info[fut]
            scores = fut.result()
            for l, score in scores.items():
                results[f'{prot}.pdbqt'][l] = score
            done += 1
            print(f'[{done}/{len(tasks)}] Done {prot} batch {bid}')

    # write results matrix
    os.makedirs(args.output_dir, exist_ok=True)
    matrix_path = os.path.join(args.output_dir, 'docking_matrix.csv')
    with open(matrix_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Receptor\\Ligand'] + ligs)
        for r in recs:
            row = [r] + [results[r].get(l, '') or '' for l in ligs]
            writer.writerow(row)

    print('All done. Matrix at:', matrix_path)

if __name__ == '__main__':
    main()
