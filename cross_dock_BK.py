#!/usr/bin/env python3
"""
cross_dock.py

Cross-docking script for Uni-Dock: docks a sampled subset of ligands
against a sampled subset of receptors, reads docking box parameters,
runs Uni-Dock, saves individual logs, and compiles a CSV matrix of best affinities.

Usage:
  python3 cross_dock.py \
    --receptor-dir receptors_pdbqt/ \
    --ligand-dir   ligands_pdbqt/ \
    --box-csv      dock_boxes.csv \
    --output-dir   crossdock_results/ \
    [--search-mode balance] \
    [--scoring vina] \
    [--num-modes 1] \
    [--receptor-percent 30.0] \
    [--ligand-percent 30.0]

python cross_dock.py \
  --receptor-dir /mnt/e/Datasbe_protein_peptide/receptors_pdbqt_filtered_test/ \
  --ligand-dir    /mnt/e/Datasbe_protein_peptide/peptide_pdbqt_test/ \
  --box-csv      dock_boxes_filtered.csv \
  --output-dir   crossdock_results/ \
  --receptor-percent 100 \
  --ligand-percent   100

"""
import os
import sys
import csv
import math
import random
import argparse
import subprocess

def parse_args():
    p = argparse.ArgumentParser(description="Cross-docking automation for Uni-Dock.")
    p.add_argument("--receptor-dir",    required=True,
                   help="Folder containing receptor PDBQT files")
    p.add_argument("--ligand-dir",      required=True,
                   help="Folder containing ligand PDBQT files")
    p.add_argument("--box-csv",         required=True,
                   help="CSV with docking box params (filename,center_*,size_*)")
    p.add_argument("--output-dir",      required=True,
                   help="Directory to store logs and CSV matrix")
    p.add_argument("--search-mode",     default="balance",
                   help="Uni-Dock search_mode (default: balance)")
    p.add_argument("--scoring",         default="vina",
                   help="Uni-Dock scoring (default: vina)")
    p.add_argument("--num-modes",       default=1, type=int,
                   help="Number of output modes per ligand (default: 1)")
    p.add_argument("--receptor-percent", default=100.0, type=float,
                   help="%% of receptors to dock (0–100, default 100)")
    p.add_argument("--ligand-percent",   default=100.0, type=float,
                   help="%% of ligands to dock (0–100, default 100)")
    return p.parse_args()

def read_box_csv(path):
    box = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = os.path.splitext(row['filename'])[0]
            box[name] = {
                'center_x': row['center_x'],
                'center_y': row['center_y'],
                'center_z': row['center_z'],
                'size_x':   row['size_x'],
                'size_y':   row['size_y'],
                'size_z':   row['size_z'],
            }
    return box

def list_pdbqt(dirpath):
    return sorted(f for f in os.listdir(dirpath) if f.lower().endswith('.pdbqt'))

def run_docking(receptor_path, ligand_path, box_params, args, log_path, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    cmd = [
        'unidock',
        '--receptor', receptor_path,
        '--gpu_batch', ligand_path,
        '--search_mode', args.search_mode,
        '--scoring', args.scoring,
        '--center_x', box_params['center_x'],
        '--center_y', box_params['center_y'],
        '--center_z', box_params['center_z'],
        '--size_x',   box_params['size_x'],
        '--size_y',   box_params['size_y'],
        '--size_z',   box_params['size_z'],
        '--num_modes', str(args.num_modes),
        '--dir', out_subdir,
    ]
    with open(log_path, 'w') as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    # Parse best score from lines starting with '1'
    best = None
    with open(log_path) as logf:
        for line in logf:
            parts = line.strip().split()
            if parts and parts[0] == '1':
                try:
                    best = float(parts[1])
                except:
                    pass
    return best

def estimate_time(n_docks, per_dock=0.5):
    secs = n_docks * per_dock
    hrs = int(secs // 3600)
    mins = int((secs % 3600) // 60)
    s   = secs % 60
    parts = []
    if hrs:  parts.append(f"{hrs}h")
    if mins: parts.append(f"{mins}m")
    parts.append(f"{s:.1f}s")
    return " ".join(parts)

def main():
    args = parse_args()
    box      = read_box_csv(args.box_csv)
    receptors = list_pdbqt(args.receptor_dir)
    ligands   = list_pdbqt(args.ligand_dir)
    if not receptors or not ligands:
        print("Error: no PDBQT in receptor or ligand dirs", file=sys.stderr)
        sys.exit(1)

    # Sample subsets
    nrec = max(1, int(math.ceil(len(receptors)*args.receptor_percent/100.0)))
    nlig = max(1, int(math.ceil(len(ligands)*  args.ligand_percent  /100.0)))
    sampled_receptors = random.sample(receptors, nrec)
    sampled_ligands   = random.sample(ligands,   nlig)

    total_docks = len(sampled_receptors)*len(sampled_ligands)
    print(f"Selected {nrec}/{len(receptors)} receptors "
          f"and {nlig}/{len(ligands)} ligands → "
          f"{total_docks} total dockings")
    print("Estimated total runtime:", estimate_time(total_docks))

    # Prepare output dirs
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    results = {r: {} for r in sampled_receptors}

    count = 0
    for r in sampled_receptors:
        prot = os.path.splitext(r)[0]
        if prot not in box:
            print(f"Warning: no box for {prot}, skipping", file=sys.stderr)
            continue
        params = box[prot]
        rpath  = os.path.join(args.receptor_dir, r)
        for l in sampled_ligands:
            lig = os.path.splitext(l)[0]
            lpath = os.path.join(args.ligand_dir, l)
            count += 1
            print(f"[{count}/{total_docks}] Docking {prot} × {lig}")

            out_subdir = os.path.join(args.output_dir, 'dock_outputs', prot)
            log_path   = os.path.join(logs_dir, f"{prot}__{lig}.log")

            score = run_docking(rpath, lpath, params, args, log_path, out_subdir)
            results[r][l] = score

    # Write CSV matrix
    matrix_path = os.path.join(args.output_dir, 'docking_matrix.csv')
    with open(matrix_path, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow([''] + sampled_ligands)
        for r in sampled_receptors:
            writer.writerow([r] + [results[r].get(l, '') for l in sampled_ligands])

    print("Done. Logs →", logs_dir)
    print("Matrix →", matrix_path)

if __name__ == '__main__':
    main()
