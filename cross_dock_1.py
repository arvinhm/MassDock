
#!/usr/bin/env python3
"""
cross_dock.py

Cross-docking script for Uni-Dock with batch GPU docking,
torsion filtering, percentage subsampling, and parallel execution.

Each GPU batch call is:
    unidock --receptor rec.pdbqt \
            --gpu_batch lig1.pdbqt lig2.pdbqt ... ligN.pdbqt \
            --search_mode fast --scoring vina \
            --center_x ... --size_x ... --dir ...

Usage example (two A100 GPUs, sample 50% of receptors & 10% of ligands):
  python3 cross_dock.py \
    --receptor-dir receptors_pdbqt_filtered/ \
    --ligand-dir   ligands_pdbqt/ \
    --box-csv      dock_boxes_filtered.csv \
    --output-dir   crossdock_results/ \
    --receptor-percent 50 \
    --ligand-percent   10 \
    --batch-size   300 \
    --gpus          0,1

python3 cross_dock_new.py \
    --receptor-dir receptors_pdbqt_filtered/ \
    --ligand-dir   ligands_pdbqt/ \
    --box-csv      dock_boxes_filtered.csv \
    --output-dir   crossdock_results/ \
    --receptor-percent 50 \
    --ligand-percent   10 \
    --batch-size   300 \
    --gpus          0,1
"""

import os
import sys
import csv
import math
import random
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--receptor-dir",      required=True,
                   help="Directory of receptor .pdbqt files")
    p.add_argument("--ligand-dir",        required=True,
                   help="Directory of ligand .pdbqt files")
    p.add_argument("--box-csv",           required=True,
                   help="CSV with columns: filename,center_x,center_y,center_z,size_x,size_y,size_z")
    p.add_argument("--output-dir",        required=True,
                   help="Where logs, outputs, and matrix will go")
    p.add_argument("--search-mode",       default="balance",
                   choices=["fast","balance","detailed"],
                   help="'fast' = nthreads=128, nsteps=20")
    p.add_argument("--scoring",           default="vina",
                   choices=["vina","vinardo","ad4"])
    p.add_argument("--exhaustiveness", type=int, default=1,
               help="Vina exhaustiveness per ligand")
    p.add_argument("--max_step",       type=int, default=8,
               help="Uni-Dock max search steps")
    p.add_argument("--spacing",  type=float, default=0.375,
                   help="% of receptors to randomly sample (0–100)")
    p.add_argument("--num-modes",         type=int,   default=1,
                   help="How many output poses per ligand")
    p.add_argument("--receptor-percent",  type=float, default=100.0,
                   help="%% of receptors to randomly sample (0–100)")
    p.add_argument("--ligand-percent",    type=float, default=100.0,
                   help="%% of ligands to randomly sample (0–100)")
    p.add_argument("--torsion-threshold", type=int,   default=43,
                   help="Skip ligands with torsion count > threshold")
    p.add_argument("--batch-size",        type=int,   default=300,
                   help="Ligands per GPU batch call (≈300 ideal for A100)")
    p.add_argument("--gpus",              type=str,   default="0",
                   help="Comma-separated CUDA device IDs to use")
    return p.parse_args()

def count_torsions(pdbqt_path):
    with open(pdbqt_path) as f:
        for L in f:
            if L.lower().startswith("torsdof"):
                parts = L.split()
                return int(parts[1]) if len(parts)>1 else 0
    return 0

def read_box_csv(path):
    box = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = os.path.splitext(row['filename'])[0]
            box[name] = row
    return box

def list_pdbqt(dirpath):
    return sorted(f for f in os.listdir(dirpath) if f.lower().endswith(".pdbqt"))

def estimate_time(batches, per_batch=0.5):
    total_s = batches * per_batch
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = total_s % 60
    return f"{h}h{m}m{s:.1f}s"

#            "--exhaustiveness", str(args.exhaustiveness),
#            "--max_step",       str(args.max_step),
def run_batch(rpath, ligand_paths, box_params, args, log_path, out_dir, gpu_id):
    # Pin this subprocess to one GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        ["unidock", "--receptor", rpath, "--gpu_batch"]
        + ligand_paths
        + [
            "--search_mode", args.search_mode,
            "--scoring",     args.scoring,
            "--center_x",    box_params['center_x'],
            "--center_y",    box_params['center_y'],
            "--center_z",    box_params['center_z'],
            "--size_x",      box_params['size_x'],
            "--size_y",      box_params['size_y'],
            "--size_z",      box_params['size_z'],
            "--num_modes",   str(args.num_modes),
            "--spacing", str(args.spacing),
            "--dir",         out_dir
        ]
    )
    with open(log_path, "w") as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, check=True)

    # Parse one “pose-1” score per ligand (in order)
    scores = {}
    names  = [os.path.basename(p) for p in ligand_paths]
    idx    = 0
    with open(log_path) as logf:
        for line in logf:
            parts = line.strip().split()
            if parts and parts[0] == "1" and idx < len(names):
                try:
                    scores[names[idx]] = float(parts[1])
                except:
                    scores[names[idx]] = None
                idx += 1
    # Ensure every ligand has an entry
    for nm in names:
        scores.setdefault(nm, None)
    return scores

def main():
    args     = parse_args()
    gpu_list = args.gpus.split(",")
    concurrency = len(gpu_list)

    # Read inputs
    box      = read_box_csv(args.box_csv)
    recs_all = list_pdbqt(args.receptor_dir)
    ligs_all = list_pdbqt(args.ligand_dir)
    if not recs_all or not ligs_all:
        sys.exit("ERROR: receptor-dir or ligand-dir is empty!")

    # Subsample receptors & ligands by percentage
    nrec = max(1, math.ceil(len(recs_all) * args.receptor_percent/100))
    nlig = max(1, math.ceil(len(ligs_all) * args.ligand_percent/100))
    recs = random.sample(recs_all, nrec)
    ligs = random.sample(ligs_all, nlig)

    # Filter ligands by torsion count
    valid = []
    for L in ligs:
        t = count_torsions(os.path.join(args.ligand_dir, L))
        if t > args.torsion_threshold:
            print(f"Skipping {L} — torsions {t} > {args.torsion_threshold}")
        else:
            valid.append(L)

    # Prepare results storage
    results = {r:{l:None for l in valid} for r in recs}

    # Build tasks: one (receptor, ligand-batch) per GPU batch
    tasks = []
    for r in recs:
        prot = os.path.splitext(r)[0]
        if prot not in box:
            continue
        rpath  = os.path.join(args.receptor_dir, r)
        params = box[prot]
        outdir = os.path.join(args.output_dir, "dock_outputs", prot)
        for i in range(0, len(valid), args.batch_size):
            batch   = valid[i:i+args.batch_size]
            paths   = [os.path.join(args.ligand_dir, L) for L in batch]
            logp    = os.path.join(args.output_dir, "logs",
                                   f"{prot}__batch{i//args.batch_size}.log")
            gpu_id  = gpu_list[(i//args.batch_size) % concurrency]
            tasks.append((r, batch, rpath, paths, params, args, logp, outdir, gpu_id))

    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    total_batches = len(tasks)
    print(f"{len(recs)} receptors × {len(valid)} ligands → {total_batches} batches")
    print("Estimated time:", estimate_time(total_batches))

    # Launch parallel batches (one worker per GPU)
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as exe:
        fut2task = {}
        for t in tasks:
            r, batch, rpath, paths, params, a, logp, outdir, gpu_id = t
            print(f"Scheduling {r} batch({len(batch)}) on GPU {gpu_id}")
            fut = exe.submit(run_batch, rpath, paths, params, a, logp, outdir, gpu_id)
            fut2task[fut] = (r, batch)
        for fut in as_completed(fut2task):
            rec, batch = fut2task[fut]
            scores = fut.result()
            for L in batch:
                results[rec][L] = scores[L]
            done += 1
            print(f"[{done}/{total_batches}] done {rec} batch")

    # Write out docking matrix
    mat_path = os.path.join(args.output_dir, "docking_matrix.csv")
    with open(mat_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow([""] + valid)
        for r in recs:
            row = [results[r].get(L, "") for L in valid]
            writer.writerow([r] + row)

    print("All done! Matrix saved to:", mat_path)

if __name__ == "__main__":
    main()


