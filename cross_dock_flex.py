#!/usr/bin/env python3
"""
cross_dock_flex_threaded.py

Batch cross-docking with Uni-Dock + flexible receptor support (flex flag):
- As soon as batch-size valid ligands found, batch is submitted for docking.
- For each receptor, uses <name>_rigid.pdbqt as rigid receptor and <name>_flex.pdbqt for --flex if it exists.
- All other logic (parallelization, matrix, logging, batching) matches your previous pipeline.

Usage example:
python3 cross_dock_flex_threaded.py \
    --receptor-dir ./run/receptors \
    --flex-dir     ./run/flex_regions \
    --ligand-dir   ./4AA_PDBQT \
    --box-csv      ./run/dock_boxes_filtered.csv \
    --output-dir   ./run/results_flex \
    --receptor-percent 100 \
    --ligand-percent   100 \
    --batch-size     500 \
    --gpus 0,1,2
"""

import os
import sys
import csv
import math
import random
import argparse
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--receptor-dir",      required=True,
                   help="Directory of *_rigid.pdbqt receptor files")
    p.add_argument("--flex-dir",          required=True,
                   help="Directory of *_flex.pdbqt flexible residue files")
    p.add_argument("--ligand-dir",        required=True,
                   help="Directory of ligand .pdbqt files")
    p.add_argument("--box-csv",           required=True,
                   help="CSV with columns: filename,center_x,center_y,center_z,size_x,size_y,size_z")
    p.add_argument("--output-dir",        required=True,
                   help="Where logs, outputs, and matrix will go")
    p.add_argument("--search-mode",       default="detailed",
                   choices=["fast","balance","detailed"])
    p.add_argument("--scoring",           default="vina",
                   choices=["vina","vinardo","ad4"])
    p.add_argument("--spacing",  type=float, default=0.375)
    p.add_argument("--num-modes",         type=int,   default=1)
    p.add_argument("--torsion-threshold", type=int,   default=43,
                   help="Skip ligands with > this many rotatable bonds")
    p.add_argument("--batch-size",        type=int,   default=300)
    p.add_argument("--receptor-percent",  type=float, default=100.0)
    p.add_argument("--ligand-percent",    type=float, default=100.0)
    p.add_argument("--gpus",              type=str,   default="0",
                   help="Comma‐separated GPU IDs")
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

def list_rigid_and_flex_receptors(receptor_dir, flex_dir):
    rigid_files = [f for f in os.listdir(receptor_dir) if f.endswith('_rigid.pdbqt')]
    rigid_dict = {}
    for rf in rigid_files:
        rec_base = rf.rsplit('_rigid.pdbqt', 1)[0]
        rigid_dict[rec_base] = os.path.join(receptor_dir, rf)
    flex_dict = {}
    for base in rigid_dict:
        flex_path = os.path.join(flex_dir, base + '_flex.pdbqt')
        flex_dict[base] = flex_path if os.path.isfile(flex_path) else None
    return rigid_dict, flex_dict

def list_pdbqt(dirpath):
    return sorted(f for f in os.listdir(dirpath) if f.lower().endswith(".pdbqt"))

def run_batch(rigid_path, flex_path, ligand_paths, box_params, args, log_path, out_dir, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["unidock", "--receptor", rigid_path]
    if flex_path:
        cmd += ["--flex", flex_path]
    cmd += ["--gpu_batch"] + ligand_paths + [
        "--search_mode", args.search_mode,
        "--scoring",     args.scoring,
        "--center_x",    box_params['center_x'],
        "--center_y",    box_params['center_y'],
        "--center_z",    box_params['center_z'],
        "--size_x",      box_params['size_x'],
        "--size_y",      box_params['size_y'],
        "--size_z",      box_params['size_z'],
        "--num_modes",   str(args.num_modes),
        "--spacing",     str(args.spacing),
        "--dir",         out_dir
    ]
    with open(log_path, "w") as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, check=True)

    # parse pose‐1 scores
    scores, names = {}, [os.path.basename(p) for p in ligand_paths]
    idx = 0
    with open(log_path) as logf:
        for line in logf:
            parts = line.split()
            if parts and parts[0] == "1" and idx < len(names):
                try:
                    scores[names[idx]] = float(parts[1])
                except:
                    scores[names[idx]] = None
                idx += 1
    for nm in names:
        scores.setdefault(nm, None)
    return scores

def main():
    args = parse_args()
    gpu_list = args.gpus.split(",")
    conc     = len(gpu_list)

    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(logs_dir, "progress.log"),
                        level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logger = logging.getLogger("crossdock")

    box = read_box_csv(args.box_csv)
    rigid_dict, flex_dict = list_rigid_and_flex_receptors(args.receptor_dir, args.flex_dir)
    recs_all = list(rigid_dict.keys())
    ligs_all = list_pdbqt(args.ligand_dir)
    if not recs_all or not ligs_all:
        sys.exit("ERROR: receptor-dir or ligand-dir is empty!")

    nrec = math.ceil(len(recs_all) * args.receptor_percent / 100)
    nlig = math.ceil(len(ligs_all) * args.ligand_percent / 100)
    recs = random.sample(recs_all, nrec)
    ligs = random.sample(ligs_all,  nlig)

    # estimate total batches for progress bar
    approx_valid = math.ceil(len(ligs) * 0.8)
    batches_per = math.ceil(approx_valid / args.batch_size)
    total_batches = batches_per * len(recs)
    results = {r:{} for r in recs}

    global_skip = 0
    submitted   = 0
    dock_pbar = tqdm(total=total_batches, desc="Batches submitted", unit="batch")

    with ThreadPoolExecutor(max_workers=conc) as exe:
        fut2task = {}

        for r in recs:
            prot   = r
            if prot not in box:
                continue
            rigid_path = rigid_dict[prot]
            flex_path  = flex_dict[prot]
            params     = box[prot]
            outdir     = os.path.join(args.output_dir, "dock_outputs", prot)

            local_skip = 0
            valid_batch = []

            filter_bar = tqdm(ligs, desc=f"Filter for {prot}", unit="lig")
            for L in filter_bar:
                t = count_torsions(os.path.join(args.ligand_dir, L))
                if t > args.torsion_threshold:
                    global_skip += 1
                    local_skip  += 1
                else:
                    valid_batch.append(L)
                filter_bar.set_postfix(passed=len(valid_batch), skipped=local_skip)
                if len(valid_batch) == args.batch_size:
                    paths = [os.path.join(args.ligand_dir, x) for x in valid_batch]
                    logp  = os.path.join(logs_dir, f"{prot}__batch{submitted}.log")
                    gpu_id = gpu_list[submitted % conc]
                    print(f"→ submitting {prot} batch#{submitted+1} ({len(valid_batch)} ligands) on GPU {gpu_id} {'(FLEX)' if flex_path else '(RIGID)'}")
                    logger.info(f"{prot} batch#{submitted+1} → {len(valid_batch)} ligands")
                    fut = exe.submit(
                        run_batch, rigid_path, flex_path, paths, params, args,
                        logp, outdir, gpu_id
                    )
                    fut2task[fut] = (r, list(valid_batch))
                    submitted += 1
                    dock_pbar.update(1)
                    valid_batch = []
            filter_bar.close()

            # leftover ligands
            if valid_batch:
                paths = [os.path.join(args.ligand_dir, x) for x in valid_batch]
                logp  = os.path.join(logs_dir, f"{prot}__batch{submitted}.log")
                gpu_id = gpu_list[submitted % conc]
                print(f"→ submitting {prot} batch#{submitted+1} ({len(valid_batch)} ligands) on GPU {gpu_id} {'(FLEX)' if flex_path else '(RIGID)'}")
                logger.info(f"{prot} batch#{submitted+1} → {len(valid_batch)} ligands")
                fut = exe.submit(
                    run_batch, rigid_path, flex_path, paths, params, args,
                    logp, outdir, gpu_id
                )
                fut2task[fut] = (r, list(valid_batch))
                submitted += 1
                dock_pbar.update(1)

        # collect results as they finish
        done = 0
        for fut in as_completed(fut2task):
            rec, batch = fut2task[fut]
            scores = fut.result()
            for L, sc in scores.items():
                results.setdefault(rec, {})[L] = sc
            done += 1
            print(f"[{done}/{total_batches}] batch completed for receptor {rec}")

    dock_pbar.close()

    os.makedirs(args.output_dir, exist_ok=True)
    mat_path = os.path.join(args.output_dir, "docking_matrix.csv")
    with open(mat_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        all_ligs = set()
        for rec_scores in results.values():
            all_ligs.update(rec_scores.keys())
        all_ligs = sorted(all_ligs)
        writer.writerow([""] + all_ligs)
        for r in recs:
            row = [ results[r].get(lig, "") for lig in all_ligs ]
            writer.writerow([r] + row)

    print(f"Done! skipped {global_skip} ligands; results in {mat_path}")

if __name__ == "__main__":
    main()
