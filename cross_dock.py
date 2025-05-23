#!/usr/bin/env python3
"""
cross_dock.py

Cross-docking script for Uni-Dock with batch GPU docking,
torsion filtering, and parallel execution.

Each batch is:
  unidock --receptor rec.pdbqt \
          --gpu_batch lig1.pdbqt lig2.pdbqt ... ligN.pdbqt \
          --search_mode ... --scoring ... --center_x ... --size_x ... --dir ...

Usage:
python3 ./cross_dock.py \
    --receptor-dir /mnt/e/Datasbe_protein_peptide/receptors_pdbqt_filtered/ \
    --ligand-dir   ./ligands_pdbqt_collab/ \
    --box-csv      ./dock_boxes_filtered.csv \
    --output-dir   ./crossdock_results/ \
    --concurrency  1
"""
import os, sys, csv, math, random, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--receptor-dir",     required=True)
    p.add_argument("--ligand-dir",       required=True)
    p.add_argument("--box-csv",          required=True)
    p.add_argument("--output-dir",       required=True)
    p.add_argument("--search-mode",      default="fast")
    p.add_argument("--scoring",          default="vina")
    p.add_argument("--exhaustiveness", type=int, default=1,
               help="Vina exhaustiveness per ligand")
    p.add_argument("--max_step",       type=int, default=1,
               help="Uni-Dock max search steps")
    p.add_argument("--num-modes",        type=int, default=1)
    p.add_argument("--receptor-percent", type=float, default=5.0)
    p.add_argument("--ligand-percent",   type=float, default=2.0)
    p.add_argument("--torsion-threshold",type=int,   default=43)
    p.add_argument("--batch-size",       type=int,   default=5000000,
                   help="How many ligands per GPU batch call")
    p.add_argument("--concurrency",      type=int,   default=5)
    return p.parse_args()

def count_torsions(pdbqt):
    with open(pdbqt) as f:
        for L in f:
            if L.lower().startswith("torsdof"):
                try: return int(L.split()[1])
                except: return 0
    return 0

def read_box_csv(path):
    box = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            nm = os.path.splitext(row['filename'])[0]
            box[nm] = {k: row[k] for k in row if k!="filename"}
    return box

def list_pdbqt(dirpath):
    return sorted(f for f in os.listdir(dirpath) if f.lower().endswith(".pdbqt"))

def run_batch(rpath, ligand_paths, params, args, log_path, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    cmd = (["unidock", "--receptor", rpath, "--gpu_batch"]
           + ligand_paths
           + ["--search_mode", args.search_mode,
              "--scoring",     args.scoring,
              "--exhaustiveness", str(args.exhaustiveness),
              "--max_step",       str(args.max_step),
              "--center_x",    params['center_x'],
              "--center_y",    params['center_y'],
              "--center_z",    params['center_z'],
              "--size_x",      params['size_x'],
              "--size_y",      params['size_y'],
              "--size_z",      params['size_z'],
              "--num_modes",   str(args.num_modes),
              "--dir",         out_subdir])
    with open(log_path,"w") as F:
        subprocess.run(cmd, stdout=F, stderr=subprocess.STDOUT)

    # parse one pose-1 score per ligand, in order:
    scores = {}
    idx = 0
    names = [os.path.basename(p) for p in ligand_paths]
    with open(log_path) as F:
        for line in F:
            parts = line.strip().split()
            if parts and parts[0]=="1" and idx < len(names):
                try: scores[names[idx]] = float(parts[1])
                except: scores[names[idx]] = None
                idx += 1
    # fill any missing
    for i,name in enumerate(names):
        if name not in scores:
            scores[name] = None
    return scores

def estimate_time(n, per=0.5):
    s = n*per
    return f"{int(s//3600)}h{int((s%3600)//60)}m{(s%60):.1f}s"

def main():
    args = parse_args()
    box       = read_box_csv(args.box_csv)
    recs      = list_pdbqt(args.receptor_dir)
    ligs      = list_pdbqt(args.ligand_dir)
    if not recs or not ligs:
        sys.exit("Empty receptor/ligand dirs")
    # sample
    nrec = max(1, math.ceil(len(recs)*args.receptor_percent/100))
    nlig = max(1, math.ceil(len(ligs)*args.ligand_percent/100))
    recs = random.sample(recs, nrec)
    ligs = random.sample(ligs,   nlig)
    # torsion filter
    valid = []
    for L in ligs:
        t = count_torsions(os.path.join(args.ligand_dir,L))
        if t>args.torsion_threshold:
            print(f"Skipping {L} (torsions {t} > {args.torsion_threshold})")
        else:
            valid.append(L)
    # prepare results
    results = {r:{l:None for l in valid} for r in recs}
    # build batch tasks
    tasks = []
    for r in recs:
        prot   = os.path.splitext(r)[0]
        if prot not in box: continue
        rpath  = os.path.join(args.receptor_dir, r)
        params = box[prot]
        outdir = os.path.join(args.output_dir, "dock_outputs", prot)
        # chunk ligands
        for i in range(0, len(valid), args.batch_size):
            batch = valid[i:i+args.batch_size]
            paths = [os.path.join(args.ligand_dir, L) for L in batch]
            logp  = os.path.join(args.output_dir, "logs",
                                  f"{prot}__batch{i//args.batch_size}.log")
            tasks.append((r, batch, rpath, paths, params, args, logp, outdir))

    total = len(tasks)
    print(f"{len(recs)} recs × {len(valid)} ligs in {total} batches; est {estimate_time(total)}")
    os.makedirs(os.path.join(args.output_dir,"logs"), exist_ok=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        fut2t = {}
        for (r,batch,rpath,paths,params,a,logp,out) in tasks:
            print(f"Scheduling {r} batch of {len(batch)} ligs")
            fut = exe.submit(run_batch, rpath, paths, params, a, logp, out)
            fut2t[fut] = (r,batch)
        for fut in as_completed(fut2t):
            rec, batch = fut2t[fut]
            scores = fut.result()
            for L in batch:
                results[rec][L] = scores[L]
            done+=1
            print(f"[{done}/{total}] done {rec} batch → {batch}")

    # write matrix
    mat = os.path.join(args.output_dir,"docking_matrix.csv")
    with open(mat,"w",newline="") as F:
        w=csv.writer(F)
        w.writerow([""]+valid)
        for r in recs:
            w.writerow([r]+[results[r][L] if results[r][L] is not None else "" for L in valid])
    print("Done. Matrix:",mat)

if __name__=="__main__":
    main()
