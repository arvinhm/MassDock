#!/usr/bin/env python3
"""
cross_dock_mcdock.py

Cross‐docking + Multi‐Conformation Docking (mcdock) driver:
 1) Adds H to receptors
 2) Converts ligands → SDF
 3) Filters & batches ligands
 4) Serially runs `unidocktools mcdock` per batch
 5) Saves each batch’s stdout/stderr to logs/<protein>_batch<N>.log
 6) Emits final docking_matrix.csv
"""

import os, sys, csv, math, random, argparse, subprocess, logging, tempfile
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--receptor-dir",    required=True, help="Directory of receptor .pdb files")
    p.add_argument("--ligand-dir",      required=True, help="Directory of ligand .pdbqt/.pdb files")
    p.add_argument("--box-csv",         required=True, help="CSV: filename,center_x,center_y,center_z,size_x,size_y,size_z")
    p.add_argument("--output-dir",      required=True, help="Root for logs/, mcdock_outputs/, docking_matrix.csv")
    p.add_argument("--torsion-threshold", type=int, default=43, help="Skip ligands with > this many rotatable bonds")
    p.add_argument("--batch-size",       type=int, default=300, help="Ligands per batch")
    p.add_argument("--receptor-percent", type=float, default=100.0, help="Use this % of receptors")
    p.add_argument("--ligand-percent",   type=float, default=100.0, help="Use this % of ligands")
    p.add_argument("--gpus",            type=str, default="0",   help="Comma‐sep GPU IDs")
    p.add_argument("--gen-conf",        action="store_true",      help="Generate ligand conformers")
    p.add_argument("--max-num-confs",   type=int, default=1000,   help="Max conformers per ligand")
    p.add_argument("--min-rmsd",        type=float, default=0.5,  help="Min RMSD for conformers")
    p.add_argument("--sf-rd",           default="vina", choices=["vina","vinardo","ad4"], help="Rigid-dock scoring")
    p.add_argument("--ex-rd",           type=int, default=128,    help="Rigid-dock exhaustiveness")
    p.add_argument("--ms-rd",           type=int, default=20,     help="Rigid-dock maxstep")
    p.add_argument("--nm-rd",           type=int, default=3,      help="Rigid-dock modes")
    p.add_argument("--rs-rd",           type=int, default=3,      help="Rigid-dock refine steps")
    p.add_argument("--topn-rd",         type=int, default=100,    help="Rigid-dock topN")
    p.add_argument("--sf-lr",           default="vina", choices=["vina","vinardo","ad4"], help="Local-refine scoring")
    p.add_argument("--ex-lr",           type=int, default=32,     help="Local-refine exhaustiveness")
    p.add_argument("--ms-lr",           type=int, default=40,     help="Local-refine maxstep")
    p.add_argument("--nm-lr",           type=int, default=1,      help="Local-refine modes")
    p.add_argument("--rs-lr",           type=int, default=5,      help="Local-refine refine steps")
    p.add_argument("--topn-lr",         type=int, default=1,      help="Local-refine topN")
    return p.parse_args()

def count_torsions(pdb_path):
    m = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=True)
    if m is None: raise ValueError(f"Cannot parse ligand {pdb_path}")
    return rdMolDescriptors.CalcNumRotatableBonds(m)

def read_box_csv(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            name = os.path.splitext(r['filename'])[0]
            d[name] = r
    return d

def list_ligands(d):
    return sorted(fn for fn in os.listdir(d) if fn.lower().endswith((".pdbqt",".pdb")))

import os
import subprocess
from rdkit import Chem

import os
import subprocess
import logging
from rdkit import Chem



def prepare_receptors(args, recs):
    prep_dir = os.path.join(args.output_dir, "prep_receptors")
    os.makedirs(prep_dir, exist_ok=True)
    logf = os.path.join(prep_dir, "prepare_receptors.log")
    logger = logging.getLogger("prep_receptors")
    if not logger.handlers:
        fh = logging.FileHandler(logf)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

    mapping = {}
    for fname in recs:
        base      = os.path.splitext(fname)[0]
        inp_pdb   = os.path.join(args.receptor_dir, fname)
        out_pdb   = os.path.join(prep_dir, f"{base}_H.pdb")
        out_pdbqt = os.path.join(prep_dir, f"{base}_H.pdbqt")

        # 1) Add H
        try:
            if not os.path.exists(out_pdb):
                mol = Chem.MolFromPDBFile(inp_pdb, removeHs=False, sanitize=False)
                molH = Chem.AddHs(mol, addCoords=True)
                with open(out_pdb,"w") as fw:
                    fw.write(Chem.MolToPDBBlock(molH))
        except Exception as e:
            logger.error(f"{base}: add H failed → {e}")
            continue

        # 2) Try prepare_receptor4.py → .pdbqt
        got_pdbqt = False
        if not os.path.exists(out_pdbqt):
            cmd = ["pythonsh","prepare_receptor4.py","-r",out_pdb,"-o",out_pdbqt,"-A","delete_nonstd_res,delete_water"]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info(f"{base}: produced pdbqt")
                mapping[base] = out_pdbqt
                got_pdbqt = True
            except Exception as e:
                logger.warning(f"{base}: pdbqt conversion failed → {e}")

        # 3) Fallback to PDB if no PDBQT
        if not got_pdbqt:
            logger.warning(f"{base}: falling back to H-added PDB (no PDBQT)")
            mapping[base] = out_pdb

    return mapping



def pdb_to_sdf(pdb):
    sd = pdb + "_SDF"
    os.makedirs(sd, exist_ok=True)
    name = os.path.splitext(os.path.basename(pdb))[0]
    fn = os.path.join(sd, f"{name}.sdf")
    if os.path.exists(fn): return fn
    m = Chem.MolFromPDBFile(pdb, removeHs=False, sanitize=True)
    if m is None: raise RuntimeError(f"RDKit read fail {pdb}")
    Chem.MolToMolFile(m, fn)
    return fn

def run_mcdock_batch(rpath, ligs, box, args, logp, outdir, gpu):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    logp = os.path.abspath(logp)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    # write command header
    with open(logp, "w") as lf:
        lf.write(f"## CMD: unidocktools mcdock -r {rpath} ...\n\n")

    # build command
    idxf = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt")
    for L in ligs:
        idxf.write(pdb_to_sdf(os.path.join(args.ligand_dir,L))+"\n")
    idxf.close()

    cmd = [
      "unidocktools","mcdock","-r",rpath,"-i",idxf.name,"-sd",outdir,
      "-cx",box["center_x"],"-cy",box["center_y"],"-cz",box["center_z"],
      "-sx",box.get("size_x","22.5"),"-sy",box.get("size_y","22.5"),"-sz",box.get("size_z","22.5"),
      "--scoring_function_rigid_docking",args.sf_rd,
      "--exhaustiveness_rigid_docking",str(args.ex_rd),
      "--max_step_rigid_docking",str(args.ms_rd),
      "--num_modes_rigid_docking",str(args.nm_rd),
      "--refine_step_rigid_docking",str(args.rs_rd),
      "--topn_rigid_docking",str(args.topn_rd),
      "--scoring_function_local_refine",args.sf_lr,
      "--exhaustiveness_local_refine",str(args.ex_lr),
      "--max_step_local_refine",str(args.ms_lr),
      "--num_modes_local_refine",str(args.nm_lr),
      "--refine_step_local_refine",str(args.rs_lr),
      "--topn_local_refine",str(args.topn_lr),
    ]
    if args.gen_conf:
        cmd += ["-g","-n",str(args.max_num_confs),"-m",str(args.min_rmsd)]

    # run & catch errors
    with open(logp,"a") as lf:
        try:
            print(f"→ [RUN] {rpath} → log: {logp}")
            subprocess.run(cmd, stdout=lf, stderr=lf, env=env, check=True)
        except subprocess.CalledProcessError as e:
            lf.write(f"\n*** FAILED *** {e}\n")
            print(f"→ [ERROR] batch failed: see {logp}")
            return {L:None for L in ligs}

    # parse scores
    scores, idx = {}, 0
    with open(logp) as lf:
        for line in lf:
            p = line.split()
            if p and p[0]=="1" and idx<len(ligs):
                try: scores[ligs[idx]] = float(p[1])
                except: scores[ligs[idx]] = None
                idx += 1
    for L in ligs:
        scores.setdefault(L,None)

    print(f"→ [DONE ] batch → {logp}")
    return scores

def main():
    args = parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare logs
    logs_dir = os.path.join(args.output_dir,"logs")
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(logs_dir,"progress.log"),
                        level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("mcdock")

    # read inputs
    box   = read_box_csv(args.box_csv)
    recs0 = [f for f in os.listdir(args.receptor_dir) if f.endswith(".pdb")]
    ligs  = list_ligands(args.ligand_dir)
    if not recs0 or not ligs:
        sys.exit("ERROR: receptors or ligands directory empty")

    # subsample
    recs = random.sample(recs0, math.ceil(len(recs0)*args.receptor_percent/100))
    ligs = random.sample(ligs,  math.ceil(len(ligs)*args.ligand_percent/100))

    # prepare receptors
    rec_map = prepare_receptors(args, recs)

    # filter out non-pdbqt
    real_recs = []
    for r in recs:
        base = os.path.splitext(r)[0]
        path = rec_map.get(base)
        if path and path.lower().endswith(".pdbqt"):
            real_recs.append(r)
        else:
            logger.warning(f"Skipping {r}: no valid .pdbqt generated")

    # loop serially over real_recs
    results, skipped, submitted = {}, 0, 0
    gpus = args.gpus.split(",")
    for r in real_recs:
        prot  = os.path.splitext(r)[0]
        rpath = rec_map[prot]
        params= box.get(prot)
        if not params: continue

        outroot = os.path.join(args.output_dir,"mcdock_outputs",prot)
        batch = []
        for L in tqdm(ligs, desc=f"Filter {prot}", unit="lig"):
            if count_torsions(os.path.join(args.ligand_dir,L))>args.torsion_threshold:
                skipped += 1
                continue
            batch.append(L)
            if len(batch)==args.batch_size:
                logp = os.path.join(logs_dir,f"{prot}_batch{submitted}.log")
                logger.info(f"START {prot}_batch{submitted}")
                scores = run_mcdock_batch(rpath,batch,params,args,logp,
                                          os.path.join(outroot,f"batch{submitted}"),
                                          gpus[submitted%len(gpus)])
                results.setdefault(r,{}).update(scores)
                submitted += 1
                batch = []

        # leftover
        if batch:
            logp = os.path.join(logs_dir,f"{prot}_batch{submitted}.log")
            logger.info(f"START {prot}_batch{submitted}")
            scores = run_mcdock_batch(rpath,batch,params,args,logp,
                                      os.path.join(outroot,f"batch{submitted}"),
                                      gpus[submitted%len(gpus)])
            results.setdefault(r,{}).update(scores)
            submitted += 1

    # write matrix
    mat = os.path.join(args.output_dir,"docking_matrix.csv")
    with open(mat,"w", newline="") as mf:
        w = csv.writer(mf)
        all_ligs = sorted({L for d in results.values() for L in d})
        w.writerow([""]+all_ligs)
        for r in real_recs:
            row = [results[r].get(L,"") for L in all_ligs]
            w.writerow([r]+row)

    print(f"Done; skipped {skipped} ligands. Results: {mat}")

if __name__=="__main__":
    main()
    