# Uni‑Dock Workflow

This repository provides scripts and step‑by‑step instructions to install and configure **Uni‑Dock** and **MGLTools**, prepare protein receptors and ligand structures for docking, compute docking box parameters, and run batch docking with Uni‑Dock on WSL Ubuntu.

---

## 1. Prerequisites

- **Windows 10/11** with WSL2 and Ubuntu installed
- NVIDIA GPU with CUDA support (compute capability ≥ 7.0)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed in WSL

---

## 2. Install Uni‑Dock via Conda

```bash
# 1) Update and install system tools
sudo apt update && sudo apt install -y build-essential cmake git wget curl

# 2) Install Miniconda (if not already) and initialize
#    (See Miniconda docs for details)

# 3) Create and activate Uni‑Dock environment
conda create -n unidock_env python=3.10 -y
conda activate unidock_env

# 4) Install Uni‑Dock from conda-forge
conda install -c conda-forge unidock -y

# 5) Verify installation
unidock --version
```

---

## 3A. Install MGLTools for receptor preparation

```bash
# 1) Download Linux tarball (108 MB)
mkdir -p ~/mgl && cd ~/mgl
wget -O mgltools.tar.gz https://ccsb.scripps.edu/mgltools/download/491

# 2) Extract and run installer
tar -xzf mgltools.tar.gz
cd mgltools_x86_64Linux2_1.5.7
./install.sh

# 3) Add MGLTools to your shell
echo ""                                            >> ~/.bashrc
echo "# MGLTools setup"                           >> ~/.bashrc
echo "export MGL_ROOT=\"\$HOME/mgl/mgltools_x86_64Linux2_1.5.7\"" >> ~/.bashrc
echo "export PATH=\"\$MGL_ROOT/bin:\$PATH\""       >> ~/.bashrc
echo "source \"\$MGL_ROOT/initMGLtools.sh\""      >> ~/.bashrc
source ~/.bashrc


# 4) Verify
which pythonsh        # MGLTools Python shell
which prepare_receptor4.py
```


---

## 4A. Prepare Ligands (.pdb → .pdbqt)

```bash
# generate peptides
./generate_peptides_from_file.py --input-file ./2to5-lib-sequences.txt --max-atoms 150 --pdb-dir ./2to5-lib-sequences
# convert pdb to sdf peptides
convert_pdb_sdf.py 
# add H to sdf
python batch_prepare_ligand_from_sdf.py --input-dir ./4AA_peptide_sequences_sdf   --output-dir ./4AA_peptide_sequences_prepared_sdf   --scratch ./tmp_meeko   --jobs 1

# confgen for sdf
export OBABEL_BIN=/usr/bin/obabel
python batch_confgen_obabel_parallel.py \
  --input_dir ./4AA_peptide_sequences_sdf \
  --output_dir ./4AA_peptide_sequences_sdf_confgen \
  --nconf 200 \
  --rmsd 0.3 \
  --cores 32

```


## 4A1 - ALL IN ONE BEST AND EASY

```bash
conda activate flex
python unified_peptide_pipeline.py     --input-file test.txt     --output merged.sdf.gz     --workers 8     --nconf 10

```


Use Open Babel to batch convert all ligand PDBs:

```bash
# Inside unidock_env
conda activate unidock_env
conda install -c conda-forge openbabel -y

# Place your ligands in `ligands/` as `.pdb`
# Use `batch_convert_ligands.py` (make executable)
chmod +x batch_convert_ligands.py
./batch_convert_ligands.py   -i  peptide_p
dbs/   -o ligands_pdbqt_collab/ -w 32

```


## 4B.create a confgen for ligands

```bash
#convert first to SDF
convert_pdb_sdf.py 


export OBABEL_BIN=/usr/bin/obabel
python batch_confgen.py \
  --input_dir ./4AA_peptide_sequences_sdf \
  --output_dir ./4AA_peptide_sequences_sdf_confgen \
  --nconf 200 \
  --rmsd 0.3 \
  --cores 32


```


Result: all `ligands_pdbqt/*.pdbqt` with polar H and Gasteiger charges.

---

## 5A. Prepare Receptors (.pdb → .pdbqt)

Use MGLTools + MolKit:

```bash
# Ensure MGLTools Python shell is in use
chmod +x batch_prepare_receptors.py

# Batch convert your receptors
pythonsh batch_prepare_receptors.py \
  --in-dir  /mnt/e/Specificty_docking/run \
  --out-dir /mnt/e/Specificty_docking/run \
  --repairs hydrogens \
  --cleanup nphs_lps_waters_nonstdres \
  --verbose
```

Output: `receptors_pdbqt/*.pdbqt` with hydrogens, Kollman/Gasteiger charges, cleaned.

---

### OR using Meeko
#### STEP 1 reducing
```bash
conda create -n flex python=3.9
conda activate flex
git clone https://github.com/forlilab/Meeko.git
cd Meeko
git checkout develop
python3 -m pip install .

./batch_prepare_receptors_meeko.py --input_dir ./run/receptors_pdb --output_dir ./prepared_receptors
```

#### OPTIONAL : STEP 2 spliting flex regions using Meeko (same as STEP 5B)

```bash
python3 meeko_flex.py \
    --in-dir ./run/receptors_pdbs_with_H \
    --out-dir ./run/meeko_flex_out \
    --csv ./run/dock_boxes_filtered.csv \
    --allow-bad-res --write-box --write-gpf --write-json

```


---

## 5B. Prepare Receptors flex using MGLtools (.pdbqt → _flex.pdbqt) ; helpful for very fine docking score (not screening)

Use MGLTools + MolKit:

```bash
# Ensure MGLTools Python shell is in use
chmod +x extract_flex_regions.py

# Batch convert your receptors
./extract_flex_regions.py --csv-file run/dock_boxes_filtered.csv \
--pdbqt-folder run/receptors \
--out-folder run/flex \
--prepare-flex-path ./prepare_flexreceptor4.py \
--max-residues 200
```

Output: `flex/*_flex.pdbqt`

---

## 5C. MCDOCK refinement

We will use MCDock for very refine docking for high quality docking

```bash
conda install -c conda-forge ambertools
python3 -m pip install pdb4amber
python3 -m pip install parmed
which pdb4amber # make sure this fucntion is ready to use


python batch_run_mcdock.py \
      --receptor_dir ./run/test/receptor \
      --receptor_csv ./run/test/dock_boxes_filtered.csv \
      --ligand_dir ./run/test/ligand \
      --output_dir ./run/test/mcdock_results \
      --batch_size 5000 \
      --seed 181129

```

## 6. Compute Docking Box Parameters

Extract center & size around ligand chain (default chain B):

```bash
chmod +x compute_dock_box.py
./compute_dock_box.py \
  --input complex.pdb \
  --chain B \
  --margin 5.0
```

Output:
```
--center_x X.xxx --center_y Y.yyy --center_z Z.zzz
--size_x   S.xxx --size_y S.yyy --size_z S.zzz
```

Use these flags in your Uni‑Dock command.

---

## 7. Run Uni‑Dock Batch Docking

Create an output folder and launch GPU batch docking:

```bash
mkdir -p results
unidock \
  --receptor receptors_pdbqt/receptor.pdbqt \
  --gpu_batch ligands_pdbqt/*.pdbqt \
  --search_mode balance \
  --scoring vina \
  --center_x X.xxx --center_y Y.yyy --center_z Z.zzz \
  --size_x S.xxx   --size_y S.yyy   --size_z S.zzz \
  --num_modes 1 \
  --dir results
```

Results: docked poses in `results/`, one `.pdbqt` per ligand.

---

## 8. Cross‑Docking All Ligands vs All Receptors

Use `cross_dock_new.py` to perform exhaustive docking of every ligand against every receptor, aggregate logs, and build a score matrix:

```bash
chmod +x cross_dock_new.py
python3 cross_dock_new.py \
  --receptor-dir receptors_pdbqt/ \
  --ligand-dir   ligands_pdbqt/ \
  --box-csv      dock_boxes.csv \
  --output-dir   crossdock_results/ \
  --search-mode  balance \
  --scoring      vina \
  --num-modes    1
```

### for flex docking

```bash
python3 cross_dock_new_2.py     --receptor-dir   ./run/flex     --flex-dir       ./run/flex     --ligand-dir     ./4AA_PDBQT     --box-csv        ./run/dock_boxes_filtered.csv     --output-dir     ./run/results_flex     --receptor-percent 100     --ligand-percent   100     --batch-size     5000
```


- **Logs**: saved per docking in `crossdock_results/logs/<receptor>__<ligand>.log`
- **Score matrix**: `crossdock_results/docking_matrix.csv` with rows = receptors, columns = ligands, entries = best affinities

---

### Collecting docking score from SDF of MCDock

```bash
python collect_docking_scores_sdf.py \
  --input_dir ./run/test/mcdock_results \
  --output_excel ./run/test/docking_scores.xlsx
```

## 9. References & Citations

- **Uni‑Dock**: Yu et al. (2023), J. Chem. Theory Comput. doi:10.1021/acs.jctc.2c01145
- **MGLTools** & **AD4ReceptorPreparation** from AutoDockTools

---

For questions or issues, open an issue at https://github.com/dptech-corp/Uni-Dock/issues

