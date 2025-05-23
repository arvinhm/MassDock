#!/usr/bin/env python3
"""
unified_peptide_pipeline.py

Complete pipeline that combines all processing steps:
  1) Read text file and generate PDB files
  2) Convert PDB files to SDF
  3) Prepare SDF input files to SDF prepared
  4) Generate conformers for SDF
  5) Merge all SDF files into SDF.gz files

Usage:
  python unified_peptide_pipeline.py \
    --input-file peptides.txt \
    --output merged.sdf.gz \
    --workers 8 \
    --nconf 10
"""

import os
import sys
import argparse
import shutil
import gzip
import subprocess
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import multiprocessing

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run(cmd, check_libraries=False):
    """Run shell command and handle errors"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = f"Command failed: {' '.join(cmd)}\nstdout:{result.stdout}\nstderr:{result.stderr}"
        
        # Special handling for Open Babel library issues
        if check_libraries and 'libopenbabel' in result.stderr:
            error_msg += "\n\nOpen Babel library issue detected. Try:\n"
            error_msg += "1. conda install -c conda-forge openbabel\n"
            error_msg += "2. sudo apt install libopenbabel-dev libopenbabel7\n"
            error_msg += "3. export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH"
            
        raise RuntimeError(error_msg)
    return result.stdout

def setup_openbabel_environment():
    """Setup Open Babel environment variables"""
    # Common library paths for Open Babel
    lib_paths = [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/lib',
        '/opt/conda/lib',
        '/home/linuxbrew/.linuxbrew/lib',
        '/usr/lib64'
    ]
    
    # Get current LD_LIBRARY_PATH
    current_path = os.environ.get('LD_LIBRARY_PATH', '')
    
    # Add library paths that actually exist
    existing_paths = [path for path in lib_paths if os.path.exists(path)]
    
    if existing_paths:
        if current_path:
            new_path = ':'.join(existing_paths) + ':' + current_path
        else:
            new_path = ':'.join(existing_paths)
        
        os.environ['LD_LIBRARY_PATH'] = new_path
        print(f"Set LD_LIBRARY_PATH to: {new_path}")

def find_obabel():
    """Find and validate Open Babel binary"""
    # Setup environment first
    setup_openbabel_environment()
    
    # Set OBABEL_BIN environment variable if not set
    if not os.environ.get('OBABEL_BIN'):
        os.environ['OBABEL_BIN'] = '/usr/bin/obabel'
    
    candidates = [
        os.environ.get('OBABEL_BIN'),
        '/usr/bin/obabel',
        '/usr/local/bin/obabel',
        '/opt/conda/bin/obabel',
        '/home/linuxbrew/.linuxbrew/bin/obabel',
        shutil.which('obabel')
    ]
    
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            # Test if obabel actually works by running simple command
            try:
                # Use a very simple test that should work even with library issues
                result = subprocess.run([candidate, '--help'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 or 'Open Babel' in result.stdout:
                    return candidate
                else:
                    print(f"Warning: {candidate} found but failed help check: {result.stderr}")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Warning: {candidate} found but failed to execute: {e}")
                continue
    
    # If we can't find a working version, try to install or provide better error message
    error_msg = """
ERROR: Cannot find working Open Babel installation.

Please try these solutions in order:

1. Install via conda (recommended):
   conda install -c conda-forge openbabel

2. Install via apt (Ubuntu/Debian):
   sudo apt update
   sudo apt install openbabel libopenbabel-dev libopenbabel7

3. If installed but not working, try setting library path:
   export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH

4. For conda environments, ensure it's activated:
   conda activate your_environment_name
   conda install -c conda-forge openbabel

5. Test installation:
   obabel --help
"""
    raise RuntimeError(error_msg)

def test_obabel_conversion(obabel_bin):
    """Test if Open Babel can actually perform conversions"""
    try:
        # Create a simple test molecule (methane) in SDF format
        test_sdf = """
  Mrv2014 01012021

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
        # Write test file
        with open('test_mol.sdf', 'w') as f:
            f.write(test_sdf)
        
        # Try conversion
        result = subprocess.run([
            obabel_bin, '-isdf', 'test_mol.sdf', '-opdb', '-O', 'test_mol.pdb'
        ], capture_output=True, text=True, timeout=30)
        
        # Cleanup
        for f in ['test_mol.sdf', 'test_mol.pdb']:
            if os.path.exists(f):
                os.remove(f)
        
        if result.returncode == 0:
            return True
        else:
            print(f"Open Babel conversion test failed: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"Open Babel test failed with exception: {e}")
        return False

# ============================================================================
# STEP 1: READ TEXT FILE AND GENERATE PDB FILES
# ============================================================================

def read_sequences(path):
    """Read peptide sequences from text file"""
    return [line.strip() for line in open(path) if line.strip()]

def filter_sequences(seqs, max_atoms, max_rot):
    """Filter sequences by atom count and rotatable bonds"""
    valid, invalid = [], []
    for seq in tqdm(seqs, desc="Filtering sequences", unit="seq"):
        try:
            mol = Chem.MolFromSequence(seq)
            if mol is None:
                invalid.append(seq)
            elif mol.GetNumAtoms() > max_atoms or Descriptors.NumRotatableBonds(mol) > max_rot:
                invalid.append(seq)
            else:
                valid.append(seq)
        except Exception as e:
            print(f"Error processing sequence {seq}: {e}")
            invalid.append(seq)
    return valid, invalid

def build_pdb(seq, out_dir):
    """Build PDB from peptide sequence using pyPept"""
    try:
        from pyPept.sequence import Sequence, correct_pdb_atoms
        from pyPept.molecule import Molecule
        from pyPept.conformer import Conformer, SecStructPredictor

        cap_seq = f"ac-{'-'.join(seq)}-am"
        seq_obj = correct_pdb_atoms(Sequence(cap_seq))
        mol_obj = Molecule(seq_obj)
        romol = mol_obj.get_molecule(fmt='ROMol')
        fasta = Conformer.get_peptide(cap_seq)
        ss = SecStructPredictor.predict_active_ss(fasta)
        conf = Conformer.generate_conformer(romol, ss, generate_pdb=False)
        pdb_block = Chem.MolToPDBBlock(conf)

        out_path = out_dir / f"{seq}.pdb"
        out_path.write_text(pdb_block)
        return out_path
    except Exception as e:
        print(f"Error building PDB for {seq}: {e}")
        return None

# ============================================================================
# STEP 2: CONVERT PDB FILES TO SDF
# ============================================================================

def pdb_to_sdf_rdkit_fallback(in_pdb, out_sdf):
    """Convert PDB to SDF using RDKit as fallback"""
    try:
        mol = Chem.MolFromPDBFile(str(in_pdb), removeHs=False)
        if mol is None:
            raise ValueError("RDKit couldn't read PDB file")
        
        # Write to SDF
        writer = Chem.SDWriter(str(out_sdf))
        writer.write(mol)
        writer.close()
        return out_sdf
    except Exception as e:
        raise RuntimeError(f"RDKit fallback failed: {e}")

def pdb_to_sdf(in_pdb, out_sdf, obabel_bin=None, use_rdkit_fallback=True):
    """Convert PDB to SDF using Open Babel, with RDKit fallback"""
    
    # Try Open Babel first if available
    if obabel_bin:
        try:
            result = subprocess.run([
                obabel_bin, '-ipdb', str(in_pdb), '-osdf', '-O', str(out_sdf)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(out_sdf) and os.path.getsize(out_sdf) > 0:
                return out_sdf
            else:
                print(f"Open Babel failed for {in_pdb}: {result.stderr}")
                if not use_rdkit_fallback:
                    raise RuntimeError(f"Open Babel conversion failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Open Babel timeout for {in_pdb}")
            if not use_rdkit_fallback:
                raise RuntimeError("Open Babel conversion timed out")
        except Exception as e:
            print(f"Open Babel error for {in_pdb}: {e}")
            if not use_rdkit_fallback:
                raise
    
    # Use RDKit fallback
    if use_rdkit_fallback:
        return pdb_to_sdf_rdkit_fallback(in_pdb, out_sdf)
    else:
        raise RuntimeError("No conversion method available")

# ============================================================================
# STEP 3: PREPARE SDF INPUT FILES TO SDF PREPARED
# ============================================================================

def process_sdf_with_meeko(sdf_path, out_dir, scratch_dir):
    """Process SDF with Meeko to add hydrogens and atom types"""
    base = os.path.splitext(os.path.basename(sdf_path))[0]
    ligand_scratch = os.path.join(scratch_dir, base)
    
    # Clean slate
    shutil.rmtree(ligand_scratch, ignore_errors=True)
    os.makedirs(ligand_scratch, exist_ok=True)

    try:
        # 1) mk_prepare_ligand → PDBQT(s) with proper prefix
        cmd_prep = [
            "mk_prepare_ligand.py",
            "-i", str(sdf_path),
            "--multimol_outdir", ligand_scratch,
            "--multimol_prefix", base
        ]
        run(cmd_prep)

        # 2) Find all produced PDBQTs
        pdbqts = glob.glob(os.path.join(ligand_scratch, f"{base}*.pdbqt"))
        if not pdbqts:
            return base, False, "no PDBQT"

        # 3) mk_export each → _prepared.sdf
        prepared_paths = []
        for pdbqt in pdbqts:
            name = os.path.splitext(os.path.basename(pdbqt))[0]
            out_sdf = os.path.join(out_dir, f"{name}_prepared.sdf")
            cmd_exp = ["mk_export.py", pdbqt, "-s", out_sdf]
            run(cmd_exp)
            prepared_paths.append(out_sdf)

        # Clean up intermediates
        shutil.rmtree(ligand_scratch, ignore_errors=True)
        return base, True, prepared_paths

    except Exception as e:
        shutil.rmtree(ligand_scratch, ignore_errors=True)
        return base, False, str(e)

# ============================================================================
# STEP 4: GENERATE CONFORMERS FOR SDF
# ============================================================================

def generate_conformers_obabel(obabel_bin, in_path, out_path, nconf, rmsd):
    """Generate conformers using Open Babel"""
    base = os.path.basename(in_path)
    try:
        result = subprocess.run([
            obabel_bin,
            '-isdf', in_path,
            '-osdf', '-O', out_path,
            '--conformer', '--nconf', str(nconf),
            '--writeconformers', '--unique',
            '--rmthresh', str(rmsd)
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return base, True, None
        else:
            return base, False, f"returncode={result.returncode}, stderr={result.stderr}"
    except subprocess.TimeoutExpired:
        return base, False, "timeout"
    except Exception as e:
        return base, False, str(e)

def generate_conformers_rdkit_fallback(in_path, out_path, nconf):
    """Generate conformers using RDKit as fallback"""
    base = os.path.basename(in_path)
    try:
        from rdkit.Chem import AllChem
        
        # Read molecule
        mol = Chem.SDMolSupplier(in_path)[0]
        if mol is None:
            return base, False, "RDKit couldn't read SDF"
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate conformers
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=nconf, randomSeed=42)
        
        if not conf_ids:
            return base, False, "No conformers generated"
        
        # Optimize conformers
        for conf_id in conf_ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
        
        # Write conformers
        writer = Chem.SDWriter(out_path)
        for conf_id in conf_ids:
            writer.write(mol, confId=conf_id)
        writer.close()
        
        return base, True, None
    except Exception as e:
        return base, False, f"RDKit fallback failed: {e}"

# ============================================================================
# STEP 5: MERGE ALL SDF FILES INTO SDF.GZ FILES
# ============================================================================

def merge_sdf_files(conf_dir, output_gz):
    """Merge all conformer SDF files into one gzipped file"""
    sdf_files = sorted(conf_dir.glob('*_conf.sdf'))
    
    total_molecules = 0
    with gzip.open(output_gz, 'wt') as out:
        for sdf_file in sdf_files:
            # Extract base name for title
            name = sdf_file.stem.replace('_conf', '')
            first_record = True
            
            with open(sdf_file, 'r') as f:
                for line in f:
                    if first_record:
                        # Replace first line (title) with filename
                        out.write(name + '\n')
                        first_record = False
                    else:
                        out.write(line)
                    
                    # Count molecules
                    if line.strip() == '$$$$':
                        total_molecules += 1
                        first_record = True
    
    return total_molecules

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified peptide processing pipeline")
    parser.add_argument('-i', '--input-file', required=True, help='Input text file with peptide sequences')
    parser.add_argument('-o', '--output', default='merged.sdf.gz', help='Output merged SDF.gz file')
    parser.add_argument('--max-atoms', type=int, default=150, help='Maximum atoms per molecule')
    parser.add_argument('--max-rot', type=int, default=43, help='Maximum rotatable bonds')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of parallel workers')
    parser.add_argument('--nconf', type=int, default=10, help='Number of conformers to generate')
    parser.add_argument('--rmsd', type=float, default=0.3, help='RMSD threshold for conformer pruning')
    parser.add_argument('--scratch', default='scratch', help='Scratch directory')
    parser.add_argument('--keep-intermediates', action='store_true', help='Keep intermediate files')
    parser.add_argument('--use-rdkit-fallback', action='store_true', default=True, help='Use RDKit as fallback for conversions')
    parser.add_argument('--force-rdkit', action='store_true', help='Force use of RDKit instead of Open Babel')
    args = parser.parse_args()

    # Check for required external tools
    obabel_bin = None
    use_obabel = not args.force_rdkit
    
    if use_obabel:
        try:
            obabel_bin = find_obabel()
            print(f"Found Open Babel binary: {obabel_bin}")
            
            # Test if it actually works
            if test_obabel_conversion(obabel_bin):
                print("✓ Open Babel conversion test passed")
            else:
                print("⚠ Open Babel test failed, will use RDKit fallback")
                if not args.use_rdkit_fallback:
                    print("ERROR: Open Babel not working and RDKit fallback disabled")
                    sys.exit(1)
        except Exception as e:
            print(f"⚠ Open Babel not available: {e}")
            if args.use_rdkit_fallback:
                print("Will use RDKit for conversions")
                obabel_bin = None
            else:
                sys.exit(1)
    else:
        print("Using RDKit for all conversions (--force-rdkit specified)")

    # Setup directories
    base_dir = Path().resolve()
    pdb_dir = base_dir / 'pdbs'
    sdf_initial_dir = base_dir / 'sdf_initial'
    sdf_prepared_dir = base_dir / 'sdf_prepared'
    conf_dir = base_dir / 'conformers'
    scratch_dir = base_dir / args.scratch
    
    # Create directories
    for d in [pdb_dir, sdf_initial_dir, sdf_prepared_dir, conf_dir, scratch_dir]:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    try:
        # ============================================================================
        # STEP 1: READ & FILTER SEQUENCES, GENERATE PDBs
        # ============================================================================
        print("=" * 60)
        print("STEP 1: Reading sequences and generating PDB files")
        print("=" * 60)
        
        sequences = read_sequences(args.input_file)
        valid_seqs, invalid_seqs = filter_sequences(sequences, args.max_atoms, args.max_rot)
        print(f"Found {len(valid_seqs)}/{len(sequences)} valid sequences, {len(invalid_seqs)} skipped")

        if invalid_seqs:
            print(f"Skipped sequences: {invalid_seqs[:5]}{'...' if len(invalid_seqs) > 5 else ''}")

        # Generate PDBs in parallel
        print("Generating PDB files...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(build_pdb, seq, pdb_dir) for seq in valid_seqs]
            pdb_results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc='PDB generation', unit='mol'):
                result = future.result()
                if result:
                    pdb_results.append(result)

        print(f"Generated {len(pdb_results)} PDB files")

        # ============================================================================
        # STEP 2: CONVERT PDB TO INITIAL SDF
        # ============================================================================
        print("\n" + "=" * 60)
        print("STEP 2: Converting PDB files to SDF")
        print("=" * 60)
        
        pdb_files = sorted(pdb_dir.glob('*.pdb'))
        conversion_failures = []
        
        for pdb_file in tqdm(pdb_files, desc='PDB → SDF conversion'):
            sdf_file = sdf_initial_dir / f"{pdb_file.stem}.sdf"
            try:
                pdb_to_sdf(pdb_file, sdf_file, obabel_bin, args.use_rdkit_fallback)
            except Exception as e:
                print(f"⚠ Failed to convert {pdb_file.name}: {e}")
                conversion_failures.append(pdb_file.name)

        if conversion_failures:
            print(f"⚠ {len(conversion_failures)} PDB files failed conversion")

        # ============================================================================
        # STEP 3: PREPARE SDF FILES WITH MEEKO
        # ============================================================================
        print("\n" + "=" * 60)
        print("STEP 3: Preparing SDF files with Meeko")
        print("=" * 60)
        
        sdf_files = sorted(sdf_initial_dir.glob('*.sdf'))
        all_prepared_files = []
        
        if not sdf_files:
            print("⚠ No SDF files found to prepare!")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_sdf_with_meeko, sdf, sdf_prepared_dir, scratch_dir): sdf 
                          for sdf in sdf_files}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing ligands"):
                    sdf_file = futures[future]
                    try:
                        base, success, result = future.result()
                        if success:
                            all_prepared_files.extend(result)
                            print(f"✓ {base}: prepared {len(result)} molecules")
                        else:
                            print(f"⚠ {base}: {result}")
                    except Exception as e:
                        print(f"⚠ {os.path.basename(sdf_file)}: ERROR: {e}")

        print(f"Prepared {len(all_prepared_files)} SDF files")

        # ============================================================================
        # STEP 4: GENERATE CONFORMERS
        # ============================================================================
        print("\n" + "=" * 60)
        print("STEP 4: Generating conformers")
        print("=" * 60)
        
        # Initialize variables to avoid UnboundLocalError
        successes = 0
        failures = []
        
        if not all_prepared_files:
            print("⚠ No prepared files found for conformer generation!")
        else:
            prepared_files = sorted(Path(f) for f in all_prepared_files)
            
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = []
                
                for sdf_file in prepared_files:
                    out_file = conf_dir / f"{sdf_file.stem}_conf.sdf"
                    
                    if obabel_bin:
                        future = executor.submit(generate_conformers_obabel, obabel_bin, str(sdf_file), str(out_file), args.nconf, args.rmsd)
                    else:
                        future = executor.submit(generate_conformers_rdkit_fallback, str(sdf_file), str(out_file), args.nconf)
                    
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Conformer generation", unit="file"):
                    base, success, error = future.result()
                    if success:
                        successes += 1
                    else:
                        failures.append((base, error))

            print(f"Generated conformers: {successes}/{len(prepared_files)} succeeded, {len(failures)} failed")
            if failures:
                print("Failures:")
                for base, err in failures[:5]:  # Show first 5 failures
                    print(f"  {base}: {err}")

        # ============================================================================
        # STEP 5: MERGE SDF FILES INTO GZIPPED OUTPUT
        # ============================================================================
        print("\n" + "=" * 60)
        print("STEP 5: Merging SDF files into compressed output")
        print("=" * 60)
        
        output_path = Path(args.output)
        total_molecules = merge_sdf_files(conf_dir, output_path)

        print(f"✔ Merged {successes} files with {total_molecules} total conformers into {args.output}")

    finally:
        # Cleanup intermediate directories unless requested to keep
        if not args.keep_intermediates:
            print("\nCleaning up intermediate files...")
            for d in [pdb_dir, sdf_initial_dir, sdf_prepared_dir, conf_dir, scratch_dir]:
                shutil.rmtree(d, ignore_errors=True)
        else:
            print(f"\nIntermediate files kept in: {', '.join([str(d) for d in [pdb_dir, sdf_initial_dir, sdf_prepared_dir, conf_dir]])}")

    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📁 Final output: {args.output}")
    
    # Final summary
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024*1024)  # MB
        print(f"📊 Output file size: {file_size:.2f} MB")

if __name__ == '__main__':
    main()