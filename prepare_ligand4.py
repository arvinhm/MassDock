#!/usr/bin/env pythonsh
# -*- coding: utf-8 -*-
# prepare_ligand4.py - patched for peptide ligands and safe list handling

import os
import sys
import getopt

from MolKit import Read
from AutoDockTools.MoleculePreparation import AD4LigandPreparation

def usage():
    """Print usage."""
    print "Usage: prepare_ligand4.py -l ligand.pdb [options]"
    print
    print "  -l  ligand filename (.pdb, .mol2 or .pdbqt)"
    print "  -v  verbose output"
    print "  -o  output PDBQT filename (default: ligand_name.pdbqt)"
    print "  -d  dictionary file for types & torsion info"
    print "  -A  repairs to make: bonds_hydrogens, bonds, hydrogens"
    print "  -C  do not add charges (default: add Gasteiger)"
    print "  -p  preserve input charges on atom type (e.g. -p Zn)"
    print "  -U  cleanup: nphs_lps, nphs, lps, '' (default: nphs_lps)"
    print "  -B  bond types to allow to rotate (default: backbone)"
    print "  -R  atom index for root (default: auto)"
    print "  -F  check/use largest non-bonded fragment"
    print "  -I  bonds to inactivate (format idx_idx_idx_idx)"
    print "  -Z  inactivate all torsions"
    print "  -g  attach non-bonded fragments"
    print "  -h  this help"

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'l:vo:d:A:Cp:U:B:R:FI:Zgh')
    except getopt.GetoptError, err:
        sys.stderr.write("Error: %s\n" % err)
        usage()
        sys.exit(2)

    ligand_filename           = None
    verbose                   = False
    repairs                   = ""
    charges_to_add            = 'gasteiger'
    preserve_charge_types     = ''
    cleanup                   = "nphs_lps"
    allowed_bonds             = "backbone"
    root                      = 'auto'
    outputfilename            = None
    check_for_fragments       = False
    bonds_to_inactivate       = ""
    inactivate_all_torsions   = False
    attach_nonbonded_fragments= False
    dict_file                 = None
    mode                      = 'automatic'

    for o, a in opts:
        if   o == '-l': ligand_filename           = a
        elif o == '-v': verbose                   = True
        elif o == '-o': outputfilename            = a
        elif o == '-d': dict_file                 = a
        elif o == '-A': repairs                   = a
        elif o == '-C': charges_to_add            = None
        elif o == '-p': preserve_charge_types    += a + ','
        elif o == '-U': cleanup                   = a
        elif o == '-B': allowed_bonds             = a
        elif o == '-R': root                      = a
        elif o == '-F': check_for_fragments       = True
        elif o == '-I': bonds_to_inactivate       = a
        elif o == '-Z': inactivate_all_torsions   = True
        elif o == '-g': attach_nonbonded_fragments= True
        elif o in ('-h','--help'):
            usage(); sys.exit(0)

    if not ligand_filename:
        sys.stderr.write("prepare_ligand4.py: ligand filename (-l) is required.\n")
        usage()
        sys.exit(2)

    # Read into a plain list to avoid slicing bugs
    raw_mols = Read(ligand_filename)
    mol_list = [m for m in raw_mols]
    if not mol_list:
        sys.stderr.write("ERROR: could not read any molecules from '%s'\n" % ligand_filename)
        sys.exit(1)

    # pick the largest molecule if multiple
    mol = mol_list[0]
    for m in mol_list[1:]:
        if len(m.allAtoms) > len(mol.allAtoms):
            mol = m

    if verbose:
        print "Loaded", ligand_filename, "→", len(mol.allAtoms), "atoms"

    mol.buildBondsByDistance()

    # preserve existing charges if requested
    preserved = {}
    if charges_to_add and preserve_charge_types:
        for t in preserve_charge_types.split(','):
            if not t: continue
            for a in mol.allAtoms.get(lambda x: x.autodock_element == t):
                if a.chargeSet is not None:
                    preserved[a] = (a.chargeSet, a.charge)

    if verbose:
        print "Running AD4LigandPreparation..."

    LPO = AD4LigandPreparation(
        mol,
        mode,
        repairs,
        charges_to_add,
        cleanup,
        allowed_bonds,
        root,
        outputfilename=outputfilename,
        dict=dict_file,
        check_for_fragments=check_for_fragments,
        bonds_to_inactivate=bonds_to_inactivate,
        inactivate_all_torsions=inactivate_all_torsions,
        attach_nonbonded_fragments=attach_nonbonded_fragments
    )

    # restore any preserved charges
    if charges_to_add:
        for atom, (setname, val) in preserved.items():
            atom._charges[setname] = val
            atom.chargeSet       = setname

    if verbose:
        print "Finished, return code:", mol.returnCode

    if mol.returnCode != 0:
        sys.stderr.write(mol.returnMsg + "\n")
    sys.exit(mol.returnCode)
