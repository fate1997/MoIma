import os
from typing import List

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem, rdPartialCharges
import subprocess
from vina import Vina
from meeko import MoleculePreparation, RDKitMolCreate


def get_charge(smiles):
    mol = Chem.MolFromSmiles(smiles)
    contribs = [x.GetFormalCharge() for x in mol.GetAtoms()]
    charge = round(sum(contribs))
    return charge


def get_cation_anion(raw_csv='example/IL_mp_processed.csv',
                     optimized_dir='example/docking/optimized',
                     pqbdt_dir='example/docking/pdbqt'):

    df = pd.read_csv(raw_csv)
    
    # Extract cation and anion SMILES {name: smiles}
    cations = {}
    anions = {}
    for i in range(len(df)):
        row = df.iloc[i]
        cation_name, anion_name = row.CATION_ANION.split('_')
        smiles1, smiles2 = row.IL_SMILES.split('.')
        if get_charge(smiles1) > 0:
            cation_smiles, anion_smiles = smiles1, smiles2
        else:
            cation_smiles, anion_smiles = smiles2, smiles1
        
        if cation_name not in cations:
            cations[cation_name] = cation_smiles
        else:
            assert cations[cation_name] == cation_smiles
        
        if anion_name not in anions:
            anions[anion_name] = anion_smiles
        else:
            assert anions[anion_name] == anion_smiles

    # Get cation and anion conformers and write to sdf files
    # Register conformers to cations and anions {name: mol}
    for name, smiles in tqdm(cations.items(), desc='get cations'):
        cation = Chem.MolFromSmiles(smiles)
        cation = Chem.AddHs(cation)
        cation.SetProp('_Name', name)
        cation_path = os.path.join(optimized_dir, f'{name}_optimized.sdf')
        if os.path.exists(cation_path):
            cations[name] = Chem.SDMolSupplier(cation_path)[0]
        else:
            cation = get_conformer(cation)
            cations[name] = cation
            if cation is None:
                print(name)
                continue
            writer = Chem.SDWriter(cation_path)
            writer.write(cation)
            writer.close()

    for name, smiles in tqdm(anions.items(), desc='get anions'):
        anion = Chem.MolFromSmiles(smiles)
        anion = Chem.AddHs(anion)
        anion.SetProp('_Name', name)
        anion_path = os.path.join(optimized_dir, f'{name}_optimized.sdf')
        if os.path.exists(anion_path):
            anions[name] = Chem.SDMolSupplier(anion_path)[0]
        else:
            anion = get_conformer(anion)
            anions[name] = anion
            if anion is None:
                print(name)
                continue
            writer = Chem.SDWriter(anion_path)
            writer.write(anion)
            writer.close()
    
    # Prepare receptor pdbqt files
    for key, cation in cations.items():
        pdb_path = os.path.join(pqbdt_dir, f'{key}.pdb')
        if cation is None:
            continue
        with open(pdb_path, 'w') as pdb_file:
            pdb_file.write(Chem.MolToPDBBlock(cation))
        # os.system(f'prepare_receptor -r {pdb_path} -o {pqbdt_dir}/{key}.pdbqt')


def get_conformer(rdmol,
                  work_dir = "./.pytest_temp"):
    original_dir = os.path.abspath(os.getcwd())
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    etkdg = AllChem.ETKDGv3()
    etkdg.randomSeed = 42
    success = AllChem.EmbedMolecule(rdmol, etkdg)
    if not success:
        AllChem.EmbedMultipleConfs(rdmol, numConfs=1)
    if rdmol.GetNumConformers() != 1:
        os.chdir(original_dir)
        return None
    AllChem.MolToXYZFile(rdmol, "temp.xyz")
    try:
        rdPartialCharges.ComputeGasteigerCharges(rdmol)
        contribs = [x.GetDoubleProp('_GasteigerCharge') for x in rdmol.GetAtoms()]
        charge = round(sum(contribs))
        process = subprocess.run(['xtb',
                    'temp.xyz',
                    r'--opt',
                    f'normal',
                    f'--charge',
                    f'{charge}'],
                    check=True,
                    capture_output=True,
                    text=True)
        mol = Chem.MolFromXYZFile("xtbopt.xyz")
        rdmol.AddConformer(mol.GetConformer(), assignId=True)
        rdmol.RemoveConformer(0)
    except:
        print(f'Cannot be optimized via XTB.')
        rdmol = None
    os.chdir(original_dir)
    return rdmol


if __name__ == '__main__':
    get_cation_anion()
    