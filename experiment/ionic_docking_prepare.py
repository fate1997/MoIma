import os
from typing import List

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem, rdPartialCharges
from oddt.toolkits.extras.rdkit import MolToPDBQTBlock
import subprocess


def get_charge(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    contribs = [x.GetFormalCharge() for x in mol.GetAtoms()]
    charge = round(sum(contribs))
    return charge


def get_cation_anion(raw_csv: str='example/IL_mp_processed.csv',
                     pqbdt_dir: str='example/PDBQT_files'):

    df = pd.read_csv(raw_csv)
    
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
    
    for name, smiles in tqdm(cations.items(), desc='get cations'):
        cation = Chem.MolFromSmiles(smiles)
        cation = Chem.AddHs(cation)
        cation = get_conformer(cation)
        with open(os.path.join(pqbdt_dir, f'{name}.pdbqt'), 'w') as f:
            f.write(MolToPDBQTBlock(cation, computeCharges=True))
        
    for name, smiles in tqdm(anions.items(), desc='get anions'):
        anion = Chem.MolFromSmiles(smiles)
        anion = Chem.AddHs(anion)
        anion = get_conformer(anion)
        with open(os.path.join(pqbdt_dir, f'{name}.pdbqt'), 'w') as f:
            f.write(MolToPDBQTBlock(anion, computeCharges=True))


def get_conformer(rdmol: Chem.Mol,
                  work_dir = r"E:\code\imperial\MoIma\.pytest_temp") -> Chem.Mol:
    original_dir = os.path.abspath(os.getcwd())
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    etkdg = AllChem.ETKDGv3()
    etkdg.randomSeed = 42
    success = AllChem.EmbedMolecule(rdmol, etkdg)
    if not success:
        AllChem.EmbedMultipleConfs(rdmol, numConfs=1)
    AllChem.MolToXYZFile(rdmol, "temp.xyz")
    rdPartialCharges.ComputeGasteigerCharges(rdmol)
    contribs = [x.GetDoubleProp('_GasteigerCharge') for x in rdmol.GetAtoms()]
    charge = round(sum(contribs))
    process = subprocess.run([r'F:\xtb-6.6.1\bin\xtb.exe',
                os.path.join(f'{work_dir}', r'temp.xyz'),
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
    os.chdir(original_dir)
    return rdmol


if __name__ == '__main__':
    get_cation_anion()
    