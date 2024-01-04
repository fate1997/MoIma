import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter
import argparse


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = np.array([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


def process_qm9(sdf_path: str = 'example/raw/gdb9.sdf',
                label_path: str = 'example/raw/gdb9.sdf.csv',
                uncharacterized_path: str = 'example/raw/uncharacterized.txt',
                new_sdf_path: str = 'example/raw/qm9.sdf',
                new_label_path: str = 'example/raw/qm9_labels.csv'):


    with open(uncharacterized_path, 'r') as f:
        skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

    targets = pd.read_csv(label_path, index_col=0)
    columns = targets.columns
    targets.drop(targets.index[skip], inplace=True)
    reordered_columns = columns[3:].tolist() + columns[:3].tolist()
    targets = targets[reordered_columns]
    targets = targets * conversion.reshape(1, -1)
    targets.to_csv(new_label_path, index_label='mol_id')

    mols = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    filtered_mols = []
    for i, mol in enumerate(mols):
        if i not in skip:
            filtered_mols.append(mol)
    writer = SDWriter(new_sdf_path)
    for mol in filtered_mols:
        writer.write(mol)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf_path', type=str, default='example/raw/gdb9.sdf')
    parser.add_argument('--label_path', type=str, default='example/raw/gdb9.sdf.csv')
    parser.add_argument('--uncharacterized_path', type=str, default='example/raw/uncharacterized.txt')
    parser.add_argument('--new_sdf_path', type=str, default='example/raw/qm9.sdf')
    parser.add_argument('--new_label_path', type=str, default='example/raw/qm9_labels.csv')
    args = parser.parse_args()

    process_qm9(args.sdf_path, args.label_path, args.uncharacterized_path, args.new_sdf_path, args.new_label_path)