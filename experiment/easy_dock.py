import pandas as pd
import yaml
import os
from rdkit.Chem import AllChem, rdPartialCharges
from rdkit import Chem
from easydock.run_dock import docking
from easydock.vina_dock import mol_dock
from tqdm import tqdm
from rdkit.Geometry import Point3D


DEFAULT_SETTING = {
    'protein': './example/PDBQT_files/cation_test.pdbqt',
    'protein_setup': './example/docking/grid.txt',
    'exhaustiveness': 8,
    'seed': 42,
    'n_poses': 1,
    'ncpu': 8,
}

def dock_ionic(raw_csv: str='example/IL_mp_processed.csv',
               pdbqt_dir: str='example/docking/pdbqt',
               optimized_dir: str='example/docking/optimized',
               skip_failed: bool=True):
    df = pd.read_csv(raw_csv)
    writer = Chem.SDWriter(os.path.join('./example/docking/', 'IL_MP.sdf'))
    if skip_failed:
        with open('./example/docking/failed_list.txt', 'r') as f:
            failed_list = eval(f.read())
    else:
        failed_list = []
    for i in tqdm(range(len(df))):
        print(i)
        if i <= int(failed_list[-1]):
            continue
        
        row = df.iloc[i]
        final_sdf_path = os.path.join('example/docking/final_sdf', f'{row.CATION_ANION}.sdf')
        if os.path.exists(final_sdf_path):
            continue
        cation_name, anion_name = row.CATION_ANION.split('_')
        cation_pqbdt = os.path.join(pdbqt_dir, f'{cation_name}.pdbqt')
        
        cation_sdf = os.path.join(optimized_dir, f'{cation_name}_optimized.sdf')
        anion_sdf = os.path.join(optimized_dir, f'{anion_name}_optimized.sdf')
        if not os.path.exists(cation_sdf) or not os.path.exists(anion_sdf):
            failed_list.append(i)
            continue
        
        cation_rdmol = Chem.SDMolSupplier(cation_sdf, removeHs=False)[0]
        rdPartialCharges.ComputeGasteigerCharges(cation_rdmol)
        contribs = [x.GetDoubleProp('_GasteigerCharge') for x in cation_rdmol.GetAtoms()]
        positions = cation_rdmol.GetConformer().GetPositions()
        pos_max_charge = positions[contribs.index(max(contribs))].tolist()
        
        # Write 'grid.txt'
        with open('./example/docking/grid.txt', 'w') as f:
            f.write(f'center_x = {pos_max_charge[0]}\n')
            f.write(f'center_y = {pos_max_charge[1]}\n')
            f.write(f'center_z = {pos_max_charge[2]}\n')
            f.write('size_x = 20\n')
            f.write('size_y = 20\n')
            f.write('size_z = 20\n')
        
        # Write config.yaml
        DEFAULT_SETTING['protein'] = cation_pqbdt
        with open('./example/docking/config.yaml', 'w') as f:
            yaml.dump(DEFAULT_SETTING, f)
        
        # Docking
        try:
            anion_rdmol = Chem.SDMolSupplier(anion_sdf, removeHs=False)[0]
            anion_rdmol.SetProp('_Name', anion_name)
            results = docking([anion_rdmol], 
                            dock_func=mol_dock, 
                            dock_config='./example/docking/config.yaml', 
                            ncpu=8)
            res_dict = {}
            for mol_id, res in results:
                res_dict[mol_id] = res
            assert len(res_dict) == 1
            anion_docked_rdmol = Chem.MolFromMolBlock(res_dict[anion_name]['mol_block'],
                                                    removeHs=False)
        except Exception as e:
            failed_list.append(i)
            with open('./example/docking/failed_list.txt', 'w') as f:
                f.write(str(failed_list))
            continue
        """positions = anion_docked_rdmol.GetConformer().GetPositions()
        # Set positions
        conf = anion_rdmol.GetConformer()
        for i in range(anion_rdmol.GetNumAtoms()):
            x,y,z = positions[i]
            conf.SetAtomPosition(i,Point3D(x,y,z))"""
        
        cation_anion = Chem.CombineMols(cation_rdmol, anion_docked_rdmol)
        cation_anion.SetProp('_Name', f'{cation_name}_{anion_name}')
        
        # Save single docking result
        single_writer = Chem.SDWriter(final_sdf_path)
        single_writer.write(cation_anion)
        single_writer.close()
        
        # Save docked complex
        writer.write(cation_anion)
    
    writer.close()
    


if __name__ == '__main__':
    dock_ionic()
        