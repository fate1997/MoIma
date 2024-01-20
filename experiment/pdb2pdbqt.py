from glob import glob
import os


pdbqt_dir = './example/docking/pdbqt'
pdb_files = glob(os.path.join(pdbqt_dir, '*.pdb'))
for i, pdb_path in enumerate(pdb_files):
    print(i)
    name = os.path.splitext(pdb_path)[0]
    try:
        os.system('prepare_receptor -r {pdb_path} -o {name}.pdbqt'.format(pdb_path=pdb_path, name=name))
    except:
        print(pdb_path)