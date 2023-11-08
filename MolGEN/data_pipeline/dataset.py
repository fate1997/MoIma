from MolGEN.data_pipeline._base import BaseDataset
from MolGEN.data_pipeline.featurizer import SMILESFeaturizer
from MolGEN.data_pipeline._base import BaseFeaturizer
import pandas as pd
from tqdm import tqdm


class SMILESDataset(BaseDataset):
    def __init__(self, 
                 raw_path: str, 
                 Featurizer: BaseFeaturizer,
                 featurizer_config: dict=None, 
                 processed_path: str = None, 
                 replace: bool = False):
        super().__init__(raw_path, 
                         Featurizer, 
                         featurizer_config,
                         processed_path,
                         replace)
        
    
    def _prepare_data(self):
        r"""Prepare data for the dataset."""
        assert self.raw_path.endswith('csv')
        df = pd.read_csv(self.raw_path)
        smiles_list = df.smiles.tolist()
        featurizer = SMILESFeaturizer.from_smiles_list(smiles_list)
        
        data_list = []
        for smiles in tqdm(smiles_list, 'SMILES featurization'):
            data = featurizer(smiles)
            data_list.append(data)

        return data_list, featurizer.__dict__()