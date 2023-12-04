from typing import Any, Dict

import torch

from moima.dataset._abc import DatasetABC
from moima.pipeline.config import DefaultConfig
from moima.pipeline.downstream.config import DownstreamPipeConfig
from moima.pipeline.pipe import PipeABC
from moima.dataset.descriptor_vec.dataset import DescDataset
from moima.dataset.descriptor_vec.featurizer import DescFeaturizer
from tqdm import tqdm
from moima.pipeline import AVAILABLE_PIPELINES


class DownstreamPipe(PipeABC):
    def __init__(self, 
                 config: DownstreamPipeConfig,  
                 model_state_dict: Dict[str, Any] = None, 
                 optimizer_state_dict: Dict[str, Any] = None,):
        super().__init__(config, model_state_dict, optimizer_state_dict)

    def _forward_batch(self, batch):
        """Train the model for one iteration."""
        batch.to(self.device)
        output = self.model(batch)
        loss = self.loss_fn(batch.y, output)
        return output, loss
    
    def _interested_info(self, batch, output):
        """Get the interested information."""
        return {}
    
    @property
    def custom_saveitems(self) -> Dict[str, Any]:
        """The items that will be saved besides `DEFAULT_SAVEITEMS."""
        return {}
    
    def _desc_from_pretrained(self):
        pretrained_pipe_class = AVAILABLE_PIPELINES[self.config.pretrained_pipe_class]
        pretrained_pipe_path = self.config.pretrained_pipe_path
        self.pretrained_pipe = pretrained_pipe_class.from_pretrained(pretrained_pipe_path)
        pretrained_pipe = self.pretrained_pipe
        pretrained_pipe.config.raw_path = self.config.raw_path
        pretrained_pipe.config.batch_size = 512
        pretrained_pipe.config.vocab_path = self.config.vocab_path
        dataset = pretrained_pipe.build_dataset()
        splitter = pretrained_pipe.build_splitter()
        loader = splitter.create_loader(dataset)
        reprs = []
        smiles_list = []
        for batch in tqdm(loader, desc='Featurizing by pretrained pipe'):
            batch.to(pretrained_pipe.device)
            reprs.append(pretrained_pipe.model.get_repr(batch))
            smiles_list.extend(batch.smiles)
        reprs = torch.cat(reprs, dim=0).detach().cpu().numpy()
        desc_dict = {smiles: reprs[i] for i, smiles in enumerate(smiles_list)}
        return desc_dict
    
    def build_featurizer(self):
        """Build the featurizer."""
        featurizer_config = self.config.featurizer
        if 'dict' in self.config.mol_desc:
            desc_dict = self._desc_from_pretrained()
            featurizer_config['additional_desc_dict'] = desc_dict
        featurizer_config.pop('name', None)
        self.featurizer = DescFeaturizer(**featurizer_config)
    
    def build_dataset(self) -> DatasetABC:
        """Build the dataset."""
        desc_dataset = DescDataset(self.config.raw_path, 
                                   self.config.label_col,
                                   self.featurizer, 
                                   processed_path=self.config.processed_path, 
                                   force_reload=self.config.force_reload, 
                                   save_processed=self.config.save_processed)
        return desc_dataset