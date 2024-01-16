from typing import Any, Dict

import torch
from tqdm import tqdm

from moima.dataset._abc import DatasetABC
from moima.dataset.descriptor_vec.dataset import DescDataset, VecBatch
from moima.dataset import FEATURIZER_REGISTRY, build_featurizer
from moima.pipeline import AVAILABLE_PIPELINES
# from moima.pipeline.downstream.config import DownstreamPipeConfig
from moima.pipeline.pipe import PipeABC, FeaturizerABC
import inspect
from moima.pipeline.config import create_config_class, ArgType, Config
from dataclasses import field
from moima.utils.evaluator.regression import RegressionMetrics


def create_downstream_config_class(class_name: str,
                        dataset_name: str,
                        model_name: str,
                        splitter_name: str,
                        loss_fn_name: str,
                        scheduler_name: str):
    r"""Create the downstream config class."""
    pretrained_pipe_class = ('pretrained_pipe_class', 
                             str, 
                             field(default='VaAEPipe',
                                   metadata={'help': 'The pretrained pipeline class.',
                                             'type': ArgType.GENERAL}))
    pretrained_pipe_path = ('pretrained_pipe_path',
                            str,
                            field(default=None,
                                  metadata={'help': 'The pretrained pipeline path.',
                                            'type': ArgType.GENERAL}))
    Config = create_config_class(class_name,
                                    dataset_name,
                                    model_name,
                                    splitter_name,
                                    loss_fn_name,
                                    scheduler_name,
                                    [pretrained_pipe_class, pretrained_pipe_path])
    setattr(inspect.getmodule(DownstreamPipe), class_name, Config)
    Config.__module__ = inspect.getmodule(DownstreamPipe).__name__
    return Config                             


class DownstreamPipe(PipeABC):
    def __init__(self, 
                 config: Config, 
                 featurizer: FeaturizerABC = None,
                 model_state_dict: Dict[str, Any] = None,
                 optimizer_state_dict: Dict[str, Any] = None,
                 scheduler_state_dict: Dict[str, Any] = None,
                 is_training: bool = True):
        super().__init__(config, 
                         featurizer,
                         model_state_dict, 
                         optimizer_state_dict, 
                         scheduler_state_dict,
                         is_training)

    def _forward_batch(self, batch, calc_loss=True):
        """Train the model for one iteration."""
        batch.to(self.device)
        output = self.model(batch)
        y = batch.y
        if calc_loss:
            loss = self.loss_fn(y, output).float()
            return output, {'loss': loss}
        return output, {}
    
    def set_interested_info(self):
        """Get the interested information."""
        results = self.batch_flatten(self.loader['val'], ['y'], return_numpy=True)
        metrics = RegressionMetrics(results['y'], results['output'])
        self.interested_info.update({'val_MAE': metrics.mae})
    
    def eval(self, loader_name: str='test'):
        """Evaluate the model."""
        self.logger.info('Evaluating'.center(60, "-"))
        loader = self.loader[loader_name]
        eval_outputs = self.batch_flatten(loader, register_items=['y'], return_numpy=True)
        metrics = RegressionMetrics(eval_outputs['y'], eval_outputs['output'])
        return metrics.get_metrics()
    
    @property
    def custom_saveitems(self) -> Dict[str, Any]:
        """The items that will be saved besides `DEFAULT_SAVEITEMS."""
        return {}
    
    def _desc_from_pretrained(self):
        r"""Get the descriptors from the pretrained pipe."""
        # Load the pretrained pipe
        pretrained_pipe_class = AVAILABLE_PIPELINES[self.config.pretrained_pipe_class]
        pretrained_pipe_path = self.config.pretrained_pipe_path
        pretrained_pipe = pretrained_pipe_class.from_pretrained(pretrained_pipe_path, is_training=False)
        
        # Build the dataset and loader for the pretrained pipe
        pretrained_pipe.config.raw_path = self.config.raw_path
        pretrained_pipe.config.force_reload = True
        pretrained_pipe.config.save_processed = False
        pretrained_pipe.config.vocab_path = 'example\ilthermo_ILs_vocab.pkl'
        dataset = pretrained_pipe.build_dataset()
        loader = dataset.create_loader(batch_size=512)
        
        # Get the descriptors dict {smiles: desc}
        reprs = []
        smiles_list = []
        for batch in tqdm(loader, desc='Featurizing by pretrained pipe'):
            batch.to(pretrained_pipe.device)
            reprs.append(pretrained_pipe.model.get_repr(batch).detach().cpu())
            smiles_list.extend(batch.smiles)
        reprs = torch.cat(reprs, dim=0).numpy()
        desc_dict = {smiles: reprs[i] for i, smiles in enumerate(smiles_list)}
        return desc_dict
    
    def build_featurizer(self):
        """Build the featurizer."""
        featurizer_config = self.config.featurizer
        if getattr(self.config, 'dataset_name') != 'desc_vec':
            return build_featurizer(**featurizer_config)
        if 'dict' in self.config.mol_desc:
            desc_dict = self._desc_from_pretrained()
            featurizer_config['addi_desc_dict'] = desc_dict
        return build_featurizer(**featurizer_config)