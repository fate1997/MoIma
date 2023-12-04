import os
import time
from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from typing import Any, Dict, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from moima.pipeline.config import DefaultConfig
from moima.dataset import DatasetFactory, FeaturizerFactory
from moima.dataset._abc import DataABC, DatasetABC
from moima.model import ModelFactory
from moima.utils._util import get_logger
from moima.utils.loss_fn import LossCalcFactory
from moima.utils.splitter import SplitterFactory
from moima.utils.splitter._abc import SplitterABC


class PipeABC(ABC):
    
    DEFAULT_SAVEITEMS = ['model_state_dict',
                         'optimizer_state_dict', 
                         'config']
    
    def __init__(self, 
                 config: DefaultConfig,
                 model_state_dict: Dict[str, Any] = None,
                 optimizer_state_dict: Dict[str, Any] = None,
                 **kwargs):
        self.config = config
        self.config.to_yaml(os.path.join(self.workspace, 'config.yaml'))
        
        self.device =  self.config.device
        self.n_epoch =  self.config.num_epochs
        
        self.logger = get_logger(f'pipe.log', 
                                 os.path.join(self.workspace))
                
        self.loss_fn = LossCalcFactory.create(**self.config.loss_fn)
        self.build_featurizer()
        self.build_loader()
        self.build_model()
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.training_trace = {}
       
    @abstractmethod
    def _forward_batch(self, batch: DataABC) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model for one iteration."""
    
    @abstractmethod
    def _interested_info(self, batch: DataABC, output: torch.Tensor) -> Dict[str, Any]:
        """Get the interested information."""
        
    @abstractproperty
    def custom_saveitems(self) -> Dict[str, Any]:
        """The items that will be saved besides `DEFAULT_SAVEITEMS."""
        
    @property
    def workspace(self) -> str:
        """The workspace of the pipeline."""
        folder = f'{self.__class__.__name__}_{self.config.desc}_{self.config.hash_key}'
        work_dir = os.path.join(self.config.output_folder, folder)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        return work_dir
    
    def build_featurizer(self):
        """Build the featurizer."""
        featurizer_config = self.config.featurizer
        self.featurizer = FeaturizerFactory.create(**featurizer_config)
    
    def build_dataset(self) -> DatasetABC:
        """Load the dataset."""
        self.logger.info("Dataset Loading".center(60, "-"))
        dataset_kwargs = self.config.dataset
        dataset_kwargs.update({'featurizer': self.featurizer})
        dataset = DatasetFactory.create(**dataset_kwargs)
        self.featurizer = dataset.featurizer
        return dataset
    
    def build_splitter(self) -> SplitterABC:
        return SplitterFactory.create(**self.config.splitter)
    
    def build_loader(self):
        dataset = self.build_dataset()
        splitter = self.build_splitter()
        train_loader, val_loader, test_loader = splitter(dataset)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    def build_model(self):
        """Load the model."""
        self.logger.info("Model Loading".center(60, "-"))
        model_config = self.config.model
        for key, value in model_config.items():
            if key in self.featurizer.__dict__:
                featurizer_value = getattr(self.featurizer, key)
                model_config[key] = featurizer_value
        self.model = ModelFactory.create(**model_config).to(self.device)
    
    def get_loader(self, split: str='val'):
        """Get the loader."""
        if split == 'train':
            return self.train_loader
        elif split == 'val':
            return self.val_loader
        elif split == 'test':
            return self.test_loader
        else:
            raise ValueError(f"Split {split} is not supported.")
    
    def batch_flatten(self, split: str='val'):
        self.model.eval()
        loader = self.get_loader(split)
        labels = []
        outputs = []
        for batch in loader:
            output, _ = self._forward_batch(batch)
            labels.append(batch.y)
            outputs.append(output)
        labels = torch.cat(labels, dim=0)
        outputs = torch.cat(outputs, dim=0)
        return labels, outputs
    
    def train(self):
        """Train the model."""
        self.logger.info(f"{repr(self.config)}")
        if 'train_loader' not in self.__dict__:
            self.build_loader()
        if 'model' not in self.__dict__:
            self.build_model()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.logger.info("Training".center(60, "-"))
        self.model.train()
        n_epoch = self.config.num_epochs
        total_iter = len(self.train_loader.dataset) // self.train_loader.batch_size * n_epoch
        starting_time = time.time()
        current_iter = 0
        for epoch in range(n_epoch):
            self.current_epoch = epoch
            for batch in self.train_loader:
                for name, params in self.model.named_parameters():
                    if torch.isnan(params).any():
                        print(current_iter, name, params)
                        raise ValueError
                current_iter += 1
                
                output, loss = self._forward_batch(batch)
                if torch.isnan(loss):
                    print(current_iter, loss)
                    raise ValueError

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 50)
                self.optimizer.step()             
                
                if current_iter % self.config.log_interval == 0 or current_iter == total_iter:
                    default_info = f'[Epoch {epoch}|{current_iter}/{total_iter}]'\
                                   f'[Loss: {loss.item():.4f}]'
                    interested_info = self._interested_info(batch, output)
                    info = default_info + ''.join([f'[{k}: {round(v, 4) if type(v) is float else v}]' for k, v in interested_info.items()])
                    self.logger.info(info)
                if current_iter % self.config.save_interval == 0:
                    self.save(**self.custom_saveitems)
        
        self.save(**self.custom_saveitems)
        
        time_elapsed = (time.time() - starting_time) / 60
        self.logger.info(f"Training finished in {time_elapsed:.2f} minutes")
    
    @classmethod
    def from_pretrained(cls, path: str):
        """Load the pretrained model."""
        results = torch.load(path)
        print(f"Pretrained model loaded.")
        return cls(**results)
    
    def save(self, **kwargs):
        """Save the necessary information to reproduce the pipeline."""
        basic_info = {
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        basic_info.update(kwargs)
        
        ckpt_folder = os.path.join(self.workspace, 'ckpt')
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        
        time = datetime.now().strftime("%H-%M-%d-%m-%Y")
        save_name = f'epoch-{self.current_epoch}_{time}.pt'
        save_path = os.path.join(ckpt_folder, save_name)
        
        torch.save(basic_info, save_path)
        self.logger.info(f"Pipeline saved to {save_path}")