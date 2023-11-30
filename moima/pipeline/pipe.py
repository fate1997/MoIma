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
from moima.dataset._abc import DataABC
from moima.model import ModelFactory
from moima.utils._util import get_logger
from moima.utils.loss_fn import LossCalcFactory
from moima.utils.splitter import SplitterFactory


class PipeABC(ABC):
    
    DEFAULT_SAVEITEMS = ['model_state_dict',
                         'optimizer_state_dict', 
                         'config']
    
    def __init__(self, config: DefaultConfig):
        self.config = config
        
        self.device =  self.config.device
        self.n_epoch =  self.config.num_epochs
        
        self.logger = get_logger(f'{self.config.desc}'+'.log', 
                                 os.path.join( self.config.output_folder, 'logs/'))
                
        self.loss_fn = LossCalcFactory.create(**self.config.loss_fn)
        self.featurizer = FeaturizerFactory.create(**self.config.featurizer)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
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
    
    def load_dataset(self):
        """Load the dataset."""
        self.logger.info("Dataset Loading".center(60, "-"))
        dataset_kwargs = self.config.dataset
        dataset_kwargs.update({'featurizer': self.featurizer})
        dataset = DatasetFactory.create(**dataset_kwargs)
        splitter = SplitterFactory.create(**self.config.splitter)
        train_loader, val_loader, test_loader = splitter(dataset)
        self.featurizer = dataset.featurizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    def load_model(self):
        """Load the model."""
        self.logger.info("Model Loading".center(60, "-"))
        model_config = self.config.model
        for key, value in model_config.items():
            if key in self.featurizer.__dict__:
                featurizer_value = getattr(self.featurizer, key)
                model_config[key] = featurizer_value
        self.model = ModelFactory.create(**model_config).to(self.device)
    
    def train(self):
        """Train the model."""
        self.logger.info(f"{repr(self.config)}")
        if 'train_loader' not in self.__dict__:
            self.load_dataset()
        if 'model' not in self.__dict__:
            self.load_model()
        
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
    
    def load_pretrained(self, path: str):
        """Load the pretrained model."""
        results = torch.load(path)
        self.config = results['config']
        self.model.load_state_dict(results['model_state_dict'])
        self.optimizer.load_state_dict(results['optimizer_state_dict'])
        for key, value in self.custom_saveitems.items():
            setattr(self, key, value)
        self.logger.info(f"Pretrained model loaded from {path}")
    
    def save(self, **kwargs):
        """Save the necessary information to reproduce the pipeline."""
        basic_info = {
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        basic_info.update(kwargs)
        
        time = datetime.now().strftime("%H-%M-%d-%m-%Y")
        save_name = f'{self.__class__.__name__}_{self.config.desc}_{time}.pt'
        save_path = os.path.join(self.config.output_folder, save_name)
        
        torch.save(basic_info, save_path)
        self.logger.info(f"Pipeline saved to {save_path}")