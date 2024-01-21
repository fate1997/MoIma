import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple
import pathlib
import inspect
import pprint

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, SequentialLR

from moima.dataset import build_dataset, build_featurizer
from moima.dataset._abc import DataABC, DatasetABC, FeaturizerABC
from moima.model import ModelFactory
from moima.pipeline.config import Config
from moima.utils._util import get_logger, EarlyStopping
from moima.utils.loss_fn import build_loss_fn
from moima.utils.splitter import build_splitter
from moima.utils.schedulers import build_scheduler
from tqdm import tqdm


class PipeABC(ABC):
    """Abstract base class for pipeline. The pipeline is used to train, evaluate, 
        or sample (for generation pipeline). The subclasses should implement the
        `_forward_batch` and `_interested_info` methods. The `_forward_batch` method
        should return the output and loss of the model. The `_interested_info` method
        should return the interested information, such as the accuracy, precision,
        recall, etc.
    
    Args:
        config: The config of the pipeline.
        model_state_dict: The state dict of the model.
        optimizer_state_dict: The state dict of the optimizer.
        is_training: Whether the pipeline is used for training. Default to True.
    
    Methods:
        __call__: Future use.
        _forward_batch: Get the output and loss of the model.
        _interested_info: Get the interested information.
        build_dataset: Load the dataset.
        build_loader: Load the dataloader.
        build_model: Load the model.
        build_loss_fn: Load the loss function.
        build_optimizer: Load the optimizer.
        batch_flatten: Flatten the batch data.
        train: Train the model.
        save: Save the necessary information to reproduce the pipeline.
        from_pretrained: Load the pretrained pipeline.
    
    Attributes:
        config: The config of the pipeline.
        device: The device of the pipeline.
        n_epoch: The number of epochs.
        logger: The logger of the pipeline.
        featurizer: The featurizer of the pipeline.
        model: The model of the pipeline.
        evaluator: The evaluator of the pipeline.
        workspace: The workspace of the pipeline.
    
    Attributes only for training:
        loss_fn: The loss function of the pipeline.
        loader: The dataloader of the pipeline.
        optimizer: The optimizer of the pipeline.
        training_trace: The training trace of the pipeline.
    """
    def __init__(self,
                 config: Config,
                 featurizer: FeaturizerABC = None,
                 model_state_dict: Dict[str, Any] = None,
                 optimizer_state_dict: Dict[str, Any] = None,
                 scheduler_state_dict: Dict[str, Any] = None,
                 is_training: bool = True):
        # Load the config and save it to the workspace
        self.config = config
        
        # Set attributes
        self.device =  self.config.device
        self.n_epoch =  self.config.num_epochs
        self.in_step_mode = self.config.in_step_mode
        
        if featurizer is not None:
            self.featurizer = featurizer
        else:
            self.featurizer = self.build_featurizer()
        self.workspace = self.build_workspace()
        
        # Load the logger
        self.logger = get_logger(f'pipe.log', self.workspace)
        self.logger.info(f"Workspace: {os.path.abspath(self.workspace)}")
        
        # Build training components
        if is_training:
            self.loss_fn = build_loss_fn(**self.config.loss_fn)
            self.loader = self.build_loader()
            self.current_epoch = 0
            self.training_trace = {}
            self.interested_info = {}
        # Update the model config according to the featurizer
        for arg, value in self.featurizer.arg4model.items():
            setattr(self.config, arg, value)
        print(self.config)
        self.model = self.build_model(model_state_dict)
        # Build the optimizer
        self.optimizer = self.build_optimizer(optimizer_state_dict)
        # Build the scheduler
        if self.config.scheduler['name'] == 'none':
            self.scheduler = None
        else:
            self.scheduler = self.build_scheduler(scheduler_state_dict)
        # Save the config
        self.config.to_yaml(os.path.join(self.workspace, 'config.yaml'))
      
    @classmethod
    def from_pretrained(cls, path: str, is_training: bool = True):
        """Load the pretrained model."""
        results = torch.load(path)
        results.update({'is_training': is_training})
        pipe = cls(**results)
        print(f"Pretrained model loaded.")
        return pipe
    
    def __call__(self, batch: DataABC):
        """Future use."""
        pass
     
    @abstractmethod
    def _forward_batch(self, batch: DataABC) -> Tuple[Tensor, Tensor]:
        r"""Get the output and loss of the model. This function could also be used
            to add logging information by setting the `self.interested_info`.
        """
    
    @abstractmethod
    def set_interested_info(self, **kwargs):
        """Get the interested information."""
        return {}
    
    def build_workspace(self) -> str:
        """The workspace of the pipeline."""
        hash_config = self.config.get_hash_key(exclude=self.featurizer.arg4model.keys())
        folder = f'{self.__class__.__name__}_{self.config.desc}_{hash_config}'
        work_dir = os.path.join(self.config.output_folder, folder)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        return work_dir
    
    def build_dataset(self) -> DatasetABC:
        """Load the dataset."""
        self.logger.info("Dataset Loading".center(60, "-"))
        dataset_kwargs = self.config.dataset
        dataset_kwargs.update({'featurizer': self.featurizer})
        
        dataset = build_dataset(**dataset_kwargs)
        self.featurizer = dataset.featurizer
        return dataset
    
    def build_splitter(self):
        """Load the splitter."""
        return build_splitter(**self.config.splitter)
    
    def build_loader(self) -> Dict[str, DataLoader]:
        dataset = self.build_dataset()
        splitter = self.build_splitter()
        train_loader, val_loader, test_loader = splitter(dataset)
        loader = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        return loader
    
    def build_model(self, state_dict: Dict[str, Any] = None) -> nn.Module:
        """Load the model."""
        self.logger.info(f"Model Loading".center(60, "-"))
        model = ModelFactory.create(**self.config.model).to(self.device)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model
    
    def build_featurizer(self):
        """Load the featurizer."""
        # Build featurizer
        featurizer_config = self.config.featurizer
        # Note that the featurizer will be updated from the dataset if the
        # dataset is loaded
        return build_featurizer(**featurizer_config)
    
    def build_loss_fn(self):
        """Load the loss function."""
        loss_fn = build_loss_fn(**self.config.loss_fn)
        return loss_fn
    
    def build_optimizer(self, state_dict: Dict[str, Any] = None) \
                                                -> torch.optim.Optimizer:
        """Load the optimizer."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        if state_dict is not None:
            optimizer.load_state_dict(state_dict)
        return optimizer
    
    def build_scheduler(self, state_dict: Dict[str, Any] = None) \
                                                -> torch.optim.lr_scheduler.LRScheduler:
        """Load the scheduler."""
        scheduler_args = self.config.scheduler
        scheduler_args.update({'optimizer': self.optimizer})
        scheduler = build_scheduler(**scheduler_args)
        if self.config.warmup_interval > 0:
            warmup_scheduler = LambdaLR(self.optimizer, 
                                        lr_lambda=lambda i: i / self.config.warmup_interval)
            scheduler = SequentialLR(optimizer=self.optimizer,
                                     schedulers=[warmup_scheduler, scheduler],
                                     milestones=[self.config.warmup_interval])
        if state_dict is not None:
            scheduler.load_state_dict(state_dict)
        return scheduler        
    
    def batch_flatten(self, 
                      loader: DataLoader, 
                      register_items: List[str]=[],
                      return_numpy: bool=False,
                      register_output: bool=True) -> Dict[str, Any]:
        """Flatten the batch data."""
        self.model.eval()
        results = defaultdict(list)
        if self.config.show_tqdm:
            loader = tqdm(loader, desc='Model running on batch')
        for batch in loader:
            if register_output:
                output, _ = self._forward_batch(batch, calc_loss=False)
                results['output'].append(output.detach().cpu())
            for item in register_items:
                value = getattr(batch, item)
                if isinstance(value, Tensor):
                    value = value.detach().cpu()
                results[item].append(value)
        for k, v in results.items():
            if not isinstance(v[0], Tensor):
                v = np.array(v, dtype=object)
                concat_values = np.concatenate(v, axis=0).tolist()
            else:
                try:
                    concat_values = torch.cat(v, dim=0)
                except RuntimeError:
                    concat_values = v
            if return_numpy and isinstance(concat_values, Tensor):
                concat_values = concat_values.detach().cpu().numpy()
            results[k] = concat_values
        return results
    
    def train(self, n_epoch: int=None):
        """Train the model."""
        self.logger.info('\n'+f"{pprint.pformat(self.config.group_dict)}")
        self.logger.info("Training".center(60, "-"))
        self.model.train()
        if n_epoch is None:
            n_epoch = self.config.num_epochs
        total_iter = len(self.loader["train"]) * n_epoch
        early_stopping = EarlyStopping(patience=self.config.patience,
                                       save_func=self.save)
        do_early_stop = self.config.patience > 0
        is_early_stop = False
        starting_time = time.time()
        current_iter = 0
        initial_epoch = self.current_epoch
        for epoch in range(n_epoch):
            self.current_epoch = epoch + initial_epoch
            for batch in self.loader["train"]:
                """
                for name, params in self.model.named_parameters():
                    if torch.isnan(params).any():
                        print(current_iter, name, params)
                        raise ValueError
                """
                current_iter += 1
                # print(current_iter, self.model.pi_.min().item())
                output, loss_dict = self._forward_batch(batch)
                loss = loss_dict['loss']
                """
                if torch.isnan(loss):
                    self.save(f'{current_iter}-loss-nan.pt')
                    print(current_iter, loss)
                    raise ValueError
                """
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_(self.model.parameters(), 50)
                self.optimizer.step()
                # Step the scheduler
                if self.in_step_mode and self.scheduler is not None and (current_iter % self.config.scheduler_interval == 0 or current_iter <= self.config.warmup_interval):
                    self.scheduler.step()
                # Save the model normally other than saving in early stopping
                if not do_early_stop and self.in_step_mode and\
                    current_iter % self.config.save_interval == 0:
                    self.save()
                # Log the information in mini-batch if in step mode
                if not self.in_step_mode:
                    continue
                if current_iter % self.config.log_interval == 0 or current_iter == total_iter:
                    self.set_interested_info()
                    self._log(self.current_epoch, current_iter, total_iter, loss_dict)
                # Early stopping in mini-batch if in step mode
                if do_early_stop: 
                    early_stopping(self.interested_info[self.config.early_stop_metric])
                if do_early_stop and early_stopping.early_stop:
                    self.logger.info(f"Early stopping at epoch {self.current_epoch}, at step {current_iter}.")
                    is_early_stop = True
                    break
            # Step the scheduler
            if not self.in_step_mode and self.scheduler is not None and ((self.current_epoch+1) % self.config.scheduler_interval == 0 or self.current_epoch < self.config.warmup_interval):
                self.scheduler.step()
            # Log the information in epoch if not in step mode
            if not self.in_step_mode and self.current_epoch % self.config.log_interval == 0 or self.current_epoch == n_epoch - 1 + initial_epoch:
                self.set_interested_info()
                self._log(self.current_epoch, current_iter, total_iter, loss_dict)
            # Save the model normally other than saving in early stopping
            if not do_early_stop and not self.in_step_mode and\
                self.current_epoch % self.config.save_interval == 0:
                self.save()
            
            # Early stopping in epoch if not in step mode
            if not self.in_step_mode and do_early_stop:
                early_stopping(self.interested_info[self.config.early_stop_metric])
                if early_stopping.early_stop:
                    self.logger.info(f"Early stopping at epoch {self.current_epoch}.")
                    is_early_stop = True
                    break
            if is_early_stop:
                break
        
        if do_early_stop:
            self.logger.info(f"Loading the best model from {early_stopping.last_save_path}")
            self.model.load_state_dict(torch.load(early_stopping.last_save_path)['model_state_dict'])
        else:
            self.save()
        
        metrics = self.eval('test')
        print_items = []
        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()
            if type(v) is float:
                print_items.append(f'[{k}: {round(v, 4)}]')
            else:
                print_items.append(f'[{k}: {v}]')
        info = ''.join(print_items)
        self.logger.info(info)
        
        time_elapsed = (time.time() - starting_time) / 60
        self.logger.info(f"Training finished in {time_elapsed:.2f} minutes")
    
    def _log(self, epoch: int, current_iter: int, total_iter: int, loss_dict: Dict[str, Any]):
        if self.scheduler is not None:
            self.interested_info.update({'lr': self.scheduler.get_last_lr()[0]})
        default_info = f'[Epoch {epoch}|{current_iter}/{total_iter}]'
        loss_dict.update(self.interested_info)
        print_items = []
        for k, v in loss_dict.items():
            if isinstance(v, Tensor):
                v = v.item()
            if type(v) is float:
                print_items.append(f'[{k}: {round(v, 4)}]')
            else:
                print_items.append(f'[{k}: {v}]')
        info = default_info + ''.join(print_items)
        self.logger.info(info)
        self.training_trace[current_iter] = loss_dict
    
    def save(self, name: str=None, verbose: bool=True) -> str:
        """Save the necessary information to reproduce the pipeline."""
        basic_info = {
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'featurizer': self.featurizer,
        }
        # basic_info.update(kwargs)
        
        ckpt_folder = os.path.join(self.workspace, 'ckpt')
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        
        if name is None:
            time = datetime.now().strftime("%H-%M-%d-%m-%Y")
            name = f'epoch-{self.current_epoch}_{time}.pt'
        save_path = os.path.join(ckpt_folder, name)
        
        torch.save(basic_info, save_path)
        if verbose:
            self.logger.info(f"Pipeline saved to {save_path}")
        return save_path