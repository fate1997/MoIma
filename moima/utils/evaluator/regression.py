import json
import os
from typing import List, Literal, Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


RDLogger.DisableLog('rdApp.*')


class RegressionMetrics:
    AVAIL_METRICS = ['mse', 'rmse', 'mae', 'aard', 'r2']
    def __init__(self, targets: torch.Tensor, outputs: torch.Tensor):
        if isinstance(targets, torch.Tensor):
            self.outputs = outputs.cpu().numpy()
            self.targets = targets.cpu().numpy()
        else:
            self.outputs = outputs
            self.targets = targets
    
    def get_metrics(self, metric_names: List[str] = None, save_path: str = None):
        if metric_names is None:
            metric_names = self.AVAIL_METRICS
        
        metrics = {}
        for metric in metric_names:
            metrics[metric] = getattr(self, metric)
        
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(metrics, f)
        return metrics
    
    def to_json(self, metric_names: List[str] = None):
        metrics = self.get_metrics(metric_names)
        file_path = os.path.join(self.pipe.config.output_folder,
                                 f'generation_metrics_{self.pipe.config.desc}.json')
        with open(file_path, 'w') as f:
            json.dump(metrics, f)
    
    @property
    def mse(self):
        return mean_squared_error(self.targets, self.outputs).item()
    
    @property
    def rmse(self):
        return np.sqrt(self.mse).item()
    
    @property
    def mae(self):
        return mean_absolute_error(self.targets, self.outputs).item()
    
    @property
    def aard(self):
        return np.mean(np.abs(self.targets-self.outputs)/self.targets).item()
    
    @property
    def r2(self):
        return r2_score(self.targets, self.outputs)