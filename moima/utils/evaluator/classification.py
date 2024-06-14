import json
import os
from typing import List, Literal, Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


RDLogger.DisableLog('rdApp.*')


class ClassificationMetrics:
    AVAIL_METRICS = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    def __init__(self, targets: torch.Tensor, outputs: torch.Tensor):
        if isinstance(targets, torch.Tensor):
            self.outputs = outputs.cpu().numpy()
            self.targets = targets.cpu().numpy()
        else:
            self.outputs = outputs
            self.targets = targets
    
    def get_metrics(self, metric_names: List[str] = None, save_path: str = None) -> dict:
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
    def accuracy(self):
        return accuracy_score(self.targets, self.outputs).item()
    
    @property
    def f1(self):
        return f1_score(self.targets, self.outputs).item()
    
    @property
    def precision(self):
        return precision_score(self.targets, self.outputs).item()
    
    @property
    def recall(self):
        return recall_score(self.targets, self.outputs).item()
    
    @property
    def roc_auc(self):
        return roc_auc_score(self.targets, self.outputs).item()