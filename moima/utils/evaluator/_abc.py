from abc import ABC, abstractmethod
from typing import List
import json


class MetricABC(ABC):
    """Abstract base class for metrics."""
    AVAIL_METRICS = []

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