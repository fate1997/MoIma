from abc import ABC, abstractmethod
import torch
from typing import Any


class LossCalcABC(ABC):
    
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Calculate the loss."""