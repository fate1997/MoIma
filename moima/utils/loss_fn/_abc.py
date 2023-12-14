from abc import ABC, abstractmethod
from typing import Any, Union

from torch import Tensor


class LossCalcABC(ABC):
    
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Union[Tensor, dict]:
        """Calculate the loss."""