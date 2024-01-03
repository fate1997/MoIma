from ._abc import LossCalcABC
from torch import nn, Tensor


class MSELossCalc(LossCalcABC):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def __call__(self, output: Tensor, target: Tensor):
        return self.loss_fn(output, target)


class L1LossCalc(LossCalcABC):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def __call__(self, output: Tensor, target: Tensor):
        return self.loss_fn(output, target)


class HuberLossCalc(LossCalcABC):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def __call__(self, output: Tensor, target: Tensor):
        return self.loss_fn(output, target)