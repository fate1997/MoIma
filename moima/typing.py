from typing import Any, List, Union

import numpy as np
import torch
from rdkit import Chem


IndexType = Union[slice, torch.Tensor, np.ndarray, List[int]]

LabelType = Union[torch.Tensor, np.ndarray, List[Any]]

MolRepr = Union[Chem.Mol, str]