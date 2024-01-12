from torch.optim import Optimizer, SGD, Adam, AdamW


OPTIMIZER_REGISTRY = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW
}


def build_optimizer(name: str=None, **kwargs) -> Optimizer:
    """Build an optimizer instance by name.
    
    Args:
        name (str): Name of the optimizer. If None, the name will be read from kwargs.
        **kwargs: Arguments for the optimizer.
    
    Returns:
        An optimizer instance.
    """
    if name is None:
        name = kwargs.pop('name')
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer {name} is not available.")
    return OPTIMIZER_REGISTRY[name](**kwargs)