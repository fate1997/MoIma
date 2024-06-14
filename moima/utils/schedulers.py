from torch.optim import lr_scheduler


SCHEDULER_REGISTRY = {
    'lambda': lr_scheduler.LambdaLR,
    'step': lr_scheduler.StepLR,
    'multi_step': lr_scheduler.MultiStepLR,
    'exp': lr_scheduler.ExponentialLR,
    'cos': lr_scheduler.CosineAnnealingLR,
    'reduce': lr_scheduler.ReduceLROnPlateau,
    'cyclic': lr_scheduler.CyclicLR,
    'cos_warm': lr_scheduler.CosineAnnealingWarmRestarts,
    'multiply': lr_scheduler.MultiplicativeLR,
    'constant': lr_scheduler.ConstantLR,
    'linear': lr_scheduler.LinearLR,
    'poly': lr_scheduler.PolynomialLR,
    'none': None
}


def build_scheduler(name: str=None, **kwargs) -> lr_scheduler.LRScheduler:
    """Build a scheduler instance by name.
    
    Args:
        name (str): Name of the scheduler. If None, the name will be read from kwargs.
        **kwargs: Arguments for the scheduler.
    
    Returns:
        A scheduler instance.
    """
    if name is None:
        name = kwargs.pop('name')
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Dataset {name} is not available.")
    return SCHEDULER_REGISTRY[name](**kwargs)