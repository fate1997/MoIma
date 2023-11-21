from moima.config._base import Config
from abc import ABC


class BasePipe(ABC):
    def __init__(self, config: Config):
        self.config = config
    
    def get_dataset(self):
        raise NotImplementedError
    
    def get_splitter(self):
        raise NotImplementedError
    
    def get_model(self):
        raise NotImplementedError
    
    def get_trainer(self):
        raise NotImplementedError
    
    def get_evaluator(self):
        raise NotImplementedError