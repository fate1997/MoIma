from dataclasses import dataclass


@dataclass
class BaseConfig:
    def from_default(cls):
        raise NotImplementedError
    
    def from_yaml(cls, path):
        raise NotImplementedError
    
    def from_json(cls, path):
        raise NotImplementedError
    
    def from_dict(cls, dict):
        raise NotImplementedError