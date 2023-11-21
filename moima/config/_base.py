from dataclasses import dataclass


@dataclass
class Config:
    
    def from_yaml(cls, path: str):
        raise NotImplementedError
    
    def from_json(cls, path: str):
        raise NotImplementedError
    
    def from_dict(cls, dict: dict):
        raise NotImplementedError