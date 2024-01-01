import argparse
from dataclasses import dataclass, field, Field
from enum import Enum
from typing import Any, Dict, List
import json
import yaml
import hashlib

from moima.dataset import DATASET_REGISTRY
from moima.model import MODEL_REGISTRY
from moima.utils.loss_fn import LOSS_FN_REGISTRY
from moima.utils.splitter import SPLITTER_REGISTRY


class ArgType(Enum):
    DATASET=0
    FEATURZIER=1
    MODEL=2
    SPLITTER=3
    LOSS_FN=4
    OPTIMIZER=5
    SCHEDULER=6
    GENERAL=7

NAME_BAG = ['dataset', 'model', 'splitter', 'loss_fn']


@dataclass
class DefaultConfig:
    r"""Default config for the pipeline.
    
    Dataset:
        dataset_name: Name of the dataset. Default: 'smiles_seq'
        raw_path: Path to the raw data. Default: None
        processed_path: Path to the processed data. Default: None
        force_reload: Whether to force reload the data. Default: False
        save_processed: Whether to save the processed data. Default: False
    
    Featurizer:
        None.
    
    Model:
        model_name: Name of the model. Default: 'chemical_vae'
    
    Splitter:
        splitter_name: Name of the splitter. Default: 'random'
        split_test: Whether to split the test set. Default: True
        batch_size: The batch size of the dataloader. Default: 128
        frac_train: The ratio or the number of the training set. Default: 0.8
        frac_val: The ratio or the number of the validation set. Default: 0.1
    
    Loss Function:
        loss_fn_name: Name of the loss function. Default: 'vae_loss'
        
    General:
        num_epochs: Number of epochs. Default: 100
        log_interval: The interval of logging. Default: 1000
        save_interval: The interval of saving. Default: 5000
        output_folder: The output folder. Default: 'output'
        desc: The description of the experiment. Default: 'default'
        device: The device to use. Default: 'cuda'
        lr: The learning rate. Default: 1e-4
    """
    # Dataset
    dataset_name: str = field(default=None, 
                              metadata={'help': 'Name of the dataset.',
                                        'choices': DATASET_REGISTRY.keys(),
                                        'type': ArgType.DATASET})
    raw_path: str = field(default=None, 
                          metadata={'help': 'Path to the raw data.',
                                    'type': ArgType.DATASET})
    processed_path: str = field(default=None, 
                                metadata={'help': 'Path to the processed data.',
                                          'type': ArgType.DATASET})
    force_reload: bool = field(default=False, 
                               metadata={'help': 'Whether to force reload the data.',
                                         'type': ArgType.DATASET})
    save_processed: bool = field(default=False, 
                                 metadata={'help': 'Whether to save the processed data.',
                                           'type': ArgType.DATASET})
    # Model
    model_name: str = field(default=None,
                            metadata={'help': 'Name of the model.',
                                        'choices': MODEL_REGISTRY.keys(),
                                        'type': ArgType.MODEL})
    
    # Splitter
    splitter_name: str = field(default='random',
                                 metadata={'help': 'Name of the splitter.',
                                           'choices': SPLITTER_REGISTRY.keys(),
                                          'type': ArgType.SPLITTER})
    split_test: bool = field(default=True,
                                metadata={'help': 'Whether to split the test set.',
                                        'type': ArgType.SPLITTER})
    batch_size: int = field(default=128,
                                metadata={'help': 'The batch size of the dataloader.',
                                        'type': ArgType.SPLITTER})
    frac_train: float = field(default=0.8,
                                metadata={'help': 'The ratio or the number of the training set.',
                                        'type': ArgType.SPLITTER})
    frac_val: float = field(default=0.1,
                                metadata={'help': 'The ratio or the number of the validation set.',
                                        'type': ArgType.SPLITTER})
    
    # Loss Function
    loss_fn_name: str = field(default=None,
                                metadata={'help': 'Name of the loss function.',
                                        'choices': LOSS_FN_REGISTRY.keys(),
                                        'type': ArgType.LOSS_FN})
    
    # General
    num_epochs: int = field(default=100,
                            metadata={'help': 'Number of epochs.',
                                        'type': ArgType.GENERAL})
    log_interval: int = field(default=1000,
                                metadata={'help': 'The interval of logging.',
                                        'type': ArgType.GENERAL})
    save_interval: int = field(default=5000,
                                metadata={'help': 'The interval of saving.',
                                        'type': ArgType.GENERAL})
    output_folder: str = field(default='output',
                                metadata={'help': 'The output folder.',
                                        'type': ArgType.GENERAL})
    desc: str = field(default='default',
                        metadata={'help': 'The description of the experiment.',
                                'type': ArgType.GENERAL})
    device: str = field(default='cuda',
                        metadata={'help': 'The device to use.',
                                'type': ArgType.GENERAL})
    lr: float = field(default=1e-3,
                        metadata={'help': 'The learning rate.',
                                'type': ArgType.GENERAL})
    
    def __post_init__(self):
        r"""Set the default value of the fields to the input value."""
        input_dict = self.__dict__
        for k, v in self.__dataclass_fields__.items():
            input_value = input_dict[k]
            field_default = v.default
            if input_value != field_default:
                v.default = input_value
    
    def get_hash_key(self, exclude: List[str]=None):
        r"""Get the hash key of the config."""
        dhash = hashlib.md5()
        hash_dict = self.__dict__
        if exclude is not None:
            for key in exclude:
                hash_dict.pop(key)
        encoded = json.dumps(hash_dict, sort_keys=True).encode()
        dhash.update(encoded)
        return int(dhash.hexdigest(), 16) % (10 ** 8)
    
    @property
    def group_dict(self):
        r"""Get the grouped dict according to the type."""
        dic = {}
        for arg in ArgType:
            dic[arg.name.lower()] = self.group(arg)
        return dic
    
    @property
    def dataset(self):
        r"""Get the dataset group."""
        return self.group(ArgType.DATASET)
    
    @property
    def splitter(self):
        r"""Get the splitter group."""
        return self.group(ArgType.SPLITTER)
    
    @property
    def model(self):
        r"""Get the model group."""
        return self.group(ArgType.MODEL)
    
    @property
    def loss_fn(self):
        r"""Get the loss function group."""
        return self.group(ArgType.LOSS_FN)
    
    @property
    def general(self):
        r"""Get the general group."""
        return self.group(ArgType.GENERAL)
    
    @property
    def optimizer(self):
        r"""Get the optimizer group."""
        return self.group(ArgType.OPTIMIZER)
    
    @property
    def scheduler(self):
        r"""Get the scheduler group."""
        return self.group(ArgType.SCHEDULER)
    
    @property
    def featurizer(self):
        r"""Get the featurizer group."""
        result_dict = self.group(ArgType.FEATURZIER)
        return result_dict
        
    def group(self, type: ArgType) -> Dict[str, Any]:
        r"""Group the config according to the type."""
        result_dict = {}
        for k, v in self.__dataclass_fields__.items():
            if v.metadata['type'] == type:
                if v.default != getattr(self, k):
                    result_dict[k] = getattr(self, k)
                else:
                    result_dict[k] = v.default
                if 'name' in k:
                    result_dict['name'] = result_dict.pop(k)
        return result_dict
        
    def __add__(self, other: 'DefaultConfig'):
        r"""Add two configs."""
        this_dict = self.__dataclass_fields__
        other_dict = other.__dataclass_fields__
        for k, v in other_dict.items():
            if k not in this_dict:
                this_dict[k] = v
            else:
                raise ValueError(f'Key {k} already exists.')
        return self.from_dict(**this_dict)
    
    def to_json(self, save_path: str):
        r"""Save the config to a json file."""
        assert save_path.endswith('.json'), 'The save path must end with .json'
        with open(save_path, 'w') as f:
            json.dump(self.group_dict, f)

    def to_yaml(self, save_path: str):
        r"""Save the config to a yaml file."""
        assert save_path.endswith('.yaml'), 'The save path must end with .yaml'
        with open(save_path, 'w') as f:
            yaml.dump(self.group_dict, f, indent=4, sort_keys=False)
    
    @classmethod
    def from_file(cls, file_path: str):
        r"""Create a config from a file."""
        assert file_path.endswith('.json') or file_path.endswith('.yaml'), \
            'The file path must end with .json or .yaml'
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                dic = json.load(f)
            else:
                dic = yaml.load(f, Loader=yaml.FullLoader)
        dic = cls._group_dict2dict(dic)
        return cls(**dic)
    
    @staticmethod
    def _group_dict2dict(group_dict: Dict[str, Any]) -> Dict[str, Any]:
        r"""Convert a group dict to a dict."""
        result_dict = {}
        for k, v in group_dict.items():
            for kk, vv in v.items():
                if k in NAME_BAG and kk == 'name':
                    result_dict[f'{k}_name'] = vv
                else:
                    result_dict[kk] = vv
        return result_dict
        
    @classmethod
    def from_args(cls) -> 'DefaultConfig':
        r"""Create a config from the command line arguments."""
        parser =  argparse.ArgumentParser(description='Parser For Arguments', 
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for k, v in cls.__dataclass_fields__.items():
            parser.add_argument(f'--{k}', 
                                type=v.type, 
                                default=v.default, 
                                help=f'{v.metadata["help"]}',
                                choices=v.metadata.get('choices', None))
        args = parser.parse_args()
        return cls(**vars(args))