import argparse
from dataclasses import dataclass, field, Field
from enum import Enum
from typing import Any, Dict

from moima.dataset._factory import DATASET_REGISTRY
from moima.model._factory import MODEL_REGISTRY
from moima.utils.loss_fn._factory import LOSS_FN_REGISTRY
from moima.utils.splitter._factory import SPLITTER_REGISTRY


class ArgType(Enum):
    DATASET=0
    FEATURZIER=1
    MODEL=2
    SPLITTER=3
    LOSS_FN=4
    OPTIMIZER=5
    SCHEDULER=6
    GENERAL=7


@dataclass
class DefaultConfig:
    # Dataset
    dataset_name: str = field(default='smiles_seq', 
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
    
    # Splitter
    splitter_name: str = field(default='random',
                                 metadata={'help': 'Name of the splitter.',
                                           'choices': SPLITTER_REGISTRY.keys(),
                                          'type': ArgType.SPLITTER})
    split_test: bool = field(default=True,
                                metadata={'help': 'Whether to split the test set.',
                                        'type': ArgType.SPLITTER})
    batch_size: int = field(default=64,
                                metadata={'help': 'The batch size of the dataloader.',
                                        'type': ArgType.SPLITTER})
    seed: int = field(default=42,
                        metadata={'help': 'The random seed.',
                                'type': ArgType.SPLITTER})
    ratios: float = field(default=(0.8, 0.1),
                                    metadata={'help': 'The train/val ratio.',
                                            'type': ArgType.SPLITTER})
    
    # Model
    model_name: str = field(default='chemical_vae',
                            metadata={'help': 'Name of the model.',
                                        'choices': MODEL_REGISTRY.keys(),
                                        'type': ArgType.MODEL})

    
    # Loss Function
    loss_fn_name: str = field(default='vae_loss',
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
        input_dict = self.__dict__
        for k, v in self.__dataclass_fields__.items():
            input_value = input_dict[k]
            field_default = v.default
            if input_value != field_default:
                v.default = input_value
    
    @property
    def dataset(self):
        return self.group(ArgType.DATASET)
    
    @property
    def splitter(self):
        return self.group(ArgType.SPLITTER)
    
    @property
    def model(self):
        return self.group(ArgType.MODEL)
    
    @property
    def loss_fn(self):
        return self.group(ArgType.LOSS_FN)
    
    @property
    def general(self):
        return self.group(ArgType.GENERAL)
    
    @property
    def optimizer(self):
        return self.group(ArgType.OPTIMIZER)
    
    @property
    def scheduler(self):
        return self.group(ArgType.SCHEDULER)
    
    @property
    def featurizer(self):
        result_dict = self.group(ArgType.FEATURZIER)
        result_dict['name'] = self.dataset['name']
        return result_dict
        
    def group(self, type: ArgType) -> Dict[str, Any]:
        """Group the config."""
        result_dict = {}
        for k, v in self.__dataclass_fields__.items():
            k = 'name' if 'name' in k else k
            if v.metadata['type'] == type:
                result_dict[k] = v.default
        return result_dict
        
    def __add__(self, other: 'DefaultConfig'):
        """Add two configs."""
        this_dict = self.__dataclass_fields__
        other_dict = other.__dataclass_fields__
        for k, v in other_dict.items():
            if k not in this_dict:
                this_dict[k] = v
            else:
                raise ValueError(f'Key {k} already exists.')
        return self.from_dict(**this_dict)
    
    @classmethod
    def from_dict(cls, **kwargs) -> 'DefaultConfig':
        """Create a config from a dictionary."""
        raise cls(**kwargs)
    
    @classmethod
    def from_args(cls) -> 'DefaultConfig':
        parser =  argparse.ArgumentParser(description='Parser For Arguments', 
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for k, v in cls.__dataclass_fields__.items():
            parser.add_argument(f'--{k}', 
                                type=v.type, 
                                default=v.default, 
                                help=f'{v.metadata["help"]}',
                                choices=v.metadata.get('choices', None))
        args = parser.parse_args()
        print(args)
        return cls(**vars(args))