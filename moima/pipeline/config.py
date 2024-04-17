import argparse
import hashlib
import inspect
import json
from dataclasses import Field, dataclass, field, make_dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

import yaml

from moima.dataset import DATASET_REGISTRY, FEATURIZER_REGISTRY
from moima.model import MODEL_REGISTRY
from moima.utils.loss_fn import LOSS_FN_REGISTRY
from moima.utils.schedulers import SCHEDULER_REGISTRY
from moima.utils.splitter import SPLITTER_REGISTRY
from moima.utils.optimizers import OPTIMIZER_REGISTRY


class ArgType(Enum):
    DATASET=0
    FEATURIZER=1
    MODEL=2
    SPLITTER=3
    LOSS_FN=4
    OPTIMIZER=5
    SCHEDULER=6
    GENERAL=7

NAME_BAG = ['dataset', 'model', 'splitter', 'loss_fn', 'featurizer', 'scheduler']


@dataclass
class Config:
    r"""Default config for the pipeline.
    
    Attributes:
        num_epochs (int): Number of epochs.
        log_interval (int): The interval of logging.
        save_interval (int): The interval of saving.
        output_folder (str): The output folder.
        lr (float): The learning rate.
        desc (str): The description of the experiment.
        device (str): The device to use.
        patience (int): The patience of early stop.
        early_stop_metric (str): The metric of early stop.
        in_step_mode (bool): Whether to log/save/early_stop in step mode.
        show_tqdm (bool): Whether to show tqdm.
        warmup_interval (int): The interval of warmup.    
    """
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
    lr: float = field(default=1e-3,
                        metadata={'help': 'The learning rate.',
                                    'type': ArgType.GENERAL})
    desc: str = field(default='default',
                      metadata={'help': 'The description of the experiment.',
                                'type': ArgType.GENERAL})
    device: str = field(default='cuda',
                        metadata={'help': 'The device to use.',
                                'type': ArgType.GENERAL})
    patience: int = field(default=-1,
                          metadata={'help': 'The patience of early stop.',
                                     'type': ArgType.GENERAL})
    early_stop_metric: str = field(default='val_MAE',
                                   metadata={'help': 'The metric of early stop.',
                                             'type': ArgType.GENERAL})
    in_step_mode: bool = field(default=False,
                               metadata={'help': 'Whether to log/save/early_stop in step mode.',
                                         'type': ArgType.GENERAL})
    show_tqdm: bool = field(default=False,
                            metadata={'help': 'Whether to show tqdm.',
                                      'type': ArgType.GENERAL})
    warmup_interval: int = field(default=0,
                                    metadata={'help': 'The interval of warmup.',
                                            'type': ArgType.GENERAL})
    scheduler_interval: int = field(default=1,
                                    metadata={'help': 'The interval of scheduler.',
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
                if key in hash_dict:
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
        result_dict = self.group(ArgType.FEATURIZER)
        return result_dict
        
    def group(self, arg_type: ArgType) -> Dict[str, Any]:
        r"""Group the config according to the type."""
        result_dict = {}
        for k, v in self.__dataclass_fields__.items():
            if v.metadata['type'] == arg_type:
                if v.default != getattr(self, k):
                    result_dict[k] = getattr(self, k)
                else:
                    result_dict[k] = v.default
                arg_name = arg_type.name.lower()+'_name'
                if arg_name == k:
                    result_dict['name'] = result_dict.pop(k)
        return result_dict
        
    def __add__(self, other: 'Config'):
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
    def from_args(cls) -> 'Config':
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
    
    def update_from_args(self):
        r"""Update the config from the command line arguments."""
        parser =  argparse.ArgumentParser(description='Parser For Arguments', 
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for k, v in self.__dataclass_fields__.items():
            parser.add_argument(f'--{k}', 
                                type=v.type, 
                                default=v.default, 
                                help=v.metadata.get('help', None),
                                choices=v.metadata.get('choices', None))
        args = parser.parse_args()
        for k, v in vars(args).items():
            setattr(self, k, v)


def get_arg_fields(arg_spec: inspect.FullArgSpec, 
                   arg_type: ArgType,
                   exclude_args: List[str]=[]) -> List[tuple]:
    r"""Get the fields from the arg spec."""
    fields_list = []
    for i, arg in enumerate(reversed(arg_spec.args)):
        if arg == 'self' or arg in exclude_args:
            continue
        i += 1
        if arg_spec.defaults is not None and len(arg_spec.defaults) >= i:
            default_value = arg_spec.defaults[-i]
        else:
            default_value = None
        if default_value == []:
            arg_field = field(default_factory=list,
                            metadata={'type': arg_type})
        elif default_value == {}:
            arg_field = field(default_factory=dict,
                            metadata={'type': arg_type})
        elif default_value == ():
            arg_field = field(default_factory=tuple,
                            metadata={'type': arg_type})
        else:
            arg_field = field(default=default_value,
                            metadata={'type': arg_type})
        annt = arg_spec.annotations[arg] if arg in arg_spec.annotations else Any
        fields_list.append((arg, annt, arg_field))
    return fields_list


def create_config_class(class_name: str,
                        dataset_name: str,
                        model_name: str,
                        splitter_name: str,
                        loss_fn_name: str,
                        scheduler_name: str,
                        featurizer_name: str=None,
                        addi_args: List[Tuple[str, type, Field]] = []):
    dataset_arg_spec = inspect.getfullargspec(DATASET_REGISTRY[dataset_name])
    model_arg_spec = inspect.getfullargspec(MODEL_REGISTRY[model_name])
    splitter_arg_spec = inspect.getfullargspec(SPLITTER_REGISTRY[splitter_name])
    loss_fn_arg_spec = inspect.getfullargspec(LOSS_FN_REGISTRY[loss_fn_name])
    featurizer_arg_spec = inspect.getfullargspec(FEATURIZER_REGISTRY[dataset_name])
    if scheduler_name != 'none':
        scheduler_arg_spec = inspect.getfullargspec(SCHEDULER_REGISTRY[scheduler_name])
        scheduler_arg_fields = get_arg_fields(scheduler_arg_spec, ArgType.SCHEDULER, ['optimizer'])
    else:
        scheduler_arg_fields = []
    #!TODO: add optimizer and check the duplicate args
    arg_fields = get_arg_fields(dataset_arg_spec, ArgType.DATASET, ['featurizer']) + \
                 get_arg_fields(model_arg_spec, ArgType.MODEL) + \
                 get_arg_fields(splitter_arg_spec, ArgType.SPLITTER) + \
                 get_arg_fields(loss_fn_arg_spec, ArgType.LOSS_FN) + \
                 get_arg_fields(featurizer_arg_spec, ArgType.FEATURIZER, ['vocab']) + \
                 scheduler_arg_fields
    loss_fn_field = ('loss_fn_name', str, field(default=loss_fn_name,
                                               metadata={'type': ArgType.LOSS_FN,
                                                         'choices': LOSS_FN_REGISTRY.keys()}))
    dataset_field = ('dataset_name', str, field(default=dataset_name,
                                                  metadata={'type': ArgType.DATASET,
                                                            'choices': DATASET_REGISTRY.keys()}))
    model_field = ('model_name', str, field(default=model_name,
                                                metadata={'type': ArgType.MODEL,
                                                            'choices': MODEL_REGISTRY.keys()}))
    splitter_field = ('splitter_name', str, field(default=splitter_name,
                                                        metadata={'type': ArgType.SPLITTER,
                                                                    'choices': SPLITTER_REGISTRY.keys()}))
    scheduler_field = ('scheduler_name', str, field(default=scheduler_name,
                                                        metadata={'type': ArgType.SCHEDULER,
                                                                    'choices': SCHEDULER_REGISTRY.keys()}))
    if featurizer_name is None:
        featurizer_name = dataset_name
    featurizer_field = ('featurizer_name', str, field(default=featurizer_name,
                                                        metadata={'type': ArgType.FEATURIZER,
                                                                    'choices': FEATURIZER_REGISTRY.keys()}))
    arg_fields = [loss_fn_field, featurizer_field, dataset_field, model_field, splitter_field, scheduler_field] + arg_fields
    arg_fields += addi_args
    
    config_class = make_dataclass(class_name, arg_fields, bases=(Config,))
    
    setattr(inspect.getmodule(Config), class_name, config_class)
    config_class.__module__ = inspect.getmodule(Config).__name__
    return config_class