from dataclasses import dataclass, field

from moima.pipeline.config import ArgType, DefaultConfig


@dataclass
class DownstreamPipeConfig(DefaultConfig):
    """The configuration for the pipeline."""
    
    # Model
    model_name: str = field(default='mlp',
                            metadata={'help': 'The model name.',
                                      'type': ArgType.MODEL})
    
    input_dim: int = field(default=50,
                            metadata={'help': 'The input dimension.',
                                      'type': ArgType.MODEL})
    hidden_dim: int = field(default=128,
                                metadata={'help': 'The hidden dimension.',
                                          'type': ArgType.MODEL})
    output_dim: int = field(default=1,
                            metadata={'help': 'The output dimension.',
                                      'type': ArgType.MODEL})
    n_layers: int = field(default=2,
                            metadata={'help': 'The number of layers.',
                                      'type': ArgType.MODEL})
    dropout: float = field(default=0.2,
                            metadata={'help': 'The dropout rate.',
                                      'type': ArgType.MODEL})
    activation: str = field(default='relu',
                            metadata={'help': 'The activation function.',
                                      'type': ArgType.MODEL})
    
    # Dataset
    label_col: str = field(default='label',
                            metadata={'help': 'The label column name.',
                                      'type': ArgType.DATASET})
    vocab_path: str = field(default=None,
                            metadata={'help': 'The path to the vocabulary.',
                                      'type': ArgType.DATASET})
    
    # Featurizer
    mol_desc: str = field(default='ecfp',
                    metadata={'help': 'The descriptor name.',
                                'type': ArgType.FEATURZIER})
    ecfp_radius: int = field(default=4,
                            metadata={'help': 'The ECFP radius.',
                                      'type': ArgType.FEATURZIER})
    ecfp_n_bits: int = field(default=2048,
                            metadata={'help': 'The ECFP number of bits.',
                                      'type': ArgType.FEATURZIER})
    desc_csv_path: str = field(default=None,
                            metadata={'help': 'The descriptor csv path.',
                                      'type': ArgType.FEATURZIER})

    
    # General
    pretrained_pipe_class: str = field(default='VaAEPipe',
                            metadata={'help': 'The pretrained pipeline class.',
                                      'type': ArgType.GENERAL})
    pretrained_pipe_path: str = field(default=None,
                            metadata={'help': 'The pretrained pipeline path.',
                                      'type': ArgType.GENERAL})
    
    # Loss
    loss_name: str = field(default='mse',
                            metadata={'help': 'The loss name.',
                                      'type': ArgType.LOSS_FN})
    
    