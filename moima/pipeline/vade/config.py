from dataclasses import dataclass, field

from moima.dataset import DATASET_REGISTRY
from moima.pipeline.config import ArgType, DefaultConfig
from moima.utils.loss_fn import LOSS_FN_REGISTRY


@dataclass
class VaDEPipeConfig(DefaultConfig):
    """The configuration for the pipeline."""
    
    # Featurizer
    seq_len: int = field(default=120,
                            metadata={'help': 'The sequence length.',
                                      'type': ArgType.FEATURZIER})
    
    # Dataset
    dataset_name: str = field(default='smiles_seq', 
                                metadata={'help': 'Name of the dataset.',
                                        'choices': DATASET_REGISTRY.keys(),
                                        'type': ArgType.DATASET})
    vocab_path: str = field(default=None,
                                metadata={'help': 'The path to the vocabulary.',
                                          'type': ArgType.DATASET})
    
    # Model
    model_name: str = field(default='vade',
                            metadata={'help': 'The model name.',
                                      'type': ArgType.MODEL})
    vocab_size: int = field(default=40,
                            metadata={'help': 'The vocabulary size.',
                                      'type': ArgType.MODEL})
    enc_hidden_dim: int = field(default=292,
                                metadata={'help': 'The hidden dimension of the encoder.',
                                          'type': ArgType.MODEL})
    latent_dim: int = field(default=292,
                            metadata={'help': 'The latent dimension.',
                                      'type': ArgType.MODEL})
    emb_dim: int = field(default=128,
                            metadata={'help': 'The embedding dimension.',
                                    'type': ArgType.MODEL})
    dec_hidden_dim: int = field(default=501,
                                metadata={'help': 'The hidden dimension of the decoder.',
                                          'type': ArgType.MODEL})
    dropout: float = field(default=0.1,
                            metadata={'help': 'The dropout rate.',
                                      'type': ArgType.MODEL})
    n_clusters: int = field(default=10,
                            metadata={'help': 'The number of clusters.',
                                      'type': ArgType.MODEL})
    
    # Loss_fn
    loss_fn_name: str = field(default='vade_loss',
                                metadata={'help': 'Name of the loss function.',
                                        'choices': LOSS_FN_REGISTRY.keys(),
                                        'type': ArgType.LOSS_FN})
    kl_weight: float = field(default=1,
                            metadata={'help': 'The kl weight.',
                                      'type': ArgType.LOSS_FN})
    
    