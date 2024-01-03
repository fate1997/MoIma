from dataclasses import dataclass, field

from moima.dataset import DATASET_REGISTRY
from moima.pipeline.config import ArgType, DefaultConfig
from moima.utils.loss_fn import LOSS_FN_REGISTRY


@dataclass
class VAEPipeConfig(DefaultConfig):
    """The configuration for the pipeline."""
    # Dataset
    vocab_path: str = field(default=None,
                        metadata={'help': 'The path to the vocabulary.',
                                  'type': ArgType.DATASET})
    dataset_name: str = field(default='smiles_seq', 
                              metadata={'help': 'Name of the dataset.',
                                        'choices': DATASET_REGISTRY.keys(),
                                        'type': ArgType.DATASET})
    
    # Featurizer
    seq_len: int = field(default=120,
                            metadata={'help': 'The sequence length.',
                                      'type': ArgType.FEATURIZER})
    
    # Model
    model_name: str = field(default='chemical_vae',
                            metadata={'help': 'The model name.',
                                      'type': ArgType.MODEL})
    vocab_size: int = field(default=38,
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
    
    # Loss_fn
    loss_fn_name: str = field(default='vae_loss',
                        metadata={'help': 'Name of the loss function.',
                                'choices': LOSS_FN_REGISTRY.keys(),
                                'type': ArgType.LOSS_FN})
    start_kl_weight: float = field(default=0.0,
                                   metadata={'help': 'The start kl weight.',
                                             'type': ArgType.LOSS_FN})
    end_kl_weight: float = field(default=0.0025,
                                    metadata={'help': 'The end kl weight.',
                                            'type': ArgType.LOSS_FN})
    