from dataclasses import dataclass, field

from moima.pipeline.config import ArgType, DefaultConfig


@dataclass
class VAEPipeConfig(DefaultConfig):
    """The configuration for the pipeline."""
    
    # Featurizer
    seq_len: int = field(default=120,
                            metadata={'help': 'The sequence length.',
                                      'type': ArgType.FEATURZIER})
    
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
    