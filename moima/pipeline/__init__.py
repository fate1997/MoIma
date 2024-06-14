from moima.pipeline.vae_pipe import VAEPipe
from moima.pipeline.vade_pipe import VaDEPipe
from typing import Dict
from moima.pipeline.pipe import PipeABC

AVAILABLE_PIPELINES: Dict[str, PipeABC] = {
    'VAEPipe': VAEPipe,
    'VaDEPipe': VaDEPipe,}