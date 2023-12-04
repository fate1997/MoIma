from moima.pipeline.vae.pipe import VAEPipe
from moima.pipeline.vade.pipe import VaDEPipe
from typing import Dict
from moima.pipeline.pipe import PipeABC

AVAILABLE_PIPELINES: Dict[str, PipeABC] = {
    'VAEPipe': VAEPipe,
    'VaDEPipe': VaDEPipe,}