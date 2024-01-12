from moima.pipeline.vae.vae import VAEPipe, VAEPipeConfig


config = VAEPipeConfig(
    raw_path='test\example\zinc1k.csv'
)

pipe = VAEPipe(config)

pipe.train()