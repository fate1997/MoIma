from moima.pipeline.vae_pipe import VAEPipeConfig, VAEPipe
import pathlib
import os

pkg_path = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    config = VAEPipeConfig(raw_path=os.path.join(pkg_path, 'example/ilthermo_ionic_liquids.csv'),
                            processed_path=os.path.join(pkg_path, 'example/il.pt'),
                            desc='IL_latent_dim256', 
                            save_processed=True,
                            force_reload=False,# `test` has passed
                            save_interval=10,
                            in_step_mode=False,
                            log_interval=5,
                            num_epochs=100,
                            batch_size=256,
                            latent_dim=256,
                            seq_len=256,
                            lr=1e-3,
                            device='cuda:0',
                            vocab_size=55)
    pipe = VAEPipe(config)
    pipe.train()