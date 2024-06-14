from moima.pipeline.vae_pipe import VAEPipeConfig, VAEPipe
import pathlib
import os

pkg_path = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    config = VAEPipeConfig(raw_path=os.path.join(pkg_path, 'example/IL4CVAE.csv'),
                            processed_path=os.path.join(pkg_path, 'example/il_cvae.pt'),
                            desc='IL_cvae_encoder',
                            additional_cols=['y'],
                            save_processed=True,
                            force_reload=False,# `test` has passed
                            consider_label=True,
                            save_interval=10,
                            in_step_mode=False,
                            log_interval=5,
                            num_epochs=100,
                            batch_size=64,
                            latent_dim=128,
                            seq_len=256,
                            lr=1e-4,
                            num_classes=3,
                            device='cuda:0',
                            vocab_size=55)
    pipe = VAEPipe(config)
    pipe.train()
