import os
import pathlib

from moima.pipeline.vae_pipe import VAEPipe, VAEPipeConfig

pkg_path = pathlib.Path(__file__).parent.parent

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == '__main__':
    config = VAEPipeConfig(
        raw_path=os.path.join(pkg_path, 'test/example/zinc1k.csv'),
        processed_path=os.path.join(pkg_path, 'example/zinc1k_vae.pt'),
        desc='zinc1k_vae',
        consider_label=False,
        vocab_path=os.path.join(pkg_path, 'example/zinc1k_vae.pkl'),
        save_processed=True,
        force_reload=True, 
        save_interval=10,
        in_step_mode=False,
        log_interval=5,
        num_epochs=100,
        enc_num_layers=3,
        dec_num_layers=3,
        batch_size=128,
        latent_dim=128,
        seq_len=128,
        lr=1e-4,
        device='cuda:0',
        vocab_size=55
    )
    pipe = VAEPipe(config)
    pipe.train()
