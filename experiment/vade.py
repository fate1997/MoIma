import os
import pathlib

from moima.pipeline.vade_pipe import VaDEPipe, VaDEPipeConfig

pkg_path = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    config = VaDEPipeConfig(
        raw_path=os.path.join(pkg_path, 'test/example/zinc1k.csv'),
        processed_path=os.path.join(pkg_path, 'example/zinc1k_vade.pt'),
        desc='zinc1k_vade', 
        save_processed=True,
        force_reload=False,# `test` has passed
        save_interval=10,
        in_step_mode=False,
        log_interval=5,
        num_epochs=100,
        batch_size=256,
        latent_dim=128,
        n_clusters=50,
        seq_len=256,
        lr=1e-3,
        device='cuda:0',
        vocab_size=55
    )
    pipe = VaDEPipe(config)
    pipe.pretrain(pre_epoch=20, retrain=False)
    pipe.train()