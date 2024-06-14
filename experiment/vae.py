from moima.pipeline.vae_pipe import VAEPipeConfig, VAEPipe
import pathlib
import os

pkg_path = pathlib.Path(__file__).parent.parent

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == '__main__':
    config = VAEPipeConfig(raw_path='/home/rengp/projects/PlayGround/score/pubchem_cations.csv',
                            processed_path=os.path.join(pkg_path, 'example/pubchem_cation_selfies.pt'),
                            desc='Pubchem_cation_selfies_ls2',
                            additional_cols=['y'],
                            consider_label=True,
                            num_classes=2,
                            vocab_path=os.path.join(pkg_path, 
                                                    'example/pubchem_cation_selfies_vocab.pkl'),
                            save_processed=True,
                            force_reload=False,# `test` has passed
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
                            vocab_size=55)
    pipe = VAEPipe(config)
    pipe.train()
