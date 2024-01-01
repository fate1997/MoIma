from moima.pipeline.vade import VaDEPipeConfig, VaDEPipe


if __name__ == '__main__':
    config = VaDEPipeConfig(raw_path='example/database/ilthermo_ILs.csv', 
                            desc='IL_latent_dim256',  # `test` has passed
                            save_interval=5000,
                            log_interval=1000,
                            num_epochs=300,
                            batch_size=800,
                            latent_dim=256,
                            n_clusters=10,
                            seq_len=256,
                            lr=2e-3,
                            device='cuda:0',
                            vocab_size=55,
                            vocab_path='example/database/ilthermo_ILs_vocab.pkl')
    pipe = VaDEPipe(config)
    pipe.pretrain(pre_epoch=20, retrain=True)
    pipe.train()
