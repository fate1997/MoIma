from moima.pipeline.downstream.pipe import DownstreamPipe, create_downstream_config_class


if __name__ == '__main__':
    DownstreamConfig = create_downstream_config_class('DownstreamConfig',
                                dataset_name='graph',
                                model_name='dimenet',
                                splitter_name='random',
                                    loss_fn_name='mse')

    config = DownstreamConfig(raw_path='example/raw/qm9.sdf',
                            label_path='example/raw/qm9_labels.csv',
                            label_col=['homo'],
                            processed_path='example/qm9_homo.pt',
                            save_processed=True,
                            force_reload=False,
                            assign_pos=True,
                            desc='dimenet',
                            atom_feature_names=['atomic_num'], 
                            save_interval=5000,
                            patience=-1,
                            log_interval=1000,
                            num_epochs=800,
                            batch_size=32)
    config.update_from_args()
    pipe = DownstreamPipe(config)
    pipe.train()