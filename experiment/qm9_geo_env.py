from moima.pipeline.downstream_pipe import DownstreamPipe, create_downstream_config_class
import pathlib
import os

pkg_path = pathlib.Path(__file__).parent.parent
print(pkg_path)

if __name__ == '__main__':
    DownstreamConfig = create_downstream_config_class('DownstreamConfig',
                                dataset_name='graph',
                                model_name='gat_v2',
                                splitter_name='random',
                                loss_fn_name='l1',
                                scheduler_name='exp')

    config = DownstreamConfig(raw_path=os.path.join(pkg_path, 'example/raw/qm9.sdf'),
                            label_path=os.path.join(pkg_path, 'example/raw/qm9_labels.csv'),
                            label_col=['homo'],
                            processed_path=os.path.join(pkg_path, 'example/qm9_homo_geo_env_64_5.pt'),
                            save_processed=True,
                            hidden_dim=16,
                            num_layers=3, 
                            num_heads=8,
                            force_reload=True,
                            assign_pos=True,
                            desc='gatv2_geo_env',
                            atom_feature_names=['all'], 
                            atom_feature_params={'geo_env_MOLINPUT':{'feature_dim': 64,
                                                                     'cutoff': 5}},
                            save_interval=10,
                            patience=500,
                            log_interval=5,
                            warmup_interval=5,
                            scheduler_interval=400,
                            frac_train=110000,
                            frac_val=10000,
                            lr=1e-3,
                            gamma=0.1,
                            in_step_mode=False,
                            num_epochs=500,
                            batch_size=128)
    config.update_from_args()
    pipe = DownstreamPipe(config)
    pipe.train()