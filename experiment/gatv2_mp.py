from moima.pipeline.downstream_pipe import DownstreamPipe, create_downstream_config_class
import pathlib
import os

pkg_path = pathlib.Path(__file__).parent.parent
print(pkg_path)

if __name__ == '__main__':
    DownstreamConfig = create_downstream_config_class('DownstreamConfig',
                                dataset_name='graph',
                                model_name='gatv2_salt',
                                splitter_name='random',
                                loss_fn_name='mse',
                                scheduler_name='none')

    config = DownstreamConfig(raw_path=os.path.join(pkg_path, 'example/database/il_mp_overall.csv'),
                            label_col=['mpK'],
                            processed_path=os.path.join(pkg_path, 'example/il_mp.pt'),
                            save_processed=True,
                            hidden_dim=16,
                            num_layers=3, 
                            num_heads=8,
                            force_reload=True,
                            assign_pos=False,
                            desc='gatv2_wo_geo',
                            atom_feature_names=['atomic_num',
                                                'degree',
                                                'chiral_tag',
                                                'num_Hs',
                                                'hybridization',
                                                'aromatic',
                                                'formal_charge',
                                                'mass'], 
                            save_interval=5000,
                            patience=-1,
                            log_interval=1000,
                            lr=1e-3,
                            in_step_mode=True,
                            num_epochs=500,
                            batch_size=128)
    config.update_from_args()
    pipe = DownstreamPipe(config)
    pipe.train()