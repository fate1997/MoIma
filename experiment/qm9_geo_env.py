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
                                loss_fn_name='mse',
                                scheduler_name='exp')

    config = DownstreamConfig(raw_path=os.path.join(pkg_path, 'example/raw/qm9.sdf'),
                            label_path=os.path.join(pkg_path, 'example/raw/qm9_labels.csv'),
                            label_col=['homo'],
                            processed_path=os.path.join(pkg_path, 'example/qm9_homo.pt'),
                            save_processed=True,
                            hidden_dim=16,
                            num_layers=3, 
                            num_heads=8,
                            force_reload=False,
                            assign_pos=True,
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
                            warmup_interval=3000,
                            scheduler_interval=2000000,
                            lr=1e-3,
                            gamma=0.1,
                            in_step_mode=True,
                            num_epochs=800,
                            batch_size=32)
    config.update_from_args()
    pipe = DownstreamPipe(config)
    pipe.train()