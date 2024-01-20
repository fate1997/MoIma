from moima.pipeline.downstream_pipe import DownstreamPipe, create_downstream_config_class
import pathlib
import os

pkg_path = pathlib.Path(__file__).parent.parent
print(pkg_path)

DownstreamConfig = create_downstream_config_class('DownstreamConfig',
                            dataset_name='graph',
                            model_name='dimenet++',
                            splitter_name='random',
                            loss_fn_name='mse',
                            scheduler_name='exp')

config = DownstreamConfig(raw_path=os.path.join(pkg_path, 'example/docking/IL_MP.sdf'),
                        label_path=os.path.join(pkg_path, 'example/docking/IL_MP.csv'),
                        label_col=['MP_K'],
                        processed_path=os.path.join(pkg_path, 'example/IL_docked.pt'),
                        save_processed=True,
                        force_reload=False,
                        assign_pos=True,
                        desc='dimenet',
                        atom_feature_names=['atomic_num'], 
                        save_interval=30,
                        patience=-1,
                        log_interval=10,
                        warmup_interval=5,
                        scheduler_interval=80,
                        lr=1e-3,
                        gamma=0.1,
                        in_step_mode=False,
                        num_epochs=300,
                        batch_size=32)
config.update_from_args()
pipe = DownstreamPipe(config)
pipe.train()