from moima.pipeline.downstream_pipe import DownstreamPipe, create_downstream_config_class
import pathlib
import os

pkg_path = pathlib.Path(__file__).parent.parent
print(pkg_path)

if __name__ == '__main__':
    DownstreamConfig = create_downstream_config_class('DownstreamConfig',
                                dataset_name='graph',
                                model_name='faenet',
                                splitter_name='random',
                                loss_fn_name='l1',
                                scheduler_name='exp')

    config = DownstreamConfig(raw_path=os.path.join(pkg_path, 'example/raw/qm9.sdf'),
                            label_path=os.path.join(pkg_path, 'example/raw/qm9_labels.csv'),
                            label_col=['homo'],
                            processed_path=os.path.join(pkg_path, 'example/qm9_homo_permute.pt'),
                            transform_names=['frame_averaging_permute'],
                            save_processed=True,
                            force_reload=False,
                            trainable_pca=True,
                            assign_pos=True,
                            desc='faenet_homo_egnn-frame',
                            atom_feature_names=['all'],
                            save_interval=10,
                            patience=500,
                            log_interval=5,
                            warmup_interval=5,
                            scheduler_interval=250,
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