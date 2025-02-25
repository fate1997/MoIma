import os
import pathlib

from moima.pipeline.downstream_pipe import (DownstreamPipe,
                                            create_downstream_config_class)

pkg_path = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    DownstreamConfig = create_downstream_config_class(
        'DownstreamConfig',
        dataset_name='desc_vec',
        model_name='mlp',
        splitter_name='random',
        loss_fn_name='l1',
        scheduler_name='cos'
    )

    config = DownstreamConfig(
        raw_path=os.path.join(pkg_path, 'test/example/zinc1k.csv'),
        label_col=['qed'],
        processed_path=os.path.join(pkg_path, 'example/zinc1k.pt'),
        save_processed=True,
        force_reload=True,
        mol_desc='ecfp',
        desc='dimenetpp',
        save_interval=10,
        patience=500,
        log_interval=5,
        warmup_interval=5,
        scheduler_interval=1,
        frac_train=0.8,
        frac_val=0.1,
        lr=1e-3,
        eta_min=1e-5,
        in_step_mode=False,
        num_epochs=500,
        T_max=50,
        batch_size=128
    )
    config.update_from_args()
    pipe = DownstreamPipe(config)
    pipe.train()