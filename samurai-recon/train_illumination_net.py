# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf

import dataflow.illumination_integration as dataflow
import utils.training_setup_utils as train_utils
from models.illumination_integration_net import IlluminationNetwork
from nn_utils.tensorboard_visualization import hdr_to_tb


def add_args(parser):
    parser.add_argument(
        "--log_step",
        type=int,
        default=100,
        help="frequency of tensorboard metric logging",
    )
    parser.add_argument(
        "--viz_step",
        type=int,
        default=2000,
        help="frequency of tensorboard train image logging",
    )
    parser.add_argument(
        "--weights_epoch", type=int, default=1, help="save weights every x epochs"
    )
    parser.add_argument(
        "--validation_epoch",
        type=int,
        default=1,
        help="render validation every x epochs",
    )
    parser.add_argument("--lrate_decay", type=int, default=500)

    return parser


def parser():
    parser = add_args(
        dataflow.add_args(IlluminationNetwork.add_args(train_utils.setup_parser()))
    )
    return parser


def run_validation(val_df, illumination_model):
    with tf.device("/device:gpu:0"):
        mseMean = tf.keras.metrics.Mean()
        for dp in val_df:
            env_map, _, roughnesses, env_map_levels = dp

            z = illumination_model.cnn_encoder(env_map)

            level_recons = []
            for i in range(roughnesses.shape[1]):
                e = illumination_model.illumination_network.eval_env_map(
                    z,
                    roughnesses[:, i : i + 1],
                    env_map.shape[1],
                )
                level_recons.append(e)

            level_recons = tf.stack(level_recons, 1)

            env_map_reshape = tf.reshape(
                env_map_levels, (-1, roughnesses.shape[1], *env_map.shape[1:])
            )

            mseMean.update_state(
                tf.keras.losses.mean_squared_error(env_map_reshape, level_recons)
            )

            gt_stack = tf.reshape(
                env_map_levels,
                (-1, roughnesses.shape[1] * env_map.shape[1], *env_map.shape[2:]),
            )
            pred_stack = tf.reshape(
                level_recons,
                (-1, roughnesses.shape[1] * env_map.shape[1], *env_map.shape[2:]),
            )

            hdr_to_tb("val_roughness", tf.concat([gt_stack, pred_stack], 2))

        mse = mseMean.result()
        tf.summary.scalar("val_loss", mse)


def main(args):
    # Setup directories, logging etc.
    with train_utils.SetupDirectory(
        args,
        copy_files=True,
        main_script=__file__,
        copy_data="data/illumination",
    ):
        strategy = (
            tf.distribute.get_strategy()
            if train_utils.get_num_gpus() <= 1
            else tf.distribute.MirroredStrategy()
        )

        global_batch_size = args.batch_size * train_utils.get_num_gpus()

        print("Start reading the dataset ...")
        train_df, val_df, _ = dataflow.get_train_val_data(
            args.datadir,
            args.steps_per_epoch,
            args.val_holdout,
            global_batch_size,
            0,
            args.num_random_roughness_samples,
        )
        print("... Dataset read!")

        # Optimizer and models
        with strategy.scope():
            illumination_model = IlluminationNetwork(args)
            eval_net = illumination_model.illumination_network

            lrate = train_utils.adjust_learning_rate_to_replica(args)
            if args.lrate_decay > 0:
                lrate = tf.keras.optimizers.schedules.ExponentialDecay(
                    lrate, decay_steps=args.lrate_decay * 1000, decay_rate=0.1
                )

            optimizer = tf.keras.optimizers.Adam(lrate, beta_1=0, beta_2=0.9)

        # Restore if possible
        start_step = illumination_model.restore()
        tf.summary.experimental.set_step(start_step)
        start_epoch = start_step // len(train_df)

        train_dist_df = strategy.experimental_distribute_dataset(train_df)

        print(
            "Starting training in epoch {} at step {}".format(start_epoch, start_step)
        )

        # initial validation to check everything is working
        run_validation(val_df, illumination_model)

        for epoch in range(start_epoch + 1, args.epochs + 1):
            pbar = tf.keras.utils.Progbar(len(train_df))
            # Iterate over the train dataset

            with strategy.scope():
                for dp in train_dist_df:
                    (
                        env_map,
                        _,
                        _,
                        directions_random,
                        roughness_random,
                        targets_random,
                    ) = dp

                    losses_per_replica = strategy.run(
                        illumination_model.train_step,
                        (
                            env_map,
                            directions_random,
                            roughness_random,
                            targets_random,
                            optimizer,
                        ),
                    )

                    losses = {
                        k: strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
                        for k, v in losses_per_replica.items()
                    }

                    losses_for_pbar = [
                        ("loss", losses["loss"].numpy()),
                        ("random_loss", losses["random_loss"].numpy()),
                        ("full_loss", losses["full_loss"].numpy()),
                    ]

                    pbar.add(
                        1,
                        values=losses_for_pbar,
                    )

                    with tf.summary.record_if(
                        tf.summary.experimental.get_step() % args.log_step == 0
                    ):
                        for k, v in losses.items():
                            tf.summary.scalar(k, v)

                        for var in eval_net.trainable_variables:
                            tf.summary.histogram(var.name, var)

                    if tf.summary.experimental.get_step() % args.viz_step == 0:
                        with tf.device("/device:gpu:0"):
                            if train_utils.get_num_gpus() > 1:
                                env_map = env_map.values[0]
                            z = illumination_model.cnn_encoder(env_map)

                            env1 = env_map[:1]
                            z1 = z[:1]

                            # Show roughness value changes
                            rgh_list = eval_net.eval_env_map_multi_rghs(
                                z1, 5, env1.shape[1]
                            )
                            hdr_to_tb("roughness", tf.concat([env1, *rgh_list], 1))

                            # Compare MLP with CNN reconstruction
                            cnn_recon = illumination_model.cnn_decoder(z1)
                            mlp_recon = eval_net.eval_env_map(z1, 0, env1.shape[1])[:1]
                            cnn_stack = tf.concat([env1, cnn_recon], 1)
                            mlp_stack = tf.concat([env1, mlp_recon], 1)
                            hdr_to_tb(
                                "reconstruction",
                                tf.concat([cnn_stack, mlp_stack], 2),
                            )

                            # Randomly sample
                            mean = tf.math.reduce_mean(z, 0)
                            std = tf.math.reduce_std(z, 0)
                            random_context = tf.random.normal(
                                shape=(1, z1.shape[-1]),
                                mean=mean,
                                stddev=std,
                                dtype=tf.float32,
                            )
                            rgh_list = eval_net.eval_env_map_multi_rghs(
                                random_context, 5, env1.shape[1]
                            )
                            cnn_r0_recon = illumination_model.cnn_decoder(
                                random_context
                            )
                            hdr_to_tb(
                                "random_sampling",
                                tf.concat([cnn_r0_recon, *rgh_list], 1),
                            )

                    tf.summary.experimental.set_step(
                        tf.summary.experimental.get_step() + 1
                    )

            # Save when a weight epoch arrives
            if epoch % args.weights_epoch == 0:
                illumination_model.save(
                    tf.summary.experimental.get_step()
                )  # Step was already incremented

            # Render validation if a validation epoch arrives
            if epoch % args.validation_epoch == 0 or epoch == 1:
                run_validation(val_df, illumination_model)


if __name__ == "__main__":
    args = parser().parse_args()
    print(args)

    main(args)