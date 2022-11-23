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


import os

import imageio
import numpy as np
from tqdm import tqdm

import utils.training_setup_utils as train_utils
import models.samurai.config as samurai_config
import dataflow.samurai.config as data_config


def add_args(parser):
    parser.add_argument(
        "--log_step",
        type=int,
        default=100,
        help="frequency of tensorboard metric logging",
    )
    parser.add_argument(
        "--weights_epoch", type=int, default=10, help="save weights every x epochs"
    )
    parser.add_argument(
        "--validation_epoch",
        type=int,
        default=5,
        help="render validation every x epochs",
    )
    parser.add_argument(
        "--testset_epoch",
        type=int,
        default=100,
        help="render testset every x epochs",
    )
    parser.add_argument(
        "--video_epoch",
        type=int,
        default=100,
        help="render video every x epochs",
    )

    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000s)",
    )

    parser.add_argument("--render_only", action="store_true")

    return parser


def parse_args():
    parser = add_args(
        data_config.add_args(
            samurai_config.add_args(
                train_utils.setup_parser(),
            ),
        ),
    )
    return train_utils.parse_args_file_without_nones(parser)


def main(args):
    # Setup directories, logging etc.
    with train_utils.SetupDirectory(
        args,
        copy_files=not args.render_only,
        main_script=__file__,
        copy_data=["data/neural_pil", "data/illumination"],
    ) as train_setup_util:
        # Delay model import until the GPUs are set
        import tensorflow as tf

        from models.samurai.samurai_model import SamuraiModel
        import dataflow.samurai as data
        from utils.camera_vis_utils import plot_camera_scene

        # Everything imported

        if args.gpu is not None and train_utils.get_num_gpus() > 1:
            strategy = tf.distribute.MirroredStrategy()
        elif args.tpu is not None:
            strategy = tf.distribute.TPUStrategy(train_setup_util.resolver)
        else:
            strategy = tf.distribute.get_strategy()

        # Create the dataflow
        (
            image_shapes,
            init_c2w,
            init_focal,
            init_directions,
            image_request_function,
            train_df,
            val_df,
            test_df,
        ) = data.create_dataflow(args)

        if args.pretrained_camera_poses_folder is not None:
            # If pretrained poses are available: use them
            assert args.random_cameras_per_view == 1

            init_c2w = np.load(
                os.path.join(args.pretrained_camera_poses_folder, "poses.npy")
            )
            # Force fx only for now
            init_focal = np.load(
                os.path.join(args.pretrained_camera_poses_folder, "focal.npy")
            )[..., :1]

        # Optimizer and models
        with strategy.scope():
            samurai = SamuraiModel(
                len(image_shapes),
                len(train_df),
                image_shapes,
                args,
                image_request_function,
                init_directions=init_directions,
                init_c2w=init_c2w,
                init_focal=init_focal,
            )

            # Ensure samurai is build and working
            rnd_batch, _, _ = samurai.build_train_batch(
                tf.constant([0]),
                32,
                min(4, samurai.random_cameras_per_view),
                tf.constant([400, 400], dtype=tf.int32),
                data.InputTargets(
                    tf.zeros((1, 400, 400, 3)), tf.zeros((1, 400, 400, 1))
                ),
            )
            samurai(rnd_batch, 0)  # Call it with random data
            full_batch, _, _ = samurai.full_image_batch_data(
                tf.constant([0]),
                32,
                1,
                tf.constant([400, 400], dtype=tf.int32),
                data.InputTargets(
                    tf.zeros((1, 400, 400, 3)), tf.zeros((1, 400, 400, 1))
                ),
            )
            samurai(full_batch, 0)  # Call it with full data

            # Good to go

            # Setup optimizers
            # Network one first
            lrate_network = train_utils.adjust_learning_rate_to_replica(args)
            if args.lrate_decay > 0:
                lrate_network = tf.keras.optimizers.schedules.ExponentialDecay(
                    lrate_network, decay_steps=args.lrate_decay * 1000, decay_rate=0.1
                )
            optimizer_network = tf.keras.optimizers.Adam(lrate_network)

            # Then the pose one (camera)
            lrate_camera = args.camera_lr
            if args.camera_lr_decay:
                lrate_camera = tf.keras.optimizers.schedules.ExponentialDecay(
                    lrate_camera,
                    decay_steps=args.camera_lr_decay * 1000,
                    decay_rate=0.1,
                )
            optimizer_camera = tf.keras.optimizers.Adam(lrate_camera)

            # Restore if possible
            start_step = samurai.restore()
            tf.summary.experimental.set_step(start_step)
            start_epoch = start_step // len(train_df)

            # Setup train distribution for multi gpu training
            train_dist_df = strategy.experimental_distribute_dataset(train_df)
            test_dist_df = strategy.experimental_distribute_dataset(test_df)

            # Set up the hyperparameter schedulers
            advanced_loss_decay_steps = (
                args.advanced_loss_done
            )  # Will be 1 magnitude lower after advanced_loss_done steps
            advanced_loss_lambda = tf.Variable(1.0, dtype=tf.float32)

            slow_fade_decay_steps = args.slow_scheduler_decay
            brdf_fade_decay_steps = args.brdf_schedule_decay
            slow_fade_1_loss_lambda = tf.Variable(1.0, dtype=tf.float32)
            slow_fade_2_loss_lambda = tf.Variable(1.0, dtype=tf.float32)

            train_step = tf.Variable(tf.summary.experimental.get_step(), dtype=tf.int32)
            profiler = None

            c2ws = samurai.camera_store.get_all_c2w().numpy()
            tf.summary.image(
                "Cameras-Initial",
                plot_camera_scene(
                    c2ws,
                    "All Positions",
                    canonical_cam=args.canonical_pose,
                )[None, ...],
            )

            print(
                "Starting training in epoch {} at step {}".format(
                    start_epoch, start_step
                )
            )
            print("Start Rendering..." if args.render_only else "Start Training...")
            for epoch in range(
                start_epoch + 1,
                args.epochs
                + (
                    2 if args.render_only else 1
                ),  # Slight hack to let this loop run when rendering is at the end
            ):
                pbar = tf.keras.utils.Progbar(len(train_df))

                # Setup the hyper parameter schedulers
                train_step.assign(tf.summary.experimental.get_step())
                advanced_loss_lambda.assign(
                    1
                    * 0.1
                    ** (tf.summary.experimental.get_step() / advanced_loss_decay_steps)
                )  # Starts with 1 goes to 0

                slow_1_fade = max(
                    1 - 1 / slow_fade_decay_steps * tf.summary.experimental.get_step(),
                    0,
                )
                slow_fade_1_loss_lambda.assign(slow_1_fade)  # Starts with 1 goes to 0
                slow_2_fade = max(
                    1 - 1 / brdf_fade_decay_steps * tf.summary.experimental.get_step(),
                    0,
                )
                slow_fade_2_loss_lambda.assign(slow_2_fade)

                # Setup the scheduled hyper parameter changes
                # First the number of cameras to render
                # Starts with all cameras - decreases to a single
                cam_m1 = samurai.random_cameras_per_view - 1
                num_target_cameras = int(round(cam_m1 * slow_1_fade) + 1)

                # The image resolution factor
                # Starts dividing the image resolution by the specified factor
                # Increases image resolution over time to full resolution
                max_factor = args.resolution_factor - 1
                current_factor = (max_factor * slow_1_fade) + 1
                max_dims = tf.constant(
                    int(round(samurai.max_resolution_dimension / current_factor)),
                    dtype=tf.int32,
                )

                # Set the softmax scaler. Describes how "peaky" the camera selection is
                # High values mean mostly a single camera takes all. Low values mean
                # the softmax is more flatter
                # Starts with flat softmax and increases overtime to a peaky one
                max_cam_scaler = args.max_softmax_scaler
                min_cam_scaler = args.min_softmax_scaler
                cam_scaler = tf.constant(
                    ((max_cam_scaler - min_cam_scaler) * (1 - slow_1_fade))
                    + min_cam_scaler,
                    dtype=tf.float32,
                )

                # Iterate over the train dataset
                if (
                    not args.render_only
                ):  # Move this behavior to a different eval script
                    with strategy.scope():
                        for i, dp in enumerate(train_dist_df):
                            if args.profile and i > 100:
                                profiler = tf.profiler.experimental.Trace(
                                    "train", step_num=i - 100, _r=1
                                )
                                profiler.__enter__()

                            img_idx, targets = dp

                            i_tensor = tf.convert_to_tensor(i, dtype=tf.int32)

                            # Setup the hyper parameter schedulers
                            train_step.assign(tf.summary.experimental.get_step())
                            advanced_loss_lambda.assign(
                                1
                                * 0.1
                                ** (
                                    tf.summary.experimental.get_step()
                                    / advanced_loss_decay_steps
                                )
                            )  # Starts with 1 goes to 0

                            slow_1_fade = max(
                                1
                                - 1
                                / slow_fade_decay_steps
                                * tf.summary.experimental.get_step(),
                                0,
                            )
                            slow_fade_1_loss_lambda.assign(
                                slow_1_fade
                            )  # Starts with 1 goes to 0
                            slow_2_fade = max(
                                1
                                - 1
                                / brdf_fade_decay_steps
                                * tf.summary.experimental.get_step(),
                                0,
                            )
                            slow_fade_2_loss_lambda.assign(slow_2_fade)

                            # Setup the scheduled hyper parameter changes
                            # First the number of cameras to render
                            # Starts with all cameras - decreases to a single
                            cam_m1 = samurai.random_cameras_per_view - 1
                            num_target_cameras = int(round(cam_m1 * slow_1_fade) + 1)

                            # The image resolution factor
                            # Starts dividing the image resolution by the specified factor
                            # Increases image resolution over time to full resolution
                            max_factor = args.resolution_factor - 1
                            current_factor = (max_factor * slow_1_fade) + 1
                            max_dims = tf.constant(
                                int(
                                    round(
                                        samurai.max_resolution_dimension
                                        / current_factor
                                    )
                                ),
                                dtype=tf.int32,
                            )

                            # Set the softmax scaler. Describes how "peaky" the camera selection is
                            # High values mean mostly a single camera takes all. Low values mean
                            # the softmax is more flatter
                            # Starts with flat softmax and increases overtime to a peaky one
                            max_cam_scaler = args.max_softmax_scaler
                            min_cam_scaler = args.min_softmax_scaler
                            cam_scaler = tf.constant(
                                ((max_cam_scaler - min_cam_scaler) * (1 - slow_1_fade))
                                + min_cam_scaler,
                                dtype=tf.float32,
                            )

                            # Ensure the image dims are None in height width
                            # Otherwise the graph will be constantly rebuild
                            dims = tf.constant(
                                targets.rgb_target.get_shape().as_list()[1:-1]
                            )
                            targets = data.InputTargets(
                                *[
                                    tf.ensure_shape(
                                        t, [t.shape[0], None, None, t.shape[-1]]
                                    )
                                    for t in [targets.rgb_target, targets.mask_target]
                                ]
                            )

                            # Run the potentially distributed train step
                            (
                                loss_per_replica,
                                loss_camera_per_replica,
                                fine_losses_per_replica,
                                global_losses_per_replica,
                            ) = strategy.run(
                                samurai.train_step,
                                (
                                    optimizer_network,
                                    optimizer_camera,
                                    img_idx,
                                    i_tensor,
                                    train_step,
                                    max_dims,
                                    num_target_cameras,
                                    cam_scaler,
                                    (
                                        tf.summary.experimental.get_step()
                                        < args.start_f_optimization
                                    ),
                                    False,
                                    advanced_loss_lambda,
                                    slow_fade_1_loss_lambda,
                                    slow_fade_2_loss_lambda,
                                    dims,
                                    targets,
                                ),
                            )

                            # Collect the losses
                            loss = strategy.reduce(
                                tf.distribute.ReduceOp.SUM, loss_per_replica, axis=None
                            )
                            loss_camera = strategy.reduce(
                                tf.distribute.ReduceOp.SUM,
                                loss_camera_per_replica,
                                axis=None,
                            )
                            # Reduce the loss dicts
                            fine_losses, global_losses = [
                                {
                                    k: strategy.reduce(
                                        tf.distribute.ReduceOp.SUM, v, axis=None
                                    )
                                    for k, v in ld.items()
                                }
                                for ld in [
                                    fine_losses_per_replica,
                                    global_losses_per_replica,
                                ]
                            ]

                            if args.profile and i > 100:
                                profiler.__exit__(None, None, None)
                                if i - 100 > 10:  # Cancel
                                    return

                            # Info for the progess bar
                            losses_for_pbar = [
                                ("loss", loss.numpy()),
                                ("loss_camera", loss_camera.numpy()),
                                ("fine_loss", fine_losses["loss"].numpy()),
                            ]
                            pbar.add(
                                1,
                                values=losses_for_pbar,
                            )

                            # Log to tensorboard
                            with tf.summary.record_if(
                                tf.summary.experimental.get_step() % args.log_step == 0
                            ):
                                tf.summary.scalar("loss", loss)
                                for k, v in fine_losses.items():
                                    tf.summary.scalar("fine_%s" % k, v)
                                for k, v in global_losses.items():
                                    tf.summary.scalar("global_%s" % k, v)

                                tf.summary.scalar(
                                    "lambda_advanced_loss", advanced_loss_lambda
                                )
                                tf.summary.scalar(
                                    "lambda_slow_fade_1_loss", slow_fade_1_loss_lambda
                                )
                                tf.summary.scalar(
                                    "lambda_slow_fade_2_loss", slow_fade_2_loss_lambda
                                )

                                tf.summary.scalar(
                                    "# Camera Multiplex", num_target_cameras
                                )
                                tf.summary.scalar("Render dimension", max_dims)
                                tf.summary.scalar(
                                    "Softmax Multiplex Multiplier", cam_scaler
                                )

                                tf.summary.histogram(
                                    "camera_loss_weights",
                                    samurai.camera_store.per_cam_weights,
                                )
                                tf.summary.histogram(
                                    "camera_loss_momentums",
                                    samurai.camera_store.momentum_velocities,
                                )
                                if samurai.camera_store.use_look_at_representation:
                                    tf.summary.histogram(
                                        "camera_position",
                                        samurai.camera_store.eye_initial
                                        + samurai.camera_store.eye_offset,
                                    )
                                    tf.summary.histogram(
                                        "camera_center",
                                        samurai.camera_store.center_initial
                                        + samurai.camera_store.center_offset,
                                    )
                                    tf.summary.histogram(
                                        "camera_up",
                                        samurai.camera_store.up_rotation_initial
                                        + samurai.camera_store.up_rotation_offset,
                                    )
                                else:
                                    tf.summary.histogram(
                                        "camera_rotations",
                                        samurai.camera_store.r_initial
                                        + samurai.camera_store.r_offset,
                                    )
                                    tf.summary.histogram(
                                        "camera_translations",
                                        samurai.camera_store.t_initial
                                        + samurai.camera_store.t_offset,
                                    )
                                tf.summary.histogram(
                                    "camera_focals",
                                    samurai.camera_store.focal_lengths_initial
                                    + samurai.camera_store.focal_lengths_offset,
                                )
                                tf.summary.histogram(
                                    "loss_buffer",
                                    samurai.circular_loss_buffer,
                                )
                                tf.summary.histogram(
                                    "per_image_loss_buffer",
                                    samurai.per_image_circular_buffer,
                                )
                                tf.summary.histogram(
                                    "appearance_embedding",
                                    samurai.appearance_store.trainable_variables,
                                )
                                if samurai.diffuse_store is not None:
                                    tf.summary.histogram(
                                        "diffuse_embedding",
                                        samurai.diffuse_store.trainable_variables,
                                    )

                            tf.summary.experimental.set_step(
                                tf.summary.experimental.get_step() + 1
                            )

                    print("Rendering last datapoint")
                    # Show last dp and render to tensorboard
                    dpNonRepl = dp
                    if train_utils.get_num_replicas(args) > 1:
                        dpNonRepl = [d.values[0] for d in dp]

                    render_full_datapoint(
                        dpNonRepl,
                        samurai,
                        max_dims,
                        strategy,
                        args,
                        tf.summary.experimental.get_step(),
                    )

                    pbar = tf.keras.utils.Progbar(len(test_df))
                    with strategy.scope():
                        print("Running test set examples")
                        for ti, dp in enumerate(test_dist_df):
                            img_idx, targets = dp

                            dims = tf.constant(
                                targets.rgb_target.get_shape().as_list()[1:-1]
                            )
                            targets = data.InputTargets(
                                *[
                                    tf.ensure_shape(
                                        t, [t.shape[0], None, None, t.shape[-1]]
                                    )
                                    for t in [targets.rgb_target, targets.mask_target]
                                ]
                            )

                            # Run the potentially distributed train step
                            (
                                loss_per_replica,
                                loss_camera_per_replica,
                                fine_losses_per_replica,
                                global_losses_per_replica,
                            ) = strategy.run(
                                samurai.train_step,
                                (
                                    optimizer_network,
                                    optimizer_camera,
                                    img_idx,
                                    i_tensor,
                                    train_step,
                                    max_dims,
                                    num_target_cameras,
                                    cam_scaler,
                                    (
                                        tf.summary.experimental.get_step()
                                        < args.start_f_optimization
                                    ),
                                    True,
                                    advanced_loss_lambda,
                                    slow_fade_1_loss_lambda,
                                    slow_fade_2_loss_lambda,
                                    dims,
                                    targets,
                                ),
                            )

                            # Collect the losses
                            loss = strategy.reduce(
                                tf.distribute.ReduceOp.SUM, loss_per_replica, axis=None
                            )
                            loss_camera = strategy.reduce(
                                tf.distribute.ReduceOp.SUM,
                                loss_camera_per_replica,
                                axis=None,
                            )

                            losses_for_pbar = [
                                ("loss", loss.numpy()),
                                ("loss_camera", loss_camera.numpy()),
                            ]
                            pbar.add(
                                1,
                                values=losses_for_pbar,
                            )
                            # Reduce the loss dicts
                            fine_losses, global_losses = [
                                {
                                    k: strategy.reduce(
                                        tf.distribute.ReduceOp.SUM, v, axis=None
                                    )
                                    for k, v in ld.items()
                                }
                                for ld in [
                                    fine_losses_per_replica,
                                    global_losses_per_replica,
                                ]
                            ]

                    # Save when a weight epoch arrives
                    if epoch % args.weights_epoch == 0:
                        samurai.save(
                            tf.summary.experimental.get_step()
                        )  # Step was already incremented

                if (
                    epoch % args.video_epoch == 0
                    or epoch == args.epochs
                    or args.render_only
                ):
                    video_dir = os.path.join(
                        args.basedir,
                        args.expname,
                        "video_{:06d}".format(tf.summary.experimental.get_step()),
                    )
                    os.makedirs(video_dir, exist_ok=True)

                    for test_idx, dp in enumerate(test_df):
                        if test_idx > len(test_df) // 20:
                            break

                        img_idx, targets = dp
                        print(img_idx)

                        render_jiggle_interpol_videos(
                            samurai,
                            img_idx.numpy()[0],
                            max_dims,
                            strategy,
                            args,
                            tf.summary.experimental.get_step(),
                            video_dir,
                        )

                        render_illumination_interpol_videos(
                            samurai,
                            img_idx.numpy()[0],
                            (len(image_shapes) - 1) - img_idx.numpy()[0],
                            max_dims,
                            strategy,
                            args,
                            tf.summary.experimental.get_step(),
                            video_dir,
                        )

                    render_video(
                        samurai,
                        max_dims,
                        strategy,
                        args,
                        tf.summary.experimental.get_step(),
                        video_dir,
                    )

                if (
                    epoch % args.testset_epoch == 0
                    or epoch == args.epochs
                    or args.render_only
                ):
                    test_dir = os.path.join(
                        args.basedir,
                        args.expname,
                        "testset_{:06d}".format(tf.summary.experimental.get_step()),
                    )
                    for test_idx, dp in enumerate(test_df):
                        if test_idx > len(test_df) // 20:
                            break
                        render_full_datapoint(
                            dp,
                            samurai,
                            max_dims,
                            strategy,
                            args,
                            tf.summary.experimental.get_step(),
                            (test_dir, test_idx),
                        )

        poses = samurai.camera_store.get_all_best_c2w().numpy()  # NumImages, 4, 4
        focals = samurai.camera_store.get_all_best_focal().numpy()  # NumImages, 2

        np.save(os.path.join(args.basedir, args.expname, "poses.npy"), poses)
        np.save(os.path.join(args.basedir, args.expname, "focal.npy"), focals)


def render_full_datapoint(dp, samurai, max_dims, strategy, args, step, save_to=None):
    import tensorflow as tf

    import nn_utils.math_utils as math_utils
    from models.samurai.input_generation_utils import scale_inputs, InputTargets
    from nn_utils.tensorboard_visualization import (
        horizontal_image_log,
    )
    from utils.camera_vis_utils import plot_camera_scene
    from utils.visualization_utils import visualize_masks

    img_idx, targets = dp

    # again set to none to keep the graph from rebuilding
    dims = tf.constant(targets.rgb_target.get_shape().as_list()[1:-1])
    targets = InputTargets(
        *[
            tf.ensure_shape(t, [t.shape[0], None, None, t.shape[-1]])
            for t in [targets.rgb_target, targets.mask_target]
        ]
    )

    full_batch, H, W = samurai.full_image_batch_data(
        img_idx, max_dims, 1, dims, targets
    )

    # Resize the targets so we can log them side by side
    _, scale_targets = scale_inputs(max_dims, dims, targets)

    fine_result = samurai.distributed_call(
        strategy,
        args.batch_size,
        full_batch,
        samurai.get_alpha(step),
    )

    brdf_keys = (
        ["basecolor", "metallic", "roughness", "normal"]
        if args.basecolor_metallic
        else ["diffuse", "specular", "roughness", "normal"]
    )

    if save_to is not None:
        save_path, save_idx = save_to

        os.makedirs(save_path, exist_ok=True)

        def save_img(suffix, img):
            return imageio.imwrite(
                os.path.join(save_path, "{:d}_{}.jpg".format(save_idx, suffix)),
                (img * 255).astype(np.uint8),
            )

        gt_rgb = tf.reshape(full_batch.rgb_targets, (H, W, 3)).numpy()
        save_img("gt_rgb", gt_rgb)
        fine_direct_rgb = tf.reshape(fine_result["direct_rgb"], (H, W, 3)).numpy()
        save_img("fine_direct_rgb", fine_direct_rgb)

        gt_alpha = tf.reshape(full_batch.mask_targets, (H, W, 1)).numpy()
        save_img("gt_alpha", gt_alpha)
        fine_alpha = tf.reshape(fine_result["acc_alpha"], (H, W, 1)).numpy()
        save_img("fine_alpha", fine_alpha)

        mask_viz_fine = visualize_masks(gt_alpha, fine_alpha)
        save_img("fine_alpha_comp", mask_viz_fine)

        fine_disparity = tf.reshape(
            fine_result["disparity"] * fine_result["acc_alpha"], (H, W, 1)
        ).numpy()
        save_img("fine_disparity", fine_disparity)

        if "rgb" in fine_result:
            fine_render_rgb = tf.reshape(fine_result["rgb"], (H, W, 3)).numpy()
            save_img("fine_rgb", fine_render_rgb)

        if all([e in fine_result for e in brdf_keys]):
            # White background is destroyed by rescaling
            normal = tf.reshape(fine_result["normal"], (H, W, 3)) * 0.5 + 0.5
            fine_alpha_clip = tf.clip_by_value(fine_alpha, 0, 1)
            normal = normal * fine_alpha_clip + tf.ones_like(normal) * (
                1 - fine_alpha_clip
            )
            normal = normal.numpy()
            save_img("normal", normal)

            if args.basecolor_metallic:
                basecolor = tf.reshape(fine_result["basecolor"], (H, W, 3)).numpy()
                save_img("basecolor", basecolor)

                metallic = tf.reshape(fine_result["metallic"], (H, W, 1)).numpy()
                save_img("metallic", metallic)
            else:
                diffuse = tf.reshape(fine_result["diffuse"], (H, W, 3)).numpy()
                save_img("diffuse", diffuse)

                specular = tf.reshape(fine_result["specular"], (H, W, 3)).numpy()
                save_img("specular", specular)

            roughness = tf.reshape(fine_result["roughness"], (H, W, 1)).numpy()
            save_img("roughness", roughness)

            env_latent, _ = samurai.illumination_embedding_store(img_idx)
            print(env_latent.shape)
            env_map_hdr = samurai.fine_model.illumination_net.eval_env_map(
                tf.reshape(env_latent, (1, -1)),
                float(0),
            )

            env_map_hdr = tf.reshape(env_map_hdr, (128, 256, 3))
            env_map = math_utils.linear_to_srgb(
                math_utils.aces_approx(env_map_hdr)
            ).numpy()  # Tone mapping
            save_img("env_map", env_map)

            env_map_hdr = env_map_hdr.numpy()
            imageio.imwrite(
                os.path.join(save_path, "{:d}_env_map.exr".format(save_idx)),
                env_map_hdr,
            )

    else:
        horizontal_image_log(
            "train/rgb",
            *(
                [
                    tf.reshape(full_batch.rgb_targets, (1, H, W, 3)),
                    tf.reshape(fine_result["direct_rgb"], (1, H, W, 3)),
                ]
                + (
                    [tf.reshape(fine_result["rgb"], (1, H, W, 3))]
                    if "rgb" in fine_result
                    else []
                )
            ),
        )

        confidence = samurai.mask_confidence_store.get_confidence_for_mask(
            img_idx, full_batch.mask_targets
        )
        confidence_mask = samurai.mask_confidence_store.apply_confidence_to_mask(
            full_batch.mask_targets, confidence
        )
        alpha = tf.reshape(fine_result["acc_alpha"], (1, H, W, 1))
        horizontal_image_log(
            "train/alpha",
            tf.reshape(full_batch.mask_targets, (1, H, W, 1)),
            tf.reshape(full_batch.gradient_targets, (1, H, W, 1)),
            tf.reshape(confidence, (1, H, W, 1)),
            tf.reshape(confidence_mask, (1, H, W, 1)),
            alpha,
        )

        fine_alpha = tf.reshape(fine_result["acc_alpha"], (H, W, 1)).numpy()
        gt_alpha = tf.reshape(full_batch.mask_targets, (H, W, 1)).numpy()

        mask_viz_fine = visualize_masks(gt_alpha, fine_alpha)

        horizontal_image_log(
            "train/mask_comparison",
            tf.convert_to_tensor(mask_viz_fine)[None, ...],
        )
        horizontal_image_log(
            "train/disparity",
            tf.reshape(
                fine_result["disparity"] * fine_result["acc_alpha"], (1, H, W, 1)
            ),
        )

        if all([e in fine_result for e in brdf_keys]):
            # White background is destroyed by rescaling
            normal = tf.reshape(fine_result["normal"], (1, H, W, 3)) * 0.5 + 0.5
            alpha_clip = tf.clip_by_value(alpha, 0, 1)
            normal = normal * alpha_clip + tf.ones_like(normal) * (1 - alpha_clip)

            # Also get the env map
            illumination_context, _ = samurai.illumination_embedding_store(img_idx)
            env_map = samurai.fine_model.illumination_net.eval_env_map(
                illumination_context, 0, alpha.shape[1]
            )

            horizontal_image_log(
                "train/brdf",
                (
                    tf.reshape(fine_result["basecolor"], (1, H, W, 3))
                    if args.basecolor_metallic
                    else tf.reshape(fine_result["diffuse"], (1, H, W, 3))
                ),
                (
                    math_utils.repeat(
                        tf.reshape(fine_result["metallic"], (1, H, W, 1)), 3, -1
                    )
                    if args.basecolor_metallic
                    else tf.reshape(fine_result["specular"], (1, H, W, 3))
                ),
                math_utils.repeat(
                    tf.reshape(fine_result["roughness"], (1, H, W, 1)), 3, -1
                ),
                normal,
                math_utils.linear_to_srgb(
                    math_utils.aces_approx(tf.reshape(env_map, (1, H, 2 * H, 3)))
                ),  # Tone mapping
            )

        c2ws = samurai.camera_store.get_all_best_c2w().numpy()
        tf.summary.image(
            "Cameras",
            plot_camera_scene(
                c2ws,
                "Top-1 Positions",
                canonical_cam=args.canonical_pose,
            )[None, ...],
        )


def render_video(samurai, max_dims, strategy, args, step, video_dir):
    import tensorflow as tf

    import nn_utils.math_utils as math_utils
    import dataflow.samurai as data
    from models.samurai.camera_store import CameraParameter

    poses, render_focal = samurai.camera_store.get_spherical_poses(40)
    render_focal = math_utils.repeat(tf.reshape(render_focal, (1, 1, 1)), 2, -1)

    pose_df = tf.data.Dataset.from_tensor_slices(poses)

    H = max_dims.numpy()
    W = max_dims.numpy()

    brdf_keys = (
        ["basecolor", "metallic", "roughness", "normal"]
        if args.basecolor_metallic
        else ["diffuse", "specular", "roughness", "normal"]
    )

    def render_pose(pose):
        pose = tf.reshape(pose, (1, 1, 4, 4))

        camera_param = CameraParameter(pose, render_focal)
        (rays, _, coords) = samurai.camera_store.build_ray_geometry(
            camera_param, (H, W), (tf.constant([1.0]), tf.constant([1.0])), False
        )

        batch_data, _, _ = samurai.full_image_batch_data(
            tf.constant([0], dtype=tf.int32),
            max_dims,
            1,
            (H, W),
            data.InputTargets(
                tf.ones((1, H, W, 3)),
                tf.ones((1, H, W, 1)),
            ),
            overwrite_rays_cw2_coordinates=(
                rays,
                camera_param,
                tf.cast(coords, tf.float32),
            ),
        )

        fine_result = samurai.distributed_call(
            strategy,
            args.batch_size,
            batch_data,
            samurai.get_alpha(step),
        )

        return fine_result

    fine_results = {}
    for pose_dp in tqdm(pose_df):
        cur_pose = pose_dp
        fine_result = render_pose(pose_dp)

        extract_keys = ["direct_rgb", "acc_alpha", "disparity", "rgb"] + brdf_keys
        for k, v in fine_result.items():
            if k in extract_keys:
                fine_results[k] = fine_results.get(k, []) + [v.numpy()]

    fine_result_np = {k: np.stack(v, 0) for k, v in fine_results.items()}

    direct_rgb = fine_result_np["direct_rgb"].reshape((-1, H, W, 3))
    imageio.mimwrite(
        os.path.join(video_dir, "direct_rgb.mp4"),
        (direct_rgb * 255).astype(np.uint8),
        fps=30,
        quality=8,
    )

    alpha = fine_result_np["acc_alpha"].reshape((-1, H, W, 1))
    imageio.mimwrite(
        os.path.join(video_dir, "alpha.mp4"),
        (np.clip(alpha, 0, 1) * 255).astype(np.uint8),
        fps=30,
        quality=8,
    )

    disparity = fine_result_np["disparity"].reshape((-1, H, W, 1)) * alpha
    imageio.mimwrite(
        os.path.join(video_dir, "disparity.mp4"),
        (disparity * 255).astype(np.uint8),
        fps=30,
        quality=8,
    )

    if "rgb" in fine_result_np:  # We did a decomposition step. Save everything
        rgb = fine_result_np["rgb"].reshape((-1, H, W, 3))
        imageio.mimwrite(
            os.path.join(video_dir, "rgb.mp4"),
            (rgb * 255).astype(np.uint8),
            fps=30,
            quality=8,
        )

        if args.basecolor_metallic:
            basecolor = fine_result_np["basecolor"].reshape((-1, H, W, 3))
            imageio.mimwrite(
                os.path.join(video_dir, "basecolor.mp4"),
                (basecolor * 255).astype(np.uint8),
                fps=30,
                quality=8,
            )

            metallic = fine_result_np["metallic"].reshape((-1, H, W, 1))
            imageio.mimwrite(
                os.path.join(video_dir, "metallic.mp4"),
                (metallic * 255).astype(np.uint8),
                fps=30,
                quality=8,
            )
        else:
            diffuse = fine_result_np["diffuse"].reshape((-1, H, W, 3))
            imageio.mimwrite(
                os.path.join(video_dir, "diffuse.mp4"),
                (diffuse * 255).astype(np.uint8),
                fps=30,
                quality=8,
            )

            specular = fine_result_np["specular"].reshape((-1, H, W, 3))
            imageio.mimwrite(
                os.path.join(video_dir, "specular.mp4"),
                (specular * 255).astype(np.uint8),
                fps=30,
                quality=8,
            )

        roughness = fine_result_np["roughness"].reshape((-1, H, W, 1))
        imageio.mimwrite(
            os.path.join(video_dir, "roughness.mp4"),
            (roughness * 255).astype(np.uint8),
            fps=30,
            quality=8,
        )

        normal = fine_result_np["normal"].reshape((-1, H, W, 3)) * 0.5 + 0.5
        alpha_clip = np.clip(alpha, 0, 1)
        normal = normal * alpha_clip + np.ones_like(normal) * (1 - alpha_clip)
        imageio.mimwrite(
            os.path.join(video_dir, "normal.mp4"),
            (normal * 255).astype(np.uint8),
            fps=30,
            quality=8,
        )


def render_illumination_interpol_videos(
    samurai,
    img_idx,
    secondary_img_idx,
    max_dims,
    strategy,
    args,
    step,
    video_dir,
):
    import tensorflow as tf

    import nn_utils.math_utils as math_utils
    import dataflow.samurai as data
    from models.samurai.camera_store import CameraParameter

    H, W = [d.numpy()[0] for d in samurai.camera_store.get_height_width(img_idx)]

    frozen_pose, _, _ = samurai.camera_store(img_idx, 1)

    (rays, _, coords) = samurai.camera_store.build_ray_geometry(
        frozen_pose, (H, W), (tf.constant([1.0]), tf.constant([1.0])), False
    )

    batch_data, _, _ = samurai.full_image_batch_data(
        tf.constant([0], dtype=tf.int32),
        max_dims,
        1,
        (H, W),
        data.InputTargets(
            tf.ones((1, H, W, 3)),
            tf.ones((1, H, W, 1)),
        ),
        overwrite_rays_cw2_coordinates=(
            rays,
            frozen_pose,
            tf.cast(coords, tf.float32),
        ),
    )

    fine_result = samurai.distributed_call(
        strategy,
        args.batch_size,
        batch_data,
        samurai.get_alpha(step),
    )

    view_direction = math_utils.normalize(-1 * rays.direction)

    (
        illumination_context_prim,
        illumination_factor_prim,
    ) = samurai.illumination_embedding_store(tf.convert_to_tensor([img_idx], tf.int32))
    (
        illumination_context_sec,
        illumination_factor_sec,
    ) = samurai.illumination_embedding_store(
        tf.convert_to_tensor([secondary_img_idx], tf.int32)
    )

    steps = 60
    fine_results = {}
    for step in range(steps):
        interpol_alpha = step / steps

        illumination_context = (
            1 - interpol_alpha
        ) * illumination_context_prim + interpol_alpha * illumination_context_sec
        illumination_factor = (
            1 - interpol_alpha
        ) * illumination_factor_prim + interpol_alpha * illumination_factor_sec

        # Render

        # First get the reflection direction
        # Add a fake sample dimension
        (
            view_direction,
            reflection_direction,
        ) = samurai.fine_model.renderer.calculate_reflection_direction(
            tf.reshape(view_direction, (1, 1, -1, 3)),
            fine_result["normal"],
            camera_pose=None,
        )

        # Illumination net expects a B, S, C shape.
        # Reflection_direction is B, C, S, 3
        batch_dim = reflection_direction.shape[0]
        diffuse_irradiance = samurai.fine_model.illumination_net.call_multi_samples(
            tf.reshape(
                reflection_direction,
                (batch_dim, -1, reflection_direction.shape[-1]),
            ),
            tf.reshape(
                tf.ones_like(fine_result["roughness"]),
                (batch_dim, -1, 1),
            ),
            illumination_context,
        )

        # Illumination net expects a B, S, C shape.
        specular_irradiance = samurai.fine_model.illumination_net.call_multi_samples(
            tf.reshape(
                reflection_direction,
                (batch_dim, -1, reflection_direction.shape[-1]),
            ),
            tf.reshape(fine_result["roughness"], (batch_dim, -1, 1)),
            illumination_context,
        )

        # Everything now should be B*S. Make sure that shapes
        # are okay
        rgb = (
            samurai.fine_model.renderer(
                *[
                    tf.reshape(e, (-1, e.shape[-1]))
                    for e in [
                        view_direction,
                        fine_result["normal"],
                        diffuse_irradiance,
                        specular_irradiance,
                        (
                            fine_result["basecolor"]
                            if args.basecolor_metallic
                            else fine_result["diffuse"]
                        ),
                        (
                            fine_result["metallic"]
                            if args.basecolor_metallic
                            else fine_result["specular"]
                        ),
                        fine_result["roughness"],
                    ]
                ]
            )
            * illumination_factor
        )
        # Reflection direction has the exact fitting shape
        rgb = tf.reshape(rgb, tf.shape(reflection_direction))

        ldr_rgb = samurai.fine_model.camera_post_processing(rgb)

        comp = math_utils.white_background_compose(
            ldr_rgb, fine_result["acc_alpha"][..., None]
        )

        fine_results["rgb"] = fine_results.get("rgb", []) + [comp.numpy()]

    fine_result_np = {k: np.stack(v, 0) for k, v in fine_results.items()}

    suffix = f"illumination_interpol_{img_idx}-{secondary_img_idx}"

    rgb = fine_result_np["rgb"].reshape((-1, H, W, 3))
    imageio.mimwrite(
        os.path.join(video_dir, f"rgb_{suffix}.mp4"),
        (rgb * 255).astype(np.uint8),
        fps=30,
        quality=8,
    )


def render_jiggle_interpol_videos(
    samurai, img_idx, max_dims, strategy, args, step, video_dir
):
    import tensorflow as tf

    import nn_utils.math_utils as math_utils
    import dataflow.samurai as data
    from models.samurai.camera_store import CameraParameter

    params = samurai.camera_store.get_jiggle_pose(img_idx, 60)
    poses = params.c2w
    render_focal = params.focal
    # Pose is N, 1, 1, 4, 4 - Focal is 1, 1, 2

    pose_df = tf.data.Dataset.from_tensor_slices(poses)

    H, W = [d.numpy()[0] for d in samurai.camera_store.get_height_width(img_idx)]

    def render_pose(pose):
        pose = tf.reshape(pose, (1, 1, 4, 4))

        camera_param = CameraParameter(pose, render_focal)
        (rays, _, coords) = samurai.camera_store.build_ray_geometry(
            camera_param, (H, W), (tf.constant([1.0]), tf.constant([1.0])), False
        )

        batch_data, _, _ = samurai.full_image_batch_data(
            tf.constant([0], dtype=tf.int32),
            max_dims,
            1,
            (H, W),
            data.InputTargets(
                tf.ones((1, H, W, 3)),
                tf.ones((1, H, W, 1)),
            ),
            overwrite_rays_cw2_coordinates=(
                rays,
                camera_param,
                tf.cast(coords, tf.float32),
            ),
        )

        fine_result = samurai.distributed_call(
            strategy,
            args.batch_size,
            batch_data,
            samurai.get_alpha(step),
        )

        return fine_result

    for freeze_pose in [True, False]:
        if freeze_pose:
            frozen_pose, _, _ = samurai.camera_store(img_idx, 1)

            (rays, _, coords) = samurai.camera_store.build_ray_geometry(
                frozen_pose, (H, W), (tf.constant([1.0]), tf.constant([1.0])), False
            )

            batch_data, _, _ = samurai.full_image_batch_data(
                tf.constant([0], dtype=tf.int32),
                max_dims,
                1,
                (H, W),
                data.InputTargets(
                    tf.ones((1, H, W, 3)),
                    tf.ones((1, H, W, 1)),
                ),
                overwrite_rays_cw2_coordinates=(
                    rays,
                    frozen_pose,
                    tf.cast(coords, tf.float32),
                ),
            )

            fine_result = samurai.distributed_call(
                strategy,
                args.batch_size,
                batch_data,
                samurai.get_alpha(step),
            )

            fine_results = {}
            for pose_dp in tqdm(pose_df):
                (rays, _, coords) = samurai.camera_store.build_ray_geometry(
                    CameraParameter(pose_dp, render_focal),
                    (H, W),
                    (tf.constant([1.0]), tf.constant([1.0])),
                    False,
                )
                # Extract modified ray direction

                view_direction = math_utils.normalize(-1 * rays.direction)

                (
                    illumination_context,
                    illumination_factor,
                ) = samurai.illumination_embedding_store(batch_data.image_idx[:, 0])

                # Render

                # First get the reflection direction
                # Add a fake sample dimension
                (
                    view_direction,
                    reflection_direction,
                ) = samurai.fine_model.renderer.calculate_reflection_direction(
                    tf.reshape(view_direction, (1, 1, -1, 3)),
                    fine_result["normal"],
                    camera_pose=None,
                )

                # Illumination net expects a B, S, C shape.
                # Reflection_direction is B, C, S, 3
                batch_dim = reflection_direction.shape[0]
                diffuse_irradiance = (
                    samurai.fine_model.illumination_net.call_multi_samples(
                        tf.reshape(
                            reflection_direction,
                            (batch_dim, -1, reflection_direction.shape[-1]),
                        ),
                        tf.reshape(
                            tf.ones_like(fine_result["roughness"]),
                            (batch_dim, -1, 1),
                        ),
                        illumination_context,
                    )
                )

                # Illumination net expects a B, S, C shape.
                specular_irradiance = (
                    samurai.fine_model.illumination_net.call_multi_samples(
                        tf.reshape(
                            reflection_direction,
                            (batch_dim, -1, reflection_direction.shape[-1]),
                        ),
                        tf.reshape(fine_result["roughness"], (batch_dim, -1, 1)),
                        illumination_context,
                    )
                )

                # Everything now should be B*S. Make sure that shapes
                # are okay
                rgb = (
                    samurai.fine_model.renderer(
                        *[
                            tf.reshape(e, (-1, e.shape[-1]))
                            for e in [
                                view_direction,
                                fine_result["normal"],
                                diffuse_irradiance,
                                specular_irradiance,
                                (
                                    fine_result["basecolor"]
                                    if args.basecolor_metallic
                                    else fine_result["diffuse"]
                                ),
                                (
                                    fine_result["metallic"]
                                    if args.basecolor_metallic
                                    else fine_result["specular"]
                                ),
                                fine_result["roughness"],
                            ]
                        ]
                    )
                    * illumination_factor
                )
                # Reflection direction has the exact fitting shape
                rgb = tf.reshape(rgb, tf.shape(reflection_direction))

                ldr_rgb = samurai.fine_model.camera_post_processing(rgb)

                comp = math_utils.white_background_compose(
                    ldr_rgb, fine_result["acc_alpha"][..., None]
                )

                fine_results["rgb"] = fine_results.get("rgb", []) + [comp.numpy()]
        else:
            fine_results = {}
            for pose_dp in tqdm(pose_df):
                fine_result = render_pose(pose_dp)

                extract_keys = ["rgb"]
                for k, v in fine_result.items():
                    if k in extract_keys:
                        fine_results[k] = fine_results.get(k, []) + [v.numpy()]

        fine_result_np = {k: np.stack(v, 0) for k, v in fine_results.items()}

        suffix = f"jiggle_frozenpose_{img_idx}" if freeze_pose else f"jiggle_{img_idx}"

        if "rgb" in fine_result_np:  # We did a decomposition step. Save everything
            rgb = fine_result_np["rgb"].reshape((-1, H, W, 3))
            imageio.mimwrite(
                os.path.join(video_dir, f"rgb_{suffix}.mp4"),
                (rgb * 255).astype(np.uint8),
                fps=30,
                quality=8,
            )


if __name__ == "__main__":
    args = parse_args()
    print(args)

    main(args)
