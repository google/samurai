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


import functools
import os
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

import nn_utils.math_utils as math_utils
from losses.samurai import (
    loss_distortion,
    normal_consistency_loss,
    normal_direction_loss,
)
from models.samurai.appearance_store import (
    AppearanceEmbeddingStore,
    DiffuseEmbeddingStore,
)
from models.samurai.camera_store import CameraStore, Ray
from models.samurai.illumination_embd_store import IlluminationEmbeddingStore
from models.samurai.input_generation_utils import (
    BatchData,
    build_train_batch,
    full_image_batch_data,
)
from models.samurai.mask_confidence_store import MaskConfidenceStore
from models.samurai.models import FineModel
from models.samurai.volume_intersection import Sphere
from nn_utils.nerf_layers import AnnealedFourierEmbedding
from utils.training_setup_utils import (
    StateRestoration,
    StateRestorationItem,
    get_num_gpus,
)


class SamuraiModel(tf.keras.Model):
    def __init__(
        self,
        num_images,
        buffer_length,
        image_dimensions: tf.Tensor,
        args,
        image_request_function,
        init_directions=None,
        init_c2w=None,
        init_focal=None,
        **kwargs,
    ):
        super(SamuraiModel, self).__init__(**kwargs)

        self.image_request_function = image_request_function

        # The main models
        self.fine_model = FineModel(args, **kwargs)

        self.color_fixed = args.fix_diffuse and args.single_env

        self.circular_loss_buffer = tf.Variable(
            initial_value=tf.zeros((buffer_length,)),
            name="circular_loss_buffer",
            trainable=False,
        )

        self.per_image_circular_buffer = tf.Variable(
            initial_value=tf.ones(
                (
                    num_images,
                    100,
                )
            )
            * 10,  # Start with a insanely high loss
            name="per_image_circular_loss_buffer",
            trainable=False,
        )
        self.per_image_write_index = tf.Variable(
            initial_value=tf.zeros(num_images, dtype=tf.int32),
            name="per_image_write_idx",
            trainable=False,
        )

        self.fourier_frequencies = args.fourier_frequency
        self.fourier_anneal_start = args.start_fourier_anneal
        self.fourier_anneal_done = args.finish_fourier_anneal
        self.camera_regularization = args.camera_regularization
        self.aim_center_regularization = args.aim_center_regularization
        self.lambda_smoothness = args.lambda_smoothness
        self.smoothness_bound_dividier = args.smoothness_bound_dividier
        self.lambda_brdf_decoder_smoothness = args.lambda_brdf_decoder_smoothness
        self.lambda_brdf_decoder_sparsity = args.lambda_brdf_decoder_sparsity

        self.coarse_distortion_lambda = args.coarse_distortion_lambda
        self.fine_distortion_lambda = args.fine_distortion_lambda
        self.normal_direction_lambda = args.normal_direction_lambda

        self.network_gradient_norm_clipping = args.network_gradient_norm_clipping
        self.camera_gradient_norm_clipping = args.camera_gradient_norm_clipping

        self.disable_posterior_scaling = args.disable_posterior_scaling
        self.disable_mask_uncertainty = args.disable_mask_uncertainty

        self.max_resolution_dimension = args.max_resolution_dimension
        self.skip_decomposition = args.skip_decomposition
        self.fixed_diffuse = args.fix_diffuse
        self.single_env = args.single_env

        # The volume bounds
        self.volume_sphere = Sphere(args.bounding_size)

        # RGB image variances between images
        self.appearance_store = AppearanceEmbeddingStore(
            num_images, args.appearance_latent_dim, self.color_fixed
        )

        # Actually build the camera optimizables
        self.random_cameras_per_view = args.random_cameras_per_view
        use_lookat = False
        if args.camera_rotation == "lookat":
            use_lookat = True

        self.camera_store = CameraStore(
            num_images,
            image_dimensions,
            args.canonical_pose,
            args.camera_weight_update_lr,
            args.camera_weight_update_momentum,
            num_cameras_per_image=args.random_cameras_per_view,
            object_height=args.bounding_size * 2,
            init_c2w=init_c2w,
            init_focal=init_focal,
            init_directions=init_directions,
            learn_f=not args.not_learn_f,
            learn_r=not args.not_learn_r,
            learn_t=not args.not_learn_t,
            use_look_at_representation=use_lookat,
            offset_learning=args.learn_camera_offsets,
            use_initializations=not args.use_fully_random_cameras,
        )

        self.mask_confidence_store = MaskConfidenceStore(num_images)

        # Useful for lighting estimation
        self.rotating_object = args.rotating_object
        self.single_env = args.single_env

        self.advanced_loss_done = args.advanced_loss_done

        # Randomize if training
        self.randomized = args.perturb == 1.0

        self.diffuse_store = None
        if (
            not args.skip_decomposition
        ):  # If decomposition is requested. Setup NeuralPIL
            illumination_latent_dim = self.fine_model.illumination_net.latent_units

            num_illuminations = 1 if self.single_env else num_images
            mean_std = np.load(
                os.path.join(
                    args.illumination_network_path, "illumination_latent_mean_std.npy"
                ),
                allow_pickle=True,
            )
            self.illumination_embedding_store = IlluminationEmbeddingStore(
                num_illuminations,
                illumination_latent_dim,
                latent_mean=mean_std[0],
                latent_std=mean_std[1],
            )
            self.illumination_embedding_store(
                tf.convert_to_tensor([0])
            )  # Ensure the store is built

            # Also diffuse store
            self.diffuse_store = DiffuseEmbeddingStore(
                num_images, args.diffuse_latent_dim
            )

        # Losses
        self.batch_size = args.batch_size

        self.random_samples = 16384
        self.num_gpu = int(max(1, get_num_gpus()))

        self.dist_loss = functools.partial(loss_distortion, self.num_gpu)
        self.normal_direction_loss = functools.partial(
            normal_direction_loss, self.num_gpu
        )
        # Setup the state restoration
        states = [
            StateRestorationItem("fine", self.fine_model),
            StateRestorationItem("appearances", self.appearance_store),
            StateRestorationItem("cameras", self.camera_store),
            StateRestorationItem(
                "circular_loss_buffer", self.circular_loss_buffer, is_variable=True
            ),
            StateRestorationItem(
                "per_image_circular_loss_buffer",
                self.per_image_circular_buffer,
                is_variable=True,
            ),
            StateRestorationItem(
                "per_image_write_index",
                self.per_image_write_index,
                is_variable=True,
            ),
            StateRestorationItem("mask_confidence_store", self.mask_confidence_store),
        ]
        if not args.skip_decomposition:
            states.append(
                StateRestorationItem(
                    "illuminations", self.illumination_embedding_store
                ),
            )
            if not self.fixed_diffuse:
                states.append(
                    StateRestorationItem("diffuse", self.diffuse_store),
                )
        self.state_restoration = StateRestoration(args, states)

        # Wrapped functions to setup build a train or eval batch
        self.build_train_batch = functools.partial(
            build_train_batch,
            self.appearance_store,
            self.camera_store,
            self.mask_confidence_store,
            self.batch_size,
        )
        self.full_image_batch_data = functools.partial(
            full_image_batch_data,
            self.appearance_store,
            self.camera_store,
            self.mask_confidence_store,
        )

    def save(self, step):
        # Save weights for step
        self.state_restoration.save(step)

    def restore(self, step: Optional[int] = None) -> int:
        # Restore weights from step or if None the latest one
        return self.state_restoration.restore(step)

    def call(
        self,
        data: BatchData,
        alpha: float,
        training: bool = False,
    ):

        backgrounds = data.rgb_targets

        # Build the near and far plane
        ray_len_to_zero = math_utils.magnitude(data.rays.origin)[
            ..., :1, :
        ]  # No sample dim

        tpmin = tf.maximum(
            (ray_len_to_zero - self.volume_sphere.radius)
            * tf.ones_like(data.rays.direction),
            1e-4,
        )
        tmin = tpmin[..., 0]

        tpmax = (ray_len_to_zero + self.volume_sphere.radius) * tf.ones_like(
            data.rays.direction
        )
        tmax = tpmax[..., 0]

        tf.debugging.assert_less(tmin, tmax)

        if self.skip_decomposition:
            illumination_embedding, illumination_factor, diffuse_embedding = (
                None,
                None,
                None,
            )
        else:
            (
                illumination_embedding,
                illumination_factor,
            ) = self.illumination_embedding_store(data.image_idx[:, 0])
            diffuse_embedding = self.diffuse_store(data.image_idx[:, 0])

        fine_payload, fine_z_samples, fine_weights = self.fine_model.render_rays(
            data.rays.origin,
            data.rays.direction,
            tmin,
            tmax,
            data.pose,
            data.appearance_embedding,
            diffuse_embedding,
            alpha,
            backgrounds,
            training,
            illumination_context=illumination_embedding,
            illumination_factor=illumination_factor,
            randomized=training and self.randomized,
        )

        return (
            fine_payload,
            fine_z_samples,
            fine_weights,
            tmin,
            tmax,
        )

    def distributed_call(
        self,
        strategy,
        chunk_size: int,
        data: BatchData,
        alpha: float,
        training: bool = False,
    ):
        """Chunks the call and in multi gpu scenarios distributes the chunks"""
        tf.debugging.assert_shapes(
            [
                (
                    data.rays.origin,
                    (1, 1, "S", 3),
                ),
                (
                    data.rays.direction,
                    (1, 1, "S", 3),
                ),
                (data.pose, (1, 1, 4, 4)),
                (data.image_idx, (1, 1)),
                (data.image_coordinates, (1, 1, "S", 2)),
                (data.appearance_embedding, (1, self.appearance_store.latent_dim)),
                (data.rgb_targets, (1, 1, "S", 3)),
                (data.mask_targets, (1, 1, "S", 1)),
                (data.mask_confidence, (1, 1, "S", 1)),
                (data.gradient_targets, (1, 1, "S", 1)),
            ]
        )

        if not isinstance(alpha, tf.Tensor):
            alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        dp_df = (
            tf.data.Dataset.from_tensor_slices(
                (
                    tf.reshape(data.rays.origin, (-1, 3)),
                    tf.reshape(data.rays.direction, (-1, 3)),
                    tf.reshape(data.image_coordinates, (-1, 2)),
                    tf.reshape(data.rgb_targets, (-1, 3)),
                    tf.reshape(data.mask_targets, (-1, 1)),
                    tf.reshape(data.mask_confidence, (-1, 1)),
                    tf.reshape(data.gradient_targets, (-1, 1)),
                )
            )
            .batch(chunk_size * get_num_gpus())
            .with_options(options)
        )
        dp_dist_df = strategy.experimental_distribute_dataset(dp_df)

        fine_payloads: Dict[str, List[tf.Tensor]] = {}

        def add_to_dict(to_add, main_dict):
            for k, v in to_add.items():
                arr = main_dict.get(
                    k,
                    [],
                )
                arr.extend(v)
                main_dict[k] = arr

            return main_dict

        for dp in dp_dist_df:
            (
                rays_o,
                rays_d,
                image_coordinates,
                rgb_targets,
                mask_targets,
                mask_confidences,
                gradients,
            ) = [tf.reshape(d, (1, 1, -1, d.shape[-1])) for d in dp]

            current_batch_data = BatchData(
                Ray(rays_o, rays_d),
                data.pose,
                data.image_idx,
                image_coordinates,
                data.appearance_embedding,
                rgb_targets,
                mask_targets,
                mask_confidences,
                gradients,
            )

            # Render image.
            (fine_result_per_replica, _, _, _, _,) = strategy.run(
                self.call,
                (
                    current_batch_data,
                    alpha,
                    training,
                ),
            )

            fine_result = {
                k: strategy.experimental_local_results(v)
                for k, v in fine_result_per_replica.items()
            }
            fine_payloads = add_to_dict(fine_result, fine_payloads)

        fine_payloads = {k: tf.concat(v, 2) for k, v in fine_payloads.items()}

        return fine_payloads

    def select_random_batch_idx(self, num_batch_idxs, select_size):
        coords = tf.reshape(tf.range(num_batch_idxs), [-1, 1])

        select_inds = tf.random.uniform_candidate_sampler(
            tf.range(coords.shape[0], dtype=tf.int64)[None, :],
            coords.shape[0],
            select_size,
            True,
            coords.shape[0],
        )[0]
        return tf.gather_nd(coords, select_inds[:, tf.newaxis])

    def get_alpha(self, step):
        return tf.cast(
            AnnealedFourierEmbedding.calculate_alpha(
                self.fourier_frequencies,
                tf.maximum(step - self.fourier_anneal_start, 1),
                self.fourier_anneal_done,
            ),
            tf.float32,
        )

    def get_mean_loss_of_image(self, img_idx):
        if img_idx is None:
            return tf.reduce_mean(self.per_image_circular_buffer, 1)
        else:
            img_idx_reshaped = tf.reshape(img_idx, (-1, 1))
            cur_values = tf.gather(self.per_image_circular_buffer, img_idx_reshaped)

            return tf.reduce_mean(
                cur_values, [i + 1 for i, _ in enumerate(cur_values.shape[1:])]
            )

    def write_to_per_image_circular_loss_buffer(self, img_idx, value):
        img_idx_reshaped = tf.reshape(img_idx, (-1, 1))

        # Get the current write index
        cur_index = tf.gather(self.per_image_write_index, img_idx_reshaped)

        # Build the buffer lookup
        img_write_index = tf.concat(
            (img_idx_reshaped, tf.reshape(cur_index, (-1, 1))), -1
        )

        # Update buffer
        new_buf = self.per_image_circular_buffer.scatter_nd_update(
            img_write_index, tf.reshape(value, (-1,))
        )
        # Bump write index
        new_idx = self.per_image_write_index.scatter_nd_update(
            img_idx_reshaped,
            tf.reshape(
                tf.math.floormod(
                    cur_index + 1, self.per_image_circular_buffer.shape[-1]
                ),
                (-1,),
            ),
        )

        # Store both
        self.per_image_circular_buffer.assign(new_buf)
        self.per_image_write_index.assign(new_idx)

    def calculate_network_loss_factor(
        self,
        fine_losses_weighted,
        epoch_step_idx,
        img_idx,
        step,
    ):
        # Now calculate the loss scaling towards the network
        # Based on "how good" the current loss compared to the moving average
        circular_buffer_losses = (
            fine_losses_weighted["alpha_loss"] + fine_losses_weighted["image_loss"]
        )

        # Write the per image circular buffer
        self.write_to_per_image_circular_loss_buffer(img_idx, circular_buffer_losses)

        new_circular_buffer = self.circular_loss_buffer.scatter_nd_update(
            tf.reshape(epoch_step_idx, (-1, 1)),
            tf.reshape(tf.reduce_mean(circular_buffer_losses), (-1,)),
        )
        self.circular_loss_buffer.assign(new_circular_buffer)

        buffer_length = self.circular_loss_buffer.shape[0]
        # If we have fewer steps taken then buffer elements select the up to steps
        # Else all buffer elements
        # Also ensure we do not try to access 0 elements
        cur_limit = tf.maximum(1, tf.minimum(buffer_length, step))
        cur_elements = self.circular_loss_buffer[:cur_limit]
        # Calculate statistics about the losses
        mean_loss = tf.reduce_mean(cur_elements)
        std_loss = tf.math.reduce_std(cur_elements)
        # And scale based on statistics
        current_factor = tf.clip_by_value(
            tf.nn.tanh(
                tf.math.divide_no_nan(
                    mean_loss - tf.reduce_mean(circular_buffer_losses), std_loss
                )
            )
            + 1.0,
            0.0,
            1.0,
        )  # If equal to mean do nothing, else scale gradients
        return current_factor

    @tf.function(
        experimental_follow_type_hints=True,
        experimental_relax_shapes=True,
    )
    def train_step(
        self,
        optimizer_network: tf.keras.optimizers.Optimizer,
        optimizer_cameras: tf.keras.optimizers.Optimizer,
        img_idx: tf.Tensor,
        epoch_step_idx: tf.Tensor,
        step: tf.Tensor,
        max_dimension_size: tf.Tensor,
        num_target_cameras: int,
        softmax_camera_scaler: tf.Tensor,
        stop_f_backprop: bool,
        is_test_set: bool,
        lambda_advanced_loss: tf.Tensor,
        lambda_slow_fade_1_loss: tf.Tensor,
        lambda_slow_fade_2_loss: tf.Tensor,
        image_dimension: tf.Tensor,
        targets: List[tf.Tensor],
    ):
        alpha = tf.convert_to_tensor(self.get_alpha(step), dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            # Build batch
            data, img_h, img_w = self.build_train_batch(
                img_idx,
                max_dimension_size,
                num_target_cameras,
                image_dimension,
                targets,
                stop_f_backprop=stop_f_backprop,
            )

            (
                fine_result,
                fine_samples,
                fine_weights,
                tmin,
                tmax,
            ) = self.call(data, alpha, training=True)

            fine_dist_loss = self.dist_loss(fine_weights, fine_samples, tmin, tmax)

            normal_dir_loss = self.normal_direction_loss(
                fine_result["individual_normal"],
                data.rays.direction,
                fine_weights,
            )

            for k in fine_result:
                tf.debugging.check_numerics(fine_result[k], "Fine output {}".format(k))

            # Get fine losses
            fine_losses = self.fine_model.calculate_losses(
                fine_result,
                data.rgb_targets,
                data.mask_targets,
                data.mask_confidence,
                lambda_advanced_loss,
                lambda_slow_fade_2_loss,
            )

            # Just directly mean everythin
            fine_losses_unweighted = {
                k: tf.reduce_mean(tf.reduce_sum(v, axis=1))
                for k, v in fine_losses.items()
            }
            # TODO fix appearance embd in the beginning with regularization

            # Add the smoothness loss
            if not self.skip_decomposition and self.lambda_smoothness > 0:
                smoothness_loss = self.fine_model.smoothness_loss(
                    data.rays.origin,
                    data.rays.direction,
                    fine_result["depth"][..., None],
                    fine_result["acc_alpha"][..., None],
                    data.gradient_targets,
                    data.appearance_embedding,
                    self.diffuse_store(tf.reshape(data.image_idx, (-1,))),
                    alpha,
                    self.volume_sphere.radius / self.smoothness_bound_dividier,
                    # Only allow a tiny sample jitter in the volume
                )
                fine_losses["smoothness"] = (
                    smoothness_loss
                    * self.lambda_smoothness
                    * tf.maximum(lambda_advanced_loss, 1e-1)
                )

            for k in fine_losses:
                tf.debugging.check_numerics(fine_losses[k], "Fine loss {}".format(k))

            # Weighting of losses for the cameras. Most of the gradient then
            # flows to the best camera
            if self.camera_store.num_cameras_per_image > 1:
                weights = self.camera_store.get_per_weight_camera_losses(
                    fine_losses["alpha_loss"] + fine_losses["image_loss"],
                    scaler=softmax_camera_scaler,
                )
                self.camera_store.update_weights(img_idx, weights)

            if self.disable_posterior_scaling:
                fine_losses_weighted = {
                    k: tf.reduce_mean(v) for k, v in fine_losses.items()
                }
            else:
                fine_losses_weighted = self.camera_store.apply_weights_to_losses(
                    fine_losses, img_idx
                )

            if not self.skip_decomposition and self.lambda_smoothness > 0.0:
                # Smoothness is not part of the regular loss. Add it after weighting
                fine_losses_weighted["loss"] = fine_losses_weighted[
                    "loss"
                ] + fine_losses_weighted["smoothness"] * self.lambda_smoothness * (
                    1 - lambda_advanced_loss
                )

            # Collect losses
            loss_network = fine_losses_weighted["loss"]
            loss_camera = fine_losses_unweighted["loss"]

            # Confidence regularizer
            confidence_regularizer = self.mask_confidence_store.get_regularization_loss(
                data.image_idx
            )
            loss_network = loss_network + 0.25 * confidence_regularizer * tf.maximum(
                lambda_slow_fade_1_loss, 0.01
            )

            aim_center_regularization = self.camera_store.aim_center_regularization(
                img_idx, num_target_cameras
            )  # B, C
            volume_padding_regularization = (
                self.camera_store.volume_padding_regularization(
                    img_idx, num_target_cameras, self.volume_sphere.radius
                )
            )

            cam_loss = {
                "aim_center": aim_center_regularization,
                "volume_padding": volume_padding_regularization,
            }

            if self.disable_posterior_scaling:
                cam_loss_weighted = {k: tf.reduce_mean(v) for k, v in cam_loss.items()}
            else:
                cam_loss_weighted = self.camera_store.apply_weights_to_losses(
                    cam_loss, img_idx
                )

            if not is_test_set:
                current_factor = self.calculate_network_loss_factor(
                    fine_losses_weighted,
                    epoch_step_idx,
                    img_idx,
                    step,
                )

            # This loss keeps the camera positions close to initial in the beginning
            camera_regularization_loss = self.camera_store.get_regularization_loss(
                img_idx  # , self.volume_sphere.radius
            )
            aim_center_regularization = tf.reduce_mean(aim_center_regularization)
            volume_padding_regularization = tf.reduce_mean(
                volume_padding_regularization
            )
            loss_camera = (
                loss_camera
                + camera_regularization_loss
                * self.camera_regularization
                * tf.maximum(lambda_advanced_loss, 0.01)
                + aim_center_regularization * self.aim_center_regularization
                + volume_padding_regularization * 100
            )

            # Decoder losses
            brdf_embedding = fine_result["brdf_embedding"]
            diffuse_context = self.diffuse_store(data.image_idx[:, 0])
            diffuse_context = tf.reshape(
                diffuse_context,
                (
                    diffuse_context.shape[0],
                    *[1 for _ in brdf_embedding.shape[1:-1]],
                    diffuse_context.shape[-1],
                ),
            ) * tf.ones_like(brdf_embedding[..., :1])

            brdf_decoder_smoothness = self.fine_model.brdf_decoder.smoothness(
                brdf_embedding[..., None, :],
                None if self.fixed_diffuse is None else diffuse_context[..., None, :],
            )
            brdf_decoder_sparsity = self.fine_model.brdf_decoder.sparsity(
                brdf_embedding[..., None, :]
            )

            global_losses = {
                "brdf_decoder_smoothness": brdf_decoder_smoothness,
                "brdf_decoder_sparsity": brdf_decoder_sparsity,
            }

            loss_network = (
                loss_network
                + self.fine_distortion_lambda
                * fine_dist_loss
                * (1 - lambda_advanced_loss)
                + self.normal_direction_lambda * normal_dir_loss
                + self.lambda_brdf_decoder_smoothness
                * global_losses["brdf_decoder_smoothness"]
                + self.lambda_brdf_decoder_sparsity
                * global_losses["brdf_decoder_sparsity"]
            )
            fine_losses_weighted["dist_loss"] = fine_dist_loss
            fine_losses_weighted["normal_dir_loss"] = normal_dir_loss

            # Apply the loss scaling factor
            if not is_test_set and not self.disable_posterior_scaling:
                loss_network = loss_network * tf.stop_gradient(current_factor)

        # Calculate the gradients
        # ... as we split the gradients for the network and cameras
        # for the two optimizers
        grad_vars_network = (
            self.fine_model.trainable_variables
            + self.appearance_store.trainable_variables
        )
        grad_vars_camera = self.camera_store.trainable_variables

        global_losses.update(
            {
                "camera_regularization": camera_regularization_loss,
                "aim_center_regularization": aim_center_regularization,
                "volume_padding_regularization": volume_padding_regularization,
                "confidence_regularization": confidence_regularizer,
            }
        )  # Losses which do not directly belong to coarse and fine

        if is_test_set:
            # We do not allow backprop to the coarse or fine network
            # Only cameras and the per object appearance/diffuse/mask confidences
            # Are adjusted
            grad_vars_network = self.appearance_store.trainable_variables

        if not self.disable_mask_uncertainty:
            grad_vars_network = (
                grad_vars_network + self.mask_confidence_store.trainable_variables
            )

        if not self.skip_decomposition:
            grad_vars_network = (
                grad_vars_network
                + self.illumination_embedding_store.trainable_variables
            )
            if not self.fixed_diffuse:
                grad_vars_network = (
                    grad_vars_network + self.diffuse_store.trainable_variables
                )

        # Calculate all gradients
        gradients_camera = tape.gradient(loss_camera, grad_vars_camera)
        gradients_network = tape.gradient(loss_network, grad_vars_network)

        def l2_norm(t):
            return math_utils.safe_sqrt(tf.reduce_sum(tf.pow(t, 2)))

        with tf.summary.record_if(step % 100 == 0):
            tf.summary.scalar("alpha", alpha, step=tf.cast(step, tf.int64))

            for gradient, variable in zip(gradients_network, grad_vars_network):
                if gradient is not None:
                    tf.summary.histogram(
                        "gradients/" + variable.name,
                        l2_norm(gradient),
                        step=tf.cast(step, tf.int64),
                    )
                tf.summary.histogram(
                    "variables/" + variable.name,
                    l2_norm(variable),
                    step=tf.cast(step, tf.int64),
                )

            for gradient, variable in zip(gradients_camera, grad_vars_camera):
                if gradient is not None:
                    tf.summary.histogram(
                        "gradients/" + variable.name,
                        l2_norm(gradient),
                        step=tf.cast(step, tf.int64),
                    )
                tf.summary.histogram(
                    "variables/" + variable.name,
                    l2_norm(variable),
                    step=tf.cast(step, tf.int64),
                )

        if self.network_gradient_norm_clipping > 0:
            gradients_network, network_norm = tf.clip_by_global_norm(
                gradients_network, self.network_gradient_norm_clipping
            )
            with tf.summary.record_if(step % 100 == 0):
                for gradient, variable in zip(gradients_network, grad_vars_network):
                    if gradient is not None:
                        tf.summary.histogram(
                            "gradients_after_clip/" + variable.name,
                            l2_norm(gradient),
                            step=tf.cast(step, tf.int64),
                        )
        if self.camera_gradient_norm_clipping > 0:
            gradients_camera, camera_norm = tf.clip_by_global_norm(
                gradients_camera, self.camera_gradient_norm_clipping
            )
            with tf.summary.record_if(step % 100 == 0):
                for gradient, variable in zip(gradients_camera, grad_vars_camera):
                    if gradient is not None:
                        tf.summary.histogram(
                            "gradients_after_clip/" + variable.name,
                            l2_norm(gradient),
                            step=tf.cast(step, tf.int64),
                        )

        # And apply optimization
        optimizer_network.apply_gradients(zip(gradients_network, grad_vars_network))
        optimizer_cameras.apply_gradients(zip(gradients_camera, grad_vars_camera))

        return (
            loss_network,
            loss_camera,
            fine_losses_weighted,
            global_losses,
        )
