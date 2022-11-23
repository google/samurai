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
from typing import Dict, Optional, Tuple

import nn_utils.math_utils as math_utils
import numpy as np
import tensorflow as tf
from losses.samurai import (
    batch_cam_bce,
    batch_cam_mae,
    batch_cam_mse,
    batch_cam_mask_loss,
    batch_cam_chabonnier,
)
from nn_utils.activations import padded_sigmoid
from nn_utils.film_siren_layers import Sine
from nn_utils.initializers import SIRENInitializer
from nn_utils.nerf_layers import (
    AnnealedFourierEmbedding,
    FourierEmbedding,
    add_gaussian_noise,
    setup_fixed_grid_sampling,
    setup_hierachical_sampling,
    split_sigma_and_payload,
    volumetric_rendering,
    cast_rays,
)
from nn_utils.preintegrated_rendering import PreintegratedRenderer
from train_illumination_net import parser as illumination_parser
from utils.training_setup_utils import get_num_gpus

import models.samurai.brdf_helper as brdf_helper
from models.illumination_integration_net import IlluminationNetwork


def slight_padded_sigmoid(x):
    return padded_sigmoid(x, 0.001)


class BRDFSegmenter(tf.keras.Model):
    def __init__(
        self,
        latent_size: int,
        basecolor_metallic: bool,
        diffuse_latent_dim: int,
        fixed_diffuse: bool,
        decoder_layer: int = 2,
        hidden_dim: int = 64,
        **kwargs
    ):
        super(BRDFSegmenter, self).__init__(**kwargs)

        self.basecolor_metallic = basecolor_metallic
        self.diffuse_latent_dim = diffuse_latent_dim
        self.fixed_diffuse = fixed_diffuse

        decoder_net = [
            tf.keras.layers.InputLayer(
                (
                    None,
                    None,
                    None,
                    latent_size,
                )
            ),
        ]
        for _ in range(decoder_layer):
            decoder_net.append(
                tf.keras.layers.Dense(
                    hidden_dim,
                    activation="relu",
                    # kernel_initializer=SIRENInitializer(w0=30.0),
                )
            )

        self.decoder_net = tf.keras.Sequential(decoder_net)

        self.diffuse_final_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    (
                        None,
                        None,
                        None,
                        hidden_dim
                        + (0 if self.fixed_diffuse else self.diffuse_latent_dim),
                    )
                ),
                tf.keras.layers.Dense(3, activation=tf.sigmoid),
            ]
        )
        if self.basecolor_metallic:
            self.spec_metal_net = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer((None, None, None, hidden_dim)),
                    tf.keras.layers.Dense(1, activation=tf.sigmoid),
                ]
            )
        else:
            self.spec_metal_net = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer((None, None, None, hidden_dim)),
                    tf.keras.layers.Dense(brdf_helper.NUM_CLASSES, activation="linear"),
                ]
            )
        self.roughness_final_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((None, None, None, hidden_dim)),
                tf.keras.layers.Dense(1, activation=tf.sigmoid),
            ]
        )

    def call(
        self,
        embedding: tf.Tensor,
        diff_embd: Optional[tf.Tensor] = None,
    ):
        main_embd = self.decoder_net(embedding)

        diff_inp = (
            tf.concat([main_embd, diff_embd], -1)
            if not self.fixed_diffuse
            else main_embd
        )
        diffuse_basecolor = self.diffuse_final_net(diff_inp)
        specular_metallic = self.spec_metal_net(main_embd)
        roughness = self.roughness_final_net(main_embd)

        return diffuse_basecolor, specular_metallic, roughness

    def smoothness(
        self, embd: tf.Tensor, diff_embd: Optional[tf.Tensor] = None, eps=1e-2
    ):
        normal = tf.concat(self(embd, diff_embd), -1)
        normal_noise = tf.concat(
            self(embd + tf.random.normal(tf.shape(embd), stddev=eps), diff_embd), -1
        )

        return tf.reduce_mean(tf.reduce_sum(tf.math.abs(normal - normal_noise), -1))

    def sparsity(self, embd: tf.Tensor):
        channel_means = tf.reduce_mean(embd, (0, 1, 2))
        return tf.reduce_mean(tf.math.abs(channel_means))


class FineModel(tf.keras.Model):
    def __init__(self, args, **kwargs):
        super(FineModel, self).__init__(**kwargs)

        self.num_samples = args.fine_samples
        self.coarse_samples = args.coarse_samples
        self.raw_noise_std = args.raw_noise_std

        self.rotating_object = args.rotating_object
        self.single_env = args.single_env

        self.bounding_size = args.bounding_size

        self.appearance_latent_dim = args.appearance_latent_dim
        self.diffuse_latent_dim = args.diffuse_latent_dim
        self.skip_decomposition = args.skip_decomposition
        self.fixed_diffuse = args.fix_diffuse

        self.compose_on_white = args.compose_on_white
        self.basecolor_metallic = args.basecolor_metallic

        self.linear_disparity_sampling = args.linear_disparity_sampling

        # Start with fourier embedding
        self.pos_embedder = AnnealedFourierEmbedding(
            args.fourier_frequency, random_offsets=args.random_encoding_offsets
        )

        main_net = [
            tf.keras.layers.InputLayer(
                (
                    None,
                    None,
                    None,
                    self.pos_embedder.get_output_dimensionality(),
                )
            ),
        ]
        # Then add the main layers
        for i in range(args.fine_net_depth // 2):
            main_net.append(
                tf.keras.layers.Dense(
                    args.fine_net_width,
                    activation="relu",
                    # kernel_initializer=SIRENInitializer(w0=30.0),
                )
            )
        # Build network stack
        self.main_net_first = tf.keras.Sequential(main_net)

        main_net = [
            tf.keras.layers.InputLayer(
                (
                    None,
                    None,
                    None,
                    args.fine_net_width + self.pos_embedder.get_output_dimensionality(),
                )
            ),
        ]
        for i in range(args.fine_net_depth // 2):
            main_net.append(
                tf.keras.layers.Dense(
                    args.fine_net_width,
                    activation="relu",
                    # kernel_initializer=SIRENInitializer(w0=30.0),
                )
            )
        self.main_net_second = tf.keras.Sequential(main_net)

        # Sigma is a own output not conditioned on the illumination
        self.sigma_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    (
                        None,
                        None,
                        None,
                        args.fine_net_width,
                    )
                ),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )

        # Build a small conditional net which gets the embedding from the main net
        # plus the apperance
        self.conditional_embedding = FourierEmbedding(args.direction_fourier_frequency)

        self.bottle_neck_layer = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    (
                        None,
                        None,
                        None,
                        args.fine_net_width,
                    )
                ),
                tf.keras.layers.Dense(
                    args.fine_net_width // 2,
                    activation="relu",
                    # kernel_initializer=SIRENInitializer(w0=30.0),
                ),
            ]
        )
        self.conditional_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    (
                        None,
                        None,
                        None,
                        args.fine_net_width // 2
                        + self.appearance_latent_dim
                        + self.conditional_embedding.get_output_dimensionality(),
                    )
                ),
                tf.keras.layers.Dense(
                    args.fine_net_width // 2,
                    activation="relu",
                    # kernel_initializer=SIRENInitializer(w0=30.0),
                ),
                tf.keras.layers.Dense(3, activation=slight_padded_sigmoid),
            ]
        )

        # Add losses
        self.num_gpu = max(1, get_num_gpus())
        self.global_batch_size = args.batch_size * self.num_gpu

        if not self.skip_decomposition:
            # Build the Illumination network
            illum_parser = illumination_parser()
            illum_args = illum_parser.parse_args(
                args="--config %s"
                % os.path.join(args.illumination_network_path, "args.txt")
            )

            illumination_net = IlluminationNetwork(illum_args, trainable=False)
            path = os.path.join(args.illumination_network_path, "network.npy")

            self.illumination_net = illumination_net.illumination_network
            self.illumination_net.set_weights(np.load(path, allow_pickle=True))

            # Extract interesting networks
            self.illum_main_net = self.illumination_net.main_network
            self.illum_conditional_mapping_net = (
                self.illumination_net.conditional_network
            )
            self.illum_mapping_net = self.illumination_net.mapping_network

            self.brdf_latent_size = 16
            self.brdf_final_net = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(
                        (
                            None,
                            None,
                            None,
                            args.fine_net_width // 2,
                        )
                    ),
                    tf.keras.layers.Dense(self.brdf_latent_size),
                ]
            )

            self.brdf_decoder = BRDFSegmenter(
                latent_size=self.brdf_latent_size,
                basecolor_metallic=self.basecolor_metallic,
                diffuse_latent_dim=(
                    0 if self.fixed_diffuse else self.diffuse_latent_dim
                ),
                fixed_diffuse=self.fixed_diffuse,
            )

            # Add the renderer
            self.renderer = PreintegratedRenderer(
                args.brdf_preintegration_path,
                basecolor_metallic=self.basecolor_metallic,
            )

        self.mse = functools.partial(batch_cam_mse, self.num_gpu)
        self.mae = functools.partial(batch_cam_mae, self.num_gpu)
        self.bce = functools.partial(batch_cam_bce, self.num_gpu)
        self.alpha_loss = functools.partial(batch_cam_mask_loss, self.num_gpu)
        self.chabonnier = functools.partial(batch_cam_chabonnier, self.num_gpu)

    def payload_to_parmeters(
        self, raymarched_payload: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        ret = {}
        start = 0  # Index where the extraction starts
        ret["direct_rgb"] = raymarched_payload[..., start : start + 3]
        start += 3

        # Ensure the value range is -1 to 1 if mlp normals are used

        if not self.skip_decomposition:
            ret["normal"] = math_utils.normalize(
                raymarched_payload[..., start : start + 3]
            )
            start += 3

            ret["brdf_embedding"] = raymarched_payload[
                ..., start : start + self.brdf_latent_size
            ]
            start += self.brdf_latent_size

            # BRDF parameters
            diff_spec_name = "basecolor" if self.basecolor_metallic else "diffuse"
            ret[diff_spec_name] = raymarched_payload[..., start : start + 3]
            start += 3

            if self.basecolor_metallic:
                ret["metallic"] = raymarched_payload[..., start : start + 1]
                start += 1
            else:
                ret["specular"] = raymarched_payload[..., start : start + 3]
                start += 3

            ret["roughness"] = raymarched_payload[..., start : start + 1]
            start += 1

        return ret

    # @tf.function
    def call(
        self,
        pts,
        appearance_context,
        diffuse_context,
        ray_directions,
        alpha,
        training: bool,
        randomized=False,
    ) -> tf.Tensor:
        """Evaluates the network for all points

        Args:
            pts (tf.Tensor(float32), [B, ..., 3]): the points where to evaluate
                the network.
            appearance_context (tf.Tensor(float32), [B, appearance_latent_dim]): the
                appearance embedding for the images.
            diffuse_context (tf.Tensor(float32), [B, diffuse_latent_dim]): the
                diffuse embedding for the images.
            ray_directions (tf.Tensor(float32), [B, ..., 3]): the ray directions for
                each sample.
            alpha: the alpha value for the fourier annealing
            randomized (bool): use randomized sigma noise. Defaults to False.

        Returns:
            sigma_payload (tf.Tensor(float32), [..., samples 1 + payload_channels]):
                the sigma and the payload.
        """
        # Tape to calculate the normal gradient
        with tf.GradientTape(watch_accessed_variables=False) as normalTape:
            if not self.skip_decomposition:
                normalTape.watch(pts)  # Watch pts as it is not a variable

            pts_embed = self.pos_embedder(pts, alpha)

            # Call the main network
            main_embd = self.main_net_first(pts_embed, training)
            main_embd = self.main_net_second(
                tf.concat([main_embd, pts_embed], -1), training
            )

            # Extract sigma
            sigma = self.sigma_net(main_embd, training)

        #########
        # Calculate rgb
        #########

        appearance_context = tf.reshape(
            appearance_context,  # B, L
            (
                appearance_context.shape[0],
                *[1 for _ in main_embd.shape[1:-1]],
                appearance_context.shape[-1],
            ),  # B, 1 (C), 1 (S), L
        ) * tf.ones_like(
            main_embd[..., :1]
        )  # B, C, S, L

        # View dependent
        view_direction = math_utils.normalize(-1 * ray_directions)
        ray_embd = self.conditional_embedding(view_direction)[
            ..., None, :
        ]  # add sample dimension
        ray_embd = ray_embd * tf.ones_like(main_embd[..., :1])  # Fill sample dimension

        # Concat main embedding and the context
        main_embd = self.bottle_neck_layer(main_embd, training)
        conditional_input = tf.concat([main_embd, appearance_context, ray_embd], -1)
        # Predict the conditional RGB
        rgb = self.conditional_net(conditional_input, training)

        # Build payload list
        # Start with direct rgb
        full_payload_list = [rgb]

        ############
        # Calculate BRDF
        ############
        if not self.skip_decomposition:
            # Ensure diffuse context has the correct shape
            if diffuse_context is not None:
                diffuse_context = tf.reshape(
                    diffuse_context,  # B, L
                    (
                        diffuse_context.shape[0],
                        *[1 for _ in main_embd.shape[1:-1]],
                        diffuse_context.shape[-1],
                    ),  # B, 1 (C), 1 (S), L
                ) * tf.ones_like(
                    main_embd[..., :1]
                )  # B, C, S, L

            brdf_embd = self.brdf_final_net(main_embd, training)
            diffuse_basecolor, specular_metallic, roughness = self.brdf_decoder(
                brdf_embd, diffuse_context
            )

            if not self.basecolor_metallic:
                specular_classes = specular_metallic

                (
                    specular,
                    diffuse_basecolor,
                ) = brdf_helper.transform_specular_and_diffuse_prediction(
                    specular_classes, diffuse_basecolor
                )

            # Normals are not directly predicted
            # Normals are derived from the gradient of sigma wrt. to the input points
            normal = math_utils.normalize(-1 * normalTape.gradient(sigma, pts))
            full_payload_list.append(normal)

            # Join for the BRDF
            full_payload_list.append(brdf_embd)
            full_payload_list.append(diffuse_basecolor)
            if self.basecolor_metallic:
                full_payload_list.append(specular_metallic)
            else:
                full_payload_list.append(specular)
            full_payload_list.append(roughness)

        # Add noise
        sigma = add_gaussian_noise(sigma, self.raw_noise_std, randomized)

        # Build the output sigma and payload
        sigma_payload = tf.concat([sigma] + full_payload_list, -1)
        return sigma_payload

    def render_rays(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        near_bound: tf.Tensor,
        far_bound: tf.Tensor,
        camera_pose: tf.Tensor,
        appearance_context: tf.Tensor,
        diffuse_context: tf.Tensor,
        alpha,
        background: tf.Tensor,
        training: bool,
        illumination_context=None,
        illumination_factor=None,
        randomized: bool = False,
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Render the rays

        Args:
            ray_origins (tf.Tensor(float32), [batch, cam, samples, 3]): the ray origin.
            ray_directions (tf.Tensor(float32), [batch, cam, samples, 3]): the ray
                direction.
            previous_z_samples (tf.Tensor(float32), [batch, cam, samples, steps]): the
                previous distances to sample along the ray.
            previous_weights (tf.Tensor(float32) [batch, cam, samples, steps]): the
                previous weights along the ray. That is the accumulated product of the
                individual alphas.
            camera_pose (tf.Tensor(float32), [batch, cam, 3, 4]): the camera matrix.
            appearance_context (tf.Tensor(float32), [batch, appearance_latent_dim]):
                the appearance embedding for the images.
            diffuse_context (tf.Tensor(float32), [batch, diffuse_latent_dim]):
                the diffuse embedding for the images.
            illumination_context (tf.Tensor(float32),
                [batch, illumination_latent_dim], optional):
                the illumination embedding for the images. Only used if decomposition
                is enabled
            illumination_factor (tf.Tensor(float32), [batch], optional): additional
                scaler which can boost the illumination for high illumination values.
                Only used if decomposition is enabled
            randomized (bool, optional): Activates noise and pertub ray features.
                Defaults to False.

        Returns:
            payload (Dict[str, tf.Tensor(float32) [batch, cam, samples, num_channels]]):
                the raymarched payload dictionary.
            z_samples (tf.Tensor(float32), [batch, cam, samples, steps]): the distances
                to sample along the ray.
            weights (tf.Tensor(float32) [batch, cam, samples, steps]): the weights
                along the ray. That is the accumulated product of the individual alphas.
        """
        tf.debugging.assert_shapes(
            [
                (ray_origins, ("B", "C", "S", 3)),
                (ray_directions, ("B", "C", "S", 3)),
                (camera_pose, ("B", "C", 4, 4)),
                (appearance_context, ("B", None)),
                (background, ("B", "C", "S", 3)),
            ]
            + ([(diffuse_context, ("B", None))] if diffuse_context is not None else [])
        )

        points, z_samples = setup_fixed_grid_sampling(
            ray_origins,
            ray_directions,
            near_bound,
            far_bound,
            self.num_samples,
            randomized=randomized,
            linear_disparity=self.linear_disparity_sampling,
        )

        raw = self.call(
            points,
            appearance_context,
            diffuse_context,
            ray_directions,
            alpha,
            training,
            randomized=randomized,
        )

        sigma, payload_raw = split_sigma_and_payload(raw)

        payload, weights = volumetric_rendering(
            sigma,
            payload_raw,
            z_samples,
            ray_directions,
            self.payload_to_parmeters,
        )

        if self.compose_on_white:
            payload["direct_rgb"] = math_utils.white_background_compose(
                payload["direct_rgb"], payload["acc_alpha"][..., None]
            )
        else:
            payload["direct_rgb"] = math_utils.background_compose(
                payload["direct_rgb"], background, payload["acc_alpha"][..., None]
            )

        if not self.skip_decomposition:
            view_direction = math_utils.normalize(-1 * ray_directions)
            # Ensure the raymarched normal is actually normalized
            payload["normal"] = math_utils.white_background_compose(
                math_utils.normalize(payload["normal"]),
                payload["acc_alpha"][..., None],
            )

            # Background mask
            brdf_keys = (
                ["basecolor", "metallic", "roughness"]
                if self.basecolor_metallic
                else ["diffuse", "specular", "roughness"]
            )
            for k in brdf_keys:
                payload[k] = math_utils.saturate(
                    math_utils.white_background_compose(
                        payload[k],
                        math_utils.saturate(payload["acc_alpha"][..., None]),
                    )
                )

            # First get the reflection direction
            # Add a fake sample dimension
            (
                view_direction,
                reflection_direction,
            ) = self.renderer.calculate_reflection_direction(
                view_direction,
                payload["normal"],
                camera_pose=camera_pose[:, :, None, :, :]
                if self.rotating_object and camera_pose is not None
                else None,
            )

            # Illumination net expects a B, S, C shape.
            # Reflection_direction is B, C, S, 3
            batch_dim = reflection_direction.shape[0]
            diffuse_irradiance = self.illumination_net.call_multi_samples(
                tf.reshape(
                    reflection_direction,
                    (batch_dim, -1, reflection_direction.shape[-1]),
                ),
                tf.reshape(
                    tf.ones_like(payload["roughness"]),
                    (batch_dim, -1, 1),
                ),
                illumination_context,
            )

            # Illumination net expects a B, S, C shape.
            specular_irradiance = self.illumination_net.call_multi_samples(
                tf.reshape(
                    reflection_direction,
                    (batch_dim, -1, reflection_direction.shape[-1]),
                ),
                tf.reshape(payload["roughness"], (batch_dim, -1, 1)),
                illumination_context,
            )

            # Everything now should be B*S. Make sure that shapes
            # are okay
            rgb = (
                self.renderer(
                    *[
                        tf.reshape(e, (-1, e.shape[-1]))
                        for e in [
                            view_direction,
                            payload["normal"],
                            diffuse_irradiance,
                            specular_irradiance,
                            (
                                payload["basecolor"]
                                if self.basecolor_metallic
                                else payload["diffuse"]
                            ),
                            (
                                payload["metallic"]
                                if self.basecolor_metallic
                                else payload["specular"]
                            ),
                            payload["roughness"],
                        ]
                    ]
                )
                * illumination_factor
            )
            # Reflection direction has the exact fitting shape
            rgb = tf.reshape(rgb, tf.shape(reflection_direction))

            payload["hdr_rgb"] = rgb

            ldr_rgb = self.camera_post_processing(rgb)
            if self.compose_on_white:
                payload["rgb"] = math_utils.white_background_compose(
                    ldr_rgb, payload["acc_alpha"][..., None]
                )
            else:
                payload["rgb"] = math_utils.background_compose(
                    ldr_rgb,
                    background,
                    payload["acc_alpha"][..., None],
                )

        return (
            payload,
            z_samples,
            weights,
        )

    def camera_post_processing(self, hdr_rgb: tf.Tensor) -> tf.Tensor:
        """Applies the camera auto-exposure post-processing

        Args:
            hdr_rgb (tf.Tensor(float32), [..., 3]): the HDR input fromt the
                rendering step.
            ev100 ([type]): [description]

        Returns:
            tf.Tensor: [description]
        """
        ldr_rgb = math_utils.linear_to_srgb(
            math_utils.soft_hdr(hdr_rgb, low_threshold=0)
        )

        return ldr_rgb

    def smoothness_loss(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        surface_intersection_depth: tf.Tensor,
        ray_alpha: tf.Tensor,
        image_gradient: tf.Tensor,
        appearance_context: tf.Tensor,
        diffuse_context: tf.Tensor,
        alpha,
        sample_stddev,
    ):
        tf.debugging.assert_shapes(
            [
                (ray_origins, ("B", "C", "S", 3)),
                (ray_directions, ("B", "C", "S", 3)),
                (surface_intersection_depth, ("B", "C", "S", 1)),
                (image_gradient, ("B", "C", "S", 1)),
                (ray_alpha, ("B", "C", "S", 1)),
                (appearance_context, ("B", None)),
            ]
        )
        # Where should we sample on the surface
        surface_intersection = tf.stop_gradient(
            cast_rays(ray_origins, ray_directions, surface_intersection_depth)
        )

        base_sample = self.call(
            surface_intersection,
            tf.stop_gradient(appearance_context),
            tf.stop_gradient(diffuse_context),
            tf.stop_gradient(ray_directions),
            alpha,
            True,
        )

        grad_scaler = tf.stop_gradient(tf.exp(-image_gradient * 3))

        # Randomly sample around that point
        jitter = tf.random.normal(surface_intersection.shape, stddev=sample_stddev)
        jitter = surface_intersection + jitter

        jitter_sample = tf.stop_gradient(
            self.call(
                jitter, appearance_context, diffuse_context, ray_directions, alpha, True
            )
        )

        # Split off sigma
        base_sigma, base_payload = split_sigma_and_payload(base_sample)
        jitter_sigma, jitter_payload = split_sigma_and_payload(jitter_sample)

        base_payload_dict = self.payload_to_parmeters(base_payload)
        jitter_payload_dict = self.payload_to_parmeters(jitter_payload)

        ray_alpha_stop_grad = tf.stop_gradient(ray_alpha)

        # Remove fake sample dimensions [..., 0, :]
        normal_loss = self.mse(
            math_utils.normalize(base_payload_dict["normal"][..., 0, :])
            * ray_alpha_stop_grad
            * grad_scaler,
            math_utils.normalize(jitter_payload_dict["normal"][..., 0, :])
            * ray_alpha_stop_grad
            * grad_scaler,
        )
        roughness_loss = self.mse(
            base_payload_dict["roughness"][..., 0, :]
            * ray_alpha_stop_grad
            * grad_scaler,
            jitter_payload_dict["roughness"][..., 0, :]
            * ray_alpha_stop_grad
            * grad_scaler,
        )
        if self.basecolor_metallic:
            specular_loss = self.mse(
                base_payload_dict["metallic"][..., 0, :]
                * ray_alpha_stop_grad
                * grad_scaler,
                jitter_payload_dict["metallic"][..., 0, :]
                * ray_alpha_stop_grad
                * grad_scaler,
            )
        else:
            specular_loss = self.mse(
                base_payload_dict["specular"][..., 0, :]
                * ray_alpha_stop_grad
                * grad_scaler,
                jitter_payload_dict["specular"][..., 0, :]
                * ray_alpha_stop_grad
                * grad_scaler,
            )

        smoothness_loss = normal_loss + roughness_loss + specular_loss
        return smoothness_loss

    def calculate_losses(
        self,
        payload: Dict[str, tf.Tensor],
        target: tf.Tensor,
        target_mask: tf.Tensor,
        target_mask_confidence: tf.Tensor,
        lambda_advanced_loss: tf.Tensor,
        lambda_slow_fade_2_loss: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Calculates the losses

        Args:
            payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the
                raymarched payload dictionary.
            target (tf.Tensor(float32), [batch, 3]): the RGB target of the
                respective ray
            target_mask (tf.Tensor(float32), [batch, 1]): the segmentation mask
                target for the respective ray.
            lambda_advanced_loss (tf.Tensor(float32), [1]): current advanced loss
                interpolation value.

        Returns:
            Dict[str, tf.Tensor]: a dict of loss names with the evaluated losses.
                "loss" stores the final loss of the layer.
        """
        target_masked = math_utils.white_background_compose(target, target_mask)

        bce_loss = self.bce(
            target_mask,
            payload["acc_alpha"][..., None],
            target_mask_confidence,
        )
        bg_alpha_loss = self.alpha_loss(target_mask, payload["individual_alphas"])
        alpha_loss = bce_loss + bg_alpha_loss

        # Calculate losses
        if self.compose_on_white:
            direct_img_loss = self.chabonnier(target_masked, payload["direct_rgb"])
        else:
            direct_img_loss = self.chabonnier(target, payload["direct_rgb"])

        final_loss = direct_img_loss + alpha_loss
        losses = {
            "loss": final_loss,
            "alpha_loss": alpha_loss,
            "bce_loss": bce_loss,
            "bg_alpha_loss": bg_alpha_loss,
            "direct_rgb_loss": direct_img_loss,
        }

        if not self.skip_decomposition:
            final_loss = alpha_loss + direct_img_loss * tf.maximum(
                lambda_slow_fade_2_loss, 0.1
            )
            if self.compose_on_white:
                image_loss = self.chabonnier(target_masked, payload["rgb"])
            else:
                image_loss = self.chabonnier(target, payload["rgb"])

            diffuse_initial_loss = self.mse(
                target_masked,
                payload["basecolor"] if self.basecolor_metallic else payload["diffuse"],
            )

            final_loss = (
                final_loss
                + image_loss * (1 - lambda_advanced_loss)
                + tf.reduce_mean(diffuse_initial_loss, -1) * lambda_advanced_loss
            )

            if not self.basecolor_metallic:
                specular_initial_loss = tf.reduce_sum(
                    tf.square(
                        math_utils.white_background_compose(
                            tf.ones_like(target_masked) * brdf_helper.NON_METAL[0],
                            target_mask,
                        )
                        - payload["specular"]
                    ),
                    axis=(-2, -1),
                ) / (self.global_batch_size * 3)
                final_loss = final_loss + specular_initial_loss * lambda_advanced_loss

            losses["image_loss"] = image_loss
            losses[
                "basecolor_initial_loss"
                if self.basecolor_metallic
                else "diffuse_initial_loss"
            ] = diffuse_initial_loss

            # And update the summed loss with the decomposition additions
            losses["loss"] = final_loss

        return losses
