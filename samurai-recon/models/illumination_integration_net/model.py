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
import tensorflow.keras as K
import tensorflow.keras.layers as layers

import nn_utils.math_utils as math_utils
from nn_utils.activations import to_hdr_activation
from nn_utils.film_siren_layers import FiLMSiren
from nn_utils.initializers import SIRENFirstLayerInitializer, SIRENInitializer
from nn_utils.nerf_layers import FourierEmbedding


class ConditionalMappingNet(K.Model):
    def __init__(self, args, **kwargs):
        super(ConditionalMappingNet, self).__init__(**kwargs)
        units = 1 if args.single_offset else args.net_width

        self.embedder = FourierEmbedding(args.cond_frequency, input_dim=1)
        self.mapping_net = K.Sequential(
            [
                layers.InputLayer(
                    (self.embedder.get_output_dimensionality(),),
                    name="ConditionalNetwork/Input",
                ),
                layers.Dense(
                    args.cond_width,
                    activation="elu",
                    kernel_initializer="he_uniform",
                    name="ConditionalNetwork/Dense1",
                ),
                layers.Dense(
                    units * 2,
                    activation="linear",
                    name="ConditionalNetwork/DenseFinal",
                ),
                layers.Reshape((2, units)),
            ]
        )
        self.mapping_net.summary()

    @tf.function(experimental_relax_shapes=True)
    def call(self, conditional):
        params = self.mapping_net(self.embedder(conditional))

        phase_shift, frequency = params[:, 0], params[:, 1]

        return phase_shift, frequency


class FiLMMappingNet(K.Model):
    def __init__(self, args, **kwargs):
        super(FiLMMappingNet, self).__init__(**kwargs)

        units = 1 if args.single_offset else args.net_width
        output_units = units * args.net_depth

        self.latent_dim = args.latent_dim

        self.mapping_net = tf.keras.Sequential(
            [
                layers.InputLayer((self.latent_dim,), name="MappingNetwork/Input"),
            ]
            + [
                layers.Dense(
                    args.mapping_width,
                    activation="elu",
                    kernel_initializer="he_uniform",
                    name="MappingNetwork/Layer_%d" % i,
                )
                for i in range(args.mapping_depth)
            ]
            + [
                layers.Dense(
                    output_units * 2,
                    activation="linear",
                    name="MappingNetwork/Final",
                ),
                layers.Reshape((2, args.net_depth, units)),
            ]
        )
        self.mapping_net.summary()

    @tf.function(experimental_relax_shapes=True)
    def call(self, latent_vector):
        params = self.mapping_net(latent_vector)

        phase_shift, frequency = params[:, 0], params[:, 1]

        return phase_shift, frequency


class MainNetwork(K.Model):
    def __init__(self, args, **kwargs):
        super(MainNetwork, self).__init__(**kwargs)

        self.film_layer = FiLMSiren(w0=1)

        self.net = []
        input_size = 3
        for i in range(args.net_depth):
            self.net.append(
                K.Sequential(
                    [
                        layers.InputLayer(
                            (input_size,), name="MainNetwork/Input%d" % i
                        ),
                        layers.Dense(
                            args.net_width,
                            activation=None,
                            kernel_initializer=SIRENFirstLayerInitializer(0.5)
                            if i == 0
                            else SIRENInitializer(w0=2),
                            name="MainNetwork/Layer%d" % i,
                        ),
                    ]
                )
            )
            input_size = args.net_width

        self.conditional_layer = K.Sequential(
            [
                layers.InputLayer((input_size,), name="MainNetwork/ConditionalInput"),
                layers.Dense(
                    args.net_width,
                    activation=None,
                    kernel_initializer=SIRENInitializer(w0=2),
                    name="MainNetwork/Conditional",
                ),
            ]
        )

        self.final_layer = K.Sequential(
            [
                layers.InputLayer((input_size,), name="MainNetwork/OutputInput"),
                layers.Dense(
                    3,
                    activation=to_hdr_activation,
                    name="MainNetwork/Output",
                ),
            ]
        )

    @tf.function(experimental_relax_shapes=True)
    def call(self, direction, main_params, conditional_params):
        main_phase_shift, main_frequency = main_params
        conditional_phase_shift, conditional_frequency = conditional_params

        x = direction
        for i, layer in enumerate(self.net):
            x = layer(x)
            x = self.film_layer(x, main_frequency[:, i], main_phase_shift[:, i])

        x = self.conditional_layer(x)
        x = self.film_layer(x, conditional_frequency, conditional_phase_shift)

        x = self.final_layer(x)

        return x


class FiLMIlluminationNetwork(K.Model):
    def __init__(self, args, **kwargs):
        super(FiLMIlluminationNetwork, self).__init__(**kwargs)

        self.latent_units = args.latent_dim

        self.main_network = MainNetwork(args, **kwargs)
        self.mapping_network = FiLMMappingNet(args, **kwargs)
        self.conditional_network = ConditionalMappingNet(args, **kwargs)

    @tf.function(experimental_relax_shapes=True)
    def call(self, direction, conditional, latent):
        main_params = self.mapping_network(latent)
        conditional_params = self.conditional_network(conditional)

        return self.main_network(direction, main_params, conditional_params)

    @tf.function(experimental_relax_shapes=True)
    def call_multi_samples(self, direction, conditional, latent):
        latent_samples = tf.expand_dims(latent, 1) * tf.ones_like(
            direction[..., :1]
        )  # B, H*W, latent_dim

        latent_flat = tf.reshape(latent_samples, (-1, self.mapping_network.latent_dim))
        directions_flat = tf.reshape(direction, (-1, 3))
        cond_flat = tf.reshape(conditional, (-1, 1))

        recon_flat = self.call(directions_flat, cond_flat, latent_flat)

        recon_shape_restored = tf.reshape(
            recon_flat,
            (
                tf.shape(latent)[0],
                direction.shape[1] if direction.shape[1] is not None else -1,
                recon_flat.shape[-1],
            ),
        )

        return recon_shape_restored

    def eval_env_map(self, latent, conditional: float, img_height=128):
        uvs = math_utils.shape_to_uv(img_height, img_height * 2)  # H, W, 2
        directions = math_utils.uv_to_direction(uvs)  # H, W, 3
        directions_flat = tf.reshape(directions, (-1, 3))

        directions_batched = math_utils.repeat(
            directions_flat[None, :], tf.shape(latent)[0], 0
        )  # B, H*W, 3
        directions_batched = tf.cast(directions_batched, tf.float32)

        cond_batched = tf.ones_like(directions_batched[..., :1]) * tf.cast(
            conditional, tf.float32
        )  # B, H*W, 1

        latent = tf.cast(latent, tf.float32)
        recon = self.call_multi_samples(directions_batched, cond_batched, latent)

        recon_shape_restore = tf.reshape(
            recon, (tf.shape(latent)[0], img_height, img_height * 2, 3)
        )

        return recon_shape_restore

    def eval_env_map_multi_rghs(self, latent, roughness_levels, img_height=128):
        import numpy as np

        rghs = np.linspace(0, 1, roughness_levels)

        ret = []

        for r in rghs:
            ret.append(self.eval_env_map(latent, r, img_height=img_height))

        return ret
