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
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from losses import multi_gpu_wrapper
from losses.illumination import MeanAbsoluteLogarithmicError
from models.illumination_integration_net.model import FiLMIlluminationNetwork
from models.illumination_integration_net.cnn import CnnDecoder, CnnEncoder
from utils.training_setup_utils import (
    StateRestoration,
    StateRestorationItem,
    get_num_gpus,
)


class IlluminationNetwork(K.Model):
    def __init__(self, args, img_height: int = 128, **kwargs):
        super(IlluminationNetwork, self).__init__(**kwargs)

        self.cnn_encoder = CnnEncoder(
            args.latent_dim, img_height=img_height, trainable=True
        )
        path = os.path.join("data/illumination/encoder.npy")
        self.cnn_encoder.set_weights(np.load(path, allow_pickle=True))

        self.cnn_decoder = CnnDecoder(
            args.latent_dim, 3, img_height=img_height, trainable=False
        )
        path = os.path.join("data/illumination/decoder.npy")
        self.cnn_decoder.set_weights(np.load(path, allow_pickle=True))

        self.illumination_network = FiLMIlluminationNetwork(args, **kwargs)

        # Setup the state restoration
        states = [
            StateRestorationItem("illumination_network", self.illumination_network),
            StateRestorationItem("cnn_encoder", self.cnn_encoder),
        ]
        self.state_restoration = StateRestoration(args, states)

        # Setup losses
        self.global_batch_size = args.batch_size * get_num_gpus()

        self.male = multi_gpu_wrapper(
            MeanAbsoluteLogarithmicError,
            self.global_batch_size,
        )

    def save(self, step):
        # Save weights for step
        self.state_restoration.save(step)

    def restore(self, step: Optional[int] = None) -> int:
        # Restore weights from step or if None the latest one
        return self.state_restoration.restore(step)

    @tf.function
    def train_step(
        self,
        environment_map: tf.Tensor,
        directions_random: tf.Tensor,
        roughness_random: tf.Tensor,
        targets_random: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        train_cnn: bool = False,
    ):
        tf.debugging.assert_shapes(
            [
                (environment_map, ("B", "H", "W", 3)),
                (directions_random, ("B", "R", 3)),
                (roughness_random, ("B", "R", 1)),
                (targets_random, ("B", "R", 3)),
            ]
        )

        with tf.GradientTape() as tape:
            z = self.cnn_encoder(environment_map)
            if not train_cnn:
                z = tf.stop_gradient(z)

            reconstruction_full = self.illumination_network.eval_env_map(
                z, 0, img_height=environment_map.shape[1]
            )

            reconstruction_random = self.illumination_network.call_multi_samples(
                directions_random, roughness_random, tf.stop_gradient(z)
            )

            loss_full = self.male(environment_map, reconstruction_full) / (
                environment_map.shape[1] * environment_map.shape[2]
            )
            loss_random = (
                self.male(targets_random, reconstruction_random)
                / directions_random.shape[1]
            )

            loss = loss_full + loss_random

        grad_vars = (
            self.illumination_network.trainable_variables
            + self.cnn_encoder.trainable_variables
        )

        gradients = tape.gradient(loss, grad_vars)

        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)

        optimizer.apply_gradients(zip(gradients, grad_vars))

        losses = {
            "loss": loss,
            "random_loss": loss_random,
            "full_loss": loss_full,
        }

        return losses

    @classmethod
    def add_args(self, parser):
        parser.add_argument("--latent_dim", type=int, default=128)

        parser.add_argument("--net_width", type=int, default=128)
        parser.add_argument("--net_depth", type=int, default=4)

        parser.add_argument("--cond_frequency", type=int, default=2)
        parser.add_argument("--cond_width", type=int, default=32)

        parser.add_argument("--mapping_width", type=int, default=128)
        parser.add_argument("--mapping_depth", type=int, default=5)

        parser.add_argument("--single_offset", action="store_true")

        return parser
