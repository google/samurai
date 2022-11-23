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
from nn_utils.activations import padded_sigmoid


def slightly_padded_tanh(x):
    return padded_sigmoid(x, 0.001) * 2 - 1


class IlluminationEmbeddingStore(tf.keras.layers.Embedding):
    def __init__(
        self, num_samples: int, latent_dim: int, latent_mean, latent_std, **kwargs
    ) -> None:
        super(IlluminationEmbeddingStore, self).__init__(
            num_samples,
            latent_dim,
            embeddings_initializer="random_normal",
            input_length=1,
            name="illumination_embeddings",
            **kwargs,
        )

        self.num_samples = num_samples
        self.latent_dim = latent_dim

        self.latent_mean = tf.convert_to_tensor(latent_mean, tf.float32)
        self.latent_std = tf.convert_to_tensor(latent_std, tf.float32)

        self.illumination_scaler = tf.Variable(
            initial_value=tf.ones((num_samples, 1)),
            name="illumination_scales",
            trainable=True,
        )

    def convert_noise_to_latent(self, noise):
        tf.debugging.assert_shapes(
            [
                (noise, ("N", self.latent_dim)),
            ]
        )
        return (
            slightly_padded_tanh(noise) * (self.latent_std[None, :] * 3)
        ) + self.latent_mean[None, :]

    def call(self, idxs):
        if self.num_samples == 1:
            return (
                self.convert_noise_to_latent(super().call(tf.convert_to_tensor([0]))),
                self.illumination_scaler[:1],
            )
        else:
            idxs_lookup = tf.reshape(
                tf.convert_to_tensor(idxs, dtype=tf.int32), (-1, 1)
            )
            scalers = tf.gather_nd(self.illumination_scaler, idxs_lookup)
            return self.convert_noise_to_latent(super().call(idxs)), scalers
