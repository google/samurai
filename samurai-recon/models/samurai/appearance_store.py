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


class AppearanceEmbeddingStore(tf.keras.layers.Embedding):
    def __init__(
        self, num_samples: int, latent_dim: int, color_fixed: bool, **kwargs
    ) -> None:
        super(AppearanceEmbeddingStore, self).__init__(
            num_samples,
            latent_dim,
            embeddings_initializer="random_normal",
            input_length=1,
            name="appearance_embeddings",
            **kwargs,
        )

        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.color_fixed = color_fixed

    def call(self, idxs):
        if self.color_fixed:
            return tf.zeros((idxs.shape[0], self.latent_dim))
        return super().call(idxs)


class DiffuseEmbeddingStore(tf.keras.layers.Embedding):
    def __init__(self, num_samples: int, latent_dim: int, **kwargs) -> None:
        super(DiffuseEmbeddingStore, self).__init__(
            num_samples,
            latent_dim,
            embeddings_initializer="zeros",
            input_length=1,
            name="diffuse_embeddings",
            **kwargs,
        )

        self.num_samples = num_samples
        self.latent_dim = latent_dim

    def call(self, idxs):
        return super().call(idxs)

    def add_regularization_loss(self, idxs):
        embds = self.call(idxs)
        return tf.reduce_sum(tf.square(embds))
