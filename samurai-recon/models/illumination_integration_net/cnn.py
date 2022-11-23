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


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from nn_utils.activations import from_hdr_activation, to_hdr_activation
from nn_utils.coord_conv import AddCoords


class CnnEncoder(tf.keras.Model):
    def __init__(
        self,
        output_units: int,
        activation="elu",
        embedding_function=None,
        img_height: int = 128,
        **kwargs,
    ):
        super(CnnEncoder, self).__init__(**kwargs)

        self.latent_dim = output_units
        activation = tf.keras.layers.Activation(activation)
        embedding_activation = tf.keras.layers.Activation(embedding_function)

        net = [
            tf.keras.layers.InputLayer((img_height, img_height * 2, 3)),
            tf.keras.layers.Lambda(lambda x: from_hdr_activation(x)),
            AddCoords(),
            tf.keras.layers.Conv2D(8, 3, activation=activation),
        ]
        downConvsNeeded = int(np.log2(img_height)) // 2
        for i in range(downConvsNeeded):
            half_output = self.latent_dim // 2
            curNf = int((half_output / downConvsNeeded) * (i + 1))
            net.append(AddCoords())
            net.append(
                tf.keras.layers.Conv2D(curNf, 4, strides=2, activation=activation)
            )
            net.append(AddCoords())
            net.append(tf.keras.layers.Conv2D(curNf, 3, activation=activation))

        net.append(tf.keras.layers.Flatten())
        net.append(
            tf.keras.layers.Dense(
                self.latent_dim,
                activation=embedding_activation,
            )
        )

        self.net = tf.keras.Sequential(net)

    @tf.function
    def call(self, img_data):
        embedding = self.net(img_data)

        return embedding


class CnnDecoder(tf.keras.Model):
    def __init__(
        self,
        input_units: int,
        output_units: int,
        activation="elu",
        output_activation=None,
        img_height: int = 128,
        **kwargs,
    ):
        super(CnnDecoder, self).__init__(**kwargs)

        activation = tf.keras.layers.Activation(activation)
        output_activation = tf.keras.layers.Activation(output_activation)

        initial_nf = input_units // 2
        net = [
            tf.keras.layers.InputLayer((input_units,)),
            tf.keras.layers.Reshape((1, 1, input_units)),
            tf.keras.layers.Conv2DTranspose(
                initial_nf,
                (1, 2),
                strides=1,
                padding="valid",
                activation=activation,
            ),  # Restore aspect ratio
        ]

        upConvsNeeded = int(
            np.log2(img_height)
        )  # Calculate how many convs are needed to go to 1x1
        for i in range(upConvsNeeded):
            cur_nf = initial_nf - int((initial_nf / (upConvsNeeded + 3)) * (i + 1))
            net.append(
                tf.keras.layers.Conv2DTranspose(
                    cur_nf,
                    4,
                    strides=2,
                    padding="same",
                    activation=activation,
                )
            )
            net.append(
                tf.keras.layers.Conv2D(
                    cur_nf,
                    3,
                    padding="same",
                    activation=activation,
                )
            )
        # Final one:
        net.append(
            tf.keras.layers.Conv2D(
                output_units,
                1,
                padding="same",
                activation=to_hdr_activation,
            )
        )

        self.net = tf.keras.Sequential(net)

    @tf.function
    def call(self, embedding):
        reconstruction = self.net(embedding)

        return reconstruction


class CnnDiscriminator(tf.keras.Model):
    def __init__(
        self,
        discriminator_units: int = 32,
        activation="relu",
        img_height: int = 128,
        **kwargs,
    ):
        super(CnnDiscriminator, self).__init__(**kwargs)

        activation = tf.keras.layers.Activation(activation)

        net = [
            tf.keras.layers.InputLayer((img_height, img_height * 2, 3)),
            tf.keras.layers.Lambda(lambda x: from_hdr_activation(x)),
            tf.keras.layers.Conv2D(8, 3, activation=activation),
        ]
        downConvsNeeded = int(np.log2(img_height)) - 3
        for i in range(downConvsNeeded):
            net.append(AddCoords())
            net.append(
                tf.keras.layers.Conv2D(
                    discriminator_units, 4, strides=2, activation=activation
                )
            )
            net.append(
                tf.keras.layers.Conv2D(
                    discriminator_units,
                    3,
                    activation=activation,
                )
            )

        net.append(tf.keras.layers.Flatten())
        net.append(tf.keras.layers.Dense(1))

        self.net = tf.keras.Sequential(net)

    @tf.function
    def call(self, img_data):
        logits = self.net(img_data)

        return logits
