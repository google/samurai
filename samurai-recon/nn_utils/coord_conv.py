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


class AddCoords(tf.keras.layers.Layer):
    """Add coords to a tensor"""

    def __init__(
        self,
    ):
        super(AddCoords, self).__init__()

    def call(self, input_tensor):
        """
        input_tensor: (batch, 1, 1, c), or (batch, x_dim, y_dim, c)
        In the first case, first tile the input_tensor to be (batch, x_dim, y_dim, c)
        In the second case, skiptile, just concat
        """
        batch_size_tensor = tf.shape(input_tensor)[0]
        x_dim = tf.shape(input_tensor)[1]
        y_dim = tf.shape(input_tensor)[2]
        xx_ones = tf.ones([batch_size_tensor, x_dim], dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
        xx_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [batch_size_tensor, 1])
        xx_range = tf.expand_dims(xx_range, 1)
        xx_channel = tf.matmul(xx_ones, xx_range)
        xx_channel = tf.expand_dims(xx_channel, -1)
        yy_ones = tf.ones([batch_size_tensor, y_dim], dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
        yy_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [batch_size_tensor, 1])
        yy_range = tf.expand_dims(yy_range, -1)
        yy_channel = tf.matmul(yy_range, yy_ones)
        yy_channel = tf.expand_dims(yy_channel, -1)

        x_dim = tf.cast(x_dim, tf.float32)
        y_dim = tf.cast(y_dim, tf.float32)

        xx_channel = tf.cast(xx_channel, tf.float32) / (x_dim - 1)
        yy_channel = tf.cast(yy_channel, tf.float32) / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)

        output_tensor = ret

        return output_tensor
