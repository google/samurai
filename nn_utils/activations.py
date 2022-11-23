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


def to_hdr_activation(x):
    return tf.exp(tf.nn.relu(x)) - 1


def from_hdr_activation(x):
    return tf.math.log(1 + tf.nn.relu(x))


def softplus_1m(x):
    return tf.math.softplus(x - 1)


def squareplus(x):
    return 0.5 * (x + tf.sqrt(tf.square(x) + 4))


def squareplus_offset(x, offset=-1):
    return squareplus(x + offset)


def padded_sigmoid(x, padding: float, upper_padding=True, lower_padding=True):
    # If padding is positive it can have values from 0-padding < x < 1+padding,
    # if negative 0+padding < x 1-padding
    x = tf.nn.sigmoid(x)

    mult = upper_padding + lower_padding  # Evil cast to int
    x = x * (1 + mult * padding)
    if lower_padding:
        x = x - padding
    return x
