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


def to_8b(x):
    return tf.cast(255 * tf.clip_by_value(x, 0, 1), tf.uint8)


def horizontal_image_log(name, *xs):
    [x.shape.assert_has_rank(4) for x in xs]
    stacked = tf.concat(xs, 2)
    tf.summary.image(name, stacked)


def vertical_image_log(name, *xs):
    [x.shape.assert_has_rank(4) for x in xs]
    stacked = tf.concat(xs, 1)
    tf.summary.image(name, stacked)


def hdr_to_tb(name, data):
    tf.summary.image(
        name,
        tf.clip_by_value(  # Just for safety
            tf.math.pow(
                data / (tf.ones_like(data) + data),
                1.0 / 2.2,
            ),
            0,
            1,
        ),  # Reinhard tone mapping
    )
