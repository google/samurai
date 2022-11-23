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
import tensorflow_io as tfio
import tensorflow_addons as tfa


def get_gradient(img):
    # L goes from 0-100 -> Scale it to 0-1
    lab_space_luminance = tfio.experimental.color.rgb_to_lab(img)[..., :1]
    lab_space_norm = tf.math.divide_no_nan(
        lab_space_luminance, tf.reduce_max(lab_space_luminance)
    )
    gradient = tf.reduce_sum(
        tf.abs(tf.stack(tf.image.image_gradients(lab_space_norm), -1)), -1
    )
    return gradient
