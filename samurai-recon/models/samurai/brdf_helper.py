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
import nn_utils.math_utils as math_utils


# All values are sRGB values
GOLD = tf.constant([255, 226, 155], dtype=tf.float32) / 255
SILVER = tf.constant([252, 250, 245], dtype=tf.float32) / 255
ALUMINIUM = tf.constant([245, 246, 246], dtype=tf.float32) / 255
IRON = tf.constant([196, 199, 199], dtype=tf.float32) / 255
COPPER = tf.constant([250, 208, 192], dtype=tf.float32) / 255

NON_METAL = tf.constant([56, 56, 56], dtype=tf.float32) / 255

CLASSES = tf.stack([GOLD, SILVER, ALUMINIUM, IRON, COPPER, NON_METAL], 0)
NUM_CLASSES = CLASSES.shape[0]


def prediction_to_specular(prediction):
    logits = tf.nn.softmax(prediction)

    # Add the required dimensions
    classes_reshape = tf.reshape(
        CLASSES, [1 for _ in logits.shape[:-1]] + [NUM_CLASSES, 3]
    )
    logits_reshape = logits[..., None]

    weighted_classes = tf.reduce_sum(classes_reshape * logits_reshape, -2)

    return weighted_classes


def transform_specular_and_diffuse_prediction(specular_class_prediction, diffuse):
    specular_color = prediction_to_specular(specular_class_prediction)

    diffuse_safety_clip = ensure_diffuse_is_safe(diffuse)

    # Ensure both colors are smaller than 1
    diff_scaled = scale_diffuse(diffuse_safety_clip, specular_color)

    return specular_color, diff_scaled


def scale_diffuse(diffuse, specular):
    max_sum = tf.stop_gradient(
        tf.maximum(tf.reduce_max(specular + diffuse, -1, keepdims=True), 1.0) - 1.0
    )
    return tf.maximum(diffuse * (1 - max_sum), 0)


def ensure_diffuse_is_safe(diffuse):
    # Diffuse can only go down to 40 and up to 240 srgb
    return diffuse * (240 / 255)
