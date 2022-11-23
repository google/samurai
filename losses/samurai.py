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
import tensorflow_addons as tfa
import nn_utils.math_utils as math_utils


def batch_cam_chi_squared_error(num_gpus, true, pred):
    se = tf.square(true - pred)
    denom = tf.math.square(true + pred)

    divide = true.shape[-2] * num_gpus
    return tf.reduce_sum(tf.math.divide_no_nan(se, denom), axis=(-2, -1)) / divide


def chi_squared_error(gpus=None):
    @tf.function
    def run(true, pred):
        se = tf.math.square(true - pred)
        denom = tf.math.square(true + pred)
        loss = tf.math.divide_no_nan(se, denom)

        if gpus is not None:
            loss_mean = tf.reduce_mean(loss)
            term_mean = loss_mean / gpus
            return term_mean

        return tf.reduce_mean(loss)

    return run


bce_tf = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, reduction=tf.keras.losses.Reduction.NONE
)


def batch_cam_bce(num_gpus, true, pred, confidence=None):
    if confidence is None:
        confidence = tf.ones_like(true)

    loss = bce_tf(true, pred)[..., None] * confidence

    shape = tf.shape(true)

    divide = tf.cast(shape[-2], tf.float32) * num_gpus
    return tf.reduce_sum(loss, axis=(-2, -1)) / divide


def batch_cam_mask_loss(num_gpus, true, pred):
    # The background loss punishes all values directly
    loss_background = tf.abs(true - pred)

    # In the foreground we do not know where information should be placed
    loss_foreground = tf.zeros_like(loss_background)

    bg_loss = loss_background * (1 - true)
    fg_loss = loss_foreground * true

    shape = tf.shape(true)
    divide = tf.cast(shape[-2], tf.float32) * num_gpus

    return (
        tf.reduce_sum(
            tf.where(
                tf.broadcast_to(tf.less(true, 0.1), tf.shape(bg_loss)), bg_loss, fg_loss
            ),
            (-2, -1),
        )
        / divide
    )


def batch_cam_mae(num_gpus, true, pred):
    ae = tf.math.abs(true - pred)
    divide = true.shape[-2] * num_gpus
    return tf.reduce_sum(ae, axis=(-2, -1)) / divide


def batch_cam_mse(num_gpus, true, pred):
    se = tf.math.square(true - pred)
    divide = true.shape[-2] * num_gpus
    return tf.reduce_sum(se, axis=(-2, -1)) / divide


def batch_cam_chabonnier(num_gpus, true, pred):
    se = math_utils.safe_sqrt(tf.math.square(true - pred) + tf.square(1e-3))
    divide = true.shape[-2] * num_gpus
    return tf.reduce_sum(se, axis=(-2, -1)) / divide


def loss_distortion(num_gpus, weights, ray_samples, near, far):
    """From Mip-NeRF 360"""
    normalized_samples = tf.math.divide_no_nan(
        ray_samples - near[..., None], far[..., None] - near[..., None]
    )

    samples_1 = normalized_samples[..., :-1]  # s_i
    samples_2 = normalized_samples[..., 1:]  # s_i+1
    weights_1 = weights[..., :-1]

    samples_mid = 0.5 * (samples_1 + samples_2)
    samples_mid_matrix = tf.abs(samples_mid[..., :, None] - samples_mid[..., None, :])
    # The last sample is missing

    loss_weighted_distance = tf.reduce_sum(
        weights_1 * tf.reduce_sum(weights_1[..., None, :] * samples_mid_matrix, -1),
        -1,
    )
    loss_weighted_size = (
        tf.reduce_sum(
            tf.square(weights_1) * (samples_2 - samples_1),
            -1,
        )
        / 3
    )

    return tf.reduce_sum(loss_weighted_distance + loss_weighted_size) / num_gpus


def batch_cam_dice(num_gpus, gamma, true, pred, confidence=None):
    dims = (-2, -1)

    if confidence is None:
        confidence = tf.ones_like(true)

    loss = tf.pow(tf.abs(true - pred), gamma) * confidence

    denom = tf.reduce_sum(tf.square(pred) * confidence, dims) + tf.reduce_sum(
        tf.square(true) * confidence, dims
    )

    calc_loss = tf.math.divide_no_nan(tf.reduce_sum(loss, dims), denom)

    return calc_loss / num_gpus


def normal_direction_loss(num_gpus, normals, ray_direction, weights):
    # Ray directions go towards the surface.
    # If the normal is in the same hemisphere as the ray direction,
    #  we hit the surface from the back
    normal_dot_ray_direction = tf.maximum(
        math_utils.dot(
            math_utils.normalize(normals),
            math_utils.normalize(ray_direction[..., None, :]),
        ),
        0,
    )

    # Therefore, we select all normals before the first full surface intersection
    # And then only gently punish normals which face towards the object
    normal_dot_ray_direction_weighted = normal_dot_ray_direction * weights[..., None]

    return tf.reduce_sum(normal_dot_ray_direction_weighted) / num_gpus


def normal_consistency_loss(num_gpus, normals, gradient_normals, weights):
    diff = tf.square(gradient_normals - normals)
    diff_weighted = diff * weights[..., None]

    return tf.reduce_sum(diff_weighted) / num_gpus
