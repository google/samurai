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
from nn_utils.math_utils import convert3x4_4x4

# Adapted from pytorch3d


def rotation_6d_to_matrix(d6: tf.Tensor) -> tf.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    tf.debugging.assert_shapes([(d6, (..., 6))])

    def proj_u2a(u, a):
        r"""
        u: batch x 3
        a: batch x 3
        """
        inner_prod = tf.reduce_sum(u * a, -1, keepdims=True)
        norm2 = tf.reduce_sum(tf.square(u), -1, keepdims=True)
        norm2 = tf.maximum(norm2, 1e-8)
        factor = inner_prod / (norm2 + 1e-10)
        return factor * u

    x_raw, y_raw = d6[..., :3], d6[..., 3:]

    x = math_utils.normalize(x_raw)
    y = math_utils.normalize(y_raw - proj_u2a(x, y_raw))
    z = math_utils.cross(x, y)

    # b1 = math_utils.normalize(a1)
    # b2 = a2 - tf.reduce_sum(b1 * a2, -1, keepdims=True) * b1
    # b2 = math_utils.normalize(b2)
    # b3 = math_utils.cross(b1, b2)
    return tf.stack((x, y, z), -1)


def matrix_to_rotation_6d(matrix: tf.Tensor) -> tf.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    tf.debugging.assert_shapes([(matrix, (..., 3, 3))])
    rot = tf.concat([matrix[..., 0], matrix[..., 1]], axis=-1)
    # rot = matrix[..., :2, :]
    return tf.reshape(rot, (*matrix.shape[:-2], 6))


def build_4x4_matrix(rotation_6d, t):
    tf.debugging.assert_shapes(
        [
            (rotation_6d, ("N", 6)),
            (t, ("N", 3)),
        ]
    )

    R = rotation_6d_to_matrix(rotation_6d)
    c2w = tf.concat([R, t[..., None]], -1)
    c2w = convert3x4_4x4(c2w)

    return c2w


def extract_6d_t_from_4x4(c2w):
    tf.debugging.assert_shapes(
        [
            (c2w, ("N", 4, 4)),
        ]
    )

    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]

    return matrix_to_rotation_6d(R), t
