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


from typing import List, Union

import numpy as np
import tensorflow as tf

import nn_utils.math_utils as math_utils
from dataflow.illumination_integration.helper import getBilinearFromUv
from nn_utils.math_utils import shape_to_uv, uv_to_direction


@tf.function
def map_levels_to_samples(
    num_roughness_0: int,
    num_random_roughness: int,
    data_levels: List[tf.Tensor],
):
    # Setup uvs
    env_shape = data_levels[0].shape
    uvs = shape_to_uv(*env_shape[:-1])
    uvs_flat = tf.reshape(uvs, [-1, 2])

    total_directions_required = num_roughness_0 + num_random_roughness

    if uvs_flat.shape[0] < total_directions_required:
        repeats_required = tf.cast(
            tf.math.ceil(total_directions_required / uvs_flat.shape[0]), tf.int32
        )
        uvs_flat = math_utils.repeat(uvs_flat, repeats_required, 0)

    uvs_shuffle = tf.random.shuffle(uvs_flat)
    uvs_random = uvs_shuffle[:total_directions_required]

    jitter = tf.random.normal(uvs_random.shape, mean=0.0, stddev=0.3)
    uvs_random = uvs_random + jitter

    # Setup roughness
    roughness_random = tf.clip_by_value(
        tf.random.uniform(
            (num_random_roughness, 1), minval=1 / 255, maxval=1 + 1 / 255
        ),
        0,
        1,
    )

    r0_uvs = uvs_random[:num_roughness_0]
    rnd_uvs = uvs_random[num_roughness_0 : num_roughness_0 + num_random_roughness]

    # Get samples
    samples_random = random_uv_roughness_access(data_levels, rnd_uvs, roughness_random)

    # Always get r0 samples
    samples_r0 = random_uv_roughness_access(
        data_levels, r0_uvs, tf.zeros_like(r0_uvs[:, :1])
    )

    ret = (
        uv_to_direction(r0_uvs),
        samples_r0,
        uv_to_direction(rnd_uvs),
        roughness_random,
        samples_random,
    )

    return (
        data_levels,
        *ret,
    )


@tf.function
def full_map_samples(num_roughness_steps: int, data_levels: List[tf.Tensor]):
    # Setup random roughnesses and get all values
    full_uvs = tf.reshape(shape_to_uv(*data_levels[0].shape[:-1]), (-1, 2))

    roughness_steps = np.linspace(0.0, 1.0, num_roughness_steps, dtype=np.float32)[
        :, None
    ]  # Add a dimension

    # Store the roughness steps
    all_samples = tf.TensorArray(
        tf.float32, size=num_roughness_steps, clear_after_read=True
    )

    for i, r in enumerate(roughness_steps):  # The dimension is removed in the for loop
        r = math_utils.repeat(
            r[:, None], full_uvs.shape[0], 0
        )  # Add a batch dimension back

        samples = random_uv_roughness_access(data_levels, full_uvs, r)
        all_samples = all_samples.write(i, samples)  # Write the sample

    ret = (
        uv_to_direction(full_uvs),
        tf.convert_to_tensor(roughness_steps),
        all_samples.stack(),
    )

    return (
        data_levels,
        *ret,
    )


@tf.function
def random_uv_roughness_access(data_levels, uvs, roughness):
    tf.debugging.assert_shapes(
        [
            (uvs, ("S", 2)),
            (roughness, ("S", 1)),
        ]
        + [(d, ("H%d" % i, "W%d" % i, 3)) for i, d in enumerate(data_levels)]
    )
    # data_levels: List[H, W, 3]
    # uvs: [S, 2]
    # Roughness: [S, 1]

    # Result: [S, 3]

    smpl_list = []
    for d in data_levels:
        samples_level = getBilinearFromUv(d[None, ...], uvs[None, ...])[0]
        smpl_list.append(samples_level)

    level_samples_batched = tf.stack(smpl_list, 0)  # M, S, 3

    return interpolate_roughness_levels(level_samples_batched, roughness)


@tf.function
def interpolate_roughness_levels(samples, roughness):
    tf.debugging.assert_shapes(
        [
            (samples, ("M", "S", 3)),
            (roughness, ("S", 1)),
        ]
    )

    # Setup the roughness interpolation
    roughness_mip_index = roughness[:, 0] * (samples.shape[0] - 1)
    # S
    lower_mip_index = tf.cast(tf.math.floor(roughness_mip_index), tf.int32)
    upper_mip_index = tf.cast(tf.math.ceil(roughness_mip_index), tf.int32)

    # Fetch the lower and upper roughness levels
    rgh_low = tf.gather(
        tf.transpose(samples, [1, 0, 2]), lower_mip_index[..., None], batch_dims=1
    )[:, 0]
    rgh_hgh = tf.gather(
        tf.transpose(samples, [1, 0, 2]), upper_mip_index[..., None], batch_dims=1
    )[:, 0]

    tf.debugging.assert_shapes(
        [
            (samples, ("M", "S", 3)),
            (roughness, ("S", 1)),
            (rgh_low, ("S", 3)),
            (rgh_hgh, ("S", 3)),
        ]
    )

    # Start interpolation
    fraction_index = roughness_mip_index - tf.cast(lower_mip_index, tf.float32)
    fraction_index = tf.reshape(fraction_index, roughness.shape)

    samples_random = rgh_low * fraction_index + rgh_hgh * (1 - fraction_index)
    return samples_random


@tf.function
def blend_two_maps(*batch_2_data):
    ret = []
    for b in batch_2_data:
        b0 = b[0]
        b1 = b[1]
        alpha = tf.random.uniform((1,))
        ret.append(alpha * b0 + (1 - alpha) * b1)
    return ret


@tf.function
def specify_mip_levels_to_fetch(
    dataset: List[Union[List[np.ndarray], np.ndarray]], idxs: List[int]
):
    random_sampled_targets = dataset[1:]
    ret = []
    for idx in idxs:
        ret.append(dataset[0][idx])

    ret.extend(random_sampled_targets)

    return (*ret,)


def random_sample_dataflow(
    dataset: List[np.ndarray],
    samples_roughness_0: int,
    samples_random_roughness: int,
    batch_size: int,
    with_blend: bool = False,
    full_l0: bool = False,
    shuffle: bool = True,
):
    dataset_len = len(dataset[0])
    ds = tf.data.Dataset.from_tensor_slices((*dataset,))
    if shuffle:
        ds = ds.shuffle(dataset_len, reshuffle_each_iteration=True)

    if with_blend:
        ds = ds.batch(2, drop_remainder=True)
        ds = ds.map(blend_two_maps)
        ds = ds.repeat(2)

    if full_l0:
        ds = ds.map(
            lambda *x: full_map_samples(5, x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda *x: map_levels_to_samples(
                samples_roughness_0, samples_random_roughness, x
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.map(lambda *x: specify_mip_levels_to_fetch(x, [0]))

    if batch_size > 0:
        ds = ds.batch(batch_size)
    ds = ds.prefetch(5)

    return ds
