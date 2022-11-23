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


import functools
import os
import math

import nn_utils.math_utils as math_utils
import numpy as np
import tensorflow as tf
from utils.training_setup_utils import get_num_replicas

from dataflow.samurai.load_nerd_blender_dataset import NerdDataset
from dataflow.samurai.load_samurai_dataset import SamuraiDataset


class InputTargets(tf.experimental.ExtensionType):
    rgb_target: tf.Tensor
    mask_target: tf.Tensor

    def __validate__(self):
        self.rgb_target.shape.assert_same_rank(self.mask_target.shape)
        assert self.rgb_target.dtype.is_floating, "rgb_target.dtype must be float"
        assert self.mask_target.dtype.is_floating, "mask_target.dtype must be float"


def resize_to_fit(x, max_dim):
    dims = tf.shape(x)
    h = dims[1]
    w = dims[2]
    max_img_dim = tf.maximum(h, w)

    factor = tf.cast(tf.divide_no_nan(max_dim, max_img_dim), tf.float32)
    new_h, new_w = (
        tf.cast(tf.cast(h, tf.float32) * factor, tf.int32),
        tf.cast(tf.cast(w, tf.float32) * factor, tf.int32),
    )
    return tf.image.resize(x, (new_h, new_w))


def load_image(filename, max_dim, channels=3):
    raw = tf.io.read_file(filename)
    image = tf.cast(
        tf.io.decode_image(raw, expand_animations=False, channels=channels) / 255,
        dtype=tf.float32,
    )

    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]
    max_img_dim = tf.maximum(h, w)

    factor = tf.cast(max_dim / max_img_dim, tf.float32)
    new_h, new_w = (
        tf.cast(tf.cast(h, tf.float32) * factor, tf.int32),
        tf.cast(tf.cast(w, tf.float32) * factor, tf.int32),
    )
    image_resize = tf.image.resize(tf.stop_gradient(image), (new_h, new_w))
    return image_resize


def idx_path_to_dp(args, channels, idx, *img_paths):
    return (
        tf.cast(idx, tf.int32),
        *[
            load_image(p, args.max_resolution_dimension, c)
            for p, c in zip(img_paths, channels)
        ],
    )


def replica_duplication(args, *dp):
    replicas = tf.maximum(get_num_replicas(args), 1)
    return (*[math_utils.repeat(d, replicas, 0) for d in dp],)


def wrap_in_namedtuple(*dp):
    return (dp[0], InputTargets(*dp[1:]))


def load_secondary_image(args, dataset, idx, max_res):
    idx_paths = dataset.get_image_path_from_indices(idx)
    idx_paths = idx_paths[1:]
    return (
        tf.cast(idx, tf.int32),
        InputTargets(
            *[
                load_image(p[0], max_res, c)
                for p, c in zip(idx_paths, dataset.channels)
            ],
        ),
    )


def dataflow(args, dataset, indices, is_train):
    train_samples = indices.shape[0]
    repeats = max(int(math.ceil(args.steps_per_epoch / train_samples)), 1)

    channels = dataset.channels
    file_flow = tf.data.Dataset.from_tensor_slices(
        dataset.get_image_path_from_indices(indices)
    )
    if is_train:
        file_flow = (
            file_flow.shuffle(min(50, train_samples))
            .repeat(repeats)
            .take(args.steps_per_epoch)
        )
    else:
        file_flow = file_flow.repeat(20)

    main_flow = file_flow.map(
        functools.partial(
            idx_path_to_dp,
            args,
            channels,
        ),
        num_parallel_calls=min(os.cpu_count(), 8),
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    return (
        main_flow.batch(1)
        .map(functools.partial(replica_duplication, args))
        .map(wrap_in_namedtuple)
        .prefetch(tf.data.AUTOTUNE)
        .with_options(options)
    )


def create_dataflow(args):
    if args.dataset == "samurai":
        mask_dir = "mask"
        dataset = SamuraiDataset(
            args.datadir, [("image", ".jpg", 3), (mask_dir, ".jpg", 1)]
        )
    elif args.dataset == "nerd":
        dataset = NerdDataset(args.datadir, args.load_gt_poses)

    image_shapes = dataset.get_image_shapes(args.max_resolution_dimension)

    # TODO make overwriteable by dataset type
    i_test = np.arange(len(image_shapes))[:: args.test_holdout]
    i_val = i_test
    i_train = np.array([i for i in np.arange(len(image_shapes)) if (i not in i_test)])

    train_df, val_df, test_df = (
        dataflow(args, dataset, i_train, True),
        dataflow(args, dataset, i_val, False),
        dataflow(args, dataset, i_test, False),
    )

    image_request_function = functools.partial(load_secondary_image, args, dataset)

    has_poses_and_focal = args.dataset in ["nerd"]
    if has_poses_and_focal:
        return (
            image_shapes,
            dataset.get_poses(),
            dataset.get_focal_length(image_shapes),
            dataset.get_directions(),
            image_request_function,
            train_df,
            val_df,
            test_df,
        )
    else:
        return (
            image_shapes,
            None,
            None,
            dataset.get_directions(),
            image_request_function,
            train_df,
            val_df,
            test_df,
        )
