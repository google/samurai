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


from collections import namedtuple
from typing import List, Tuple, Optional

import nn_utils.math_utils as math_utils
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_graphics as tfg
import tensorflow_graphics.image.pyramid as tfgp
from dataflow.samurai import InputTargets

from models.samurai.appearance_store import AppearanceEmbeddingStore
from models.samurai.camera_store import CameraStore, Ray, CameraParameter
from models.samurai.mask_confidence_store import MaskConfidenceStore
from nn_utils.gradient_image import get_gradient


class BatchData(tf.experimental.ExtensionType):
    rays: Ray
    pose: tf.Tensor
    image_idx: tf.Tensor
    image_coordinates: tf.Tensor
    appearance_embedding: tf.Tensor
    rgb_targets: tf.Tensor
    mask_targets: tf.Tensor
    mask_confidence: tf.Tensor
    gradient_targets: tf.Tensor

    def __validate__(self):
        tf.debugging.assert_shapes(
            [
                (
                    self.rays.origin,
                    ("B", "C", "S", 3),
                ),
                (
                    self.rays.direction,
                    ("B", "C", "S", 3),
                ),
                (self.pose, ("B", "C", 4, 4)),
                (self.image_idx, ("B", 1)),
                (self.image_coordinates, ("B", "C", "S", 2)),
                (self.appearance_embedding, ("B", None)),
                (self.rgb_targets, ("B", "C", "S", 3)),
                (self.mask_targets, ("B", "C", "S", 1)),
                (self.mask_confidence, ("B", "C", "S", 1)),
                (self.gradient_targets, ("B", "C", "S", 1)),
            ]
        )


def scale_inputs(max_dimension_size: int, dims, targets) -> List[tf.Tensor]:
    # assert all([targets[0].shape[:-1] == t.shape[:-1] for t in targets])

    height, width = dims[0], dims[1]
    max_dim = tf.maximum(height, width)
    factor = tf.cast(max_dimension_size / max_dim, tf.float32)

    new_height = tf.cast(tf.cast(height, tf.float32) * factor, tf.int32)
    new_width = tf.cast(tf.cast(width, tf.float32) * factor, tf.int32)
    new_dim = (new_height, new_width)

    resized = [
        tf.image.resize(t, new_dim, antialias=True)
        for t in [targets.rgb_target, targets.mask_target]
    ]

    return new_dim, resized


def get_interpolated_targets_from_coordinates(coordinates, *targets):
    ret = []

    tf.debugging.assert_shapes(
        [
            (coordinates, ("B", "N", 2)),
        ]
        + [(t, ("B", "H", "W", None)) for t in targets]
    )

    for t in targets:
        samples = tfa.image.interpolate_bilinear(t, coordinates, indexing="xy")
        ret.append(samples)

    return ret


def get_targets_from_coordinates(coordinates, *targets):
    tf.debugging.assert_shapes(
        [
            (coordinates, ("B", "C", "N", 2)),  # Coordinates is indexed xy instead ij
        ]
        + [(t, ("B", "C", "H", "W", None)) for t in targets]
    )
    result = [
        tf.gather_nd(
            t,
            coordinates,
            batch_dims=2,
        )
        for t in targets
    ]

    tf.debugging.assert_shapes(
        [
            (coordinates, ("B", "C", "N", 2)),
        ]
        + [(t, ("B", "C", "H", "W", None)) for t in targets]
        + [(i, ("B", "C", "N", None)) for i in result]
    )
    return result


def random_no_replace(logits, batch):
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 1e-4, 1)))
    _, indices = tf.nn.top_k(tf.math.log(logits) + z, batch)
    return indices


def build_weighted_random_index_select(
    batch_size, height: int, width: int, mask, gradients
):
    """This function creates a weighted sampling pattern. It samples based on the following
    importance:
        1. Edges inside the object
        2. General areas inside the object
        3. Edges in the background
        4. The background

    Arguments:
        batch_size {[type]} -- [description]
        height {int} -- [description]
        width {int} -- [description]
        mask {[type]} -- [description]
        rgb {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if len(mask.shape) == 4:
        mask = mask[0]  # Remove batch dim
    mask.shape.assert_has_rank(3)

    if len(gradients.shape) == 4:
        gradients = gradients[0]  # Remove batch dim

    coordsFull = tf.stack(
        tf.meshgrid(tf.range(height), tf.range(width), indexing="ij"), -1
    )
    coords = tf.reshape(coordsFull, [height * width, 2])

    mask_logit = tf.where(
        mask > 0.5, tf.ones_like(mask) * 1.0, tf.ones_like(mask) * 0.5
    )
    logits = mask_logit

    idxs = random_no_replace(tf.reshape(logits, (-1,)), batch_size)

    return tf.gather_nd(coords, idxs[:, tf.newaxis])


def build_random_index_select(batch_size: int, height: int, width: int):
    coordsFull = tf.stack(
        tf.meshgrid(tf.range(height), tf.range(width), indexing="ij"), -1
    )
    coords = tf.reshape(coordsFull, [height * width, 2])

    select_inds = tf.random.uniform(
        (batch_size,), minval=0, maxval=height * width, dtype=tf.int32
    )

    return tf.gather_nd(coords, select_inds[:, tf.newaxis])  # ij indexed list


def update_mask_with_confidence(targets, confidence_store, idxs):
    mask = targets[1]
    mask = confidence_store.apply_confidence_idx_to_masks(idxs, mask)
    targets[1] = mask
    return targets


def add_mask_confidence(targets, confidence_store, mask, idxs):
    confs = confidence_store.get_confidence_for_mask(idxs, mask)
    targets.append(confs)
    return targets


def add_gradient(targets):
    rgb = targets[0]
    mask = targets[1]

    gradient = get_gradient(rgb * mask)
    targets.append(gradient)  # build_mipmap(gradient, 4))

    return targets


def build_mipmap(img, levels):
    return tfgp.downsample(img, levels)


def interpolate_from_mipmap(mips, full_scale_coordinates):
    main_shape = tf.shape(mips[0])
    return_samples = []
    for m in mips:
        cur_shape = tf.shape(m)
        scaler = tf.cast(cur_shape[1] / main_shape[1], tf.float32)

        cur_scale_coordinates = full_scale_coordinates * scaler
        return_samples.append(
            tfa.image.interpolate_bilinear(m, cur_scale_coordinates, indexing="xy")
        )

    return return_samples


def full_image_batch_data(
    appearance_store: AppearanceEmbeddingStore,
    camera_store: CameraStore,
    confidence_store: MaskConfidenceStore,
    img_idx,
    max_dimension_size: int,
    num_cameras: int,
    dims,
    targets: InputTargets,
    stop_f_backprop: bool = False,
    overwrite_rays_cw2_coordinates: Optional[
        Tuple[Ray, CameraParameter, tf.Tensor]
    ] = None,
) -> Tuple[BatchData, int, int]:
    idxs = tf.reshape(tf.convert_to_tensor(img_idx, dtype=tf.int32), (-1, 1))
    appearance_embd = appearance_store(img_idx)

    # GET TARGETS
    new_dims, targets = scale_inputs(max_dimension_size, dims, targets)
    h, w = new_dims

    targets = update_mask_with_confidence(targets, confidence_store, idxs)
    targets = add_mask_confidence(targets, confidence_store, targets[1], idxs)
    targets = add_gradient(targets)

    # GET RAYS
    if overwrite_rays_cw2_coordinates is not None:
        rays, pose, coordinates = overwrite_rays_cw2_coordinates
    else:
        (rays, _, coordinates) = camera_store.generate_rays_for_pose(
            (h, w),
            img_idx,
            num_cameras,
            add_jitter=False,
            stop_f_backprop=stop_f_backprop,
        )
        pose, _, cam_idxs = camera_store(img_idx, num_cameras)

    # Flatten rays, coordinates, targets
    rays = Ray(
        tf.reshape(rays.origin, (*rays.origin.shape[:2], h * w, 3)),  # B, C, H*W, 3
        tf.reshape(rays.direction, (*rays.direction.shape[:2], h * w, 3)),
    )
    coordinates = math_utils.repeat(
        tf.reshape(coordinates, (coordinates.shape[0], 1, h * w, 2)), num_cameras, 1
    )

    targets = [
        math_utils.repeat(
            tf.reshape(t, (coordinates.shape[0], 1, h * w, t.shape[-1])), num_cameras, 1
        )
        for t in targets
    ]

    return (
        BatchData(
            rays,
            pose.c2w,
            idxs,
            coordinates,  # xy indexing
            appearance_embd,
            *targets,
        ),
        h,
        w,
    )


def build_train_batch(
    appearance_store: AppearanceEmbeddingStore,
    camera_store: CameraStore,
    confidence_store: MaskConfidenceStore,
    batch_size: int,
    img_idx,
    max_dimension_size: int,
    num_target_cameras: int,
    dims,
    targets: InputTargets,
    stop_f_backprop: bool = False,
) -> BatchData:
    idxs = tf.reshape(tf.convert_to_tensor(img_idx, dtype=tf.int32), (-1, 1))
    appearance_embd = appearance_store(img_idx)

    # GET TARGETS
    new_dims, targets = scale_inputs(max_dimension_size, dims, targets)
    h, w = new_dims

    targets = update_mask_with_confidence(targets, confidence_store, idxs)
    targets = add_mask_confidence(targets, confidence_store, targets[1], idxs)
    targets = add_gradient(targets)

    # GET RAYS
    (rays, jitter_coordinates, coordinates,) = camera_store.generate_rays_for_pose(
        (h, w), img_idx, num_target_cameras, stop_f_backprop=stop_f_backprop
    )  # Jitter_coordinates and coordinates are xy indexed

    pose, _, cam_idxs = camera_store(img_idx, num_target_cameras)

    # START SETTING UP THE SELECTION INDICES
    select_indices = build_weighted_random_index_select(
        batch_size // num_target_cameras, h, w, targets[1], targets[-1]  # // 5,
    )

    # Add cam and batch_dim to select indices
    select_indices = tf.reshape(select_indices, (1, 1, *select_indices.shape))
    select_indices = tf.tile(
        select_indices,
        (
            idxs.shape[0],  # Tile for each batch...
            num_target_cameras,  # .. and camera
            *[1 for _ in select_indices.shape[2:]],
        ),
    )  # Fill batch and cam dimensions as defined

    # SELECT SAMPLES
    jitter_coordinates = tf.tile(
        jitter_coordinates[:, None, ...],
        (
            1,
            num_target_cameras,  # add cam dim to jitter_coordinates
            *[1 for _ in jitter_coordinates.shape[1:]],
        ),  # All other stay the same
    )

    (
        rays_origin,
        rays_direction,
        select_jitter_coordinates,
    ) = get_targets_from_coordinates(
        select_indices, rays.origin, rays.direction, jitter_coordinates
    )

    # Interpolate requires a B, N, 2 shape
    # So flatten cam and sample dimension
    jitter_flat = tf.reshape(select_jitter_coordinates, (idxs.shape[0], -1, 2))
    # Perform the interpolation
    selected_targets = get_interpolated_targets_from_coordinates(jitter_flat, *targets)
    # selected_targets.append(interpolate_from_mipmap(targets[-1], jitter_flat))
    # And restore the dimensions
    selected_targets = [
        tf.reshape(
            t,
            tf.concat([tf.shape(select_jitter_coordinates)[:-1], tf.shape(t)[-1:]], 0),
        )
        for t in selected_targets
    ]

    return (
        BatchData(
            Ray(rays_origin, rays_direction),
            pose.c2w,
            idxs,
            select_jitter_coordinates,  # xy indexing
            appearance_embd,
            *selected_targets,
        ),
        h,
        w,
    )
