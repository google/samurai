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


import math
from typing import Tuple

import tensorflow as tf
import numpy as np
from nn_utils.math_utils import magnitude, repeat, normalize, dot, l2Norm
from nn_utils.camera.camera_utils import build_look_at_matrix, c2w_to_lookat, r_t_to_c2w
from nn_utils.camera.rotation_sixd import (
    build_4x4_matrix,
    matrix_to_rotation_6d,
)

# Useful tuples


class CameraParameter(tf.experimental.ExtensionType):
    c2w: tf.Tensor
    focal: tf.Tensor

    def __validate__(self):
        assert self.c2w.dtype.is_floating, "c2w.dtype must be float"
        assert self.focal.dtype.is_floating, "focal.dtype must be float"


class Ray(tf.experimental.ExtensionType):
    origin: tf.Tensor
    direction: tf.Tensor

    def __validate__(self):
        self.origin.shape.assert_is_compatible_with(self.direction.shape)
        assert self.origin.dtype.is_floating, "origin.dtype must be float"
        assert self.direction.dtype.is_floating, "direction.dtype must be float"


class CameraStore(tf.keras.layers.Layer):
    def __init__(
        self,
        num_images: int,
        image_dimensions,
        canonical_pose_idx: int,
        weight_update_lr: float,
        weight_update_momentum: float,
        init_directions=None,
        init_c2w=None,
        init_focal=None,
        num_cameras_per_image: int = 8,
        object_height: float = 1.0,
        angular_fov: float = 53.13,
        learn_r: bool = True,
        learn_t: bool = True,
        learn_f: bool = True,
        fy_only: bool = True,
        squish_focal_range: bool = True,
        use_look_at_representation: bool = False,
        offset_learning: bool = True,
        use_initializations: bool = True,
        **kwargs
    ):
        super(CameraStore, self).__init__(**kwargs)

        self.weight_update_lr = weight_update_lr
        self.weight_update_momentum = weight_update_momentum

        # Image 0 always fixes the optimization
        self.canonical_pose_idx = canonical_pose_idx

        # Calculate the distance required so the targets fits in view
        self.distance = (object_height / 2) / (math.tan(math.radians(angular_fov) / 2))
        if isinstance(image_dimensions, list):
            image_dimensions = np.asarray(image_dimensions)
        self.image_dimensions = image_dimensions
        height, width = image_dimensions[:, 0], image_dimensions[:, 1]

        self.num_cameras_per_image = num_cameras_per_image if use_initializations else 6
        self.num_images = num_images
        total_poses = self.num_images * self.num_cameras_per_image

        self.fy_only = fy_only
        self.use_look_at_representation = use_look_at_representation
        self.offset_learning = offset_learning

        self.per_cam_weights = tf.Variable(
            initial_value=(
                tf.ones((num_images, self.num_cameras_per_image))
                / self.num_cameras_per_image
            ),
            name="camera_weights",
            trainable=False,
        )  # Every camera contributes equally in the beginning
        self.momentum_velocities = tf.Variable(
            initial_value=tf.zeros((num_images, self.num_cameras_per_image)),
            name="momentum_velocities",
            trainable=False,
        )  # Every camera contributes equally in the beginning

        # Setup the focal lengths
        self.setup_focal_length(
            fy_only,
            width,
            angular_fov,
            height,
            self.distance,
            self.num_cameras_per_image,
            squish_focal_range,
            num_images,
            learn_f,
            offset_learning,
            init_focal,
        )

        # Setup the poses
        self.setup_poses(
            total_poses,
            self.distance,
            num_images,
            self.num_cameras_per_image,
            learn_t,
            learn_r,
            offset_learning,
            init_directions if use_initializations else None,
            init_c2w if use_initializations else None,
        )

    def setup_poses(
        self,
        total_poses,
        distance,
        num_images,
        num_cameras_per_image,
        learn_t,
        learn_r,
        offset_learning,
        init_directions=None,
        init_c2w=None,
    ):
        # First check if we already have poses
        if init_c2w is not None:
            r, t = init_c2w[..., :3, :3], init_c2w[..., :3, 3]
            self.distance = tf.reduce_mean(magnitude(t))
        elif init_directions is not None:
            t = init_directions * distance
            t = tf.expand_dims(t, 1)

            # If we use camera multiplex add a bit of noise
            if num_cameras_per_image > 1:
                t = repeat(t, num_cameras_per_image, 1)
                t_noise = tf.random.normal(
                    (num_images, num_cameras_per_image, 3), stddev=distance / 5
                )  # Add random noise
                # Only n-1 should have random noise applied. The first index is
                # not perturbed
                t_noise = tf.concat(
                    [tf.zeros_like(t_noise[:, :1, :]) + 1e-4, t_noise[:, 1:, :]], 1
                )

                t = normalize(t + t_noise) * distance

            random_up = tf.random.normal(t[..., :1].shape, mean=0.0, stddev=np.pi / 3)
            random_up = tf.concat(
                (tf.zeros_like(random_up[:, :1, :]), random_up[:, 1:, :]), 1
            )
            random_up = tf.reshape(random_up, (-1, 1))
            t = tf.reshape(t, (-1, 3))

            r = build_look_at_matrix(t, tf.zeros_like(t), up_rotation=random_up)
        else:
            raise Exception("No initialization provided")

        if self.use_look_at_representation:
            c2w = r_t_to_c2w(r, t)
            eye_pos, center = c2w_to_lookat(c2w)

            eye_pos = tf.reshape(eye_pos, (num_images, num_cameras_per_image, 3))
            center = tf.reshape(center, (num_images, num_cameras_per_image, 3))

            if offset_learning:
                learn_initial_eye = False
                learn_initial_center = False
                learn_offset_eye = learn_t
                learn_offset_center = learn_r
            else:
                learn_initial_eye = learn_t
                learn_initial_center = learn_r
                learn_offset_eye = False
                learn_offset_center = False

            self.eye_initial = tf.Variable(
                initial_value=eye_pos,
                name="camera_eye_initial",
                dtype=tf.float32,
                trainable=learn_initial_eye,
            )
            self.eye_offset = tf.Variable(
                initial_value=tf.zeros_like(eye_pos),
                name="camera_eye",
                dtype=tf.float32,
                trainable=learn_offset_eye,
            )

            self.up_rotation_initial = tf.Variable(
                initial_value=tf.zeros_like(eye_pos[..., :1]),
                name="camera_up_rotation_initial",
                dtype=tf.float32,
                trainable=learn_initial_center,
            )
            self.up_rotation_offset = tf.Variable(
                initial_value=tf.zeros_like(eye_pos[..., :1]),
                name="camera_up_rotation",
                dtype=tf.float32,
                trainable=learn_offset_center,
            )

            self.center_initial = tf.Variable(
                initial_value=center,
                dtype=tf.float32,
                name="camera_center_initial",
                trainable=learn_initial_center,
            )
            self.center_offset = tf.Variable(
                initial_value=tf.zeros_like(center),
                dtype=tf.float32,
                name="camera_center",
                trainable=learn_offset_center,
            )
        else:
            # Then split based on the rotation representation and compress the rotation matrix
            rot = tf.reshape(
                matrix_to_rotation_6d(r), (num_images, num_cameras_per_image, 6)
            )
            t = tf.reshape(t, (num_images, num_cameras_per_image, 3))

            # Decide which variable we train
            if offset_learning:
                learn_initial_r = False
                learn_initial_t = False
                learn_offset_r = learn_r
                learn_offset_t = learn_t
            else:
                learn_initial_r = learn_r
                learn_initial_t = learn_t
                learn_offset_r = False
                learn_offset_t = False

            # And store variables
            self.t_initial = tf.Variable(
                initial_value=t,
                name="camera_t_initial",
                dtype=tf.float32,
                trainable=learn_initial_t,
            )
            self.t_offset = tf.Variable(
                initial_value=tf.zeros_like(t),
                name="camera_t",
                dtype=tf.float32,
                trainable=learn_offset_t,
            )

            self.r_initial = tf.Variable(
                initial_value=rot,
                dtype=tf.float32,
                name="camera_r_initial",
                trainable=learn_initial_r,
            )
            self.r_offset = tf.Variable(
                initial_value=tf.zeros_like(rot),
                dtype=tf.float32,
                name="camera_r",
                trainable=learn_offset_r,
            )

    def setup_focal_length(
        self,
        fy_only,
        width,
        angular_fov,
        height,
        distance,
        num_cameras_per_image,
        squish_focal_range,
        num_images,
        learn_f,
        offset_learning,
        init_focal=None,
    ):
        def dim_to_f(dim, afov: float):
            fov = 2 * distance * math.tan(math.radians(afov) / 2)
            return (tf.cast(dim, tf.float32) * distance) / fov

        # init_focal can be either a single float, a tuple of two floats (fx, fy)
        # Or a numpy array (num_images, 1|2)
        if init_focal is not None:
            if isinstance(init_focal, float):
                init_f = init_focal * np.ones_like(height).reshape((-1,))[:, None]
            elif isinstance(init_focal, tuple):
                assert len(init_focal) == 2

                init_f = np.stack(init_f).reshape((1, 2))
                init_f = init_f * np.ones_like(height).reshape((-1,))[:, None]
            elif isinstance(init_focal, np.ndarray):
                assert len(init_focal.shape) == 2
                assert init_focal.shape[0] == num_images
                assert init_focal.shape[1] in [1, 2]
                assert init_focal.dtype == np.float32

                init_f = init_focal
            else:
                raise ValueError(
                    "The focal lengths init can either be float, a tuple of 2 floats "
                    "or numpy array containing the focal lengths per image."
                )
        else:
            self.fy_only = fy_only
            init_f = dim_to_f(height, angular_fov)[:, None]
            if not fy_only:
                init_f = np.stack([init_f, init_f], -1)

        focal_lengths = repeat(
            tf.convert_to_tensor(init_f, dtype=tf.float32)[:, None, :],
            num_cameras_per_image,
            1,
        )  # N, C, 1|2

        self.widths = tf.convert_to_tensor(width, dtype=tf.int32)
        self.heights = tf.convert_to_tensor(height, dtype=tf.int32)

        self.squish_focal_range = squish_focal_range
        if squish_focal_range:
            # We first divide by the image dimensions
            if fy_only:
                focal_lengths = focal_lengths / tf.cast(
                    tf.reshape(self.heights, (-1, 1, 1)), tf.float32
                )
            else:
                focal_lengths = tf.stack(
                    [
                        focal_lengths[..., 0]
                        / tf.cast(tf.reshape(self.widths, (-1, 1)), tf.float32),
                        focal_lengths[..., 1]
                        / tf.cast(tf.reshape(self.heights, (-1, 1)), tf.float32),
                    ],
                    -1,
                )

            # Additionally we store the sqrt of the range
            focal_lengths = tf.sqrt(focal_lengths)

        # Decide which variable we train
        if offset_learning:
            learn_initial_f = False
            learn_offset_f = learn_f
        else:
            learn_initial_f = learn_f
            learn_offset_f = False

        self.focal_lengths_initial = tf.Variable(
            initial_value=focal_lengths,
            dtype=tf.float32,
            name="camera_focal_lengths_initial",
            trainable=learn_initial_f,
        )
        self.focal_lengths_offset = tf.Variable(
            initial_value=tf.zeros_like(focal_lengths),
            name="camera_focal_lengths",
            dtype=tf.float32,
            trainable=learn_offset_f,
        )

    def get_regularization_loss(self, idxs):
        idxs = tf.reshape(tf.convert_to_tensor(idxs), (-1, 1))

        if self.use_look_at_representation:
            eye_offset = tf.reduce_sum(tf.abs(tf.gather_nd(self.eye_offset, idxs)))
            center_offset = tf.reduce_sum(
                tf.abs(tf.gather_nd(self.center_offset, idxs))
            )
            up_offset = tf.reduce_sum(
                tf.abs(tf.gather_nd(self.up_rotation_offset, idxs))
            )
            pos_offset = eye_offset + center_offset * 10 + up_offset * np.pi
        else:
            pos_offset = tf.reduce_sum(tf.abs(tf.gather_nd(self.t_offset, idxs)))

        f_offset = (
            tf.reduce_sum(tf.abs(tf.gather_nd(self.focal_lengths_offset, idxs))) * 10
        )

        return pos_offset + f_offset

    def get_height_width(self, idxs) -> Tuple[tf.Tensor, tf.Tensor]:
        idxs = tf.reshape(tf.convert_to_tensor(idxs), (-1,))

        height = tf.gather_nd(self.heights, idxs[:, None])
        width = tf.gather_nd(self.widths, idxs[:, None])

        return height, width

    def get_per_weight_camera_losses(self, losses, scaler=20):
        tf.debugging.assert_shapes([(losses, ("B", "C"))])
        return tf.nn.softmax(scaler * (-losses), axis=-1)

    def apply_weights_to_losses(self, losses, idx):
        losses_ret = {}

        idxs = tf.reshape(tf.convert_to_tensor(idx, dtype=tf.int32), (-1, 1))

        weights = None
        for k, v in losses.items():
            if weights is None:
                _, weights, _ = self(idxs, v.shape[-1])
                # Just accept the weights as they are
                weights = tf.stop_gradient(weights)

            tf.debugging.assert_shapes(
                [
                    (v, ("B", "C")),
                    (weights, ("B", "C")),
                ]
            )
            if v.shape[-1] == 1:
                losses_ret[k] = tf.reduce_mean(v[:, 0])
            else:
                losses_ret[k] = tf.reduce_mean(tf.reduce_sum(v * weights, axis=1))

        return losses_ret

    def update_weights(self, idx, new_weights):
        idxs = tf.reshape(tf.convert_to_tensor(idx, dtype=tf.int32), (-1, 1))

        tf.debugging.assert_shapes([(idxs, ("B", 1)), (new_weights, ("B", "C"))])

        _, current_weights, camera_idxs = self(idxs, new_weights.shape[-1])

        # Gradient in a sense
        gradient = new_weights - current_weights
        lr_gradient = self.weight_update_lr * gradient

        prev_velocity = tf.gather_nd(self.momentum_velocities, idxs)
        prev_velocity = tf.gather_nd(
            prev_velocity, camera_idxs[..., None], batch_dims=1
        )

        tf.debugging.assert_shapes(
            [
                (gradient, ("B", "K")),
                (prev_velocity, ("B", "K")),
                (new_weights, ("B", "K")),
                (current_weights, ("B", "K")),
            ]
        )

        new_velocity = self.weight_update_momentum * prev_velocity + lr_gradient

        # Ensure we do not go negative
        new_weights = tf.nn.relu(
            current_weights + self.weight_update_momentum * new_velocity + lr_gradient
        )
        # Ensure everything sums up to 1
        new_weights_norm = new_weights / tf.reduce_sum(new_weights, -1, keepdims=True)

        # scatter here if fewer weights then num_cameras were calculated
        if new_weights_norm.shape[-1] != self.num_cameras_per_image:
            # We need to add the batch indexing
            b_idx = repeat(
                tf.range(camera_idxs.shape[0])[None, ...], camera_idxs.shape[1], 1
            )
            camera_idxs = tf.stack([b_idx, camera_idxs], -1)  # "B", "K", 2

            new_weights_norm = tf.scatter_nd(
                camera_idxs,
                new_weights_norm,
                (*current_weights.shape[:-1], self.num_cameras_per_image),
            )
            new_velocity = tf.scatter_nd(
                camera_idxs,
                new_velocity,
                (*gradient.shape[:-1], self.num_cameras_per_image),
            )

        # Update self.per_cam_weights and self.momentum_deltas
        new_weights = self.per_cam_weights.scatter_nd_update(
            tf.reshape(idx, [-1, 1]), new_weights_norm
        )
        new_momentum_velocity = self.momentum_velocities.scatter_nd_update(
            tf.reshape(idx, [-1, 1]), new_velocity
        )
        self.per_cam_weights.assign(new_weights)
        self.momentum_velocities.assign(new_momentum_velocity)

    def get_all_best_c2w(self):
        c2ws = []
        for i in range(self.num_images):
            parameters, _, _ = self(i, 1)
            c2ws.append(parameters.c2w[0, 0])  # Remove batch and camera dimension

        return tf.stack(c2ws)

    def handle_keyframe_camera(
        self,
        all_origins,
        current_origin,
        current_idx,
        max_similarity_to_escape_keyframe_radius,
        get_mean_loss,
    ):
        with tf.name_scope("KeyframeCameraSelect"):
            # Get the mean losses for each image
            mean_losses = get_mean_loss(None)
            # Also find the closest pose
            similarity = tf.reshape(
                dot(
                    normalize(current_origin),
                    normalize(all_origins),
                )
                + 1,
                (-1,),
            )  # Here high is best. 2 Ist max value
            similarity_reverse = 2 - similarity  # 0 is the best, 2 is the worst

            def normalize_values(val):
                return tf.math.divide_no_nan(
                    val - tf.reduce_min(val), tf.reduce_max(val) - tf.reduce_min(val)
                )

            # Build a joint metric which goes from 0 to 2. Lower is better.
            normalize_losses = normalize_values(
                tf.minimum(mean_losses, 5.0)
            )  # 0 is the best
            normalize_similarity = normalize_values(similarity_reverse)  # 0 is the best
            joint_metric = (normalize_losses * 2 + normalize_similarity) / 3

            # Filter out samples we do not want
            indices = tf.range(similarity.shape[0], dtype=tf.int32)
            # Filter out values which are too close
            values_too_large = similarity < max_similarity_to_escape_keyframe_radius
            # But always include the canonical pose
            always_canonical = indices == self.canonical_pose_idx
            either_far_away_or_canonical = tf.logical_or(
                values_too_large, always_canonical
            )
            # Except we have the same pose
            same_idx_mask = tf.cast(indices, tf.int32) != tf.cast(
                tf.reshape(current_idx, []), tf.int32
            )
            # This filter removes all images in the keyframe radius, except the canonical image
            # And removes the current image
            joint_filter = tf.logical_and(same_idx_mask, either_far_away_or_canonical)

            # Get all values and indices where the filter is true
            filtered_indices = tf.boolean_mask(indices, joint_filter)
            filtered_values = tf.boolean_mask(joint_metric, joint_filter)

            # Now check which values have the lowest metric
            top_indices = tf.argsort(filtered_values)  # argsort sorts from low to high
            # This indexes into the filter_indices array. Gather the lowest index to find the
            # image index
            best_index = tf.gather(
                filtered_indices, tf.reshape(top_indices[0], (-1, 1))
            )
            return tf.reshape(best_index, ())  # And flatten it to a scalar

    def get_closest_best_camera(self, idx, get_mean_loss):
        with tf.name_scope("ClosestCameraFinding"):
            current_parameter, _, _ = self(idx, 1)
            current_c2w = current_parameter.c2w
            all_pose_c2w = self.get_all_best_c2w()

            current_origin = tf.reshape(current_c2w[..., :3, -1], (1, 3))
            all_origin = tf.reshape(all_pose_c2w[..., :3, -1], (-1, 3))

            similarity = tf.reshape(
                dot(
                    normalize(current_origin),
                    normalize(all_origin),
                )
                + 1,
                (-1,),
            )  # 2 is best, 0 worst

            # Heurisitc to find a fixed keypoint radius
            closest_origins = tf.math.top_k(similarity, k=similarity.shape[0] // 12)

            min_similairty = tf.reduce_min(closest_origins.values)

            mean_losses = get_mean_loss(closest_origins.indices)
            # Lowest loss is the best image
            top_indices = tf.argsort(mean_losses)  # argsort sorts from low to high

            best_index = tf.gather(
                closest_origins.indices, tf.reshape(top_indices[0], (-1, 1))
            )
            best_index = tf.reshape(best_index, ())  # Turn to scalar

        with tf.name_scope("ClosestCameraSelection"):
            return tf.cond(
                # Our current pose is the lowest loss image
                best_index == tf.reshape(idx, ()),
                # Then we have a keyframe and we try to align to other keyframes
                lambda: self.handle_keyframe_camera(
                    all_origin, current_origin, idx, min_similairty, get_mean_loss
                ),
                # Our image is not the keyframe and we can select it
                lambda: best_index,
            )

    def get_all_best_focal(self):
        focals = []
        for i in range(self.num_images):
            parameters, _, _ = self(i, 1)
            focals.append(parameters.focal[0, 0])  # Remove batch and camera dimension

        return tf.stack(focals)

    def get_all_c2w(self):
        c2ws = []
        for i in range(self.num_images):
            parameters, _, _ = self(i, self.num_cameras_per_image)
            for j in range(self.num_cameras_per_image):
                c2ws.append(parameters.c2w[0, j])  # Remove batch and camera dimension

        return tf.stack(c2ws)

    def generate_rays_for_pose(
        self,
        image_dimensions: Tuple[int, int],
        idxs,
        k,
        add_jitter: bool = True,
        stop_f_backprop: bool = False,
    ) -> Tuple[Ray, tf.Tensor]:
        idxs = tf.reshape(tf.convert_to_tensor(idxs, dtype=tf.int32), (-1, 1))
        parameters, weights, _ = self.call(idxs, k, stop_f_backprop)

        H, W = image_dimensions
        org_h, org_w = self.get_height_width(idxs)

        H_s = H / org_h
        W_s = W / org_w

        return self.build_ray_geometry(parameters, (H, W), (H_s, W_s), add_jitter)

    def build_ray_geometry(
        self,
        parameters: CameraParameter,
        HW: Tuple[int, int],
        HW_scale: Tuple[float, float],
        add_jitter: bool = True,
    ):
        H, W = HW
        H_s, W_s = HW_scale
        tf.debugging.assert_shapes(
            [(parameters.c2w, ("B", "C", 4, 4)), (parameters.focal, ("B", "C", 2))]
        )

        batch_dim = parameters.c2w.shape[0]

        x, y = tf.meshgrid(
            tf.range(W, dtype=tf.float32) + 0.5,
            tf.range(H, dtype=tf.float32) + 0.5,
            indexing="xy",
        )
        # Add batch dim
        x = repeat(x[None, ...], batch_dim, 0)
        y = repeat(y[None, ...], batch_dim, 0)

        # Remove the 0.5 offset and cast to int
        sample_coords = tf.cast(
            tf.stack([x, y], -1) - 0.5, tf.int32
        )  # B, Cams, H, W, 2

        # add the jitter to i and j
        if add_jitter:
            jitter_coords = tf.random.uniform(
                [batch_dim, H, W, 2], minval=-0.5, maxval=0.5
            )
        else:  # or not...
            jitter_coords = tf.zeros([batch_dim, H, W, 2])
        x = x + jitter_coords[..., 0]
        y = y + jitter_coords[..., 1]

        # Add camera dim
        x = repeat(x[:, None, ...], parameters.focal.shape[1], 1)
        y = repeat(y[:, None, ...], parameters.focal.shape[1], 1)  # B, Cams, H, W

        jitter_sample_coords = tf.stack([x, y], -1)  # B, Cams, H, W, 2

        focal = parameters.focal
        fx = focal[..., 0] * tf.cast(W_s[:, None], tf.float32)
        fy = focal[..., 1] * tf.cast(H_s[:, None], tf.float32)

        dirs = tf.stack(
            [
                (x - float(W) * float(0.5)) / tf.reshape(fx, (*fx.shape[:2], 1, 1)),
                -(y - float(H) * float(0.5)) / tf.reshape(fy, (*fy.shape[:2], 1, 1)),
                -tf.ones_like(x),
            ],
            -1,
        )  # B, C, H, W, 3

        c2w_reshape = tf.reshape(
            parameters.c2w, (*parameters.c2w.shape[:2], 1, 1, *parameters.c2w.shape[2:])
        )
        rays_d = tf.reduce_sum(dirs[..., None, :] * c2w_reshape[..., :3, :3], -1)
        rays_o = tf.broadcast_to(c2w_reshape[..., :3, -1], tf.shape(rays_d))

        jitter_sample_coords_no_c = jitter_sample_coords[:, 0]  # B, H, W, 2

        tf.debugging.assert_shapes(
            [
                (rays_o, ("B", "C", "H", "W", 3)),
                (rays_d, ("B", "C", "H", "W", 3)),
                (jitter_sample_coords_no_c, ("B", "H", "W", 2)),
                (sample_coords, ("B", "H", "W", 2)),
            ]
        )

        return (
            Ray(rays_o, rays_d),  # B, Cams, H, W, 3
            # We used the center coordinates. For interpolation subtract -.5 again
            # otherwise we always blend in parts of the neighboring pixels
            jitter_sample_coords_no_c - 0.5,
            sample_coords,
        )

    def volume_padding_regularization(self, idxs, k, radius):
        if self.use_look_at_representation:
            idxs_shaped = tf.reshape(
                tf.convert_to_tensor(idxs, dtype=tf.int32), (-1, 1)
            )

            eye_init = tf.gather_nd(self.eye_initial, idxs_shaped)
            eye_offset = tf.gather_nd(self.eye_offset, idxs_shaped)
            eye = eye_init + eye_offset

            # Get the weights
            weights = tf.gather_nd(self.per_cam_weights, idxs_shaped)

            top_k = tf.math.top_k(weights, k=k)

            magn = magnitude(eye)

            # First find the loss of going into the volume bounds
            loss = tf.square(tf.maximum(radius - magn, 0) / radius)
            # Then from straying away to far
            loss = loss + tf.square(tf.maximum(magn - radius * 3, 0))

            loss = tf.ensure_shape(
                tf.gather_nd(loss, top_k.indices[..., None], batch_dims=1),
                (idxs_shaped.shape[0], k, 1),
            )

            return loss[..., 0]
        else:
            parameters, weights, _ = self.call(idxs, k, True)
            c2w = parameters.c2w

            t = c2w[..., :3, 3]
            magn = magnitude(t)

            loss = tf.square(tf.maximum(radius - magn, 0))
            return loss[..., 0]

    def aim_center_regularization(self, idxs, k):
        if self.use_look_at_representation:
            idxs_shaped = tf.reshape(
                tf.convert_to_tensor(idxs, dtype=tf.int32), (-1, 1)
            )

            center_init = tf.gather_nd(self.center_initial, idxs_shaped)
            center_offset = tf.gather_nd(self.center_offset, idxs_shaped)
            center = center_init + center_offset

            # Get the weights
            weights = tf.gather_nd(self.per_cam_weights, idxs_shaped)

            top_k = tf.math.top_k(weights, k=k)

            center_dist_loss = l2Norm(center)

            center_dist_loss = tf.ensure_shape(
                tf.gather_nd(center_dist_loss, top_k.indices[..., None], batch_dims=1),
                (idxs_shaped.shape[0], k, 1),
            )

            to_center_dist = center_dist_loss[..., 0]

            tf.debugging.assert_shapes(
                [
                    (idxs_shaped, ("B", 1)),
                    (to_center_dist, ("B", k)),
                ]
            )

            return to_center_dist
        else:
            parameters, weights, _ = self.call(idxs, k, True)
            idxs_shaped = tf.reshape(
                tf.convert_to_tensor(idxs, dtype=tf.int32), (-1, 1)
            )
            c2w = parameters.c2w

            tf.debugging.assert_shapes(
                [
                    (idxs_shaped, ("B", 1)),
                    (c2w, ("B", k, "4", "4")),
                ]
            )

            dir_shape = tf.zeros((*c2w.shape[:2], 1), dtype=tf.float32)

            dirs = tf.concat(
                [  # TODO enable optimizing cx/cy?
                    tf.zeros_like(dir_shape),
                    tf.zeros_like(dir_shape),
                    -tf.ones_like(dir_shape),
                ],
                -1,
            )  # B, C, 3

            rays_d = tf.reduce_sum(dirs[..., None, :] * c2w[..., :3, :3], -1)  # B, C, 3
            rays_o = tf.stop_gradient(
                tf.broadcast_to(c2w[..., :3, -1], tf.shape(rays_d))
            )  # B, C, 3

            ray_to_center_magn = magnitude(rays_o)  # B, C, 1
            center_proj = rays_o + normalize(rays_d) * ray_to_center_magn

            to_center_dist = l2Norm(center_proj)[..., 0]

            tf.debugging.assert_shapes(
                [
                    (idxs_shaped, ("B", 1)),
                    (c2w, ("B", k, "4", "4")),
                    (dirs, ("B", k, 3)),
                    (rays_d, ("B", k, 3)),
                    (rays_o, ("B", k, 3)),
                    (to_center_dist, ("B", k)),
                ]
            )

            return to_center_dist

    def get_camera_parameters(
        self, idxs, stop_f_backprop: bool = False
    ) -> CameraParameter:
        idxs = tf.reshape(tf.convert_to_tensor(idxs, dtype=tf.int32), (-1, 1))
        # Get the focal
        focal_init = tf.gather_nd(
            self.focal_lengths_initial, idxs
        )  # Index the batch dimensions
        focal_offset = tf.gather_nd(
            self.focal_lengths_offset, idxs
        )  # Index the batch dimensions
        focal = focal_init + focal_offset

        # B, "C", 2

        tf.debugging.assert_shapes(
            [
                (idxs, ("B", 1)),
                (focal, ("B", self.num_cameras_per_image, 1 if self.fy_only else 2)),
            ]
        )

        # Desquish if necessary
        def fx_desquish(x):
            return x

        def fy_desquish(x):
            return x

        if self.squish_focal_range:
            height, width = self.get_height_width(idxs)

            def fy_desquish(x):
                return x**2 * tf.cast(
                    tf.reshape(height, (idxs.shape[0], 1)), tf.float32
                )

            if focal.shape[-1] == 2:

                def fx_desquish(x):
                    return x**2 * tf.cast(
                        tf.reshape(width, (idxs.shape[0], 1)), tf.float32
                    )

        if focal.shape[-1] == 2:
            fx = fx_desquish(focal[..., 0])
            fy = fy_desquish(focal[..., 1])
        else:
            fy = fy_desquish(focal[..., 0])
            fx = fy

        focal = tf.stack([fx, fy], -1)

        if stop_f_backprop:
            focal = tf.stop_gradient(focal)

        # Get the poses
        if self.use_look_at_representation:
            eye_init = tf.gather_nd(self.eye_initial, idxs)
            eye_offset = tf.gather_nd(self.eye_offset, idxs)
            eye = eye_init + eye_offset

            center_init = tf.gather_nd(self.center_initial, idxs)
            center_offset = tf.gather_nd(self.center_offset, idxs)
            center = center_init + center_offset

            up_rotation_init = tf.gather_nd(self.up_rotation_initial, idxs)
            up_rotation_offset = tf.gather_nd(self.up_rotation_offset, idxs)
            up_rotation = up_rotation_init + up_rotation_offset * np.pi

            tf.debugging.assert_shapes(
                [
                    (eye, ("B", "C", 3)),
                    (center, ("B", "C", 3)),
                    (up_rotation, ("B", "C", 1)),
                ]
            )

            eye_flat = tf.reshape(eye, (-1, 3))
            center_flat = tf.reshape(center, (-1, 3))
            up_flat = tf.reshape(up_rotation, (-1, 1))

            r_flat = build_look_at_matrix(eye_flat, center_flat, up_rotation=up_flat)
            c2w_flat = r_t_to_c2w(r_flat, eye_flat)
        else:
            r_init = tf.gather_nd(self.r_initial, idxs)
            r_offset = tf.gather_nd(self.r_offset, idxs)
            r = r_init + r_offset

            t_init = tf.gather_nd(self.t_initial, idxs)
            t_offset = tf.gather_nd(self.t_offset, idxs)
            t = t_init + t_offset

            tf.debugging.assert_shapes(
                [
                    (r, ("B", "C", 6)),
                    (t, ("B", "C", 3)),
                ]
            )

            r_flat = tf.reshape(r, (-1, 6))
            t_flat = tf.reshape(t, (-1, 3))

            # Build extrinsic matrix
            c2w_flat = build_4x4_matrix(r_flat, t_flat)

        batch_shape = tf.shape(idxs)[0]
        c2w = tf.reshape(c2w_flat, (batch_shape, self.num_cameras_per_image, 4, 4))

        tf.debugging.assert_shapes([(c2w, ("B", "C", 4, 4)), (focal, ("B", "C", 2))])

        return CameraParameter(c2w, focal)

    def call(self, idx, k, stop_f_backprop: bool = False):
        idx = tf.convert_to_tensor(idx, dtype=tf.int32)
        idxs = tf.reshape(idx, (-1, 1))

        k = max(1, k)  # 0 cameras is not possible
        # Also not more then the specified
        k = min(k, self.num_cameras_per_image)
        if k == self.num_cameras_per_image:  # All are requested
            parameters = self.get_camera_parameters(idxs, stop_f_backprop)
            weights = tf.gather_nd(self.per_cam_weights, idxs)

            if self.num_cameras_per_image == 1:
                # For a single image always ensure weights of 1
                weights = tf.ones_like(weights)

            idxs = repeat(
                tf.reshape(
                    tf.range(self.num_cameras_per_image, dtype=tf.int32), (1, -1)
                ),
                idxs.shape[0],
                0,
            )
        else:  # Select the number of requested cameras
            idxs = tf.reshape(idx, (-1, 1))
            weights = tf.gather_nd(self.per_cam_weights, idxs)

            tf.debugging.assert_shapes(
                [(idxs, ("B", 1)), (weights, ("B", self.num_cameras_per_image))]
            )

            # Select the best (based on weights)
            top_k = tf.math.top_k(weights, k=k)

            cams = self.get_camera_parameters(idxs, stop_f_backprop)

            tf.debugging.assert_shapes(
                [
                    (idxs, ("B", 1)),
                    (cams.c2w, ("B", "C", 4, 4)),
                    (cams.focal, ("B", "C", 2)),
                ]
            )

            c2w = tf.ensure_shape(
                tf.gather_nd(cams.c2w, top_k.indices[..., None], batch_dims=1),
                (idxs.shape[0], k, 4, 4),
            )
            focal = tf.ensure_shape(
                tf.gather_nd(cams.focal, top_k.indices[..., None], batch_dims=1),
                (idxs.shape[0], k, 2),
            )

            tf.debugging.assert_shapes(
                [
                    (idxs, ("B", 1)),
                    (c2w, ("B", "K", 4, 4)),
                    (focal, ("B", "K", 2)),
                ]
            )

            parameters = CameraParameter(c2w, focal)
            weights = tf.math.divide_no_nan(
                top_k.values, tf.reduce_sum(top_k.values, -1, keepdims=True)
            )
            idxs = top_k.indices

        # Ensure the shapes always match
        in_idx = tf.reshape(idx, (-1, 1))

        tf.debugging.assert_shapes(
            [
                (in_idx, ("B", 1)),
                (idxs, ("B", "K")),
                (weights, ("B", "K")),
                (parameters.c2w, ("B", "K", 4, 4)),
                (parameters.focal, ("B", "K", 2)),
            ]
        )
        return parameters, weights, idxs

    def get_jiggle_pose(self, cam_idx, num_frames, theta_scale=0.05):
        thetas = np.arange(num_frames) / num_frames * 2 * np.pi
        thetas_sin = tf.asin(tf.sin(thetas) * theta_scale)
        thetas_cos = tf.asin(tf.cos(thetas) * theta_scale)

        anchor_pos, _, _ = self(cam_idx, 1)
        anchor_c2w = tf.squeeze(anchor_pos.c2w)  # 4,4
        anchor_focal = tf.squeeze(anchor_pos.focal)  # 2,

        def rot_phi(phis):
            return tf.convert_to_tensor(
                [
                    [
                        [1, 0, 0, 0],
                        [0, tf.cos(phi), -tf.sin(phi), 0],
                        [0, tf.sin(phi), tf.cos(phi), 0],
                        [0, 0, 0, 1],
                    ]
                    for phi in phis
                ],
                dtype=tf.float32,
            )

        def rot_theta(ths):
            return tf.convert_to_tensor(
                [
                    [
                        [tf.cos(th), 0, tf.sin(th), 0],
                        [0, 1, 0, 0],
                        [-tf.sin(th), 0, tf.cos(th), 0],
                        [0, 0, 0, 1],
                    ]
                    for th in ths
                ],
                dtype=tf.float32,
            )

        joined_c2w = rot_phi(thetas_cos) @ anchor_c2w[None, ...]
        joined_c2w = rot_theta(thetas_sin) @ joined_c2w

        return CameraParameter(
            tf.reshape(joined_c2w, (num_frames, 1, 1, 4, 4)),
            tf.reshape(anchor_focal, (1, 1, 2)),
        )

    def get_spherical_poses(self, num_poses):
        c2ws = self.get_all_best_c2w()
        t = c2ws[..., :3, 3]
        _, center = c2w_to_lookat(c2ws)
        mean_center = tf.reduce_mean(center, 0)
        mean_radius = tf.reduce_mean(magnitude(t))

        focals = self.get_all_best_focal()
        mean_focal = tf.reduce_mean(focals)

        def trans_t(t):
            return tf.convert_to_tensor(
                [
                    [1, 0, 0, mean_center[0]],
                    [0, 1, 0, mean_center[1]],
                    [0, 0, 1, mean_center[2] + t],
                    [0, 0, 0, 1],
                ],
                dtype=tf.float32,
            )

        def rot_phi(phi):
            return tf.convert_to_tensor(
                [
                    [1, 0, 0, 0],
                    [0, tf.cos(phi), -tf.sin(phi), 0],
                    [0, tf.sin(phi), tf.cos(phi), 0],
                    [0, 0, 0, 1],
                ],
                dtype=tf.float32,
            )

        def rot_theta(th):
            return tf.convert_to_tensor(
                [
                    [tf.cos(th), 0, tf.sin(th), 0],
                    [0, 1, 0, 0],
                    [-tf.sin(th), 0, tf.cos(th), 0],
                    [0, 0, 0, 1],
                ],
                dtype=tf.float32,
            )

        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
            c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
            return c2w

        render_poses = tf.stack(
            [
                pose_spherical(angle, -30.0, mean_radius)
                for angle in np.linspace(-180, 180, num_poses + 1)[:-1]
            ],
            0,
        )

        return render_poses, mean_focal

    def get_intrinsic_matrix(self, idxs, image_dimensions) -> tf.Tensor:
        idxs = tf.reshape(idxs, (-1, 1))
        img_dim = tf.cast(tf.gather_nd(self.image_dimensions, idxs), tf.float32)
        org_h, org_w = img_dim[:, 0], img_dim[:, 1]

        H, W = image_dimensions

        H = tf.reshape(tf.cast(H, tf.float32), (-1,)) * tf.ones_like(org_h)
        W = tf.reshape(tf.cast(W, tf.float32), (-1,)) * tf.ones_like(org_w)

        H_s = tf.cast(H / org_h, tf.float32)
        W_s = tf.cast(W / org_w, tf.float32)

        parameters, _, _ = self.call(idxs, 1)

        fx = parameters.focal[:, 0, 1]
        fy = parameters.focal[:, 0, 0]  # Shape is just batched

        o = tf.zeros_like(fx)
        l = tf.ones_like(fx)
        intrinsic = tf.stack(
            [
                tf.stack([fx * W_s, o, -W / 2], 1),
                tf.stack([o, -fy * H_s, -H / 2], 1),
                tf.stack([o, o, -l], 1),
            ],
            1,
        )

        tf.debugging.assert_shapes(
            [
                (intrinsic, ("B", 3, 3)),
                (parameters.c2w, ("B", 1, 4, 4)),
                (parameters.focal, ("B", 1, 2)),
            ]
        )

        return intrinsic
