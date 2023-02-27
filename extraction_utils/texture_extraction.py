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


import os

import imageio
import numpy as np
import tensorflow as tf
from models.samurai.samurai_model import SamuraiModel
from nn_utils.nerf_layers import (
    split_sigma_and_payload,
    setup_fixed_grid_sampling,
    volumetric_rendering,
)
from nn_utils.math_utils import normalize, dot
from tqdm import tqdm
from extraction_utils.dilation_fill import dilate

from utils.decorator import timing


@timing
def texture_query_network(
    args,
    samurai: SamuraiModel,
    extraction_img: int,
    chunk_size: int,
    ray_samples: int,
    output_dir: str,
    positions: np.ndarray,
    normals: np.ndarray,
    cast_rays: bool = True,
):
    positions_flat = positions.reshape(-1, 3)
    normals_flat = normals.reshape(-1, 3)

    positions_normals_flat = np.concatenate((positions_flat, normals_flat), -1).astype(np.float32)

    appearance_context = samurai.appearance_store(
        tf.convert_to_tensor([extraction_img])
    )
    diffuse_context = (
        None
        if samurai.fixed_diffuse
        else samurai.diffuse_store(tf.convert_to_tensor([extraction_img]))
    )

    move_along_normal = samurai.volume_sphere.radius * 0.02
    tmin = 0
    tmax = move_along_normal * 4

    print(
        f"Creating the camera in {move_along_normal} units away from the surface. Sampling for {tmax} total distance. Volume sphere radius is {samurai.volume_sphere.radius} units"
    )

    parameters = {
        "normal": [],
        "basecolor": [],
        "metallic": [],
        "roughness": [],
        "acc_alpha": [],
    }
    print(positions_normals_flat.shape)
    if not cast_rays:
        ray_samples = 1
    ray_chunk = (positions_normals_flat.shape[0] // chunk_size) * ray_samples
    print(ray_chunk, chunk_size, ray_samples)
    for position_normal_sample in tqdm(
        np.array_split(
            positions_normals_flat,
            ray_chunk,
        )
    ):
        position_sample = position_normal_sample[..., :3]
        normal_sample = position_normal_sample[..., 3:]

        if cast_rays:
            position_sample = tf.convert_to_tensor(
                position_normal_sample[..., :3], tf.float32
            )
            normal_sample = tf.convert_to_tensor(
                position_normal_sample[..., 3:], tf.float32
            )

            cur_origin = position_sample + normal_sample * move_along_normal
            cur_direction = normalize(
                position_sample - (position_sample + normal_sample)
            )

            cur_origin = tf.convert_to_tensor(tf.reshape(cur_origin, (1, 1, -1, 3)), tf.float32)
            cur_direction = normalize(tf.reshape(cur_direction, (1, 1, -1, 3)))

            points, z_samples = setup_fixed_grid_sampling(
                cur_origin,
                cur_direction,
                tmin,
                tmax,
                ray_samples,
                randomized=False,
                linear_disparity=False,
            )

            raw = samurai.fine_model(
                points,
                appearance_context,
                diffuse_context,
                cur_direction,
                samurai.fourier_frequencies + 1,
                randomized=False,
            )
            sigma, payload_raw = split_sigma_and_payload(raw)  # B, C, S, 3|1

            payload_dict, weights = volumetric_rendering(
                sigma,
                payload_raw,
                z_samples,
                cur_direction,
                samurai.fine_model.payload_to_parmeters,
            )

            surface_normal = normalize(payload_dict["normal"][0, 0])

            alpha = payload_dict["acc_alpha"].numpy().reshape((-1,))
            # Cur direction faces towards the surface
            # A visible surface can only face in the oposite direction
            # In other words if the dot product between surface normal and
            # direction is negative
            normal_filter = dot(surface_normal, cur_direction[0, 0])[..., 0] < 0

            alpha_filter = alpha > 0.7
            point_filter = alpha_filter & normal_filter

            payload_dict["normal"] = tf.reshape(
                payload_dict["normal"], normal_sample.shape
            )
            payload_dict["normal"] = tf.reshape(
                tf.where(point_filter[:, None], payload_dict["normal"], normal_sample),
                (1, 1, *normal_sample.shape),
            )
        else:
            ray_dir = tf.reshape(
                tf.cast(
                    tf.concat(
                        [
                            tf.zeros_like(position_sample)[:1, :-1],
                            tf.ones_like(position_sample)[:1, -1:],
                        ],
                        -1,
                    ),
                    tf.float32,
                ),
                (1, 1, -1, 3),
            )
            samples = tf.convert_to_tensor(position_sample.reshape((1, 1, -1, 1, 3)))
            raw = samurai.fine_model(
                samples,
                appearance_context,
                diffuse_context,
                ray_dir,
                samurai.fourier_frequencies + 1,
                False,
            )
            sigma, payload = split_sigma_and_payload(raw)  # B, S, C

            sigma_np_flat = sigma.numpy().reshape((-1,))
            positions_non_sigma_extract = position_sample[
                sigma_np_flat < args.threshold
            ]
            if positions_non_sigma_extract.shape[0] > 0:
                print(
                    f"Found {positions_non_sigma_extract.shape[0]} positions where the sigma is too low (of {sigma_np_flat.shape[0]} poses)"
                )

            payload_dict = samurai.fine_model.payload_to_parmeters(payload[..., 0, :])

            surface_normal = payload_dict["normal"][0, 0]

            # Cur direction faces towards the surface
            # A visible surface can only face in the oposite direction
            # In other words if the dot product between surface normal and
            # direction is negative
            normal_same = dot(surface_normal, normal_sample)[..., 0].numpy() > 0

            if np.size(normal_same) - np.count_nonzero(normal_same) > 0:
                print(
                    f"{np.size(normal_same) - np.count_nonzero(normal_same)} normals are different"
                )

            payload_dict["normal"] = tf.reshape(
                payload_dict["normal"], normal_sample.shape
            )
            payload_dict["normal"] = tf.reshape(
                tf.where(normal_same[:, None], payload_dict["normal"], normal_sample),
                (1, 1, *normal_sample.shape),
            )

        parameters = {
            k: v + [payload_dict[k].numpy()[0, 0]] for k, v in parameters.items()
        }

    parameters_np = {
        k: np.concatenate(v, 0).reshape(
            (*positions.shape[:-1], v[0].shape[-1] if len(v[0].shape) == 2 else 1)
        )
        for k, v in parameters.items()
    }

    normal = parameters_np["normal"] * parameters_np["acc_alpha"]
    normal = dilate(normal * 2 - 1, np.all(normal != 0, -1)) * 0.5 + 0.5

    imageio.imwrite(os.path.join(output_dir, "normal.exr"), normal.astype(np.float32))
    imageio.imwrite(
        os.path.join(output_dir, "normal.jpg"),
        (np.clip((normal * 0.5 + 0.5), 0, 1) * 255).astype(np.uint8),
    )
    for mtype in ["basecolor", "metallic", "roughness"]:
        imageio.imwrite(
            os.path.join(output_dir, mtype + ".jpg"),
            (np.clip(parameters_np[mtype], 0, 1) * 255).astype(np.uint8),
        )


# Usage:
# wsp = pyexr.open(os.path.join(extract_dir, WORLD_SPACE_POSITION)).get()
# wsn = pyexr.open(os.path.join(extract_dir, WORLD_SPACE_NORMAL)).get()
# texture_query_network(
#     samurai,
#     args.extraction_image,
#     args.chunk_size,
#     extract_dir,
#     wsp, #world_space_position,
#     wsn, #world_space_normal,
# )
