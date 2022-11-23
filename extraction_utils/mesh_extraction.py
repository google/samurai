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
import subprocess

import mcubes
import numpy as np
import open3d as o3d
import tensorflow as tf
import trimesh
from nn_utils.nerf_layers import split_sigma_and_payload
from tqdm import tqdm

from extraction_utils.const import *
from extraction_utils.sampling_utils import regular_3d_grid
from extraction_utils.utils import Rotation, RotationDefinition, build_rotation_matrix
from utils.decorator import timing


@timing
def perform_marching_cubes(
    sample_resolution,
    chunk_size,
    threshold,
    samurai,
    appearance_context,
    diffuse_context,
) -> trimesh.Trimesh:
    sample_bounds = samurai.volume_sphere.radius
    sample_grid = regular_3d_grid(
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        sample_bounds,
        sample_resolution,
    )

    sample_grid_flat = sample_grid.reshape((-1, 3)).astype(np.float32)

    all_sigma = []
    for samples in tqdm(
        np.array_split(sample_grid_flat, sample_grid_flat.shape[0] // chunk_size)
    ):
        ray_dir = tf.cast(
            tf.concat(
                [tf.zeros_like(samples)[:1, :-1], tf.ones_like(samples)[:1, -1:]],
                -1,
            ),
            tf.float32,
        )
        samples = tf.convert_to_tensor(samples.reshape((1, -1, 3)))
        raw = samurai.fine_model(
            samples,
            appearance_context,
            diffuse_context,
            ray_dir,
            samurai.fourier_frequencies + 1,
            False,
        )
        sigma, _ = split_sigma_and_payload(raw)  # B, S, C
        all_sigma.append(sigma.numpy()[0])  # S, C

    all_sigma = np.concatenate(all_sigma, 0).reshape(sample_grid.shape[:-1])

    print(
        np.any(all_sigma > threshold),
        all_sigma.mean(),
        all_sigma.min(),
        all_sigma.max(),
    )

    print("running marching cubes...")
    vertices, triangles = mcubes.marching_cubes(all_sigma, threshold)
    vertices_centered = (vertices / sample_resolution - 0.5) * 2
    vertices_centered_scaled = vertices_centered * sample_bounds

    # Rotate z 90
    rot_mat = build_rotation_matrix([Rotation(RotationDefinition.Z_AXIS, 90)])
    rot_shaped = rot_mat.reshape(
        (*[1 for _ in vertices_centered_scaled.shape[:-1]], *rot_mat.shape)
    )

    vertices_rotated = np.sum(vertices_centered_scaled[..., None, :] * rot_shaped, -1)
    # Flip x
    flip_vec = np.array([-1.0, 1.0, 1.0], dtype=np.float32).reshape(
        (*[1 for _ in vertices_centered_scaled.shape[:-1]], 3)
    )
    vertices_flipped = vertices_rotated * flip_vec

    mesh = trimesh.Trimesh(vertices_flipped, triangles)
    trimesh.repair.fix_normals(mesh)

    return mesh


def select_main_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    meshes = mesh.split(only_watertight=True)
    vertex_len_list = [m.vertices.shape[0] for m in meshes]
    index_max = max(range(len(vertex_len_list)), key=vertex_len_list.__getitem__)
    mesh = meshes[index_max]

    return mesh


@timing
def refine_mesh(
    extract_dir,
    samurai,
    args,
    total_samples,
    appearance_context,
    diffuse_context,
):
    mesh = o3d.io.read_triangle_mesh(os.path.join(extract_dir, INITIAL_MESH))

    mesh = mesh.compute_triangle_normals()
    pcd = mesh.sample_points_uniformly(
        number_of_points=total_samples, use_triangle_normal=True
    )
    points = np.asarray(pcd.points).astype(np.float32)  # N, 3
    normals = np.asarray(pcd.normals).astype(np.float32)  # N, 3

    imp_points, imp_normal = perform_fine_sampling(
        points,
        normals,
        args.chunk_size,
        args.ray_samples,
        samurai,
        appearance_context,
        diffuse_context,
    )

    print(imp_points.shape)
    print(imp_normal.shape)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(imp_points.astype(np.float64))
    pc.normals = o3d.utility.Vector3dVector(imp_normal.astype(np.float64))
    pc.normalize_normals()  # Point cloud is now setup

    pc, _ = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pc, depth=8
        )

    mesh_path = os.path.join(extract_dir, MESH_IMPROVED)
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    mesh = trimesh.load_mesh(mesh_path)
    meshes = mesh.split(only_watertight=True)
    vertex_len_list = [m.vertices.shape[0] for m in meshes]
    index_max = max(range(len(vertex_len_list)), key=vertex_len_list.__getitem__)
    mesh = meshes[index_max]

    return mesh


def perform_fine_sampling(
    points,
    normals,
    chunk_size,
    ray_samples,
    samurai,
    appearance_context,
    diffuse_context,
):
    from nn_utils.math_utils import normalize, dot
    from nn_utils.nerf_layers import volumetric_rendering, setup_fixed_grid_sampling

    all_sigma = []
    all_points = []
    all_normals = []

    move_along_normal = samurai.volume_sphere.radius * 0.05
    tmin = 0
    tmax = move_along_normal * 3.5

    print(
        f"Creating the camera in {move_along_normal} units away from the surface. Sampling for {tmax} total distance. Volume sphere radius is {samurai.volume_sphere.radius} units"
    )

    point_normals = np.concatenate((points, normals), -1)

    samples_per_batch = chunk_size // ray_samples
    for position_normal_sample in tqdm(
        np.array_split(point_normals, point_normals.shape[0] // samples_per_batch)
    ):
        position_sample = position_normal_sample[..., :3]
        normal_sample = position_normal_sample[..., 3:]

        cur_origin = position_sample + normal_sample * move_along_normal
        cur_direction = normalize(position_sample - (position_sample + normal_sample))

        cur_origin = tf.convert_to_tensor(tf.reshape(cur_origin, (1, 1, -1, 3)))
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
        expected_depth = payload_dict["depth"][0, 0]  # B, C removed
        alpha = payload_dict["acc_alpha"].numpy().reshape((-1,))

        surface_point = cur_origin[0, 0] + cur_direction[0, 0] * expected_depth[:, None]
        surface_normal = payload_dict["normal"][0, 0]

        alpha_filter = alpha > 0.7
        # Cur direction faces towards the surface
        # A visible surface can only face in the oposite direction
        # In other words if the dot product between surface normal and
        # direction is negative
        normal_filter = dot(surface_normal, cur_direction[0, 0])[..., 0] < 0

        point_filter = alpha_filter & normal_filter

        all_points.append(surface_point.numpy()[point_filter])
        all_normals.append(surface_normal.numpy()[point_filter])

    all_points = np.concatenate(all_points, 0)
    all_normals = np.concatenate(all_normals, 0)

    return all_points, all_normals


@timing
def bake_world_space_positions(args, extract_dir: str) -> bool:
    blender_run_command = (
        f"{args.blender_path} --python extraction_utils/export_world_space_position.py -b -- "
        f"--mesh_path {os.path.join(extract_dir, MESH_IMPROVED)} "
        f"--mesh_uved_path {os.path.join(extract_dir, UVED_MESH)} "
        f"--world_space_position_path {os.path.join(extract_dir, WORLD_SPACE_POSITION)} "
        f"--world_space_normal_path {os.path.join(extract_dir, WORLD_SPACE_NORMAL)} "
        f"--resolution {args.texture_resolution} "
        f"--decimate_ratio {args.decimate_ratio}"
    )
    print("Running...")
    print(blender_run_command)
    out = subprocess.run(blender_run_command, capture_output=True, shell=True)

    print(out)

    if out.returncode != 0:
        print("Error:", out)
        return False

    return True


# Usage
# mesh = perform_marching_cubes(
#     args.sample_resolution,
#     args.chunk_size,
#     args.threshold,
#     samurai,
#     appearance_context,
#     diffuse_context,
# )
# mesh_path = os.path.join(extract_dir, INITIAL_MESH)
# mesh.export(mesh_path)

# bake_world_space_positions(args, extract_dir)

# print("Done")
