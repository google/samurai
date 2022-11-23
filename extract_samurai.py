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

import dataflow.samurai.config as data_config
import models.samurai.config as samurai_config
import utils.training_setup_utils as train_utils
from extraction_utils.const import *
from utils.decorator import timing
import trimesh


def parse_args():
    parser = data_config.add_args(
        samurai_config.add_args(
            train_utils.setup_parser(),
        ),
    )

    parser.add_argument("--sample_resolution", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=25)
    parser.add_argument("--chunk_size", type=int, default=16384)

    parser.add_argument("--extraction_image", type=int, default=0)

    parser.add_argument("--blender_path", type=str, required=True)
    parser.add_argument("--texture_resolution", type=int, default=1024)
    parser.add_argument("--decimate_ratio", type=float, default=0.2)

    parser.add_argument("--ray_samples", type=int, default=32)

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--gpus", type=str)

    return train_utils.parse_args_file_without_nones(parser)


def load_obj(path):
    return trimesh.load_mesh(path)


@timing
def main(args):
    run_dir = os.path.join(
        args.basedir,
        args.expname,
    )
    extract_dir = os.path.join(run_dir, "mesh_extract")
    os.makedirs(extract_dir, exist_ok=True)

    if args.gpus is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Load the tf dependencies just in time
    # This ensures the correct gpus are setup
    from extraction_utils.utils import get_samurai_model
    import tensorflow as tf

    # STEP 1
    # Load the model

    samurai, train_df, test_df = get_samurai_model(args)

    appearance_context = samurai.appearance_store(
        tf.convert_to_tensor([args.extraction_image])
    )
    diffuse_context = (
        None
        if samurai.fixed_diffuse
        else samurai.diffuse_store(tf.convert_to_tensor([args.extraction_image]))
    )

    from extraction_utils.mesh_extraction import (
        bake_world_space_positions,
        perform_marching_cubes,
        select_main_mesh,
        refine_mesh,
    )

    # STEP 2
    # Extract the mesh

    mesh_path = os.path.join(extract_dir, INITIAL_MESH)
    if not os.path.exists(mesh_path) or args.force:
        mesh = perform_marching_cubes(
            args.sample_resolution,
            args.chunk_size,
            args.threshold,
            samurai,
            appearance_context,
            diffuse_context,
        )
        mesh = select_main_mesh(mesh)

        mesh.export(mesh_path)
    else:
        mesh = trimesh.load_mesh(mesh_path)

    # STEP 3
    # Refine the coarse marching cubes mesh

    mesh_path = os.path.join(extract_dir, MESH_IMPROVED)
    if not os.path.exists(mesh_path) or args.force:
        mesh = refine_mesh(
            extract_dir,
            samurai,
            args,
            2_000_000,
            appearance_context,
            diffuse_context,
        )
        mesh.export(mesh_path)
    else:
        mesh = trimesh.load_mesh(mesh_path)

    # STEP 4
    # Bake world space positions

    bake_success = bake_world_space_positions(args, extract_dir)

    if not bake_success:
        return

    from extraction_utils.texture_extraction import texture_query_network

    # STEP 3
    # Extract the texture for all positions

    wsp = imageio.imread(os.path.join(extract_dir, WORLD_SPACE_POSITION))
    wsn = imageio.imread(os.path.join(extract_dir, WORLD_SPACE_NORMAL))
    texture_query_network(
        args,
        samurai,
        args.extraction_image,
        args.chunk_size,
        args.ray_samples,
        extract_dir,
        wsp,
        wsn,
    )

    # STEP 4
    # Optimize the extracted textures

    from extraction_utils.glb_extraction import convert_model, extract_glb

    # STEP 5
    # Save the model in an optimized glb model for mobile rendering

    convert_model(args, extract_dir)
    export_success = extract_glb(args, extract_dir)

    if not export_success:
        return


if __name__ == "__main__":
    args = parse_args()
    print(args)

    main(args)
