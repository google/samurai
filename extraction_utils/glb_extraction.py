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

import imageio
import numpy as np

from extraction_utils.const import *
from utils.decorator import timing


@timing
def extract_glb(args, extract_dir):
    blender_run_command = (
        f"{args.blender_path} extraction_utils/BlenderGLTFExportSetup.blend "
        "--python extraction_utils/export_gltf.py -b -- "
        f"--output_path {os.path.join(extract_dir, FINAL_GLB)} "
        f"--mesh_path {os.path.join(extract_dir, UVED_MESH)} "
        f"--basecolor_path {os.path.join(extract_dir, BASECOLOR)} "
        f"--rough_metallic_path {os.path.join(extract_dir, ROUGH_METALLIC)} "
        f"--ws_normal_path {os.path.join(extract_dir, NORMAL)} "
        f"--tangent_normal_path {os.path.join(extract_dir, TANGENT_NORMAL)} "
    )
    print("Running...")
    print(blender_run_command)
    out = subprocess.run(blender_run_command, capture_output=True, shell=True)

    print(out)

    if out.returncode != 0:
        print("Error:", out)
        return False

    return True


def convert_model(args, extract_dir):
    brdf = [
        imageio.imread(os.path.join(extract_dir, mtype)) / 255
        for mtype in [METALLIC, ROUGHNESS]
    ]
    brdf = [a[..., None] if len(a.shape) == 2 else a for a in brdf]
    mtl, rough = brdf

    rough = rough[:, :, :1]
    rgh_mtl = np.concatenate((np.zeros_like(rough), rough, mtl), -1)

    imageio.imwrite(
        os.path.join(extract_dir, ROUGH_METALLIC), (rgh_mtl * 255).astype(np.uint8)
    )


# Usage
# convert_model(args, extract_dir)
# extract_glb(args, extract_dir)
