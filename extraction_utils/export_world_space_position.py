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
import os
import sys

import bpy
from mathutils import *


def render_world_space_coordinates(
    obj_path: str,
    obj_uv_path: str,
    world_space_position_path: str,
    world_space_normal_path: str,
    resolution: int = 1024,
    gpu_id: int = -1,
    decimate_ratio: float = 1.0,
    smooth_iterations: int = 10,
):
    # Cycles is required for World Space Position baking
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
    bpy.context.scene.render.image_settings.color_depth = "32"
    bpy.context.scene.render.image_settings.color_mode = "RGB"
    bpy.context.scene.render.image_settings.exr_codec = "NONE"

    # Setup GPU
    if gpu_id != -1:
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.feature_set = "EXPERIMENTAL"

        prefs = bpy.context.preferences
        cprefs = prefs.addons["cycles"].preferences
        cprefs.compute_device_type = "CUDA"
        cudaDevices = []
        # This returns a tuple for whatever reason
        for device in cprefs.get_devices()[0]:
            device.use = False
            if device.type == "CUDA":
                cudaDevices.append(device)
        cudaDevices[gpu_id].use = True
        print("USING DEVICES:", [c for c in cudaDevices if c.use])

    # Setup a simple color space
    bpy.context.scene.display_settings.display_device = "None"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.sequencer_colorspace_settings.name = "Raw"

    # Cleanup first
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)
    [bpy.data.objects.remove(o, do_unlink=True) for o in list(bpy.data.objects)]

    # Import the obj
    imported_object = bpy.ops.import_scene.obj(filepath=obj_path, split_mode="OFF")
    obj = bpy.data.objects[-1]
    obj.name = "ObjectToExport"

    # Transform opengl to blender
    obj.rotation_euler = (0, 0, 0)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # UV unwrap
    print("UV unwrapping...")
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action="SELECT")

    # Ensure normals are okay
    bpy.ops.mesh.normals_make_consistent()

    bpy.ops.object.editmode_toggle()

    # TODO decimate and smooth
    if smooth_iterations > 0:
        print("Decimating the mesh")
        smooth = obj.modifiers.new(name="Smooth", type="SMOOTH")
        smooth.iterations = smooth_iterations

        bpy.ops.object.modifier_apply(modifier=smooth.name)

    if decimate_ratio < 1 and decimate_ratio > 0:
        decimate = obj.modifiers.new(name="Decimate", type="DECIMATE")
        decimate.ratio = decimate_ratio

        bpy.ops.object.modifier_apply(modifier=decimate.name)

    # UV
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action="SELECT")

    bpy.ops.uv.smart_project(
        # angle_limit=40.0,
        island_margin=0.001,
        correct_aspect=True,
        scale_to_bounds=True,
    )
    bpy.ops.object.editmode_toggle()

    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # World Space baking

    # Cleanup
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    for image in bpy.data.images:
        # don't do anything if the image has any users.
        if image.users:
            continue

        # remove the image otherwise
        bpy.data.images.remove(image)

    # Create new material
    mat = bpy.data.materials.new(name="BakeMat")
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    for node in nodes:
        nodes.remove(node)

    # Create bake target
    print("Creating image...")
    bakeImg = nodes.new(type="ShaderNodeTexImage")

    image = bpy.data.images.new(
        "BakeImg",
        width=resolution,
        height=resolution,
        alpha=False,
        float_buffer=True,
        is_data=True,
    )
    image.file_format = "OPEN_EXR"
    image.colorspace_settings.name = "Raw"

    geometry = nodes.new(type="ShaderNodeNewGeometry")
    geometry.location = Vector((bakeImg.location[0] + 250, bakeImg.location[1]))

    emission = nodes.new(type="ShaderNodeEmission")
    emission.location = Vector((geometry.location[0] + 250, geometry.location[1]))

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = Vector((emission.location[0] + 250, emission.location[1]))

    links = mat.node_tree.links
    geom_link = links.new(geometry.outputs[1], emission.inputs[0])
    emit_link = links.new(emission.outputs[0], output.inputs[0])

    bakeImg.image = image

    # Append Material
    if obj.data.materials:
        # assign to 1st material slot
        obj.data.materials[0] = mat
    else:
        # no slots
        obj.data.materials.append(mat)

    # Handle bake image selection
    for node in nodes:
        node.select = False

    bakeImg.select = True
    nodes.active = bakeImg

    bpy.context.view_layer.objects.active = obj

    print("Baking...")
    bpy.ops.object.bake(
        type="POSITION",
        width=resolution,
        height=resolution,
        margin=16,
        use_clear=True,
    )
    image.save_render(world_space_position_path)

    bpy.ops.object.bake(
        type="EMIT",
        width=resolution,
        height=resolution,
        margin=16,
        use_clear=True,
    )
    image.save_render(world_space_normal_path)

    # World space baking done

    # Re-export mesh with UV
    bpy.ops.export_scene.obj(
        filepath=obj_uv_path,
        check_existing=False,
        use_selection=True,
        use_normals=True,
        use_uvs=True,
        use_materials=False,
        keep_vertex_order=True,
        axis_forward="Y",
        axis_up="Z",  # Export to fit opengl again
    )


# Start with:
# blender --python export_world_space_position.py -b -- --mesh_path test.obj --mesh_uved_path test_uv.obj --world_space_position_path wsp.exr --world_space_normal_path wsn.exr --gpu 0
# Everything after -- is ignored for blender and we can use it here
argv = sys.argv
argv = argv[argv.index("--") + 1 :]
print(argv)


def query_argv(name, default_val):
    if name in argv:
        ret = argv[argv.index(name) + 1]
    else:
        if default_val == None:
            raise Exception("You need to provide a value for %s" % name)
        ret = default_val
    return ret


gpu_id = int(query_argv("--gpu", "-1"))
mesh_location = query_argv("--mesh_path", None)
mesh_uved_location = query_argv("--mesh_uved_path", None)
world_space_position_location = query_argv("--world_space_position_path", None)
world_space_normal_location = query_argv("--world_space_normal_path", None)
resolution = int(query_argv("--resolution", "2048"))

decimate = float(query_argv("--decimate_ratio", "1"))
smooth = float(query_argv("--smooth", "0"))

render_world_space_coordinates(
    mesh_location,
    mesh_uved_location,
    world_space_position_location,
    world_space_normal_location,
    resolution=resolution,
    gpu_id=gpu_id,
    decimate_ratio=decimate,
    smooth_iterations=smooth,
)
