import math
import os
import sys

import bpy


def create_or_fetch_texture(name, path):
    try:
        # tex = bpy.data.textures[name]
        img = bpy.data.images[name]
        bpy.data.images.remove(img, do_unlink=True)
        print("Texture %s already created" % name)
    except KeyError:
        print("Creating texture %s" % name)

    img = bpy.data.images.load(path)
    img.name = name
    print("Image loaded:", name, path)

    return img


def export_to_gltf(
    output_path: str,
    obj_path: str,
    basecolor_path: str,
    roughness_metallic_path: str,
    world_space_normal_path: str,
    tangent_space_normal_path: str,
):
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
    # Remove the not needed obj material
    bpy.data.materials.remove(obj.data.materials[0], do_unlink=True)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Smooth shading
    bpy.ops.object.shade_smooth()
    bpy.context.object.data.use_auto_smooth = True

    mat = bpy.data.materials.get("GLTFExport")
    print(mat)
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    print(nodes)

    # Apply material
    if obj.data.materials:
        # assign to 1st material slot
        obj.data.materials[0] = mat
    else:
        # no slots
        obj.data.materials.append(mat)

    # Set the textures
    basecolor = nodes["Basecolor"]
    roughmetallic = nodes["RoughnessMetal"]
    world_space_normal = nodes["WorldSpaceNormal"]

    bsctex = create_or_fetch_texture("basecolor_texture", basecolor_path)
    mrtex = create_or_fetch_texture("roughness_metal_texture", roughness_metallic_path)
    nrmtex = create_or_fetch_texture(
        "world_space_normal_texture", world_space_normal_path
    )

    basecolor.image = bsctex
    roughmetallic.image = mrtex
    world_space_normal.image = nrmtex

    mrtex.colorspace_settings.name = "Non-Color"
    nrmtex.colorspace_settings.name = "Non-Color"

    # Bake the world space normal to tangent space
    bakeImg = nodes.new(type="ShaderNodeTexImage")

    image = bpy.data.images.new(
        "BakeImg",
        width=world_space_normal.image.size[0],
        height=world_space_normal.image.size[1],
        alpha=False,
        float_buffer=True,
        is_data=True,
    )

    image.colorspace_settings.name = "Non-Color"

    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_depth = "16"
    bpy.context.scene.render.image_settings.color_mode = "RGB"

    bakeImg.image = image
    bakeImg.select = True
    nodes.active = bakeImg
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.bake(
        type="NORMAL",
        normal_space="TANGENT",
        normal_r="POS_X",
        normal_g="POS_Y",
        normal_b="POS_Z",
        width=world_space_normal.image.size[0],
        height=world_space_normal.image.size[1],
        margin=16,
        use_clear=True,
    )
    image.save_render(tangent_space_normal_path)

    nodes.remove(bakeImg)

    brdf = nodes["Principled BSDF"]

    tangent_normal_node = nodes.new(type="ShaderNodeNormalMap")
    tangent_normal_img = nodes.new(type="ShaderNodeTexImage")

    tnrmtex = create_or_fetch_texture(
        "tangent_normal_texture", tangent_space_normal_path
    )
    tnrmtex.colorspace_settings.name = "Non-Color"
    tangent_normal_img.image = tnrmtex

    # Now swap the normals
    links = mat.node_tree.links
    image_tangent_link = links.new(
        tangent_normal_img.outputs["Color"], tangent_normal_node.inputs["Color"]
    )
    tangent_brdf_link = links.new(
        tangent_normal_node.outputs["Normal"], brdf.inputs["Normal"]
    )

    nodes.remove(world_space_normal)

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        use_selection=True,
        export_normals=True,
        export_materials="EXPORT",
        export_yup=False,
    )


# Start with:
# blender BlenderGLTFExportSetup.blend --python export_gltf.py -b -- --output_path test.glb --mesh_path test.obj --diffuse_path ... --specular_path ... --roughness_path ... --ws_normal_path ...
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


output_path = query_argv("--output_path", None)
mesh_obj = query_argv("--mesh_path", None)
basecolor = query_argv("--basecolor_path", None)
rough_metallic = query_argv("--rough_metallic_path", None)
normal = query_argv("--ws_normal_path", None)
tangent_normal = query_argv("--tangent_normal_path", None)

export_to_gltf(output_path, mesh_obj, basecolor, rough_metallic, normal, tangent_normal)
