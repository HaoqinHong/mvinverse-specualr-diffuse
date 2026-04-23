import bpy
from mathutils import Vector

import argparse
import cv2
from datetime import datetime
import glob
import json
import logging
import math
import numpy as np
import os
import random
import subprocess
import sys
import time


# +++ +++ +++ +++
# Common
# +++ +++ +++ +++
def get_logger(log_filepath):
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)

    # Set a file handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Set a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set a formatter
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Set handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def log_runtime_gpu_info(logger):
    logger.info("GPU diagnostics: collecting nvidia-smi info")
    try:
        gpu_query = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ]
        gpu_output = subprocess.check_output(gpu_query, stderr=subprocess.STDOUT, text=True).strip()
        if gpu_output:
            for line in gpu_output.splitlines():
                logger.info(f"GPU: {line}")
        else:
            logger.info("GPU: nvidia-smi returned empty output")
    except Exception as e:
        logger.warning(f"GPU diagnostics failed: {e}")

    try:
        proc_query = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
        proc_output = subprocess.check_output(proc_query, stderr=subprocess.STDOUT, text=True).strip()
        if proc_output:
            for line in proc_output.splitlines():
                logger.info(f"GPU process: {line}")
        else:
            logger.info("GPU process: no active compute apps")
    except Exception as e:
        logger.warning(f"GPU process diagnostics failed: {e}")


def _deg_to_rad(*values):
    return tuple(math.radians(_value) for _value in values)


# +++ +++ +++ +++
# Camera
# +++ +++ +++ +++
def _az_el_to_position(azimuths, elevations):
    x = np.cos(azimuths) * np.cos(elevations)
    y = np.sin(azimuths) * np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1)


def _create_cameras(positions):
    # Create an empty object to track
    empty = bpy.data.objects.new("3DAT_Empty", None)
    bpy.context.scene.collection.objects.link(empty)

    # Create cameras
    for position in positions:
        # Create a camera
        camera = bpy.data.cameras.new("Camera")
        object = bpy.data.objects.new("Camera", camera)
        x, y, z = position
        object.location  = x, y, z
        object.data.type = "PERSP"
        bpy.context.scene.collection.objects.link(object)

        # Create a camera constraint
        camera_constraint = object.constraints.new(type="TRACK_TO")
        camera_constraint.track_axis = "TRACK_NEGATIVE_Z"
        camera_constraint.up_axis    = "UP_Y"
        camera_constraint.target     = empty

        bpy.context.view_layer.update()


# +++ +++ +++ +++
# Env Light
# +++ +++ +++ +++
def _create_lights():
    # Remove lights
    for object in bpy.data.objects:
        if object.type == "LIGHT" or object.name.startswith("3DAT_Empty_SpotLight"):
            bpy.data.objects.remove(object, do_unlink=True)
    for light  in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)

    # Create an empty object to track
    empty = bpy.data.objects.new("3DAT_Empty_SpotLight", None)
    bpy.context.scene.collection.objects.link(empty)

    # Create a point light
    pointlight_data   = bpy.data.lights.new(name="3DAT_PointLight", type="POINT")
    pointlight_object = bpy.data.objects.new(name="3DAT_PointLight", object_data=pointlight_data)
    bpy.context.collection.objects.link(pointlight_object)

    pointlight_object.location       = ( 4.08, 1.01, 5.90)
    pointlight_object.rotation_mode  = "XYZ"
    pointlight_object.rotation_euler = _deg_to_rad(37.26, 3.164, 106.94)
    pointlight_object.scale          = (  1.0,  1.0,  1.0)

    pointlight_data.energy  = 1000
    # 'PointLight' object has no attribute 'use_soft_falloff'
    #pointlight_data.use_soft_falloff = False
    pointlight_data.shadow_soft_size = 0.1
    pointlight_data.use_shadow               = True
    pointlight_data.shadow_buffer_clip_start = 0.05
    pointlight_data.shadow_buffer_bias       = 1.0
    pointlight_data.use_contact_shadow       = True
    pointlight_data.contact_shadow_distance  = 0.2
    pointlight_data.contact_shadow_bias      = 0.03
    pointlight_data.contact_shadow_thickness = 0.2

    # Create a spot light
    spotlight_data   = bpy.data.lights.new(name="3DAT_SpotLight", type="SPOT")
    spotlight_object = bpy.data.objects.new(name="3DAT_SpotLight", object_data=spotlight_data)
    bpy.context.scene.collection.objects.link(spotlight_object)

    spotlight_constraint = spotlight_object.constraints.new(type="TRACK_TO")
    spotlight_constraint.track_axis = "TRACK_NEGATIVE_Z"
    spotlight_constraint.up_axis    = "UP_Y"
    spotlight_constraint.target     = empty

    spotlight_object.location       = (3.06, -3.06, 2.5)
    spotlight_object.rotation_mode  = "XYZ"
    spotlight_object.rotation_euler = _deg_to_rad(60.0, 0.0, 45.0)
    spotlight_object.scale          = (0.56, 0.56, 0.56)

    spotlight_data.energy = 2000
    # 'SpotLight' object has no attribute 'use_soft_falloff'
    #spotlight_data.use_soft_falloff = False
    spotlight_data.shadow_soft_size = 2.21
    spotlight_data.spot_size  = math.radians(24.7)
    spotlight_data.spot_blend = 1.0
    spotlight_data.use_shadow                = True
    spotlight_data.shadow_buffer_clip_start  = 0.05
    spotlight_data.shadow_buffer_bias        = 0.1
    spotlight_data.use_contact_shadow        = True
    spotlight_data.contact_shadow_distance   = 7.2
    spotlight_data.contact_shadow_bias       = 0.03
    spotlight_data.contact_shadow_thickness  = 0.03

    # Create an area light
    arealight_data   = bpy.data.lights.new(name="3DAT_AreaLight", type="AREA")
    arealight_object = bpy.data.objects.new(name="3DAT_AreaLight", object_data=arealight_data)
    bpy.context.collection.objects.link(arealight_object)

    arealight_object.location       = (-3.58, 1.28, 0.86)
    arealight_object.rotation_mode  = "XYZ"
    arealight_object.rotation_euler = _deg_to_rad(72.30, -2.71, -100.0)
    arealight_object.scale          = (  1.0, 1.0,  1.0)

    arealight_data.energy  = 95
    arealight_data.shape  = "RECTANGLE"
    arealight_data.size   = 3.55
    arealight_data.size_y = 1.52
    arealight_data.use_shadow                = True
    arealight_data.shadow_buffer_clip_start  = 0.05
    arealight_data.shadow_buffer_bias        = 1.0
    arealight_data.use_contact_shadow        = True
    arealight_data.contact_shadow_distance   = 0.2
    arealight_data.contact_shadow_bias       = 0.03
    arealight_data.contact_shadow_thickness  = 0.2

    # Create a sun light
    sunlight_data1   = bpy.data.lights.new(name="3DAT_SunLight1", type="SUN")
    sunlight_object1 = bpy.data.objects.new(name="3DAT_SunLight1", object_data=sunlight_data1)
    bpy.context.collection.objects.link(sunlight_object1)

    sunlight_object1.location       = ( 2.22, -4.53, 5.90)
    sunlight_object1.rotation_mode  = "XYZ"
    sunlight_object1.rotation_euler = _deg_to_rad(37.26, 3.16, 53.67)
    sunlight_object1.scale          = (  1.0,   1.0,  1.0)

    sunlight_data1.energy = 2.0
    sunlight_data1.angle  = math.radians(0.0)
    sunlight_data1.use_shadow                  = True
    sunlight_data1.shadow_buffer_bias          = 0.1
    sunlight_data1.shadow_cascade_count        = 4
    sunlight_data1.shadow_cascade_fade         = 0.1
    sunlight_data1.shadow_cascade_max_distance = 1000
    sunlight_data1.shadow_cascade_exponent     = 0.8
    sunlight_data1.use_contact_shadow          = True
    sunlight_data1.contact_shadow_distance     = 0.2
    sunlight_data1.contact_shadow_bias         = 0.03
    sunlight_data1.contact_shadow_thickness    = 0.2

    # Create a sun light
    sunlight_data2   = bpy.data.lights.new(name="3DAT_SunLight2", type="SUN")
    sunlight_object2 = bpy.data.objects.new(name="3DAT_SunLight2", object_data=sunlight_data2)
    bpy.context.collection.objects.link(sunlight_object2)

    sunlight_object2.location       = ( 4.08, 1.01, 5.90)
    sunlight_object2.rotation_mode  = "XYZ"
    sunlight_object2.rotation_euler = _deg_to_rad(37.26, 3.16, 106.94)
    sunlight_object2.scale          = (  1.0,  1.0,  1.0)

    sunlight_data2.energy = 3.0
    sunlight_data2.angle  = math.radians(12.0)
    sunlight_data2.use_shadow                  = True
    sunlight_data2.shadow_buffer_bias          = 0.1
    sunlight_data2.shadow_cascade_count        = 4
    sunlight_data2.shadow_cascade_fade         = 0.1
    sunlight_data2.shadow_cascade_max_distance = 1000
    sunlight_data2.shadow_cascade_exponent     = 0.8
    sunlight_data2.use_contact_shadow          = True
    sunlight_data2.contact_shadow_distance     = 7.2
    sunlight_data2.contact_shadow_bias         = 0.03
    sunlight_data2.contact_shadow_thickness    = 0.03


def _create_envlight():
    # Remove all nodes of current world
    cur_world = bpy.data.worlds[0]
    cur_world.use_nodes = True

    nodes = cur_world.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    # Create the nodes
    texcoord_node = nodes.new(type="ShaderNodeTexCoord")
    mapping_node  = nodes.new(type="ShaderNodeMapping")
    mapping_node.name = "3DAT_EnvLightMappingNode"
    mapping_node.vector_type = "POINT"
    tex_node = nodes.new(type="ShaderNodeTexEnvironment")
    tex_node.name = "3DAT_EnvLightImageNode"
    bg_node  = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs[1].default_value = 2.0
    out_node = nodes.new(type="ShaderNodeOutputWorld")

    # Link the nodes
    links = cur_world.node_tree.links
    links.new(texcoord_node.outputs[0], mapping_node.inputs[0])
    links.new(mapping_node.outputs[0],  tex_node.inputs[0])
    links.new(tex_node.outputs[0],      bg_node.inputs[0])
    links.new(bg_node.outputs[0],       out_node.inputs[0])

def execute_setting_envlight(envlight_filepaths):
    envlight_filepath = envlight_filepaths[random.randrange(0, len(envlight_filepaths))]
    bpy.data.worlds[0].node_tree.nodes["3DAT_EnvLightImageNode"].image = bpy.data.images.load(envlight_filepath)
    bpy.data.worlds[0].node_tree.nodes["3DAT_EnvLightMappingNode"].inputs[2].default_value[2] = math.radians(random.uniform(0.0, 360.0))


def execute_setting_white_envlight(strength):
    cur_world = bpy.data.worlds[0]
    cur_world.use_nodes = True

    nodes = cur_world.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs[1].default_value = strength
    out_node = nodes.new(type="ShaderNodeOutputWorld")

    links = cur_world.node_tree.links
    links.new(bg_node.outputs[0], out_node.inputs[0])

# +++ +++ +++ +++
# Asset
# +++ +++ +++ +++
def _get_mesh_bbox(object):
    bbox_min = ( math.inf,) * 3
    bbox_max = (-math.inf,) * 3

    for coord in object.bound_box:
        coord = Vector(coord)
        coord = object.matrix_world @ coord
        bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
        bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    return Vector(bbox_min), Vector(bbox_max)


def _normalize_mesh():
    # Merge meshes
    bpy.ops.object.select_all(action="DESELECT")
    for object in bpy.data.objects:
        if isinstance(object.data, bpy.types.Mesh):
            bpy.context.view_layer.objects.active = object
            break
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.join()

    # Normalize the mesh
    for object in bpy.data.objects:
        if isinstance(object.data, bpy.types.Mesh):
            # Scale the mesh
            bbox_min, bbox_max = _get_mesh_bbox(object)
            object.scale *= ((1.0 / max(bbox_max - bbox_min)) * 2.0)

            bpy.context.view_layer.update()

            # Translate the mesh
            bbox_min, bbox_max = _get_mesh_bbox(object)
            object.matrix_world.translation += (-(bbox_min + bbox_max) / 2.0)

            # Smooth the mesh
            bpy.ops.object.shade_smooth()
            break

    bpy.ops.object.select_all(action="DESELECT")


def _load_gltf(asset_filepath):
    bpy.ops.import_scene.gltf(filepath=asset_filepath)
    _normalize_mesh()


# +++ +++ +++ +++
# Compositor
# +++ +++ +++ +++
def _get_scene_compositor_node_tree(scene):
    scene.use_nodes = True

    for attr_name in ("node_tree", "compositor_node_tree", "compositing_node_tree"):
        node_tree = getattr(scene, attr_name, None)
        if node_tree is not None:
            return node_tree

    raise RuntimeError("Compositor node tree is unavailable in this Blender version.")


def _create_compositor_for_generating_images():
    scene = bpy.context.scene
    node_tree = _get_scene_compositor_node_tree(scene)

    # Remove nodes
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # Create nodes
    renderlayers_node   = node_tree.nodes.new(type="CompositorNodeRLayers")
    composite_node      = node_tree.nodes.new(type="CompositorNodeComposite")
    separatecolor_node1 = node_tree.nodes.new(type="CompositorNodeSepRGBA")
    combinecolor_node1  = node_tree.nodes.new(type="CompositorNodeCombRGBA")

    # If view_transform is "AgX", using [100.0, 100.0, 100.0] as the background color, which will be mapped to [1.0, 1.0, 1.0], can prevent losing pure white background.
    # If view_transform is "Raw" or "Standard", using [100.0, 100.0, 100.0] as the background color, which will be reduced to [1.0, 1.0, 1.0], is harmless.
    subtract_node1 = node_tree.nodes.new(type="CompositorNodeMath")
    subtract_node1.operation = "SUBTRACT"
    subtract_node1.inputs[1].default_value = 100.0
    subtract_node2 = node_tree.nodes.new(type="CompositorNodeMath")
    subtract_node2.operation = "SUBTRACT"
    subtract_node2.inputs[1].default_value = 100.0
    subtract_node3 = node_tree.nodes.new(type="CompositorNodeMath")
    subtract_node3.operation = "SUBTRACT"
    subtract_node3.inputs[1].default_value = 100.0
    # If view_transform is "AgX", using [100.0, 100.0, 100.0] as the background color, which will be mapped to [1.0, 1.0, 1.0], can prevent losing pure white background.
    # If view_transform is "Raw" or "Standard", using [100.0, 100.0, 100.0] as the background color, which will be reduced to [1.0, 1.0, 1.0], is harmless.
    multiplyadd_node1 = node_tree.nodes.new(type="CompositorNodeMath")
    multiplyadd_node1.operation = "MULTIPLY_ADD"
    multiplyadd_node1.inputs[2].default_value = 100.0
    multiplyadd_node2 = node_tree.nodes.new(type="CompositorNodeMath")
    multiplyadd_node2.operation = "MULTIPLY_ADD"
    multiplyadd_node2.inputs[2].default_value = 100.0
    multiplyadd_node12 = node_tree.nodes.new(type="CompositorNodeMath")
    multiplyadd_node12.operation = "MULTIPLY_ADD"
    multiplyadd_node12.inputs[2].default_value = 100.0

    # Link nodes
    node_tree.links.new(separatecolor_node1.outputs[0], subtract_node1.inputs[0])
    node_tree.links.new(separatecolor_node1.outputs[1], subtract_node2.inputs[0])
    node_tree.links.new(separatecolor_node1.outputs[2], subtract_node3.inputs[0])
    node_tree.links.new(subtract_node1.outputs[0],      multiplyadd_node1.inputs[0])
    node_tree.links.new(subtract_node2.outputs[0],      multiplyadd_node2.inputs[0])
    node_tree.links.new(subtract_node3.outputs[0],      multiplyadd_node12.inputs[0])
    node_tree.links.new(renderlayers_node.outputs[1],   multiplyadd_node1.inputs[1])
    node_tree.links.new(renderlayers_node.outputs[1],   multiplyadd_node2.inputs[1])
    node_tree.links.new(renderlayers_node.outputs[1],   multiplyadd_node12.inputs[1])
    node_tree.links.new(multiplyadd_node1.outputs[0],   combinecolor_node1.inputs[0])
    node_tree.links.new(multiplyadd_node2.outputs[0],   combinecolor_node1.inputs[1])
    node_tree.links.new(multiplyadd_node12.outputs[0],  combinecolor_node1.inputs[2])
    node_tree.links.new(renderlayers_node.outputs[0],   separatecolor_node1.inputs[0])
    node_tree.links.new(combinecolor_node1.outputs[0],  composite_node.inputs[0])


# +++ +++ +++ +++
# Operation
# +++ +++ +++ +++
def execute_setting_scene(
    camera_resolutions,
    camera_distances,
    camera_azimuths,
    camera_elevations,
    eevee_render_samples=64,
    eevee_use_gtao=True,
    eevee_use_ssr=True,
    eevee_shadow_cube_size="1024",
    eevee_shadow_cascade_size="2048",
):
    # Remove all objects, cameras, meshes, materials, textures and images
    for object   in bpy.data.objects:
        bpy.data.objects.remove(object, do_unlink=True)
    for camera   in bpy.data.cameras:
        bpy.data.cameras.remove(camera, do_unlink=True)
    for mesh     in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture  in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image    in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

    # Configure current scene
    bpy.context.scene.render.resolution_x = camera_resolutions[0][0]
    bpy.context.scene.render.resolution_y = camera_resolutions[0][1]
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.use_nodes     = True
    eevee = bpy.context.scene.eevee

    # Blender EEVEE API differs between versions (e.g. 4.x vs 5.x).
    # Set options only when available to keep this script portable.
    def _set_eevee_if_exists(attr_name, value):
        if hasattr(eevee, attr_name):
            setattr(eevee, attr_name, value)

    _set_eevee_if_exists("taa_render_samples", int(max(1, eevee_render_samples)))

    _set_eevee_if_exists("use_gtao", eevee_use_gtao)
    _set_eevee_if_exists("gtao_distance", 0.2)
    _set_eevee_if_exists("gtao_factor", 0.59)
    _set_eevee_if_exists("gtao_quality", 0.25)
    _set_eevee_if_exists("use_gtao_bent_normals", eevee_use_gtao)
    _set_eevee_if_exists("use_gtao_bounce", eevee_use_gtao)

    # SSR was replaced/renamed in newer Blender versions.
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr = eevee_use_ssr
    elif hasattr(eevee, "use_raytracing"):
        eevee.use_raytracing = eevee_use_ssr
    _set_eevee_if_exists("use_ssr_halfres", False)
    _set_eevee_if_exists("ssr_border_fade", 0.093)

    _set_eevee_if_exists("shadow_cube_size", eevee_shadow_cube_size)
    _set_eevee_if_exists("shadow_cascade_size", eevee_shadow_cascade_size)
    _set_eevee_if_exists("use_shadow_high_bitdepth", True)

    _camera_distances  = camera_distances[:,None]
    _camera_azimuths   = np.deg2rad(camera_azimuths.reshape(-1))
    _camera_elevations = np.deg2rad(camera_elevations.reshape(-1))
    _camera_positions  = _az_el_to_position(_camera_azimuths, _camera_elevations) * _camera_distances
    _create_cameras(_camera_positions)


def execute_loading_asset(asset_filepath):
    if asset_filepath.split(".")[-1].lower() == "glb":
        _load_gltf(asset_filepath)

        _update_material_for_generating_images()
        _create_compositor_for_generating_images()


def _execute_rendering(camera_resolutions, dst_folderpath, envlight_filepaths=None, use_white_envlight=False, white_env_strength=1.0):
    i = 0
    for object in bpy.data.objects:
        if isinstance(object.data, bpy.types.Camera):
            if use_white_envlight:
                execute_setting_white_envlight(white_env_strength)
            elif envlight_filepaths != None and len(envlight_filepaths) > 0:
                _create_lights()
                _create_envlight()
                execute_setting_envlight(envlight_filepaths)
            bpy.context.scene.camera = object
            bpy.context.scene.render.resolution_x = camera_resolutions[i][0]
            bpy.context.scene.render.resolution_y = camera_resolutions[i][1]
            bpy.context.scene.render.filepath = os.path.join(dst_folderpath, f"{i:03d}_{camera_resolutions[i][0]}x{camera_resolutions[i][1]}")
            bpy.ops.render.render(write_still=True)
            i = i + 1


def execute_export_camera_metadata(camera_resolutions, dst_folderpath):
    cameras = []
    i = 0
    for obj in bpy.data.objects:
        if isinstance(obj.data, bpy.types.Camera):
            width = int(camera_resolutions[i][0])
            height = int(camera_resolutions[i][1])

            fx = 0.5 * width / math.tan(obj.data.angle_x * 0.5)
            fy = 0.5 * height / math.tan(obj.data.angle_y * 0.5)
            cx = width * 0.5
            cy = height * 0.5

            c2w = obj.matrix_world.copy()
            w2c = c2w.inverted()

            cameras.append({
                "view_id": i,
                "image_name": f"{i:03d}_{width}x{height}.png",
                "width": width,
                "height": height,
                "intrinsics": {
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "K": [
                        [float(fx), 0.0, float(cx)],
                        [0.0, float(fy), float(cy)],
                        [0.0, 0.0, 1.0],
                    ],
                },
                "extrinsics": {
                    "c2w": [[float(v) for v in row] for row in c2w],
                    "w2c": [[float(v) for v in row] for row in w2c],
                },
            })
            i = i + 1

    os.makedirs(dst_folderpath, exist_ok=True)
    with open(os.path.join(dst_folderpath, "camera_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"cameras": cameras}, f, indent=2)


def execute_export_pointcloud(dst_folderpath):
    points = []
    normals = []

    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.visible_get():
            continue

        mesh = obj.data
        world_mat = obj.matrix_world
        normal_mat = world_mat.to_3x3().inverted().transposed()

        for v in mesh.vertices:
            world_pos = world_mat @ v.co
            world_nrm = (normal_mat @ v.normal).normalized()
            points.append((world_pos.x, world_pos.y, world_pos.z))
            normals.append((world_nrm.x, world_nrm.y, world_nrm.z))

    if len(points) == 0:
        return

    os.makedirs(dst_folderpath, exist_ok=True)
    ply_path = os.path.join(dst_folderpath, "pointcloud.ply")
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")

        for (x, y, z), (nx, ny, nz) in zip(points, normals):
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {nx:.8f} {ny:.8f} {nz:.8f}\n")


# +++ +++ +++ +++
# Special Use
# +++ +++ +++ +++
def compute_target_filepaths(src_folderpath, err_filepath):
    src_filepaths = glob.glob(os.path.join(src_folderpath, "*", "*.glb"))

    src_info = {}
    for path in src_filepaths:
        id = os.path.basename(path).split(".")[0]
        src_info[id] = path

    src_ids = set(src_info.keys())
    err_ids = set()
    if os.path.exists(err_filepath):
        with open(err_filepath, "r", encoding="utf-8") as f:
            err_ids = set([line.strip() for line in f.readlines()])

    tgt_ids = src_ids - err_ids

    tgt_filepaths = []
    for id in tgt_ids:
        if os.path.exists(src_info[id]):
            tgt_filepaths.append(src_info[id])

    return sorted(tgt_filepaths)


def _update_material_for_generating_images():
    #assert len(bpy.data.materials) == 1 and bpy.data.materials[0].use_nodes, "One using-node material is expected."
    # material = bpy.data.materials[0]
    def _update_material_for_generating_images_internal(material):
        if not material:
            return

        flag_node_name = "3DAT_SkipMaterialSetupFlag"
        for node in material.node_tree.nodes:
            if node.name == flag_node_name:
                print(f"Skip material {material.name} because it has already been set up.")
                return
        flag_node = material.node_tree.nodes.new("ShaderNodeValue")
        flag_node.name = flag_node_name

        # Rename nodes
        found_bsdf = False
        for node in material.node_tree.nodes:
            if type(node) == bpy.types.ShaderNodeBsdfPrincipled:
                found_bsdf = True
                node.name = "3DAT_BsdfPrincipledNode"
            if type(node) == bpy.types.ShaderNodeOutputMaterial:
                node.name = "3DAT_MaterialOutputNode"
        assert found_bsdf, "A principled BSDF material is expected."

        # Create nodes
        value_node = material.node_tree.nodes.new("ShaderNodeValue")
        value_node.name = "3DAT_ValueNode"
        value_node.outputs[0].default_value = 0.0
        geometry_node = material.node_tree.nodes.new("ShaderNodeNewGeometry")

        vectortransform_node = material.node_tree.nodes.new("ShaderNodeVectorTransform")
        vectortransform_node.vector_type  = "NORMAL"
        vectortransform_node.convert_from = "WORLD"
        vectortransform_node.convert_to   = "CAMERA"
        multiply_node = material.node_tree.nodes.new("ShaderNodeVectorMath")
        multiply_node.operation = "MULTIPLY"
        multiply_node.inputs[1].default_value = Vector((1.0, 1.0, -1.0))
        multiplyadd_node2 = material.node_tree.nodes.new("ShaderNodeVectorMath")
        multiplyadd_node2.operation = "MULTIPLY_ADD"
        multiplyadd_node2.inputs[1].default_value = Vector((0.5, 0.5, 0.5))
        multiplyadd_node2.inputs[2].default_value = Vector((0.5, 0.5, 0.5))

        vectortransform_node2 = material.node_tree.nodes.new("ShaderNodeVectorTransform")
        vectortransform_node2.vector_type  = "NORMAL"
        vectortransform_node2.convert_from = "WORLD"
        vectortransform_node2.convert_to   = "CAMERA"
        multiply_node2 = material.node_tree.nodes.new("ShaderNodeVectorMath")
        multiply_node2.operation = "MULTIPLY"
        multiply_node2.inputs[1].default_value = Vector((1.0, 1.0, -1.0))
        multiplyadd_node3 = material.node_tree.nodes.new("ShaderNodeVectorMath")
        multiplyadd_node3.operation = "MULTIPLY_ADD"
        multiplyadd_node3.inputs[1].default_value = Vector((0.5, 0.5, 0.5))
        multiplyadd_node3.inputs[2].default_value = Vector((0.5, 0.5, 0.5))

        vectortransform_node3 = material.node_tree.nodes.new("ShaderNodeVectorTransform")
        vectortransform_node3.vector_type  = "POINT"
        vectortransform_node3.convert_from = "WORLD"
        vectortransform_node3.convert_to   = "CAMERA"
        multiply_node3 = material.node_tree.nodes.new("ShaderNodeVectorMath")
        multiply_node3.operation = "MULTIPLY"
        multiply_node3.inputs[1].default_value = Vector((-1.0, -1.0, 1.0))
        normalize_node2 = material.node_tree.nodes.new("ShaderNodeVectorMath")
        normalize_node2.operation = "NORMALIZE"
        multiplyadd_node4 = material.node_tree.nodes.new("ShaderNodeVectorMath")
        multiplyadd_node4.operation = "MULTIPLY_ADD"
        multiplyadd_node4.inputs[1].default_value = Vector((0.5, 0.5, 0.5))
        multiplyadd_node4.inputs[2].default_value = Vector((0.5, 0.5, 0.5))

        emission_node1 = material.node_tree.nodes.new("ShaderNodeEmission")
        emission_node1.name = "3DAT_BasecolorEmissionNode"
        emission_node2 = material.node_tree.nodes.new("ShaderNodeEmission")
        emission_node2.name = "3DAT_MetallicEmissionNode"
        emission_node3 = material.node_tree.nodes.new("ShaderNodeEmission")
        emission_node3.name = "3DAT_RoughnessEmissionNode"
        emission_node4 = material.node_tree.nodes.new("ShaderNodeEmission")
        emission_node4.name = "3DAT_NormalEmissionNode"
        emission_node5 = material.node_tree.nodes.new("ShaderNodeEmission")
        emission_node5.name = "3DAT_Normal2EmissionNode"
        emission_node6 = material.node_tree.nodes.new("ShaderNodeEmission")
        emission_node6.name = "3DAT_Normal3EmissionNode"
        emission_node7 = material.node_tree.nodes.new("ShaderNodeEmission")
        emission_node7.name = "3DAT_ViewEmissionNode"
        emission_node8 = material.node_tree.nodes.new("ShaderNodeEmission")
        emission_node8.name = "3DAT_DepthEmissionNode"
        emission_node9 = material.node_tree.nodes.new("ShaderNodeEmission")
        emission_node9.name = "3DAT_MaskEmissionNode"

        separatexyz_node = material.node_tree.nodes.new("ShaderNodeSeparateXYZ")
        multiply_node4 = material.node_tree.nodes.new("ShaderNodeMath")
        multiply_node4.operation = "MULTIPLY"
        multiply_node4.inputs[1].default_value = -1.0

        # Create links and set defaults
        material.node_tree.links.new(geometry_node.outputs[1],         vectortransform_node.inputs[0])
        material.node_tree.links.new(vectortransform_node.outputs[0],  multiply_node.inputs[0])
        material.node_tree.links.new(multiply_node.outputs[0],         multiplyadd_node2.inputs[0])
        material.node_tree.links.new(multiplyadd_node2.outputs[0],     emission_node5.inputs[0])
        material.node_tree.links.new(vectortransform_node2.outputs[0], multiply_node2.inputs[0])
        material.node_tree.links.new(multiply_node2.outputs[0],        multiplyadd_node3.inputs[0])
        material.node_tree.links.new(multiplyadd_node3.outputs[0],     emission_node6.inputs[0])
        material.node_tree.links.new(geometry_node.outputs[0],         vectortransform_node3.inputs[0])
        material.node_tree.links.new(vectortransform_node3.outputs[0], multiply_node3.inputs[0])
        material.node_tree.links.new(multiply_node3.outputs[0],        normalize_node2.inputs[0])
        material.node_tree.links.new(normalize_node2.outputs[0],       multiplyadd_node4.inputs[0])
        material.node_tree.links.new(multiplyadd_node4.outputs[0],     emission_node7.inputs[0])
        material.node_tree.links.new(vectortransform_node3.outputs[0], separatexyz_node.inputs[0])
        material.node_tree.links.new(separatexyz_node.outputs[2],      multiply_node4.inputs[0])
        material.node_tree.links.new(multiply_node4.outputs[0],        emission_node8.inputs[0])
        emission_node9.inputs[0].default_value = Vector((1.0, 1.0, 1.0, 1.0))

        found_basecolor = False
        found_metallic  = False
        found_roughness = False
        found_normal    = False
        for link in material.node_tree.links:
            if link.to_socket == material.node_tree.nodes["3DAT_BsdfPrincipledNode"].inputs[0]:
                found_basecolor = True
                material.node_tree.links.new(link.from_socket, material.node_tree.nodes["3DAT_BasecolorEmissionNode"].inputs[0])
            if link.to_socket == material.node_tree.nodes["3DAT_BsdfPrincipledNode"].inputs[1]:
                found_metallic = True
                material.node_tree.links.new(link.from_socket, material.node_tree.nodes["3DAT_MetallicEmissionNode"].inputs[0])
            if link.to_socket == material.node_tree.nodes["3DAT_BsdfPrincipledNode"].inputs[2]:
                found_roughness = True
                material.node_tree.links.new(link.from_socket, material.node_tree.nodes["3DAT_RoughnessEmissionNode"].inputs[0])
            if link.to_socket == material.node_tree.nodes["3DAT_BsdfPrincipledNode"].inputs[5]:
                found_normal = True
                # Create nodes to transform normal from world space to geometry tangent space
                crossproduct_node = material.node_tree.nodes.new("ShaderNodeVectorMath")
                crossproduct_node.operation = "CROSS_PRODUCT"
                normalize_node = material.node_tree.nodes.new("ShaderNodeVectorMath")
                normalize_node.operation = "NORMALIZE"
                dotproduct_node1 = material.node_tree.nodes.new("ShaderNodeVectorMath")
                dotproduct_node1.operation = "DOT_PRODUCT"
                dotproduct_node2 = material.node_tree.nodes.new("ShaderNodeVectorMath")
                dotproduct_node2.operation = "DOT_PRODUCT"
                dotproduct_node3 = material.node_tree.nodes.new("ShaderNodeVectorMath")
                dotproduct_node3.operation = "DOT_PRODUCT"
                combinexyz_node = material.node_tree.nodes.new("ShaderNodeCombineXYZ")
                multiplyadd_node = material.node_tree.nodes.new("ShaderNodeVectorMath")
                multiplyadd_node.operation = "MULTIPLY_ADD"
                multiplyadd_node.inputs[1].default_value = Vector((0.5, 0.5, 0.5))
                multiplyadd_node.inputs[2].default_value = Vector((0.5, 0.5, 0.5))

                # Link nodes
                material.node_tree.links.new(geometry_node.outputs[1],  crossproduct_node.inputs[0])
                material.node_tree.links.new(geometry_node.outputs[2],  crossproduct_node.inputs[1])
                material.node_tree.links.new(crossproduct_node.outputs[0], normalize_node.inputs[0])
                material.node_tree.links.new(geometry_node.outputs[2],  dotproduct_node1.inputs[0])
                material.node_tree.links.new(normalize_node.outputs[0], dotproduct_node2.inputs[0])
                material.node_tree.links.new(geometry_node.outputs[1],  dotproduct_node3.inputs[0])
                material.node_tree.links.new(link.from_socket, dotproduct_node1.inputs[1])
                material.node_tree.links.new(link.from_socket, dotproduct_node2.inputs[1])
                material.node_tree.links.new(link.from_socket, dotproduct_node3.inputs[1])
                material.node_tree.links.new(dotproduct_node1.outputs["Value"], combinexyz_node.inputs[0])
                material.node_tree.links.new(dotproduct_node2.outputs["Value"], combinexyz_node.inputs[1])
                material.node_tree.links.new(dotproduct_node3.outputs["Value"], combinexyz_node.inputs[2])
                material.node_tree.links.new(combinexyz_node.outputs[0],  multiplyadd_node.inputs[0])
                material.node_tree.links.new(multiplyadd_node.outputs[0], material.node_tree.nodes["3DAT_NormalEmissionNode"].inputs[0])
                material.node_tree.links.new(link.from_socket, vectortransform_node2.inputs[0])

        assert found_basecolor, "The basecolor of the Principled BSDF node of material is not expected to be a default value."
        if not found_metallic:
            emission_node2.inputs[0].default_value = Vector((0.0, 0.0, 0.0, 1.0))
        if not found_roughness:
            emission_node3.inputs[0].default_value = Vector((0.5, 0.5, 0.5, 1.0))
        if not found_normal:
            emission_node4.inputs[0].default_value = Vector((0.5, 0.5, 1.0, 1.0))
            material.node_tree.links.new(multiplyadd_node2.outputs[0], emission_node6.inputs[0]) # If no normal map is found, high-poly and low-poly are the same one.

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_images_internal(material)


def _update_material_for_generating_occupation_images():
    def _update_material_for_generating_occupation_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_ValueNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_occupation_images_internal(material)


def _update_material_for_generating_basecolor_images():
    def _update_material_for_generating_basecolor_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_BasecolorEmissionNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_basecolor_images_internal(material)


def _update_material_for_generating_metallic_images():
    def _update_material_for_generating_metallic_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_MetallicEmissionNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_metallic_images_internal(material)


def _update_material_for_generating_roughness_images():
    def _update_material_for_generating_roughness_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_RoughnessEmissionNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_roughness_images_internal(material)


def _update_material_for_generating_normal_images():
    def _update_material_for_generating_normal_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_NormalEmissionNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_normal_images_internal(material)


def _update_material_for_generating_normal2_images():
    def _update_material_for_generating_normal2_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_Normal2EmissionNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_normal2_images_internal(material)


def _update_material_for_generating_normal3_images():
    def _update_material_for_generating_normal3_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_Normal3EmissionNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_normal3_images_internal(material)


def _update_material_for_generating_shaded_images():
    def _update_material_for_generating_shaded_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_BsdfPrincipledNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_shaded_images_internal(material)


def _update_material_for_generating_view_images():
    def _update_material_for_generating_view_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_ViewEmissionNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_view_images_internal(material)


def _update_material_for_generating_depth_images():
    def _update_material_for_generating_depth_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_DepthEmissionNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_depth_images_internal(material)


def _update_material_for_generating_mask_images():
    def _update_material_for_generating_mask_images_internal(material):
        if not material:
            return
        material.node_tree.links.new(material.node_tree.nodes["3DAT_MaskEmissionNode"].outputs[0], material.node_tree.nodes["3DAT_MaterialOutputNode"].inputs[0])

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.visible_get():
            for material in obj.data.materials:
                _update_material_for_generating_mask_images_internal(material)


def execute_rendering(camera_resolutions, dst_folderpath, envlight_filepaths, need_occupation, need_basecolor, need_metallic, need_roughness, need_normal, need_normal2, need_normal3, need_view, need_shaded, shaded_exposure=0.8, use_white_envlight=False, white_env_strength=1.0, need_depth=False, need_mask=False, normal_png_only=False):
    # Render occupation
    if need_occupation:
        folderpath = os.path.join(dst_folderpath, "occupation")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Raw"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode  = "BW"
        bpy.context.scene.render.image_settings.color_depth = "8"

        _update_material_for_generating_occupation_images()
        _execute_rendering(camera_resolutions, folderpath)

    # Render basecolor
    if need_basecolor:
        folderpath = os.path.join(dst_folderpath, "basecolor")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Standard"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode  = "RGB"
        bpy.context.scene.render.image_settings.color_depth = "8"

        _update_material_for_generating_basecolor_images()
        _execute_rendering(camera_resolutions, folderpath)

    # Render metallic
    if need_metallic:
        folderpath = os.path.join(dst_folderpath, "metallic")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Raw"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode  = "BW"
        bpy.context.scene.render.image_settings.color_depth = "8"

        _update_material_for_generating_metallic_images()
        _execute_rendering(camera_resolutions, folderpath)

    # Render roughness
    if need_roughness:
        folderpath = os.path.join(dst_folderpath, "roughness")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Raw"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode  = "BW"
        bpy.context.scene.render.image_settings.color_depth = "8"

        _update_material_for_generating_roughness_images()
        _execute_rendering(camera_resolutions, folderpath)

    # Render normal, which is normal in geometry tangent space of high-poly geometry
    if need_normal and not normal_png_only:
        folderpath = os.path.join(dst_folderpath, "normal")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Raw"
        bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
        bpy.context.scene.render.image_settings.color_mode  = "RGB"
        bpy.context.scene.render.image_settings.color_depth = "16"
        bpy.context.scene.render.image_settings.exr_codec   = "ZIP"

        _update_material_for_generating_normal_images()
        _execute_rendering(camera_resolutions, folderpath)

    if need_normal:
        # Extra PNG export for quick inspection / pipelines that require png.
        folderpath_png = os.path.join(dst_folderpath, "normal_png")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Raw"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode  = "RGB"
        bpy.context.scene.render.image_settings.color_depth = "8"

        _update_material_for_generating_normal_images()
        _execute_rendering(camera_resolutions, folderpath_png)

    # Render normal2, which is normal in view space of low-poly geometry
    if need_normal2:
        folderpath = os.path.join(dst_folderpath, "normal2")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Raw"
        # comment this to generate png file for normal2
        # bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
        bpy.context.scene.render.image_settings.color_mode  = "RGB"
        # bpy.context.scene.render.image_settings.color_depth = "16"
        # bpy.context.scene.render.image_settings.exr_codec   = "ZIP"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_depth = "8"

        _update_material_for_generating_normal2_images()
        _execute_rendering(camera_resolutions, folderpath)

    # Render normal3, which is normal in view space of high-poly geometry
    if need_normal3:
        folderpath = os.path.join(dst_folderpath, "normal3")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        # comment this to generate png file for normal3
        # bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
        # bpy.context.scene.render.image_settings.color_depth = "16"
        # bpy.context.scene.render.image_settings.exr_codec   = "ZIP"
        bpy.context.scene.view_settings.view_transform      = "Raw"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode  = "RGB"
        bpy.context.scene.render.image_settings.color_depth = "8"

        _update_material_for_generating_normal3_images()
        _execute_rendering(camera_resolutions, folderpath)

    # Render view direction, which points from a shading point to the camera in view space
    if need_view:
        folderpath = os.path.join(dst_folderpath, "view")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Raw"
        bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
        bpy.context.scene.render.image_settings.color_mode  = "RGB"
        bpy.context.scene.render.image_settings.color_depth = "16"
        bpy.context.scene.render.image_settings.exr_codec   = "ZIP"

        _update_material_for_generating_view_images()
        _execute_rendering(camera_resolutions, folderpath)

    # Render depth in camera space (positive forward depth)
    if need_depth:
        folderpath = os.path.join(dst_folderpath, "depth")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Raw"
        bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
        bpy.context.scene.render.image_settings.color_mode  = "BW"
        bpy.context.scene.render.image_settings.color_depth = "16"
        bpy.context.scene.render.image_settings.exr_codec   = "ZIP"

        _update_material_for_generating_depth_images()
        _execute_rendering(camera_resolutions, folderpath)

    # Render foreground mask (white object on black background)
    if need_mask:
        folderpath = os.path.join(dst_folderpath, "mask")
        bpy.context.scene.view_settings.exposure            = 0.0
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Raw"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode  = "BW"
        bpy.context.scene.render.image_settings.color_depth = "8"

        _update_material_for_generating_mask_images()
        _execute_rendering(camera_resolutions, folderpath)

    # Render shaded
    if need_shaded:
        folderpath = os.path.join(dst_folderpath, "shaded")
        bpy.context.scene.view_settings.exposure            = shaded_exposure
        bpy.context.scene.view_settings.gamma               = 1.0
        bpy.context.scene.view_settings.view_transform      = "Standard"
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode  = "RGB"
        bpy.context.scene.render.image_settings.color_depth = "8"

        _update_material_for_generating_shaded_images()
        _execute_rendering(camera_resolutions, folderpath, envlight_filepaths, use_white_envlight, white_env_strength)


def get_optimal_camera_resolutions(origin_camera_resolutions, tmp_folderpath):
    target_camera_resolutions = [
        (1328, 1328),  # 1:1
        (1056, 1584),  # 2:3
        (1584, 1056),  # 3:2
        (1104, 1472),  # 3:4
        (1472, 1104),  # 4:3
        (928, 1664),  # 9:16
        (1664, 928),  # 16:9
    ]
    optimal_camera_resolutions = []

    i = 0
    for object in bpy.data.objects:
        if isinstance(object.data, bpy.types.Camera):
            filepath = os.path.join(os.path.join(tmp_folderpath, "occupation"), f"{i:03d}_{origin_camera_resolutions[i][0]}x{origin_camera_resolutions[i][1]}.png")
            image = cv2.bitwise_not(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE))
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects_xmin = []
            rects_ymin = []
            rects_xmax = []
            rects_ymax = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                rects_xmin.append(x)
                rects_ymin.append(y)
                rects_xmax.append(x + w)
                rects_ymax.append(y + h)
            bounding_width = max(rects_xmax) - min(rects_xmin)
            bounding_height = max(rects_ymax) - min(rects_ymin)
            bounding_aspect_ratio = bounding_width / bounding_height
            optimal_camera_resolution = None
            min_diff = float('inf')

            for target_camera_resolution in target_camera_resolutions:
                # Calculate target aspect ratio
                target_width, target_height = target_camera_resolution
                target_aspect_ratio = target_width / target_height

                # Calculate the difference in aspect ratios
                diff = abs(bounding_aspect_ratio - target_aspect_ratio)

                # Update the closest resolution if the current one is closer
                if diff < min_diff:
                    min_diff = diff
                    optimal_camera_resolution = (target_width, target_height)

            optimal_camera_resolutions.append(optimal_camera_resolution)

            i = i + 1

    return np.asarray(optimal_camera_resolutions)


def execute_updating_camera_fov(camera_resolutions, tmp_folderpath):
    def _compute_fov(margin, height, old_fov):
        return 2.0 * math.atan((0.5 * height - margin) / (0.5 * height / math.tan(0.5 * old_fov)))

    execute_rendering(camera_resolutions, tmp_folderpath, None, True, False, False, False, False, False, False, False, False)

    i = 0
    for object in bpy.data.objects:
        if isinstance(object.data, bpy.types.Camera):
            filepath = os.path.join(os.path.join(tmp_folderpath, "occupation"), f"{i:03d}_{camera_resolutions[i][0]}x{camera_resolutions[i][1]}.png")
            image    = cv2.bitwise_not(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE))
            height, width = image.shape
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects_xmin = []
            rects_ymin = []
            rects_xmax = []
            rects_ymax = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                rects_xmin.append(x)
                rects_ymin.append(y)
                rects_xmax.append(x + w)
                rects_ymax.append(y + h)
            rect = [min(rects_xmin), min(rects_ymin), max(rects_xmax), max(rects_ymax)]
            margin_left   = rect[0]
            margin_top    = rect[1]
            margin_right  = width  - rect[2]
            margin_bottom = height - rect[3]
            margin_horizontal = min(margin_left, margin_right)
            margin_vertical   = min(margin_top, margin_bottom)

            object.data.lens_unit = "FOV"
            if (margin_horizontal / width) < (margin_vertical / height):
                # Adjust FOV based on width direction
                angle_width  = 2.0 * math.atan(width / height * math.tan(object.data.angle / 2.0))
                angle_width  = _compute_fov(margin_horizontal, width, angle_width)
                object.data.angle = 2.0 * math.atan(height / width * math.tan(angle_width / 2.0))
            else:
                # Adjust FOV based on height direction
                object.data.angle = _compute_fov(margin_vertical, height, object.data.angle)

            i = i + 1


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folderpath",  type=str, required=True)
    parser.add_argument("--dst_folderpath",  type=str, required=True)
    parser.add_argument("--tmp_folderpath",  type=str, required=True)
    parser.add_argument("--light_folderpath", type=str, default="")
    parser.add_argument("--need_occupation", action="store_true")
    parser.add_argument("--need_basecolor",  action="store_true")
    parser.add_argument("--need_metallic",   action="store_true")
    parser.add_argument("--need_roughness",  action="store_true")
    parser.add_argument("--need_normal",     action="store_true") # normal in geometry tangent space of high-poly geometry
    parser.add_argument("--normal_png_only", action="store_true") # render only normal_png and skip normal EXR
    parser.add_argument("--need_normal2",    action="store_true") # normal in view space of low-poly geometry
    parser.add_argument("--need_normal3",    action="store_true") # normal in view space of high-poly geometry
    parser.add_argument("--need_view",       action="store_true") # view-direction from a shading point to camera in view space
    parser.add_argument("--need_shaded",     action="store_true")
    parser.add_argument("--need_camera_meta", action="store_true")
    parser.add_argument("--need_pointcloud", action="store_true")
    parser.add_argument("--need_depth",      action="store_true")
    parser.add_argument("--need_mask",       action="store_true")
    parser.add_argument("--use_white_envlight", action="store_true")
    parser.add_argument("--white_env_strength", type=float, default=2.0)
    parser.add_argument("--shaded_exposure", type=float, default=0.8)
    parser.add_argument("--random_seed",     type=int, default=42)
    parser.add_argument("--camera_count",    type=int, default=12)
    parser.add_argument("--eevee_render_samples", type=int, default=64)
    parser.add_argument("--eevee_disable_gtao", action="store_true")
    parser.add_argument("--eevee_disable_ssr", action="store_true")
    parser.add_argument("--fast_mode", action="store_true")
    argv = sys.argv[sys.argv.index("--")+1:]
    args = parser.parse_args(argv)

    # Set up variables
    src_folderpath  = args.src_folderpath
    dst_folderpath  = args.dst_folderpath
    tmp_folderpath  = args.tmp_folderpath
    env_folderpath  = args.light_folderpath
    need_occupation = args.need_occupation
    need_basecolor  = args.need_basecolor
    need_metallic   = args.need_metallic
    need_roughness  = args.need_roughness
    need_normal     = args.need_normal
    normal_png_only = args.normal_png_only
    need_normal2    = args.need_normal2
    need_normal3    = args.need_normal3
    need_view       = args.need_view
    need_shaded     = args.need_shaded
    need_camera_meta = args.need_camera_meta
    need_pointcloud = args.need_pointcloud
    need_depth      = args.need_depth
    need_mask       = args.need_mask
    use_white_envlight = args.use_white_envlight
    white_env_strength = args.white_env_strength
    shaded_exposure = args.shaded_exposure
    random_seed     = args.random_seed
    camera_count    = args.camera_count
    eevee_render_samples = args.eevee_render_samples
    eevee_disable_gtao = args.eevee_disable_gtao
    eevee_disable_ssr = args.eevee_disable_ssr
    fast_mode = args.fast_mode

    if normal_png_only:
        need_normal = True

    log_filepath    = os.path.join(tmp_folderpath, f"{timestamp}.log")
    error_filepath  = os.path.join(tmp_folderpath, "error.txt")

    os.makedirs(dst_folderpath, exist_ok=True)
    os.makedirs(tmp_folderpath, exist_ok=True)

    # Ensure reproducible random view sampling and env-light randomness across runs.
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Set up cameras
    auto_multiple_resolutions = False
    camera_resolutions = np.asarray([(1024, 1024) for _ in range(10 + 2)])
    camera_distances   = np.asarray([5.0 for _ in range(10 + 2)])
    camera_azimuths    = np.asarray([(360.0 / 10 * i) for i in range(10)] + [-45.0, 45.0])
    camera_elevations  = np.asarray([30.0] * 10 + [-30.0, -30.0])

    if fast_mode and eevee_render_samples == 64:
        eevee_render_samples = 8
    if fast_mode:
        eevee_disable_gtao = True
        eevee_disable_ssr = True

    camera_count = max(1, min(camera_count, len(camera_resolutions)))
    eevee_shadow_cube_size = "512" if fast_mode else "1024"
    eevee_shadow_cascade_size = "1024" if fast_mode else "2048"

    # Start generation
    logger = get_logger(log_filepath)
    logger.info(f"Start generation (random_seed={random_seed})")
    logger.info(
        "Render settings: "
        f"camera_count={camera_count}, "
        f"eevee_render_samples={eevee_render_samples}, "
        f"gtao={not eevee_disable_gtao}, "
        f"ssr={not eevee_disable_ssr}, "
        f"shaded_exposure={shaded_exposure}, "
        f"white_env_strength={white_env_strength}, "
        f"fast_mode={fast_mode}"
    )
    log_runtime_gpu_info(logger)

    tgt_filepaths = compute_target_filepaths(src_folderpath, error_filepath)
    if len(tgt_filepaths) > 0:
        envlight_filepaths = glob.glob(os.path.join(env_folderpath, "*.exr")) if env_folderpath else []

        for tgt_filepath in tgt_filepaths:
            tgt_filename   = os.path.basename(tgt_filepath).split(".")[0]
            tgt_folderpath = os.path.join(os.path.join(dst_folderpath, os.path.basename(os.path.dirname(tgt_filepath))), tgt_filename)
            tgt_statepath  = os.path.join(tgt_folderpath, "success.txt")

            if os.path.exists(tgt_statepath):
                logger.info(f"Skipped {tgt_filename}")
            else:
                try:
                    start_time = time.time()
                    os.makedirs(tgt_folderpath, exist_ok=True)

                    # Make per-asset randomness deterministic, even when resuming/skipping.
                    random.seed(f"{random_seed}:{tgt_filename}")

                    # Start generation for an asset file
                    indices = random.sample(range(len(camera_resolutions)), camera_count)
                    current_resolutions = camera_resolutions[indices]
                    current_distances   = camera_distances[indices]
                    current_azimuths    = camera_azimuths[indices]
                    current_elevations  = camera_elevations[indices]

                    execute_setting_scene(
                        current_resolutions,
                        current_distances,
                        current_azimuths,
                        current_elevations,
                        eevee_render_samples,
                        not eevee_disable_gtao,
                        not eevee_disable_ssr,
                        eevee_shadow_cube_size,
                        eevee_shadow_cascade_size,
                    )
                    execute_loading_asset(tgt_filepath)

                    if need_occupation:
                        if auto_multiple_resolutions:
                            execute_rendering(current_resolutions, tmp_folderpath, None, True, False, False, False, False, False, False, False, False)
                            current_resolutions = get_optimal_camera_resolutions(current_resolutions, tmp_folderpath)
                        execute_updating_camera_fov(current_resolutions, tmp_folderpath)

                    if need_camera_meta:
                        execute_export_camera_metadata(current_resolutions, tgt_folderpath)

                    if need_pointcloud:
                        execute_export_pointcloud(tgt_folderpath)

                    execute_rendering(current_resolutions, tgt_folderpath, envlight_filepaths, False, need_basecolor, need_metallic, need_roughness, need_normal, need_normal2, need_normal3, need_view, need_shaded, shaded_exposure, use_white_envlight, white_env_strength, need_depth, need_mask, normal_png_only)

                    with open(tgt_statepath, "w") as f:
                        pass

                    end_time = time.time()
                    logger.info(f"Processing {tgt_filename} took {end_time - start_time:.2f} seconds")
                except Exception as e:
                    with open(error_filepath, "a", encoding="utf-8") as f:
                        f.write(tgt_filename + "\n")
                    logger.error(f"Processing {tgt_filename} encountered an error: {e}")

    logger.info("End generation")
