import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import bpy
import bpy_extras.object_utils
import mathutils
import numpy as np


BIG_SCREEN_IMAGE = Path("/root/autodl-tmp/DJLaction4/image.png")
SMALL_SCREEN_IMAGE = Path("/root/autodl-tmp/DJLaction4/image2.png")
DJI_LABEL_IMAGE = None
BIG_SCREEN_INSET = 0.08
SMALL_SCREEN_SIDE_RATIO = 0.88
SMALL_SCREEN_X_MARGIN_RATIO = 0.075
SMALL_SCREEN_Y_SCALE = 1.0
SMALL_SCREEN_OVERALL_SCALE = 0.95
SMALL_SCREEN_OFFSET_RATIO = 0.0015
DJI_LABEL_YMAX_WIDTH_RATIO = 0.51
DJI_LABEL_YMAX_CENTER_X_RATIO = 0.56
DJI_LABEL_YMAX_CENTER_Z_RATIO = 0.84
DJI_LABEL_YMAX_OFFSET_RATIO = 0.0012


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(path: Path):
    before = set(bpy.data.objects)
    if path.suffix.lower() == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(path))
    elif path.suffix.lower() == ".obj":
        bpy.ops.import_scene.obj(filepath=str(path))
    elif path.suffix.lower() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh format: {path}")
    after = set(bpy.data.objects)
    new_objects = [obj for obj in after - before if obj.type == "MESH"]
    if not new_objects:
        raise RuntimeError(f"No mesh object imported from {path}")
    if len(new_objects) == 1:
        return new_objects[0]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in new_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = new_objects[0]
    bpy.ops.object.join()
    return bpy.context.object



def normalize_object(obj, target_diameter=120.0):
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    center = sum(corners, mathutils.Vector((0.0, 0.0, 0.0))) / 8.0
    dims = mathutils.Vector((
        max(c.x for c in corners) - min(c.x for c in corners),
        max(c.y for c in corners) - min(c.y for c in corners),
        max(c.z for c in corners) - min(c.z for c in corners),
    ))
    scale = target_diameter / max(dims.length, 1e-6)
    obj.location -= center
    obj.scale = tuple(float(v * scale) for v in obj.scale)
    bpy.context.view_layer.update()


def local_bbox_corners(obj):
    xs = [corner[0] for corner in obj.bound_box]
    ys = [corner[1] for corner in obj.bound_box]
    zs = [corner[2] for corner in obj.bound_box]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    return [
        mathutils.Vector((minx, miny, minz)),
        mathutils.Vector((minx, miny, maxz)),
        mathutils.Vector((minx, maxy, minz)),
        mathutils.Vector((minx, maxy, maxz)),
        mathutils.Vector((maxx, miny, minz)),
        mathutils.Vector((maxx, miny, maxz)),
        mathutils.Vector((maxx, maxy, minz)),
        mathutils.Vector((maxx, maxy, maxz)),
    ]




def make_principled_material(name, color, roughness=0.55, emission=None, emission_strength=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    mat.use_backface_culling = False
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        if "Emission" in bsdf.inputs and emission is not None:
            bsdf.inputs["Emission"].default_value = emission
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def make_image_emission_material(name, image_path, emission_strength=0.9, roughness=0.18):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    mat.blend_method = "BLEND"
    mat.use_screen_refraction = False
    mat.show_transparent_back = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path))
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf.inputs["Roughness"].default_value = roughness
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission" in bsdf.inputs:
        links.new(tex.outputs["Color"], bsdf.inputs["Emission"])
    elif "Emission Color" in bsdf.inputs:
        links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    if "Alpha" in bsdf.inputs:
        links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def add_screen_image_on_zmin_face(obj, image_path, inset=0.08, offset_ratio=0.006):
    xs = [corner[0] for corner in obj.bound_box]
    ys = [corner[1] for corner in obj.bound_box]
    zs = [corner[2] for corner in obj.bound_box]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    dx, dy, dz = maxx - minx, maxy - miny, maxz - minz
    pad_x = dx * inset
    pad_z = dz * inset
    z = minz - max(dx, dy, dz) * offset_ratio
    verts = [
        (minx + pad_x, miny + dy * inset, z),
        (minx + pad_x, maxy - dy * inset, z),
        (maxx - pad_x, maxy - dy * inset, z),
        (maxx - pad_x, miny + dy * inset, z),
    ]
    faces = [(0, 1, 2, 3)]
    mesh = bpy.data.meshes.new("screenImageMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.polygons[0].loop_indices, [(1, 0), (1, 1), (0, 1), (0, 0)]):
        uv_layer.data[loop].uv = uv

    plane = bpy.data.objects.new("screen_image_zmin_face", mesh)
    bpy.context.collection.objects.link(plane)
    plane.parent = obj
    plane.data.materials.append(make_image_emission_material("screen_image_material", image_path))
    return plane


def add_front_small_screen_on_zmax_face(
    obj,
    image_path,
    side_ratio=0.88,
    x_margin_ratio=0.075,
    y_scale=1.0,
    overall_scale=0.95,
    offset_ratio=0.0015,
):
    xs = [corner[0] for corner in obj.bound_box]
    ys = [corner[1] for corner in obj.bound_box]
    zs = [corner[2] for corner in obj.bound_box]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    dx, dy, dz = maxx - minx, maxy - miny, maxz - minz
    max_dim = max(dx, dy, dz)

    base_width_x = min(dy * side_ratio, dx * 0.42)
    base_height_y = min(dy * side_ratio * y_scale, dy * 0.96)
    cx = minx + dx * x_margin_ratio + base_width_x * 0.5
    cy = (miny + maxy) * 0.5
    width_x = base_width_x * overall_scale
    height_y = base_height_y * overall_scale
    x_lo = cx - width_x * 0.5
    x_hi = cx + width_x * 0.5
    y_lo = cy - height_y * 0.5
    y_hi = cy + height_y * 0.5
    local_surface_z = []
    for vertex in obj.data.vertices:
        x, y, z = vertex.co
        if x_lo <= x <= x_hi and y_lo <= y <= y_hi:
            local_surface_z.append(float(z))
    surface_z = max(local_surface_z) if local_surface_z else maxz
    z = surface_z + max_dim * offset_ratio
    verts = [
        (x_lo, y_lo, z),
        (x_hi, y_lo, z),
        (x_hi, y_hi, z),
        (x_lo, y_hi, z),
    ]
    faces = [(0, 1, 2, 3)]
    mesh = bpy.data.meshes.new("frontSmallScreenImageMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.polygons[0].loop_indices, [(1, 0), (0, 0), (0, 1), (1, 1)]):
        uv_layer.data[loop].uv = uv

    plane = bpy.data.objects.new("front_small_screen_zmax_face", mesh)
    bpy.context.collection.objects.link(plane)
    plane.parent = obj
    plane.data.materials.append(make_image_emission_material("front_small_screen_image_material", image_path, emission_strength=1.05))
    return plane


def add_dji_label_on_ymax_face(obj, text="dji-004", offset_ratio=0.012):
    xs = [corner[0] for corner in obj.bound_box]
    ys = [corner[1] for corner in obj.bound_box]
    zs = [corner[2] for corner in obj.bound_box]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    dx, dy, dz = maxx - minx, maxy - miny, maxz - minz
    max_dim = max(dx, dy, dz)
    y = maxy + max_dim * offset_ratio

    label_w = dx * 0.36
    label_h = dz * 0.14
    cx = minx + dx * 0.56
    cz = maxz - dz * 0.16
    verts = [
        (cx - label_w * 0.5, y, cz - label_h * 0.5),
        (cx - label_w * 0.5, y, cz + label_h * 0.5),
        (cx + label_w * 0.5, y, cz + label_h * 0.5),
        (cx + label_w * 0.5, y, cz - label_h * 0.5),
    ]
    mesh = bpy.data.meshes.new("djiLabelMesh")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    label = bpy.data.objects.new("dji_label_plate", mesh)
    bpy.context.collection.objects.link(label)
    label.parent = obj
    label.data.materials.append(make_principled_material("dji_label_white", (0.92, 0.92, 0.88, 1.0), roughness=0.55))

    font_curve = bpy.data.curves.new("djiLabelTextCurve", "FONT")
    font_curve.body = text
    font_curve.align_x = "CENTER"
    font_curve.align_y = "CENTER"
    font_curve.size = label_h * 0.58
    font_curve.extrude = max_dim * 0.0008
    txt = bpy.data.objects.new("dji_label_text", font_curve)
    bpy.context.collection.objects.link(txt)
    txt.location = (cx, y + max_dim * 0.002, cz)
    txt.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    txt.parent = obj
    txt.data.materials.append(make_principled_material("dji_label_black", (0.045, 0.045, 0.04, 1.0), roughness=0.6))
    return label, txt


def add_dji_label_image_on_ymax_face(
    obj,
    image_path,
    width_ratio=DJI_LABEL_YMAX_WIDTH_RATIO,
    center_x_ratio=DJI_LABEL_YMAX_CENTER_X_RATIO,
    center_z_ratio=DJI_LABEL_YMAX_CENTER_Z_RATIO,
    offset_ratio=DJI_LABEL_YMAX_OFFSET_RATIO,
):
    xs = [corner[0] for corner in obj.bound_box]
    ys = [corner[1] for corner in obj.bound_box]
    zs = [corner[2] for corner in obj.bound_box]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    dx, dy, dz = maxx - minx, maxy - miny, maxz - minz
    max_dim = max(dx, dy, dz)

    aspect = 3.1
    try:
        img = bpy.data.images.load(str(image_path), check_existing=True)
        aspect = max(float(img.size[0]), 1.0) / max(float(img.size[1]), 1.0)
    except Exception:
        img = None

    label_w = dx * width_ratio
    label_h = min(label_w / aspect, dz * 0.20)
    cx = minx + dx * center_x_ratio
    cz = minz + dz * center_z_ratio
    y = maxy + max_dim * offset_ratio

    x_lo, x_hi = cx - label_w * 0.5, cx + label_w * 0.5
    z_lo, z_hi = cz - label_h * 0.5, cz + label_h * 0.5
    verts = [
        (x_lo, y, z_lo),
        (x_hi, y, z_lo),
        (x_hi, y, z_hi),
        (x_lo, y, z_hi),
    ]
    mesh = bpy.data.meshes.new("djiLabelImageMesh")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.polygons[0].loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv_layer.data[loop].uv = uv

    label = bpy.data.objects.new("dji_label_image_ymax_face", mesh)
    bpy.context.collection.objects.link(label)
    label.parent = obj
    label.data.materials.append(make_image_emission_material("dji_label_image_material", image_path, emission_strength=0.25, roughness=0.45))
    return label


def add_local_box(obj, name, center, size, mat):
    sx, sy, sz = size
    cx, cy, cz = center
    verts = [
        (cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz), (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz), (cx - sx, cy + sy, cz + sz),
    ]
    faces = [(0,1,2,3), (4,7,6,5), (0,4,5,1), (1,5,6,2), (2,6,7,3), (3,7,4,0)]
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    decal = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(decal)
    decal.parent = obj
    decal.data.materials.append(mat)
    return decal


def add_photo_decals(obj):
    xs = [corner[0] for corner in obj.bound_box]
    ys = [corner[1] for corner in obj.bound_box]
    zs = [corner[2] for corner in obj.bound_box]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    dx, dy, dz = maxx - minx, maxy - miny, maxz - minz
    screen_mat = make_principled_material("screen_soft_emission", (0.45, 0.50, 0.56, 1.0), roughness=0.35, emission=(0.22, 0.28, 0.34, 1.0), emission_strength=0.45)
    white_mat = make_principled_material("marker_white", (0.92, 0.92, 0.88, 1.0), roughness=0.62)
    black_mat = make_principled_material("marker_black", (0.02, 0.02, 0.018, 1.0), roughness=0.55)
    green_mat = make_principled_material("screen_green", (0.08, 0.85, 0.20, 1.0), roughness=0.45, emission=(0.02, 0.9, 0.14, 1.0), emission_strength=0.9)
    thickness = max(dx, dy, dz) * 0.004
    for face_y, sign in ((maxy + thickness, 1.0), (miny - thickness, -1.0)):
        add_local_box(obj, "screen_panel", (minx + dx * 0.56, face_y, minz + dz * 0.53), (dx * 0.30, thickness, dz * 0.30), screen_mat)
        add_local_box(obj, "marker_outer", (minx + dx * 0.28, face_y + sign * thickness, minz + dz * 0.48), (dx * 0.16, thickness, dz * 0.18), white_mat)
        add_local_box(obj, "marker_inner", (minx + dx * 0.28, face_y + sign * thickness * 2, minz + dz * 0.48), (dx * 0.105, thickness, dz * 0.12), black_mat)
        for ox, oz in [(-0.055, -0.055), (0.03, -0.02), (-0.02, 0.04), (0.065, 0.065)]:
            add_local_box(obj, "marker_cell", (minx + dx * (0.28 + ox), face_y + sign * thickness * 3, minz + dz * (0.48 + oz)), (dx * 0.025, thickness, dz * 0.025), white_mat)
        add_local_box(obj, "green_status", (minx + dx * 0.66, face_y + sign * thickness * 2, minz + dz * 0.33), (dx * 0.075, thickness, dz * 0.018), green_mat)
        add_local_box(obj, "screen_highlight", (minx + dx * 0.50, face_y + sign * thickness * 2, minz + dz * 0.70), (dx * 0.06, thickness, dz * 0.012), white_mat)
def ensure_reasonable_materials(obj):
    if obj.data.materials:
        for mat in obj.data.materials:
            if mat and mat.use_nodes:
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    if "Base Color" in bsdf.inputs:
                        col = bsdf.inputs["Base Color"].default_value
                        bsdf.inputs["Base Color"].default_value = (max(col[0], 0.055), max(col[1], 0.055), max(col[2], 0.052), col[3])
                    if "Roughness" in bsdf.inputs:
                        bsdf.inputs["Roughness"].default_value = min(max(bsdf.inputs["Roughness"].default_value, 0.36), 0.70)
                    if "Metallic" in bsdf.inputs:
                        bsdf.inputs["Metallic"].default_value = min(bsdf.inputs["Metallic"].default_value, 0.1)
        return
    mat = bpy.data.materials.new("mat_action4_dark_plastic")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.065, 0.065, 0.06, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.58
    obj.data.materials.append(mat)

def setup_camera(width, height, fx, fy, cx, cy):
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = width
    cam_data.lens = fx * cam_data.sensor_width / width
    cam_data.shift_x = (cx - width * 0.5) / width
    cam_data.shift_y = (height * 0.5 - cy) / width
    return cam


def add_lights(scale=1.0):
    bpy.ops.object.light_add(type="AREA", location=(0, -300, 520))
    key = bpy.context.object
    key.name = "KeyLight"
    key.data.energy = random.uniform(1500, 2300) * scale
    key.data.size = random.uniform(430, 720)
    bpy.ops.object.light_add(type="AREA", location=(-320, 240, 320))
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = random.uniform(650, 1100) * scale
    fill.data.size = random.uniform(420, 700)
    bpy.ops.object.light_add(type="AREA", location=(260, -120, 220))
    rim = bpy.context.object
    rim.name = "RimLight"
    rim.data.energy = random.uniform(450, 850) * scale
    rim.data.size = random.uniform(120, 260)


def create_background(level=0.60):
    bpy.ops.mesh.primitive_plane_add(size=900, location=(0, 80, -74))
    plane = bpy.context.object
    plane.name = "matte_table_background"
    mat = bpy.data.materials.new("matte_warm_gray")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        v = random.uniform(max(0.35, level - 0.08), min(0.90, level + 0.08))
        bsdf.inputs["Base Color"].default_value = (v, v * random.uniform(0.98, 1.04), v * random.uniform(0.96, 1.02), 1.0)
        bsdf.inputs["Roughness"].default_value = random.uniform(0.72, 0.95)
    plane.data.materials.append(mat)
    return plane


def camera_look_at(camera, target):
    direction = mathutils.Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def world_to_pixel(scene, camera, point):
    co = bpy_extras.object_utils.world_to_camera_view(scene, camera, point)
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    return [float(co.x * width), float((1.0 - co.y) * height), float(co.z)]


def keep_zmin_face_down(scene, camera, world_corners):
    """Roll the camera so the z_min face reads as a lower/bottom face in the image."""
    zmin_center = sum((world_corners[i] for i in (0, 2, 6, 4)), mathutils.Vector()) / 4.0
    zmax_center = sum((world_corners[i] for i in (1, 3, 7, 5)), mathutils.Vector()) / 4.0
    edge_02_center = (world_corners[0] + world_corners[2]) / 2.0
    edge_46_center = (world_corners[4] + world_corners[6]) / 2.0
    base_quat = camera.rotation_euler.to_quaternion()
    best_rotation = camera.rotation_euler.copy()
    best_score = -1e9
    for step in range(180):
        angle = math.tau * step / 180.0
        camera.rotation_euler = (base_quat @ mathutils.Quaternion((0.0, 0.0, 1.0), angle)).to_euler()
        bpy.context.view_layer.update()
        zmin_px = world_to_pixel(scene, camera, zmin_center)
        zmax_px = world_to_pixel(scene, camera, zmax_center)
        e02_px = world_to_pixel(scene, camera, edge_02_center)
        e46_px = world_to_pixel(scene, camera, edge_46_center)
        face_dx = zmin_px[0] - zmax_px[0]
        face_dy = zmin_px[1] - zmax_px[1]
        edge_dx = e46_px[0] - e02_px[0]
        edge_dy = e46_px[1] - e02_px[1]
        # Make the z_min face's 0-2 / 4-6 edge pair separate vertically
        # instead of left-right; either edge can be above depending on view.
        score = abs(edge_dy) * 3.0 - abs(edge_dx) * 4.0 + face_dy * 0.15 - abs(face_dx) * 0.05
        if score > best_score:
            best_score = score
            best_rotation = camera.rotation_euler.copy()
    camera.rotation_euler = best_rotation
    bpy.context.view_layer.update()


def bbox_corners_from_info(info):
    xs = [info["min_x"], info["max_x"]]
    ys = [info["min_y"], info["max_y"]]
    zs = [info["min_z"], info["max_z"]]
    return [mathutils.Vector((x, y, z)) for x in xs for y in ys for z in zs]



def make_random_background(bg_dir, width, height):
    import cv2

    bg_paths = sorted(Path(bg_dir).glob("*.png")) + sorted(Path(bg_dir).glob("*.jpg"))
    bg_paths = [p for p in bg_paths if "contact_sheet" not in p.stem]
    if not bg_paths:
        raise RuntimeError(f"No background images found in {bg_dir}")
    bg = cv2.imread(str(random.choice(bg_paths)), cv2.IMREAD_COLOR)
    if bg is None:
        raise RuntimeError(f"Failed to read background image from {bg_dir}")
    bh, bw = bg.shape[:2]
    scale = max(width / max(1, bw), height / max(1, bh))
    resized = cv2.resize(bg, (int(round(bw * scale)), int(round(bh * scale))), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    x = 0 if rw == width else random.randint(0, rw - width)
    y = 0 if rh == height else random.randint(0, rh - height)
    crop = resized[y:y + height, x:x + width].copy()
    if random.random() < 0.45:
        crop = cv2.GaussianBlur(crop, (3, 3), random.uniform(0.0, 0.7))
    return crop


def composite_on_background(image_path, bg_dir):
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read rendered image: {image_path}")
    if img.ndim != 3 or img.shape[2] < 4:
        return
    h, w = img.shape[:2]
    fg = img[:, :, :3].astype(np.float32)
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    bg = make_random_background(bg_dir, w, h).astype(np.float32)
    comp = np.clip(fg * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)
    cv2.imwrite(str(image_path), comp)


def composite_cropped_foreground_on_background(image_path, bg_dir):
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read cropped foreground: {image_path}")
    if img.ndim != 3 or img.shape[2] < 4:
        return None
    h, w = img.shape[:2]
    fg = img[:, :, :3].astype(np.float32)
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    bg = make_random_background(bg_dir, w, h).astype(np.float32)
    comp = np.clip(fg * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)
    cv2.imwrite(str(image_path), comp)
    return True


def post3s_roi_side(sample_index):
    if sample_index % 16 < 3:
        return None
    return "left" if (sample_index - 3) % 2 == 0 else "right"


def select_background_dir(args, sample_index):
    if args.view_mode != "front_then_sides":
        return args.background_dir
    cycle = sample_index % 16
    front_dir = Path("assets/yolo_roi_bg_front3s_grabcut_light_preview_8")
    side_dir = Path("assets/yolo_roi_bg_after3s_side_clean")
    left_dir = Path("assets/yolo_roi_bg_after3s_side_clean_left")
    right_dir = Path("assets/yolo_roi_bg_after3s_side_clean_right")
    if cycle < 3 and front_dir.exists():
        return front_dir
    side = post3s_roi_side(sample_index)
    # The extracted ROI background folders were named opposite to the
    # user's left/right hand interpretation, so intentionally swap them here.
    if side == "left" and right_dir.exists():
        return right_dir
    if side == "right" and left_dir.exists():
        return left_dir
    if side_dir.exists():
        return side_dir
    return args.background_dir


def crop_render_and_points(image_path, corners_2d, out_path, pad_ratio, out_size, post_effects=False, pad_px=None, final_pad_px=None, square_crop=True):
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read rendered image: {image_path}")
    pts = np.asarray([[p[0], p[1]] for p in corners_2d], dtype=np.float32)
    h, w = img.shape[:2]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    if final_pad_px is not None:
        final_pad = max(0.0, min(float(final_pad_px), (out_size - 1) * 0.45))
        target_content = max(1.0, out_size - final_pad * 2.0)
        if square_crop:
            side_target = max(box_w, box_h) * out_size / target_content
            crop_w_target = side_target
            crop_h_target = side_target
        else:
            crop_w_target = box_w * out_size / target_content
            crop_h_target = box_h * out_size / target_content
        pad_x = max(0.0, (crop_w_target - box_w) * 0.5)
        pad_y = max(0.0, (crop_h_target - box_h) * 0.5)
    else:
        pad = float(pad_px) if pad_px is not None else max(box_w, box_h) * pad_ratio
        pad_x = pad
        pad_y = pad
    x1 = x1 - pad_x
    y1 = y1 - pad_y
    x2 = x2 + pad_x
    y2 = y2 + pad_y
    if square_crop:
        side = max(x2 - x1, y2 - y1)
        cx_box = (x1 + x2) * 0.5
        cy_box = (y1 + y2) * 0.5
        x1 = cx_box - side * 0.5
        x2 = cx_box + side * 0.5
        y1 = cy_box - side * 0.5
        y2 = cy_box + side * 0.5
        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if x2 > w - 1:
            x1 -= x2 - (w - 1)
            x2 = float(w - 1)
        if y2 > h - 1:
            y1 -= y2 - (h - 1)
            y2 = float(h - 1)
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(w - 1), x2)
    y2 = min(float(h - 1), y2)
    x1i, y1i = int(math.floor(x1)), int(math.floor(y1))
    x2i, y2i = int(math.ceil(x2)), int(math.ceil(y2))
    crop = img[y1i:y2i + 1, x1i:x2i + 1]
    crop_h, crop_w = crop.shape[:2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    sx = out_size / max(1.0, float(crop_w))
    sy = out_size / max(1.0, float(crop_h))
    if post_effects:
        alpha = crop[:, :, 3:4] if crop.ndim == 3 and crop.shape[2] == 4 else None
        color = crop[:, :, :3] if alpha is not None else crop
        if random.random() < 0.28:
            color = cv2.GaussianBlur(color, (3, 3), random.uniform(0.20, 0.60))
        crop_f = color.astype(np.float32)
        if random.random() < 0.60:
            crop_f = crop_f * random.uniform(0.78, 1.20) + random.uniform(-16.0, 12.0)
        if random.random() < 0.22:
            gray = crop_f.mean(axis=2, keepdims=True)
            crop_f = gray + (crop_f - gray) * random.uniform(0.82, 1.14)
        if random.random() < 0.30:
            noise = np.random.normal(0.0, random.uniform(0.8, 2.4), crop_f.shape).astype(np.float32)
            crop_f = np.clip(crop_f + noise, 0, 255)
        crop_f = np.clip(crop_f, 0, 255)
        color = crop_f.astype(np.uint8)
        crop = np.concatenate([color, alpha], axis=2) if alpha is not None else color
    cv2.imwrite(str(out_path), crop)
    transformed = []
    for x, y, z in corners_2d:
        transformed.append([(x - x1i) * sx, (y - y1i) * sy, z])
    crop_info = {
        "xyxy": [x1i, y1i, x2i, y2i],
        "source_width": int(w),
        "source_height": int(h),
        "crop_width": int(crop_w),
        "crop_height": int(crop_h),
        "scale_x": float(sx),
        "scale_y": float(sy),
        "pad_x": 0.0,
        "pad_y": 0.0,
        "square_crop": bool(square_crop),
        "final_pad_px": None if final_pad_px is None else float(final_pad_px),
    }
    return transformed, crop_info


def setup_world(strength):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        c = bpy.context.scene.world.color
        bg.inputs["Color"].default_value = (c[0], c[1], c[2], 1.0)
        bg.inputs["Strength"].default_value = strength


def add_camera_softbox(camera, scale=1.0):
    bpy.ops.object.light_add(type="AREA", location=camera.location)
    light = bpy.context.object
    light.name = "CameraSoftbox"
    light.data.energy = 1700 * scale
    light.data.size = 260
    return light


def sync_camera_softbox(light, camera, target):
    direction = (mathutils.Vector(target) - camera.location).normalized()
    light.location = camera.location + direction * 55.0 + mathutils.Vector((0.0, 0.0, 22.0))
    camera_look_at(light, target)


def balanced_camera_direction(sample_index, jitter=0.22):
    """Cycle through a full-sphere view bank so every major side is seen often."""
    base_dirs = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
        (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
        (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
        (1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1),
    ]
    v = mathutils.Vector(base_dirs[sample_index % len(base_dirs)])
    v += mathutils.Vector((
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
    ))
    if v.length < 1e-6:
        v = mathutils.Vector((1.0, 0.0, 0.0))
    return v.normalized()


def real_video_like_camera_direction(sample_index, jitter=0.12):
    """Bias previews/training toward the horizontal Action 4 poses seen in the target video."""
    base_dirs = [
        (0.35, -1.00, 0.12), (-0.35, -1.00, 0.12),
        (0.55, -0.85, 0.20), (-0.55, -0.85, 0.20),
        (0.85, -0.55, 0.16), (-0.85, -0.55, 0.16),
        (0.25, -1.00, -0.18), (-0.25, -1.00, -0.18),
        (0.65, -0.75, -0.18), (-0.65, -0.75, -0.18),
        (0.15, -0.95, 0.45), (-0.15, -0.95, 0.45),
        (0.95, -0.25, 0.12), (-0.95, -0.25, 0.12),
        (0.55, -0.65, 0.52), (-0.55, -0.65, 0.52),
        (0.45, -0.65, -0.45), (-0.45, -0.65, -0.45),
        (0.00, -1.00, 0.00), (0.00, -1.00, 0.28),
    ]
    v = mathutils.Vector(base_dirs[sample_index % len(base_dirs)])
    v += mathutils.Vector((
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
        random.uniform(-jitter * 0.8, jitter * 0.8),
    ))
    if v.length < 1e-6:
        v = mathutils.Vector((0.0, -1.0, 0.0))
    return v.normalized()


def real_csv_face_camera_direction(sample_index, jitter=0.28):
    """Sample major visible faces according to the remapped real CSV labels."""
    face_to_dir = {
        "x_min": (-1.0, 0.0, 0.0),  # 0-1-3-2
        "x_max": (1.0, 0.0, 0.0),   # 4-5-7-6
        "y_max": (0.0, 1.0, 0.0),   # 2-3-7-6
        "z_min": (0.0, 0.0, -1.0),  # 0-2-6-4
        "z_max": (0.0, 0.0, 1.0),   # 1-3-7-5
        "y_min": (0.0, -1.0, 0.0),  # 0-1-5-4
    }
    base = mathutils.Vector(face_to_dir[real_csv_target_face(sample_index)])

    side_dirs = [
        mathutils.Vector((1.0, 0.0, 0.0)),
        mathutils.Vector((-1.0, 0.0, 0.0)),
        mathutils.Vector((0.0, 1.0, 0.0)),
        mathutils.Vector((0.0, -1.0, 0.0)),
        mathutils.Vector((0.0, 0.0, 1.0)),
        mathutils.Vector((0.0, 0.0, -1.0)),
    ]
    side_dirs = [v for v in side_dirs if abs(v.dot(base)) < 0.5]
    side = side_dirs[sample_index % len(side_dirs)]
    v = base * random.uniform(0.78, 1.0) + side * random.uniform(0.18, 0.45)
    v += mathutils.Vector((
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
    ))
    if v.length < 1e-6:
        v = base
    return v.normalized()


def direct_side_face_camera_direction(sample_index, jitter=0.035):
    """Camera directions that look almost straight at the two side faces."""
    base_dirs = [
        mathutils.Vector((1.0, 0.0, 0.0)),   # x_max: 4-5-7-6
        mathutils.Vector((-1.0, 0.0, 0.0)),  # x_min: 0-1-3-2
    ]
    base = base_dirs[sample_index % len(base_dirs)]
    # Keep only mild off-axis variation so the side face remains dominant.
    v = base + mathutils.Vector((
        0.0,
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
    ))
    return v.normalized()


def tilted_side_face_camera_direction(sample_index, jitter=0.16):
    """Side-dominant views with enough tilt to reveal adjacent faces."""
    base_dirs = [
        mathutils.Vector((1.0, 0.26, 0.16)),
        mathutils.Vector((1.0, -0.24, -0.14)),
        mathutils.Vector((-1.0, 0.26, -0.16)),
        mathutils.Vector((-1.0, -0.24, 0.14)),
        mathutils.Vector((1.0, 0.38, -0.08)),
        mathutils.Vector((-1.0, -0.38, 0.08)),
    ]
    base = base_dirs[sample_index % len(base_dirs)]
    v = base + mathutils.Vector((
        random.uniform(-jitter * 0.35, jitter * 0.35),
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
    ))
    return v.normalized()


def tilted_screen_front_camera_direction(sample_index, jitter=0.14):
    """Large-screen-front views with slight yaw/pitch variation."""
    base_dirs = [
        mathutils.Vector((0.16, 0.18, -1.0)),
        mathutils.Vector((-0.16, 0.16, -1.0)),
        mathutils.Vector((0.25, -0.08, -1.0)),
        mathutils.Vector((-0.25, -0.06, -1.0)),
        mathutils.Vector((0.08, 0.30, -1.0)),
        mathutils.Vector((-0.08, -0.24, -1.0)),
    ]
    base = base_dirs[sample_index % len(base_dirs)]
    v = base + mathutils.Vector((
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
        random.uniform(-jitter * 0.35, jitter * 0.35),
    ))
    return v.normalized()


def front_then_side_camera_direction(sample_index, jitter=0.16, roi_side=None):
    """A deterministic preview/training mix: a few screen-front views, then side-dominant views."""
    cycle = sample_index % 16
    if cycle < 3:
        return tilted_screen_front_camera_direction(cycle, jitter=jitter * 0.75)
    if roi_side == "left":
        base_pool = [
            mathutils.Vector((1.0, 0.34, 0.06)),
            mathutils.Vector((1.0, -0.38, -0.05)),
            mathutils.Vector((1.0, 0.50, -0.10)),
            mathutils.Vector((1.0, -0.54, 0.10)),
            mathutils.Vector((1.0, 0.18, 0.18)),
            mathutils.Vector((1.0, -0.20, -0.18)),
            mathutils.Vector((1.0, 0.62, 0.04)),
            mathutils.Vector((0.82, 0.52, -0.36)),
            mathutils.Vector((0.82, -0.48, 0.34)),
            mathutils.Vector((0.72, 0.18, -0.62)),
            mathutils.Vector((0.72, -0.16, 0.58)),
            mathutils.Vector((0.58, 0.74, -0.18)),
            mathutils.Vector((0.58, -0.72, 0.16)),
            mathutils.Vector((0.42, -1.0, 0.16)),
            mathutils.Vector((0.38, -1.0, -0.28)),
            mathutils.Vector((0.62, -0.86, 0.42)),
        ]
        base = base_pool[(cycle - 3) % len(base_pool)]
        v = base + mathutils.Vector((
            random.uniform(-jitter * 0.35, jitter * 0.35),
            random.uniform(-jitter, jitter),
            random.uniform(-jitter, jitter),
        ))
        return v.normalized()
    if roi_side == "right":
        base_pool = [
            mathutils.Vector((-1.0, 0.34, -0.06)),
            mathutils.Vector((-1.0, -0.38, 0.05)),
            mathutils.Vector((-1.0, 0.50, 0.10)),
            mathutils.Vector((-1.0, -0.54, -0.10)),
            mathutils.Vector((-1.0, -0.18, -0.18)),
            mathutils.Vector((-1.0, 0.20, 0.18)),
            mathutils.Vector((-1.0, 0.62, -0.04)),
            mathutils.Vector((-0.82, 0.52, 0.36)),
            mathutils.Vector((-0.82, -0.48, -0.34)),
            mathutils.Vector((-0.72, 0.18, 0.62)),
            mathutils.Vector((-0.72, -0.16, -0.58)),
            mathutils.Vector((-0.58, 0.74, 0.18)),
            mathutils.Vector((-0.58, -0.72, -0.16)),
            mathutils.Vector((-0.42, -1.0, -0.16)),
            mathutils.Vector((-0.38, -1.0, 0.28)),
            mathutils.Vector((-0.62, -0.86, -0.42)),
        ]
        base = base_pool[(cycle - 3) % len(base_pool)]
        v = base + mathutils.Vector((
            random.uniform(-jitter * 0.35, jitter * 0.35),
            random.uniform(-jitter, jitter),
            random.uniform(-jitter, jitter),
        ))
        return v.normalized()
    side_dirs = [
        mathutils.Vector((1.0, 0.34, 0.06)),
        mathutils.Vector((-1.0, 0.34, -0.06)),
        mathutils.Vector((1.0, -0.38, -0.05)),
        mathutils.Vector((-1.0, -0.38, 0.05)),
        mathutils.Vector((1.0, 0.50, -0.10)),
        mathutils.Vector((-1.0, 0.50, 0.10)),
        mathutils.Vector((1.0, -0.54, 0.10)),
        mathutils.Vector((-1.0, -0.54, -0.10)),
        mathutils.Vector((1.0, 0.18, 0.18)),
        mathutils.Vector((-1.0, -0.18, -0.18)),
        mathutils.Vector((1.0, -0.20, -0.18)),
        mathutils.Vector((-1.0, 0.20, 0.18)),
        mathutils.Vector((1.0, 0.62, 0.04)),
    ]
    base = side_dirs[(cycle - 3) % len(side_dirs)]
    v = base + mathutils.Vector((
        random.uniform(-jitter * 0.45, jitter * 0.45),
        random.uniform(-jitter, jitter),
        random.uniform(-jitter, jitter),
    ))
    return v.normalized()


def ymin_side_camera_direction(sample_index, jitter=0.16, roi_side=None):
    """Views that deliberately expose the 0-1-5-4 / y_min face while keeping side context."""
    if roi_side == "right":
        x_sign = -1.0
    elif roi_side == "left":
        x_sign = 1.0
    else:
        x_sign = 1.0 if sample_index % 2 == 0 else -1.0
    base_pool = [
        mathutils.Vector((0.12 * x_sign, -1.0, 0.06)),
        mathutils.Vector((0.20 * x_sign, -1.0, -0.14)),
        mathutils.Vector((0.28 * x_sign, -1.0, 0.26)),
        mathutils.Vector((0.36 * x_sign, -0.94, -0.30)),
        mathutils.Vector((0.44 * x_sign, -0.90, 0.18)),
        mathutils.Vector((0.18 * x_sign, -1.0, 0.42)),
    ]
    base = base_pool[sample_index % len(base_pool)]
    v = base + mathutils.Vector((
        random.uniform(-jitter * 0.55, jitter * 0.55),
        random.uniform(-jitter * 0.35, jitter * 0.35),
        random.uniform(-jitter, jitter),
    ))
    return v.normalized()


def real_roi_shape_ok(corners_2d, sample_index, width, height, aspect_min, aspect_max, front_aspect_min, front_aspect_max):
    pts = np.asarray([[p[0], p[1]] for p in corners_2d], dtype=np.float32)
    bw = max(1.0, float(pts[:, 0].max() - pts[:, 0].min()))
    bh = max(1.0, float(pts[:, 1].max() - pts[:, 1].min()))
    aspect = bw / bh
    if sample_index % 16 < 3:
        return front_aspect_min <= aspect <= front_aspect_max
    return aspect_min <= aspect <= aspect_max


def polygon_area_2d(points):
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def face_weights_from_corners(corners_2d):
    pts = np.asarray([[p[0], p[1]] for p in corners_2d], dtype=np.float32)
    faces = {
        "x_min": [0, 1, 3, 2],
        "x_max": [4, 5, 7, 6],
        "y_min": [0, 1, 5, 4],
        "y_max": [2, 3, 7, 6],
        "z_min": [0, 2, 6, 4],
        "z_max": [1, 3, 7, 5],
    }
    areas = {name: polygon_area_2d(pts[idxs]) for name, idxs in faces.items()}
    total = sum(areas.values()) + 1e-6
    weights = {name: area / total for name, area in areas.items()}
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return weights, ranked


def real_csv_target_face(sample_index):
    # Interleave faces so small previews do not collapse to a single face.
    sequence = [
        "x_max", "x_min", "y_max", "x_max",
        "x_min", "z_max", "y_max", "x_max",
        "x_min", "z_min", "y_max", "x_max",
        "x_min", "z_max", "y_max", "y_min",
    ]
    return sequence[sample_index % len(sequence)]


def corner_shape_descriptor(corners_2d, width, height):
    pts = np.asarray([[p[0] / float(width), p[1] / float(height)] for p in corners_2d], dtype=np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw = max(1e-6, float(x2 - x1))
    bh = max(1e-6, float(y2 - y1))
    norm = pts.copy()
    norm[:, 0] = (norm[:, 0] - x1) / bw
    norm[:, 1] = (norm[:, 1] - y1) / bh
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    edge_lengths = [float(np.linalg.norm(norm[a] - norm[b])) for a, b in edges]
    face_weights, ranked = face_weights_from_corners([[p[0] * width, p[1] * height, 1.0] for p in pts])
    face_vec = [face_weights[k] for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")]
    parts = [
        (pts - 0.5).reshape(-1) * 0.65,
        (norm - 0.5).reshape(-1) * 1.0,
        np.asarray([bw, bh, bw / bh], dtype=np.float32) * 0.7,
        np.asarray(edge_lengths, dtype=np.float32) * 0.35,
        np.asarray(face_vec, dtype=np.float32) * 0.55,
    ]
    return np.concatenate(parts).astype(np.float32)


def load_real_corner_shape_bank(csv_paths):
    by_img = defaultdict(list)
    for path in csv_paths:
        if not path or not Path(path).exists():
            continue
        with Path(path).open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                try:
                    label = int(row[0])
                    x = float(row[1])
                    y = float(row[2])
                    w = float(row[4])
                    h = float(row[5])
                except ValueError:
                    continue
                by_img[row[3]].append((label, x, y, w, h))
    bank = []
    for img_name, rows in by_img.items():
        counts = Counter(r[0] for r in rows)
        if len(rows) != 8 or any(counts[i] != 1 for i in range(8)):
            continue
        ordered = sorted(rows, key=lambda r: r[0])
        w = ordered[0][3]
        h = ordered[0][4]
        corners = [[x, y, 1.0] for _label, x, y, _w, _h in ordered]
        bank.append({"image": img_name, "descriptor": corner_shape_descriptor(corners, w, h)})
    return bank


def nearest_real_corner_shape(corners_2d, width, height, shape_bank):
    if not shape_bank:
        return None, 0.0
    desc = corner_shape_descriptor(corners_2d, width, height)
    dists = [(float(np.linalg.norm(desc - item["descriptor"])), item) for item in shape_bank]
    dists.sort(key=lambda x: x[0])
    return dists[0][1], dists[0][0]


def corner_shape_distance(corners_2d, width, height, shape_item):
    if shape_item is None:
        return None
    desc = corner_shape_descriptor(corners_2d, width, height)
    return float(np.linalg.norm(desc - shape_item["descriptor"]))


def make_mesh_materials_visible(obj, min_base=0.16):
    # Blender's material preview uses a bright studio HDRI. In scripted renders,
    # pure-black imported PBR bases need a neutral floor to show plastic detail.
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if not bsdf:
            continue
        if "Base Color" in bsdf.inputs:
            col = bsdf.inputs["Base Color"].default_value
            bsdf.inputs["Base Color"].default_value = (max(col[0], min_base), max(col[1], min_base), max(col[2], min_base * 0.95), col[3])
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.65
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.65
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = min(max(bsdf.inputs["Roughness"].default_value, 0.30), 0.58)


def set_input_if_present(bsdf, names, value):
    for name in names:
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value
            return True
    return False


def darken_base_color_input(mat, bsdf, scale):
    if "Base Color" not in bsdf.inputs:
        return
    base_input = bsdf.inputs["Base Color"]
    col = base_input.default_value
    base_input.default_value = (
        max(0.004, col[0] * scale),
        max(0.004, col[1] * scale),
        max(0.0035, col[2] * scale),
        col[3],
    )
    if not base_input.links:
        return

    tree = mat.node_tree
    original_link = base_input.links[0]
    source_socket = original_link.from_socket
    try:
        tree.links.remove(original_link)
    except RuntimeError:
        return

    mix = tree.nodes.new(type="ShaderNodeMixRGB")
    mix.name = "BodyTextureDarken"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    mix.inputs[2].default_value = (scale, scale, scale, 1.0)
    tree.links.new(source_socket, mix.inputs[1])
    tree.links.new(mix.outputs[0], base_input)


def enhance_natural_reflections(obj, body_roughness=0.42, glass_roughness=0.16, body_specular=0.58, glass_specular=0.88, body_color_scale=0.45):
    glass_keys = ("boli", "glass", "siyin", "screen", "lens")
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if not bsdf:
            continue
        name = mat.name.lower()
        is_glass = any(k in name for k in glass_keys)
        rough = glass_roughness if is_glass else body_roughness
        spec = glass_specular if is_glass else body_specular
        if not is_glass:
            darken_base_color_input(mat, bsdf, body_color_scale)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = min(bsdf.inputs["Roughness"].default_value, rough)
        set_input_if_present(bsdf, ("Specular IOR Level", "Specular"), spec)
        if is_glass:
            set_input_if_present(bsdf, ("Alpha",), max(0.55, bsdf.inputs["Alpha"].default_value if "Alpha" in bsdf.inputs else 1.0))


def add_reflection_strip_lights(scale=1.0):
    specs = [
        ("TopReflectionStrip", (0, -170, 210), 180, 34, 230),
        ("LeftReflectionStrip", (-210, -130, 70), 120, 28, 130),
        ("ScreenCatchlight", (135, -155, 95), 65, 18, 85),
    ]
    lights = []
    for name, loc, size, size_y, energy in specs:
        bpy.ops.object.light_add(type="AREA", location=loc)
        light = bpy.context.object
        light.name = name
        light.data.shape = "RECTANGLE"
        light.data.size = size
        light.data.size_y = size_y
        light.data.energy = energy * scale
        camera_look_at(light, (0, 0, 0))
        lights.append(light)
    return lights


def enable_cycles_gpu(device_type="OPTIX"):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for candidate in (device_type, "CUDA"):
        try:
            prefs.compute_device_type = candidate
            prefs.get_devices()
            enabled = []
            for device in prefs.devices:
                use_device = device.type != "CPU"
                device.use = use_device
                if use_device:
                    enabled.append(device.name)
            if enabled:
                bpy.context.scene.cycles.device = "GPU"
                print(f"Cycles GPU enabled ({candidate}): {', '.join(enabled)}")
                return True
        except Exception as exc:
            print(f"Cycles GPU setup failed for {candidate}: {exc}")
    bpy.context.scene.cycles.device = "CPU"
    print("Cycles GPU unavailable; falling back to CPU")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=Path("datasets/dji_action4"), type=Path)
    parser.add_argument("--model", default=Path("/root/autodl-fs/official.glb"), type=Path)
    parser.add_argument("--out-dir", default=Path("datasets/dji_action4_official_glb_gray_mean110"), type=Path)
    parser.add_argument("--num-images", default=32, type=int)
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--height", default=960, type=int)
    parser.add_argument("--crop-output", action="store_true", default=True)
    parser.add_argument("--no-crop-output", dest="crop_output", action="store_false")
    parser.add_argument("--crop-size", default=1024, type=int)
    parser.add_argument("--crop-pad", default=0.02, type=float)
    parser.add_argument("--crop-pad-px", default=None, type=float)
    parser.add_argument("--final-pad-px", default=8.0, type=float)
    parser.add_argument("--rect-crop-output", action="store_true")
    parser.add_argument("--target-diameter", default=120.0, type=float)
    parser.add_argument("--add-decals", action="store_true")
    parser.add_argument("--dji-label", default=None)
    parser.add_argument("--post-effects", action="store_true", default=True)
    parser.add_argument("--no-post-effects", dest="post_effects", action="store_false")
    parser.add_argument("--dof", action="store_true")
    parser.add_argument("--engine", default="cycles", choices=["eevee", "cycles"])
    parser.add_argument("--cycles-device", default="OPTIX", choices=["OPTIX", "CUDA"])
    parser.add_argument("--ambient-strength", default=0.32, type=float)
    parser.add_argument("--exposure", default=-0.62, type=float)
    parser.add_argument("--gamma", default=1.08, type=float)
    parser.add_argument("--material-min-base", default=0.0, type=float)
    parser.add_argument("--light-scale", default=0.16, type=float)
    parser.add_argument("--background-level", default=0.60, type=float)
    parser.add_argument("--background-plane", action="store_true")
    parser.add_argument("--background-dir", default=Path("assets/yolo_roi_bg_grabcut_light_mix_preview"), type=Path)
    parser.add_argument("--samples", default=24, type=int)
    parser.add_argument("--body-roughness", default=0.50, type=float)
    parser.add_argument("--body-color-scale", default=0.0001, type=float)
    parser.add_argument("--glass-roughness", default=0.045, type=float)
    parser.add_argument("--glass-specular", default=1.0, type=float)
    parser.add_argument("--reflection-scale", default=0.42, type=float)
    parser.add_argument("--orbit-camera", action="store_true", default=True)
    parser.add_argument("--front-camera", dest="orbit_camera", action="store_false")
    parser.add_argument("--view-mode", default="front_then_sides", choices=["random_orbit", "balanced_faces", "real_video_like", "real_csv_faces", "real_corner_shapes", "direct_side_faces", "tilted_side_faces", "tilted_screen_front", "front_then_sides"])
    parser.add_argument("--ymin-heavy", action="store_true")
    parser.add_argument("--view-jitter", default=0.16, type=float)
    parser.add_argument("--real-like-filter", action="store_true", default=False)
    parser.add_argument("--no-real-like-filter", dest="real_like_filter", action="store_false")
    parser.add_argument("--real-like-aspect-min", default=1.08, type=float)
    parser.add_argument("--real-like-aspect-max", default=1.65, type=float)
    parser.add_argument("--real-like-hfrac-min", default=0.52, type=float)
    parser.add_argument("--real-like-hfrac-max", default=0.82, type=float)
    parser.add_argument("--real-roi-shape-filter", action="store_true", default=True)
    parser.add_argument("--no-real-roi-shape-filter", dest="real_roi_shape_filter", action="store_false")
    parser.add_argument("--real-roi-aspect-min", default=0.90, type=float)
    parser.add_argument("--real-roi-aspect-max", default=1.75, type=float)
    parser.add_argument("--real-roi-front-aspect-min", default=0.82, type=float)
    parser.add_argument("--real-roi-front-aspect-max", default=1.45, type=float)
    parser.add_argument("--multi-face-filter", action="store_true", default=True)
    parser.add_argument("--no-multi-face-filter", dest="multi_face_filter", action="store_false")
    parser.add_argument("--top-face-weight-max", default=0.52, type=float)
    parser.add_argument("--second-face-weight-min", default=0.12, type=float)
    parser.add_argument("--real-corner-csv", nargs="*", default=[
        "labels_my-project-name_2026-05-25-02-16-54.csv",
        "labels_my-project-name_2026-05-25-03-59-55.csv",
        "labels_my-project-name_2026-05-25-11-21-48.csv",
    ])
    parser.add_argument("--corner-shape-filter", action="store_true", default=False)
    parser.add_argument("--no-corner-shape-filter", dest="corner_shape_filter", action="store_false")
    parser.add_argument("--corner-shape-max-dist", default=0.90, type=float)
    parser.add_argument("--corner-shape-target-mode", default="nearest", choices=["cycle", "nearest"])
    parser.add_argument("--corner-shape-max-per-real", default=0, type=int)
    parser.add_argument("--camera-radius-min", default=270.0, type=float)
    parser.add_argument("--camera-radius-max", default=370.0, type=float)
    parser.add_argument("--camera-height-min", default=-190.0, type=float)
    parser.add_argument("--camera-height-max", default=220.0, type=float)
    parser.add_argument("--fx", default=572.411363389757, type=float)
    parser.add_argument("--fy", default=573.5704328585578, type=float)
    parser.add_argument("--cx", default=325.2611083984375, type=float)
    parser.add_argument("--cy", default=242.04899588216654, type=float)
    args = parser.parse_args()
    real_shape_bank = load_real_corner_shape_bank([Path(p) for p in args.real_corner_csv]) if args.corner_shape_filter else []
    if args.view_mode == "real_corner_shapes":
        print(f"loaded real corner shape bank: {len(real_shape_bank)} samples")

    clear_scene()
    scene = bpy.context.scene
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        enable_cycles_gpu(args.cycles_device)
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 96
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 3
        scene.eevee.gtao_factor = 0.8
        scene.eevee.use_soft_shadows = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.film_transparent = bool(args.background_dir)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = args.exposure
    scene.view_settings.gamma = args.gamma
    scene.world.color = (args.background_level, args.background_level, args.background_level)
    setup_world(args.ambient_strength)

    mesh_path = args.model if args.model else args.dataset_dir / "models" / "obj_000001.ply"
    obj = import_mesh(mesh_path)
    obj.name = "dji_action4"
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)

    normalize_object(obj, target_diameter=args.target_diameter)
    ensure_reasonable_materials(obj)
    make_mesh_materials_visible(obj, min_base=args.material_min_base)
    enhance_natural_reflections(
        obj,
        body_roughness=args.body_roughness,
        glass_roughness=args.glass_roughness,
        glass_specular=args.glass_specular,
        body_color_scale=args.body_color_scale,
    )
    if args.add_decals:
        add_photo_decals(obj)

    local_corners = local_bbox_corners(obj)
    add_screen_image_on_zmin_face(obj, BIG_SCREEN_IMAGE, inset=BIG_SCREEN_INSET)
    add_front_small_screen_on_zmax_face(
        obj,
        SMALL_SCREEN_IMAGE,
        side_ratio=SMALL_SCREEN_SIDE_RATIO,
        x_margin_ratio=SMALL_SCREEN_X_MARGIN_RATIO,
        y_scale=SMALL_SCREEN_Y_SCALE,
        overall_scale=SMALL_SCREEN_OVERALL_SCALE,
        offset_ratio=SMALL_SCREEN_OFFSET_RATIO,
    )
    if DJI_LABEL_IMAGE is not None and DJI_LABEL_IMAGE.exists():
        add_dji_label_image_on_ymax_face(obj, DJI_LABEL_IMAGE)
    if args.dji_label:
        add_dji_label_on_ymax_face(obj, text=args.dji_label)

    cam = setup_camera(args.width, args.height, args.fx, args.fy, args.cx, args.cy)
    add_lights(scale=args.light_scale)
    add_reflection_strip_lights(scale=args.reflection_scale)
    camera_softbox = add_camera_softbox(cam, scale=args.light_scale)
    if args.background_plane:
        create_background(level=args.background_level)

    color_dir = args.out_dir / "rgb"
    label_dir = args.out_dir / "labels"
    color_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    if args.append and (args.out_dir / "labels.json").exists():
        all_records = json.loads((args.out_dir / "labels.json").read_text(encoding="utf-8"))

    idx = int(args.start_index)
    target_idx = idx + args.num_images
    attempts = 0
    max_attempts = args.num_images * 80
    corner_shape_match_counts = Counter()
    while idx < target_idx and attempts < max_attempts:
        attempts += 1
        obj.rotation_euler = (random.uniform(-0.45, 0.45), random.uniform(-2.75, 2.75), random.uniform(-0.55, 0.55))
        obj.location = (
            random.uniform(-10, 10),
            random.uniform(-6, 6),
            random.uniform(-10, 10),
        )

        sample_no = args.start_index + idx
        # Keep the final output index for front/side scheduling, but let side
        # templates rotate across rejected attempts so strict shape filtering
        # does not get stuck on one impossible direction.
        direction_sample_no = sample_no
        if args.view_mode == "front_then_sides" and sample_no % 16 >= 3:
            direction_sample_no = 3 + ((attempts - 1) % 16)
        roi_side = post3s_roi_side(sample_no) if args.view_mode == "front_then_sides" else None
        target_shape = None
        sample_background_dir = select_background_dir(args, sample_no)
        if args.view_mode == "real_corner_shapes" and real_shape_bank and args.corner_shape_target_mode == "cycle":
            target_shape = real_shape_bank[sample_no % len(real_shape_bank)]
        if args.orbit_camera and args.view_mode == "balanced_faces":
            direction = balanced_camera_direction(sample_no, jitter=args.view_jitter)
            radius = random.uniform(args.camera_radius_min, args.camera_radius_max)
            cam.location = obj.location + direction * radius
        elif args.orbit_camera and args.view_mode == "real_video_like":
            direction = real_video_like_camera_direction(sample_no, jitter=args.view_jitter)
            radius = random.uniform(args.camera_radius_min, args.camera_radius_max)
            cam.location = obj.location + direction * radius
        elif args.orbit_camera and args.view_mode in {"direct_side_faces", "tilted_side_faces", "tilted_screen_front", "front_then_sides"}:
            if args.view_mode == "direct_side_faces":
                direction = direct_side_face_camera_direction(sample_no, jitter=args.view_jitter)
            elif args.view_mode == "tilted_side_faces":
                direction = tilted_side_face_camera_direction(sample_no, jitter=args.view_jitter)
            elif args.view_mode == "tilted_screen_front":
                direction = tilted_screen_front_camera_direction(sample_no, jitter=args.view_jitter)
            elif args.ymin_heavy:
                direction = ymin_side_camera_direction(direction_sample_no, jitter=args.view_jitter, roi_side=roi_side)
            else:
                direction = front_then_side_camera_direction(direction_sample_no, jitter=args.view_jitter, roi_side=roi_side)
            direction = (obj.matrix_world.to_3x3() @ direction).normalized()
            radius = random.uniform(args.camera_radius_min, args.camera_radius_max)
            cam.location = obj.location + direction * radius
        elif args.orbit_camera and args.view_mode == "real_csv_faces":
            direction = real_csv_face_camera_direction(sample_no, jitter=args.view_jitter)
            direction = (obj.matrix_world.to_3x3() @ direction).normalized()
            radius = random.uniform(args.camera_radius_min, args.camera_radius_max)
            cam.location = obj.location + direction * radius
        elif args.orbit_camera and args.view_mode == "real_corner_shapes":
            direction = real_csv_face_camera_direction(sample_no, jitter=args.view_jitter)
            direction = (obj.matrix_world.to_3x3() @ direction).normalized()
            radius = random.uniform(args.camera_radius_min, args.camera_radius_max)
            cam.location = obj.location + direction * radius
        elif args.orbit_camera:
            azimuth = random.uniform(0.0, math.tau)
            radius = random.uniform(args.camera_radius_min, args.camera_radius_max)
            cam.location = obj.location + mathutils.Vector(
                (
                    math.cos(azimuth) * radius,
                    math.sin(azimuth) * radius,
                    random.uniform(args.camera_height_min, args.camera_height_max),
                )
            )
        else:
            cam.location = mathutils.Vector(
                (
                    random.uniform(-80, 80),
                    random.uniform(-360, -250),
                    random.uniform(45, 135),
                )
            )
        camera_look_at(cam, obj.location)
        sync_camera_softbox(camera_softbox, cam, obj.location)
        cam.data.dof.use_dof = bool(args.dof)
        cam.data.dof.focus_object = obj
        cam.data.dof.aperture_fstop = random.uniform(8.0, 16.0)
        bpy.context.view_layer.update()

        world_corners = [obj.matrix_world @ c for c in local_corners]
        if args.view_mode == "front_then_sides" and sample_no % 16 >= 3:
            keep_zmin_face_down(scene, cam, world_corners)
            sync_camera_softbox(camera_softbox, cam, obj.location)
            bpy.context.view_layer.update()
        corners_2d = []
        for c in world_corners:
            co = bpy_extras.object_utils.world_to_camera_view(scene, cam, c)
            corners_2d.append([float(co.x * args.width), float((1.0 - co.y) * args.height), float(co.z)])

        if args.view_mode == "front_then_sides" and args.real_roi_shape_filter:
            if not args.ymin_heavy:
                if not real_roi_shape_ok(
                    corners_2d,
                    sample_no,
                    args.width,
                    args.height,
                    args.real_roi_aspect_min,
                    args.real_roi_aspect_max,
                    args.real_roi_front_aspect_min,
                    args.real_roi_front_aspect_max,
                ):
                    continue
            if roi_side is not None:
                _face_weights, ranked_faces = face_weights_from_corners(corners_2d)
                expected_face = "y_min" if args.ymin_heavy else ("x_max" if roi_side == "left" else "x_min")
                top_n = 2 if args.ymin_heavy else 3
                top_faces = {ranked_faces[i][0] for i in range(top_n)}
                if expected_face not in top_faces:
                    continue

        points = np.array([[p[0], p[1]] for p in corners_2d], dtype=np.float32)
        margin = 8
        if (
            points[:, 0].min() < margin
            or points[:, 0].max() > args.width - margin
            or points[:, 1].min() < margin
            or points[:, 1].max() > args.height - margin
        ):
            continue

        stem = f"{idx:06d}"
        raw_path = color_dir / f"{stem}_full.png"
        final_path = color_dir / f"{stem}.png"
        scene.render.filepath = str(raw_path if args.crop_output else final_path)
        bpy.ops.render.render(write_still=True)

        label_corners = corners_2d
        crop_info = None
        cam_width = args.width
        cam_height = args.height
        cam_fx = args.fx
        cam_fy = args.fy
        cam_cx = args.cx
        cam_cy = args.cy
        if args.crop_output:
            label_corners, crop_info = crop_render_and_points(
                raw_path,
                corners_2d,
                final_path,
                args.crop_pad,
                args.crop_size,
                post_effects=args.post_effects,
                pad_px=args.crop_pad_px,
                final_pad_px=args.final_pad_px,
                square_crop=not args.rect_crop_output,
            )
            raw_path.unlink(missing_ok=True)
            if args.view_mode in {"real_video_like", "real_csv_faces", "real_corner_shapes"} and args.real_like_filter:
                crop_pts = np.asarray([[p[0], p[1]] for p in label_corners], dtype=np.float32)
                cx1, cy1 = crop_pts.min(axis=0)
                cx2, cy2 = crop_pts.max(axis=0)
                bw = max(1.0, float(cx2 - cx1))
                bh = max(1.0, float(cy2 - cy1))
                aspect = bw / bh
                h_frac = bh / float(args.crop_size)
                if (
                    aspect < args.real_like_aspect_min
                    or aspect > args.real_like_aspect_max
                    or h_frac < args.real_like_hfrac_min
                    or h_frac > args.real_like_hfrac_max
                ):
                    final_path.unlink(missing_ok=True)
                    continue
            if args.view_mode == "real_csv_faces" and args.multi_face_filter:
                face_weights, ranked_faces = face_weights_from_corners(label_corners)
                target_face = real_csv_target_face(sample_no)
                if (
                    ranked_faces[0][0] != target_face
                    or ranked_faces[0][1] > args.top_face_weight_max
                    or ranked_faces[1][1] < args.second_face_weight_min
                ):
                    final_path.unlink(missing_ok=True)
                    continue
            nearest_shape = None
            nearest_shape_dist = None
            if args.view_mode == "real_corner_shapes" and args.corner_shape_filter:
                if target_shape is not None:
                    nearest_shape = target_shape
                    nearest_shape_dist = corner_shape_distance(label_corners, args.crop_size, args.crop_size, target_shape)
                else:
                    nearest_shape, nearest_shape_dist = nearest_real_corner_shape(label_corners, args.crop_size, args.crop_size, real_shape_bank)
                if nearest_shape is None or nearest_shape_dist is None or nearest_shape_dist > args.corner_shape_max_dist:
                    final_path.unlink(missing_ok=True)
                    continue
                if args.corner_shape_max_per_real > 0 and corner_shape_match_counts[nearest_shape["image"]] >= args.corner_shape_max_per_real:
                    final_path.unlink(missing_ok=True)
                    continue
            x1, y1, _x2, _y2 = crop_info["xyxy"]
            cam_width = args.crop_size
            cam_height = args.crop_size
            cam_fx = args.fx * crop_info["scale_x"]
            cam_fy = args.fy * crop_info["scale_y"]
            cam_cx = (args.cx - x1) * crop_info["scale_x"] + crop_info.get("pad_x", 0.0)
            cam_cy = (args.cy - y1) * crop_info["scale_y"] + crop_info.get("pad_y", 0.0)
            if sample_background_dir:
                composite_cropped_foreground_on_background(final_path, sample_background_dir)
        elif sample_background_dir:
            composite_on_background(final_path, sample_background_dir)

        record = {
            "image": f"rgb/{stem}.png",
            "obj_id": 1,
            "corners_3d": [[float(v) for v in c] for c in local_corners],
            "corners_2d": label_corners,
            "full_corners_2d": corners_2d,
            "crop": crop_info,
            "model": str(mesh_path),
            "camera": {
                "width": cam_width,
                "height": cam_height,
                "full_width": args.width,
                "full_height": args.height,
                "fx": cam_fx,
                "fy": cam_fy,
                "cx": cam_cx,
                "cy": cam_cy,
            },
            "object_location": [float(v) for v in obj.location],
            "object_rotation_euler": [float(v) for v in obj.rotation_euler],
            "background_dir": None if sample_background_dir is None else str(sample_background_dir),
        }
        if args.view_mode == "real_corner_shapes":
            if nearest_shape is not None:
                corner_shape_match_counts[nearest_shape["image"]] += 1
            record["corner_shape_match"] = {
                "target_mode": args.corner_shape_target_mode,
                "target_real_image": None if target_shape is None else target_shape["image"],
                "matched_real_image": None if nearest_shape is None else nearest_shape["image"],
                "distance": nearest_shape_dist,
                "max_distance": args.corner_shape_max_dist,
            }
        (label_dir / f"{stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        all_records.append(record)
        idx += 1

    (args.out_dir / "labels.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"rendered {len(all_records)} images to {args.out_dir} after {attempts} attempts")


if __name__ == "__main__":
    main()
