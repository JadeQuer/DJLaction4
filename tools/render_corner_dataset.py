import argparse
import json
import math
import random
from pathlib import Path

import bpy
import bpy_extras.object_utils
import mathutils
import numpy as np


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


def bbox_corners_from_info(info):
    xs = [info["min_x"], info["max_x"]]
    ys = [info["min_y"], info["max_y"]]
    zs = [info["min_z"], info["max_z"]]
    return [mathutils.Vector((x, y, z)) for x in xs for y in ys for z in zs]



def crop_render_and_points(image_path, corners_2d, out_path, pad_ratio, out_size, post_effects=False, pad_px=None):
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read rendered image: {image_path}")
    pts = np.asarray([[p[0], p[1]] for p in corners_2d], dtype=np.float32)
    h, w = img.shape[:2]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    pad = float(pad_px) if pad_px is not None else max(x2 - x1, y2 - y1) * pad_ratio
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(float(w - 1), x2 + pad)
    y2 = min(float(h - 1), y2 + pad)
    x1i, y1i = int(math.floor(x1)), int(math.floor(y1))
    x2i, y2i = int(math.ceil(x2)), int(math.ceil(y2))
    crop = img[y1i:y2i + 1, x1i:x2i + 1]
    crop_h, crop_w = crop.shape[:2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    if post_effects:
        if random.random() < 0.35:
            crop = cv2.GaussianBlur(crop, (3, 3), random.uniform(0.20, 0.45))
        crop_f = crop.astype(np.float32)
        crop_f = np.clip(crop_f * random.uniform(1.06, 1.18) + random.uniform(2.0, 8.0), 0, 255)
        if random.random() < 0.35:
            noise = np.random.normal(0.0, random.uniform(0.8, 2.0), crop_f.shape).astype(np.float32)
            crop_f = np.clip(crop_f + noise, 0, 255)
        crop = crop_f.astype(np.uint8)
    cv2.imwrite(str(out_path), crop)
    sx = out_size / max(1.0, float(crop_w))
    sy = out_size / max(1.0, float(crop_h))
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


def enhance_natural_reflections(obj, body_roughness=0.42, glass_roughness=0.16, body_specular=0.58, glass_specular=0.88):
    glass_keys = ("boli", "glass", "pbr", "siyin", "screen", "lens")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=Path("datasets/dji_action4"), type=Path)
    parser.add_argument("--model", default=Path("/root/autodl-fs/official.glb"), type=Path)
    parser.add_argument("--out-dir", default=Path("datasets/dji_action4_official_glb_gray_mean110"), type=Path)
    parser.add_argument("--num-images", default=32, type=int)
    parser.add_argument("--width", default=2048, type=int)
    parser.add_argument("--height", default=1536, type=int)
    parser.add_argument("--crop-output", action="store_true", default=True)
    parser.add_argument("--no-crop-output", dest="crop_output", action="store_false")
    parser.add_argument("--crop-size", default=1024, type=int)
    parser.add_argument("--crop-pad", default=0.02, type=float)
    parser.add_argument("--crop-pad-px", default=None, type=float)
    parser.add_argument("--target-diameter", default=120.0, type=float)
    parser.add_argument("--add-decals", action="store_true")
    parser.add_argument("--post-effects", action="store_true")
    parser.add_argument("--dof", action="store_true")
    parser.add_argument("--engine", default="cycles", choices=["eevee", "cycles"])
    parser.add_argument("--ambient-strength", default=0.92, type=float)
    parser.add_argument("--exposure", default=-0.18, type=float)
    parser.add_argument("--gamma", default=1.08, type=float)
    parser.add_argument("--material-min-base", default=0.12, type=float)
    parser.add_argument("--light-scale", default=0.38, type=float)
    parser.add_argument("--background-level", default=0.60, type=float)
    parser.add_argument("--background-plane", action="store_true")
    parser.add_argument("--samples", default=48, type=int)
    parser.add_argument("--body-roughness", default=0.42, type=float)
    parser.add_argument("--glass-roughness", default=0.16, type=float)
    parser.add_argument("--reflection-scale", default=0.45, type=float)
    parser.add_argument("--orbit-camera", action="store_true", default=True)
    parser.add_argument("--front-camera", dest="orbit_camera", action="store_false")
    parser.add_argument("--camera-radius-min", default=270.0, type=float)
    parser.add_argument("--camera-radius-max", default=370.0, type=float)
    parser.add_argument("--camera-height-min", default=45.0, type=float)
    parser.add_argument("--camera-height-max", default=135.0, type=float)
    parser.add_argument("--fx", default=572.411363389757, type=float)
    parser.add_argument("--fy", default=573.5704328585578, type=float)
    parser.add_argument("--cx", default=325.2611083984375, type=float)
    parser.add_argument("--cy", default=242.04899588216654, type=float)
    args = parser.parse_args()

    clear_scene()
    scene = bpy.context.scene
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 96
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 3
        scene.eevee.gtao_factor = 0.8
        scene.eevee.use_soft_shadows = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.film_transparent = False
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
    enhance_natural_reflections(obj, body_roughness=args.body_roughness, glass_roughness=args.glass_roughness)
    if args.add_decals:
        add_photo_decals(obj)

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

    local_corners = local_bbox_corners(obj)
    all_records = []

    idx = 0
    attempts = 0
    max_attempts = args.num_images * 80
    while idx < args.num_images and attempts < max_attempts:
        attempts += 1
        obj.rotation_euler = (random.uniform(-0.45, 0.45), random.uniform(-2.75, 2.75), random.uniform(-0.55, 0.55))
        obj.location = (
            random.uniform(-10, 10),
            random.uniform(-6, 6),
            random.uniform(-10, 10),
        )

        if args.orbit_camera:
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
        corners_2d = []
        for c in world_corners:
            co = bpy_extras.object_utils.world_to_camera_view(scene, cam, c)
            corners_2d.append([float(co.x * args.width), float((1.0 - co.y) * args.height), float(co.z)])

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
            )
            raw_path.unlink(missing_ok=True)
            x1, y1, _x2, _y2 = crop_info["xyxy"]
            cam_width = args.crop_size
            cam_height = args.crop_size
            cam_fx = args.fx * crop_info["scale_x"]
            cam_fy = args.fy * crop_info["scale_y"]
            cam_cx = (args.cx - x1) * crop_info["scale_x"]
            cam_cy = (args.cy - y1) * crop_info["scale_y"]

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
        }
        (label_dir / f"{stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        all_records.append(record)
        idx += 1

    (args.out_dir / "labels.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"rendered {len(all_records)} images to {args.out_dir} after {attempts} attempts")


if __name__ == "__main__":
    main()
