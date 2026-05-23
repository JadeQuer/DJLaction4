import argparse
import json
import math
import random
from pathlib import Path

import bpy
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


def add_lights():
    bpy.ops.object.light_add(type="AREA", location=(0, -500, 700))
    key = bpy.context.object
    key.name = "KeyLight"
    key.data.energy = 450
    key.data.size = 500
    bpy.ops.object.light_add(type="POINT", location=(-300, 300, 500))
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 90


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-images", default=32, type=int)
    parser.add_argument("--width", default=640, type=int)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--fx", default=572.411363389757, type=float)
    parser.add_argument("--fy", default=573.5704328585578, type=float)
    parser.add_argument("--cx", default=325.2611083984375, type=float)
    parser.add_argument("--cy", default=242.04899588216654, type=float)
    args = parser.parse_args()

    import bpy_extras.object_utils

    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.film_transparent = False
    scene.world.color = (0.78, 0.78, 0.78)

    mesh_path = args.dataset_dir / "models" / "obj_000001.ply"
    info_path = args.dataset_dir / "models" / "models_info.json"
    info = json.loads(info_path.read_text())["1"]
    obj = import_mesh(mesh_path)
    obj.name = "dji_action4"
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)

    # Make generated meshes visible even if vertex colors/materials are sparse.
    mat = bpy.data.materials.new("mat_action4")
    mat.diffuse_color = (0.025, 0.025, 0.025, 1.0)
    obj.data.materials.append(mat)

    cam = setup_camera(args.width, args.height, args.fx, args.fy, args.cx, args.cy)
    add_lights()

    color_dir = args.out_dir / "rgb"
    label_dir = args.out_dir / "labels"
    color_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    local_corners = bbox_corners_from_info(info)
    all_records = []

    idx = 0
    attempts = 0
    max_attempts = args.num_images * 80
    while idx < args.num_images and attempts < max_attempts:
        attempts += 1
        obj.rotation_euler = (
            random.uniform(-math.pi, math.pi),
            random.uniform(-math.pi, math.pi),
            random.uniform(-math.pi, math.pi),
        )
        obj.location = (
            random.uniform(-3, 3),
            random.uniform(-3, 3),
            random.uniform(-3, 3),
        )

        cam.location = mathutils.Vector(
            (
                random.uniform(-70, 70),
                random.uniform(-760, -520),
                random.uniform(40, 160),
            )
        )
        camera_look_at(cam, obj.location)
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
        scene.render.filepath = str(color_dir / f"{stem}.png")
        bpy.ops.render.render(write_still=True)

        record = {
            "image": f"rgb/{stem}.png",
            "obj_id": 1,
            "corners_3d": [[float(v) for v in c] for c in local_corners],
            "corners_2d": corners_2d,
            "camera": {
                "width": args.width,
                "height": args.height,
                "fx": args.fx,
                "fy": args.fy,
                "cx": args.cx,
                "cy": args.cy,
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
