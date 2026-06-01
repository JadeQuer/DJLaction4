import argparse
import math
from pathlib import Path

import bpy
import mathutils

from render_corner_dataset import (
    add_front_small_screen_on_zmax_face,
    add_lights,
    add_reflection_strip_lights,
    add_dji_label_on_ymax_face,
    add_screen_image_on_zmin_face,
    camera_look_at,
    clear_scene,
    enhance_natural_reflections,
    ensure_reasonable_materials,
    import_mesh,
    make_mesh_materials_visible,
    normalize_object,
    setup_camera,
    setup_world,
)


VIEWS = [
    ("x_pos", (1.0, 0.0, 0.0)),
    ("x_neg", (-1.0, 0.0, 0.0)),
    ("y_pos", (0.0, 1.0, 0.0)),
    ("y_neg", (0.0, -1.0, 0.0)),
    ("z_pos", (0.0, 0.0, 1.0)),
    ("z_neg", (0.0, 0.0, -1.0)),
    ("diag_pos", (1.0, 1.0, 1.0)),
    ("diag_neg", (-1.0, -1.0, -1.0)),
]


def setup_scene(width, height):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.05
    scene.view_settings.gamma = 1.05
    scene.world.color = (0.60, 0.60, 0.60)
    setup_world(0.95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=Path("/root/autodl-fs/official.glb"), type=Path)
    parser.add_argument("--screen-image", default=Path("/root/autodl-tmp/DJLaction4/image.png"), type=Path)
    parser.add_argument("--front-small-screen-image", default=None, type=Path)
    parser.add_argument("--front-small-screen-x-margin-ratio", default=0.075, type=float)
    parser.add_argument("--front-small-screen-y-scale", default=1.0, type=float)
    parser.add_argument("--front-small-screen-overall-scale", default=0.95, type=float)
    parser.add_argument("--front-small-screen-offset-ratio", default=0.0015, type=float)
    parser.add_argument("--dji-label", default=None)
    parser.add_argument("--out-dir", default=Path("runs/screen_image_8views"), type=Path)
    parser.add_argument("--width", default=1400, type=int)
    parser.add_argument("--height", default=1000, type=int)
    parser.add_argument("--distance", default=360.0, type=float)
    args = parser.parse_args()

    clear_scene()
    setup_scene(args.width, args.height)

    obj = import_mesh(args.model)
    obj.name = "dji_action4"
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    normalize_object(obj, target_diameter=120.0)
    ensure_reasonable_materials(obj)
    make_mesh_materials_visible(obj, min_base=0.12)
    enhance_natural_reflections(obj)
    add_screen_image_on_zmin_face(obj, args.screen_image, inset=0.08)
    if args.front_small_screen_image:
        add_front_small_screen_on_zmax_face(
            obj,
            args.front_small_screen_image,
            x_margin_ratio=args.front_small_screen_x_margin_ratio,
            y_scale=args.front_small_screen_y_scale,
            overall_scale=args.front_small_screen_overall_scale,
            offset_ratio=args.front_small_screen_offset_ratio,
        )
    if args.dji_label:
        add_dji_label_on_ymax_face(obj, args.dji_label)

    cam = setup_camera(args.width, args.height, 980.0, 980.0, args.width * 0.5, args.height * 0.5)
    add_lights(scale=0.55)
    add_reflection_strip_lights(scale=0.35)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, direction in VIEWS:
        v = mathutils.Vector(direction).normalized()
        cam.location = v * args.distance
        camera_look_at(cam, (0.0, 0.0, 0.0))
        bpy.context.view_layer.update()
        bpy.context.scene.render.filepath = str(args.out_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"wrote {args.out_dir / f'{name}.png'}")


if __name__ == "__main__":
    main()
