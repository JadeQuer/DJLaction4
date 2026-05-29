import argparse
from pathlib import Path

import bpy
import mathutils

from render_corner_dataset import (
    add_lights,
    add_reflection_strip_lights,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=Path("/root/autodl-fs/official.glb"), type=Path)
    parser.add_argument("--screen-image", default=Path("/root/autodl-tmp/DJLaction4/image.png"), type=Path)
    parser.add_argument("--out", default=Path("runs/screen_image_face_check.png"), type=Path)
    parser.add_argument("--width", default=1600, type=int)
    parser.add_argument("--height", default=1200, type=int)
    args = parser.parse_args()

    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.05
    scene.view_settings.gamma = 1.05
    scene.world.color = (0.60, 0.60, 0.60)
    setup_world(0.95)

    obj = import_mesh(args.model)
    obj.name = "dji_action4"
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    normalize_object(obj, target_diameter=120.0)
    ensure_reasonable_materials(obj)
    make_mesh_materials_visible(obj, min_base=0.12)
    enhance_natural_reflections(obj)
    add_screen_image_on_zmin_face(obj, args.screen_image, inset=0.08)

    cam = setup_camera(args.width, args.height, 980.0, 980.0, args.width * 0.5, args.height * 0.5)
    cam.location = mathutils.Vector((0.0, 0.0, -360.0))
    camera_look_at(cam, (0.0, 0.0, 0.0))
    add_lights(scale=0.55)
    add_reflection_strip_lights(scale=0.35)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.out)
    bpy.ops.render.render(write_still=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
