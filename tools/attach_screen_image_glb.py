import argparse
from pathlib import Path

import bpy


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(path: Path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [obj for obj in set(bpy.data.objects) - before if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = "dji_action4"
    bpy.context.view_layer.update()
    return obj


def make_screen_material(image_path: Path):
    mat = bpy.data.materials.new("screen_photo_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.location = (-700, 0)
    tex.image = bpy.data.images.load(str(image_path))

    uv = nodes.new(type="ShaderNodeUVMap")
    uv.location = (-900, 0)
    uv.uv_map = "UVMap"

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (-250, 0)
    bsdf.inputs["Roughness"].default_value = 0.18
    if "Emission" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.65
    elif "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.65

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (50, 0)

    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission" in bsdf.inputs:
        links.new(tex.outputs["Color"], bsdf.inputs["Emission"])
    elif "Emission Color" in bsdf.inputs:
        links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def add_screen_plane(parent, image_path: Path):
    xs = [corner[0] for corner in parent.bound_box]
    ys = [corner[1] for corner in parent.bound_box]
    zs = [corner[2] for corner in parent.bound_box]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    dx, dy, dz = maxx - minx, maxy - miny, maxz - minz
    thickness = max(dx, dy, dz) * 0.006

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    plane.name = "screen_photo_plane"
    plane.parent = parent
    plane.location = ((minx + maxx) * 0.5, miny - thickness, (minz + maxz) * 0.5)
    plane.rotation_euler = (1.57079632679, 0.0, 0.0)
    plane.scale = (dx * 0.5, dz * 0.5, 1.0)

    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.reset()
    bpy.ops.object.mode_set(mode="OBJECT")

    mat = make_screen_material(image_path)
    plane.data.materials.append(mat)
    return plane


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=Path("/root/autodl-fs/official.glb"), type=Path)
    parser.add_argument("--image", default=Path("/root/autodl-tmp/DJLaction4/image.png"), type=Path)
    parser.add_argument("--output", default=Path("assets/official_screen_image.glb"), type=Path)
    args = parser.parse_args()

    clear_scene()
    obj = import_glb(args.input)
    add_screen_plane(obj, args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.output), export_format="GLB", use_selection=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
