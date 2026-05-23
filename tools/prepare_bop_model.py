import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


DJI_ACTION4_SIZE_MM = np.array([70.5, 32.8, 44.2], dtype=np.float64)


def _as_mesh(loaded):
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
                meshes.append(geom)
        if not meshes:
            raise ValueError("No mesh geometry found in scene")
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise TypeError(f"Unsupported mesh type: {type(loaded)!r}")


def _model_info(mesh):
    vertices = np.asarray(mesh.vertices)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extents = vmax - vmin
    diameter = float(np.linalg.norm(extents))
    return {
        "diameter": diameter,
        "max_x": float(vmax[0]),
        "max_y": float(vmax[1]),
        "max_z": float(vmax[2]),
        "min_x": float(vmin[0]),
        "min_y": float(vmin[1]),
        "min_z": float(vmin[2]),
        "size_x": float(extents[0]),
        "size_y": float(extents[1]),
        "size_z": float(extents[2]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--obj-id", default=1, type=int)
    parser.add_argument(
        "--size-mm",
        default="70.5,32.8,44.2",
        help="Target XYZ extents in millimeters. Default maps X=width, Y=depth, Z=height.",
    )
    args = parser.parse_args()

    target_size = np.array([float(x) for x in args.size_mm.split(",")], dtype=np.float64)
    if target_size.shape != (3,):
        raise ValueError("--size-mm must contain exactly three comma-separated numbers")

    loaded = trimesh.load(args.input, force=None)
    mesh = _as_mesh(loaded)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    mesh.apply_translation(-center)

    extents = mesh.extents.astype(np.float64)
    if np.any(extents <= 0):
        raise ValueError(f"Invalid mesh extents: {extents}")

    # Use anisotropic scaling so the coarse generated mesh matches the official
    # DJI Action 4 bounding-box dimensions used later for corner labels/PnP.
    scale = target_size / extents
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    models_dir = args.dataset_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_ply = models_dir / f"obj_{args.obj_id:06d}.ply"
    mesh.export(out_ply)

    info = {str(args.obj_id): _model_info(mesh)}
    (models_dir / "models_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    camera = {
        "cx": 325.2611083984375,
        "cy": 242.04899588216654,
        "depth_scale": 0.1,
        "fx": 572.411363389757,
        "fy": 573.5704328585578,
        "height": 480,
        "width": 640,
    }
    (args.dataset_dir / "camera.json").write_text(json.dumps(camera, indent=2), encoding="utf-8")

    print(f"wrote {out_ply}")
    print(f"extents_mm {mesh.extents.tolist()}")
    print(f"models_info {models_dir / 'models_info.json'}")


if __name__ == "__main__":
    main()
