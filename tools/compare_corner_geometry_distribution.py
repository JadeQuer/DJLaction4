import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


FACES = {
    "x_min_0-1-3-2": [0, 1, 3, 2],
    "x_max_4-5-7-6": [4, 5, 7, 6],
    "y_min_0-1-5-4": [0, 1, 5, 4],
    "y_max_2-3-7-6": [2, 3, 7, 6],
    "z_min_0-2-6-4": [0, 2, 6, 4],
    "z_max_1-3-7-5": [1, 3, 7, 5],
}


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def summarize(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def load_records(root):
    root = Path(root)
    rows = []
    for lp in sorted((root / "labels").glob("*.json")):
        rec = json.loads(lp.read_text(encoding="utf-8"))
        img = cv2.imread(str(root / rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        pts = np.asarray(rec["corners_2d"], dtype=np.float32)[:, :2]
        if pts.shape != (8, 2):
            continue
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        bw = max(float(x2 - x1), 1e-6)
        bh = max(float(y2 - y1), 1e-6)
        face_areas = {name: polygon_area(pts[idxs]) for name, idxs in FACES.items()}
        total_area = sum(face_areas.values()) + 1e-6
        face_weights = {name: val / total_area for name, val in face_areas.items()}
        top_face = max(face_weights, key=face_weights.get)
        rel = np.stack([(pts[:, 0] - x1) / bw, (pts[:, 1] - y1) / bh], axis=1)
        src = rec.get("source", {})
        rows.append({
            "name": lp.stem,
            "w": w,
            "h": h,
            "bbox_w_frac": bw / w,
            "bbox_h_frac": bh / h,
            "bbox_aspect": bw / bh,
            "bbox_cx": ((x1 + x2) * 0.5) / w,
            "bbox_cy": ((y1 + y2) * 0.5) / h,
            "top_face": top_face,
            "face_weights": face_weights,
            "rel_corners": rel.tolist(),
            "frame_id": src.get("frame_id"),
            "roi_index": src.get("roi_index"),
        })
    return rows


def summarize_dataset(rows):
    out = {}
    for key in ["bbox_w_frac", "bbox_h_frac", "bbox_aspect", "bbox_cx", "bbox_cy"]:
        out[key] = summarize([r[key] for r in rows])
    top_counts = Counter(r["top_face"] for r in rows)
    out["top_face_counts"] = dict(top_counts)
    out["top_face_fraction"] = {k: v / max(1, len(rows)) for k, v in top_counts.items()}
    out["face_weight_mean"] = {
        face: float(np.mean([r["face_weights"][face] for r in rows])) if rows else 0.0
        for face in FACES
    }
    rel = np.asarray([r["rel_corners"] for r in rows], dtype=np.float32)
    if rel.size:
        out["relative_corner_mean"] = rel.mean(axis=0).tolist()
        out["relative_corner_std"] = rel.std(axis=0).tolist()
    by_phase = defaultdict(list)
    for r in rows:
        frame_id = r.get("frame_id")
        if frame_id is None:
            phase = "unknown"
        elif frame_id < 90:
            phase = "front_0_3s"
        else:
            phase = "post_3s"
        by_phase[phase].append(r)
    out["by_phase"] = {k: summarize_dataset_shallow(v) for k, v in by_phase.items()}
    return out


def summarize_dataset_shallow(rows):
    return {
        "count": len(rows),
        "bbox_aspect": summarize([r["bbox_aspect"] for r in rows]),
        "bbox_w_frac": summarize([r["bbox_w_frac"] for r in rows]),
        "bbox_h_frac": summarize([r["bbox_h_frac"] for r in rows]),
        "top_face_fraction": {
            k: v / max(1, len(rows)) for k, v in Counter(r["top_face"] for r in rows).items()
        },
        "face_weight_mean": {
            face: float(np.mean([r["face_weights"][face] for r in rows])) if rows else 0.0
            for face in FACES
        },
    }


def draw_mean_shape(rows, out_path, title):
    rel = np.asarray([r["rel_corners"] for r in rows], dtype=np.float32)
    if rel.size == 0:
        return
    mean = rel.mean(axis=0)
    std = rel.std(axis=0)
    canvas = np.full((520, 520, 3), 245, dtype=np.uint8)
    pts = np.stack([70 + mean[:, 0] * 380, 70 + mean[:, 1] * 380], axis=1)
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        cv2.line(canvas, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), (0, 160, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        radius = int(max(4, min(24, 4 + 80 * float(std[i].mean()))))
        p_int = tuple(np.round(p).astype(int))
        cv2.circle(canvas, p_int, radius, (180, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p_int, 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (p_int[0] + 9, p_int[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2, cv2.LINE_AA)
    cv2.putText(canvas, title[:54], (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", action="append", required=True)
    ap.add_argument("--synth-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    real_rows = []
    for root in args.real_root:
        real_rows.extend(load_records(root))
    synth_rows = load_records(args.synth_root)

    report = {
        "real_roots": args.real_root,
        "synth_root": args.synth_root,
        "real_count": len(real_rows),
        "synth_count": len(synth_rows),
        "real": summarize_dataset(real_rows),
        "synth": summarize_dataset(synth_rows),
    }
    (out / "report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    (out / "real_rows.json").write_text(json.dumps(json_safe(real_rows), indent=2), encoding="utf-8")
    (out / "synth_rows.json").write_text(json.dumps(json_safe(synth_rows), indent=2), encoding="utf-8")
    draw_mean_shape(real_rows, out / "real_mean_corner_shape.png", "real manual labels mean relative corner shape")
    draw_mean_shape(synth_rows, out / "synth_mean_corner_shape.png", "synthetic train mean relative corner shape")
    print(json.dumps({"out": str(out), "real": len(real_rows), "synth": len(synth_rows), "report": str(out / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
