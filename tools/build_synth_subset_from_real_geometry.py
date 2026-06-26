import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from compare_corner_geometry_distribution import FACES


EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def descriptor(corners, width, height):
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw = max(float(x2 - x1), 1e-6)
    bh = max(float(y2 - y1), 1e-6)
    rel = np.stack([(pts[:, 0] - x1) / bw, (pts[:, 1] - y1) / bh], axis=1)
    face_areas = {name: polygon_area(pts[idxs]) for name, idxs in FACES.items()}
    total = sum(face_areas.values()) + 1e-6
    face_vec = np.asarray([face_areas[k] / total for k in FACES], dtype=np.float32)
    edge_lens = np.asarray([np.linalg.norm(rel[a] - rel[b]) for a, b in EDGES], dtype=np.float32)
    return np.concatenate([
        (rel - 0.5).reshape(-1) * 1.0,
        np.asarray([bw / bh], dtype=np.float32) * 1.8,
        face_vec * 3.2,
        edge_lens * 0.20,
    ])


def load_items(root, is_real=False):
    root = Path(root)
    out = []
    for lp in sorted((root / "labels").glob("*.json")):
        rec = json.loads(lp.read_text(encoding="utf-8"))
        img_path = root / rec["image"]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        corners = rec.get("corners_2d")
        if not corners or len(corners) != 8:
            continue
        desc = descriptor(corners, w, h)
        pts = np.asarray(corners, dtype=np.float32)[:, :2]
        bw = float(pts[:, 0].max() - pts[:, 0].min())
        bh = float(pts[:, 1].max() - pts[:, 1].min())
        face_areas = {name: polygon_area(pts[idxs]) for name, idxs in FACES.items()}
        top_face = max(face_areas, key=face_areas.get)
        out.append({
            "root": str(root),
            "label_path": str(lp),
            "image_path": str(img_path),
            "record": rec,
            "desc": desc,
            "bbox_aspect": bw / max(1e-6, bh),
            "top_face": top_face,
            "is_real": is_real,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", action="append", required=True)
    ap.add_argument("--synth-root", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=600)
    ap.add_argument("--per-real", type=int, default=8)
    ap.add_argument("--max-per-source", type=int, default=260)
    ap.add_argument("--target-zmax-frac", type=float, default=0.66)
    ap.add_argument("--target-zmin-frac", type=float, default=0.34)
    ap.add_argument("--max-x-face-frac", type=float, default=0.18)
    ap.add_argument("--aspect-min", type=float, default=0.95)
    ap.add_argument("--aspect-max", type=float, default=1.60)
    args = ap.parse_args()

    real = []
    for root in args.real_root:
        real.extend(load_items(root, is_real=True))
    synth = []
    for root in args.synth_root:
        synth.extend(load_items(root, is_real=False))
    if not real or not synth:
        raise RuntimeError(f"need real and synth, got {len(real)} real {len(synth)} synth")

    real_desc = np.stack([x["desc"] for x in real], axis=0)
    synth_desc = np.stack([x["desc"] for x in synth], axis=0)
    d = np.linalg.norm(synth_desc[:, None, :] - real_desc[None, :, :], axis=2)
    source_counts = {}
    selected = []
    selected_ids = set()
    top_counts = {}

    def face_quota_ok(item):
        n = max(1, len(selected))
        top = item["top_face"]
        if top in ("x_min_0-1-3-2", "x_max_4-5-7-6"):
            x_count = top_counts.get("x_min_0-1-3-2", 0) + top_counts.get("x_max_4-5-7-6", 0)
            return (x_count + 1) / n <= args.max_x_face_frac
        if top == "z_max_1-3-7-5":
            return top_counts.get(top, 0) < args.count * args.target_zmax_frac
        if top == "z_min_0-2-6-4":
            return top_counts.get(top, 0) < args.count * args.target_zmin_frac
        return True

    def add_selected(dist, real_item, item, synth_idx):
        selected.append((dist, real_item, item))
        selected_ids.add(synth_idx)
        src = item["root"]
        source_counts[src] = source_counts.get(src, 0) + 1
        top_counts[item["top_face"]] = top_counts.get(item["top_face"], 0) + 1

    nearest_order = np.argsort(d, axis=0)
    for real_idx in range(len(real)):
        added = 0
        for synth_idx in nearest_order[:, real_idx]:
            synth_idx = int(synth_idx)
            if synth_idx in selected_ids:
                continue
            item = synth[synth_idx]
            src = item["root"]
            if not (args.aspect_min <= item["bbox_aspect"] <= args.aspect_max):
                continue
            if not face_quota_ok(item):
                continue
            if source_counts.get(src, 0) >= args.max_per_source:
                continue
            add_selected(float(d[synth_idx, real_idx]), real[real_idx], item, synth_idx)
            added += 1
            if added >= args.per_real or len(selected) >= args.count:
                break
        if len(selected) >= args.count:
            break
    if len(selected) < args.count:
        best_any = np.min(d, axis=1)
        for synth_idx in np.argsort(best_any):
            synth_idx = int(synth_idx)
            if synth_idx in selected_ids:
                continue
            item = synth[synth_idx]
            src = item["root"]
            if not (args.aspect_min <= item["bbox_aspect"] <= args.aspect_max):
                continue
            if not face_quota_ok(item):
                continue
            if source_counts.get(src, 0) >= args.max_per_source:
                continue
            real_idx = int(np.argmin(d[synth_idx]))
            add_selected(float(best_any[synth_idx]), real[real_idx], item, synth_idx)
            if len(selected) >= args.count:
                break

    out = Path(args.out_dir)
    rgb = out / "rgb"
    labels = out / "labels"
    rgb.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    records = []
    report_rows = []
    for i, (dist, real_item, item) in enumerate(selected):
        stem = f"{i:06d}"
        dst_img = rgb / f"{stem}.png"
        shutil.copy2(item["image_path"], dst_img)
        rec = json.loads(json.dumps(item["record"]))
        rec["image"] = f"rgb/{stem}.png"
        rec["source_subset"] = {
            "source_root": item["root"],
            "source_label": item["label_path"],
            "matched_real_root": real_item["root"],
            "matched_real_label": real_item["label_path"],
            "geometry_distance": dist,
        }
        (labels / f"{stem}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        records.append(rec)
        report_rows.append({
            "name": stem,
            "distance": dist,
            "source_root": item["root"],
            "source_label": item["label_path"],
            "matched_real_label": real_item["label_path"],
            "top_face": item["top_face"],
            "aspect": item["bbox_aspect"],
        })
    (out / "labels.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    report = {
        "out": str(out),
        "count": len(records),
        "real_count": len(real),
        "synth_count": len(synth),
        "source_counts": source_counts,
        "top_face_counts": top_counts,
        "mean_distance": float(np.mean([r["distance"] for r in report_rows])) if report_rows else None,
        "rows": report_rows,
    }
    (out / "subset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("out", "count", "real_count", "synth_count", "source_counts", "mean_distance")}, indent=2))


if __name__ == "__main__":
    main()
