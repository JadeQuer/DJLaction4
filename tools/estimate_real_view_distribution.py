import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


FACES = {
    "x_min_0-1-3-2": [0, 1, 3, 2],
    "x_max_4-5-7-6": [4, 5, 7, 6],
    "y_min_0-1-5-4": [0, 1, 5, 4],
    "y_max_2-3-7-6": [2, 3, 7, 6],
    "z_min_0-2-6-4": [0, 2, 6, 4],
    "z_max_1-3-7-5": [1, 3, 7, 5],
}


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def face_summary(corners):
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    areas = {name: polygon_area(pts[idxs]) for name, idxs in FACES.items()}
    total = sum(areas.values()) + 1e-6
    weights = {name: val / total for name, val in areas.items()}
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "face_weights": weights,
        "top_face": ranked[0][0],
        "second_face": ranked[1][0],
        "top_face_weight": ranked[0][1],
    }


def square_box(x1, y1, x2, y2, w, h, pad=0.08):
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = min(max(bw, bh) * (1.0 + 2.0 * pad), w - 1, h - 1)
    x1, x2 = cx - side * 0.5, cx + side * 0.5
    y1, y2 = cy - side * 0.5, cy + side * 0.5
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w - 1:
        x1 -= x2 - (w - 1)
        x2 = w - 1
    if y2 > h - 1:
        y1 -= y2 - (h - 1)
        y2 = h - 1
    return [int(max(0, np.floor(x1))), int(max(0, np.floor(y1))), int(min(w - 1, np.ceil(x2))), int(min(h - 1, np.ceil(y2)))]


def descriptor(img):
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
    hog_feat = hog.compute(gray).reshape(-1)
    hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1)
    edge_small = cv2.resize(edges, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32) / 255.0
    feat = np.concatenate([
        hog_feat.astype(np.float32) * 0.35,
        hist_h.astype(np.float32),
        hist_s.astype(np.float32),
        hist_v.astype(np.float32),
        edge_small * 20.0,
        np.array([gray.mean(), gray.std(), (edges > 0).mean() * 4096.0], dtype=np.float32),
    ])
    return feat / (np.linalg.norm(feat) + 1e-6)


def load_synth(root, max_items):
    root = Path(root)
    items = []
    for lp in sorted((root / "labels").glob("*.json")):
        rec = json.loads(lp.read_text(encoding="utf-8"))
        img = cv2.imread(str(root / rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        fs = face_summary(rec["corners_2d"])
        items.append({
            "label": str(lp),
            "name": lp.stem,
            "image": img,
            "desc": descriptor(img),
            **fs,
        })
        if max_items and len(items) >= max_items:
            break
    return items


def collect_real(video, detector_path, max_frames, stride, max_rois, conf, iou, roi_pad):
    detector = YOLO(detector_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    rows = []
    frame_idx = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        h, w = frame.shape[:2]
        res = detector.predict(frame, conf=conf, iou=iou, max_det=max_rois, verbose=False)
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            boxes = res[0].boxes.xyxy.detach().cpu().numpy()
            scores = res[0].boxes.conf.detach().cpu().numpy()
            for roi_id, bi in enumerate(np.argsort(-scores)[:max_rois]):
                x1, y1, x2, y2 = boxes[bi]
                sx1, sy1, sx2, sy2 = square_box(x1, y1, x2, y2, w, h, pad=roi_pad)
                crop = frame[sy1:sy2 + 1, sx1:sx2 + 1]
                if crop.size == 0:
                    continue
                rows.append({
                    "frame_idx": frame_idx,
                    "roi_id": roi_id,
                    "score": float(scores[bi]),
                    "bbox": [sx1, sy1, sx2, sy2],
                    "image": crop,
                    "desc": descriptor(crop),
                })
        written += 1
        frame_idx += 1
        if written >= max_frames:
            break
    cap.release()
    return rows


def make_sheet(matches, out_path, thumb=(256, 256), cols=3):
    if not matches:
        return
    tw, th = thumb
    label_h = 48
    rows = (len(matches) + cols - 1) // cols
    sheet = np.full((rows * (th * 2 + label_h), cols * tw, 3), 235, np.uint8)
    for i, m in enumerate(matches):
        r, c = divmod(i, cols)
        x = c * tw
        y = r * (th * 2 + label_h)
        real = cv2.resize(m["real_image"], (tw, th), interpolation=cv2.INTER_AREA)
        synth = cv2.resize(m["synth_image"], (tw, th), interpolation=cv2.INTER_AREA)
        sheet[y:y + label_h, x:x + tw] = 28
        text = f"f{m['frame_idx']} r{m['roi_id']} {m['top_face']} d={m['distance']:.3f}"
        cv2.putText(sheet, text[:34], (x + 5, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, m["synth"][:34], (x + 5, y + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        sheet[y + label_h:y + label_h + th, x:x + tw] = real
        sheet[y + label_h + th:y + label_h + th * 2, x:x + tw] = synth
    cv2.imwrite(str(out_path), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/root/autodl-fs/head_left_rgb_raw.mp4")
    ap.add_argument("--detector", default="runs_pre/detect/runs/dji_action4_yolo_real_full_ft/weights/best.pt")
    ap.add_argument("--synth-root", default="datasets/dji_action4_defaults_locked_check_12_mytry4")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-synth", type=int, default=400)
    ap.add_argument("--real-frames", type=int, default=160)
    ap.add_argument("--real-stride", type=int, default=6)
    ap.add_argument("--max-rois", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.5)
    ap.add_argument("--roi-pad", type=float, default=0.08)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synth = load_synth(args.synth_root, args.max_synth)
    real = collect_real(args.video, args.detector, args.real_frames, args.real_stride, args.max_rois, args.det_conf, args.det_iou, args.roi_pad)
    if not synth or not real:
        raise RuntimeError(f"Need synth and real, got {len(synth)} synth and {len(real)} real")

    face_counts = {name: 0.0 for name in FACES}
    top_counts = {name: 0 for name in FACES}
    rows = []
    sheet_items = []
    synth_desc = np.stack([x["desc"] for x in synth], axis=0)
    for rr in real:
        dists = np.linalg.norm(synth_desc - rr["desc"][None, :], axis=1)
        order = np.argsort(dists)[:args.top_k]
        local = dists[order]
        tau = max(float(np.std(local)), 0.03)
        weights = np.exp(-(local - float(local.min())) / tau)
        weights = weights / (weights.sum() + 1e-6)
        agg = {name: 0.0 for name in FACES}
        for wi, idx in zip(weights, order):
            item = synth[int(idx)]
            for face, val in item["face_weights"].items():
                agg[face] += float(wi) * float(val)
        top_face = max(agg, key=agg.get)
        top_counts[top_face] += 1
        for face, val in agg.items():
            face_counts[face] += val
        best = synth[int(order[0])]
        rec = {
            "frame_idx": rr["frame_idx"],
            "roi_id": rr["roi_id"],
            "det_score": rr["score"],
            "bbox": rr["bbox"],
            "top_face": top_face,
            "face_scores": agg,
            "best_synth": best["name"],
            "best_synth_label": best["label"],
            "best_synth_top_face": best["top_face"],
            "distance": float(dists[order[0]]),
        }
        rows.append(rec)
        if len(sheet_items) < 36:
            sheet_items.append({
                **rec,
                "real_image": rr["image"],
                "synth_image": best["image"],
                "synth": best["name"],
            })

    total = max(1, len(rows))
    summary = {
        "num_real_rois": len(rows),
        "num_synth": len(synth),
        "weighted_face_distribution": {k: float(v / total) for k, v in face_counts.items()},
        "top_face_counts": top_counts,
        "top_face_fraction": {k: float(v / total) for k, v in top_counts.items()},
    }
    (out_dir / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    make_sheet(sheet_items, out_dir / "matched_view_contact_sheet.png")
    print(json.dumps({"out": str(out_dir), "summary": summary, "contact_sheet": str(out_dir / "matched_view_contact_sheet.png")}, indent=2))


if __name__ == "__main__":
    main()
