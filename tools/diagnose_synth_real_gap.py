import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from corner_pose_resnet import CornerDataset, decode_heatmaps, preprocess_frame, ResNetCornerNet
from infer_video_yolo_resnet_pose import project_heatmap_points_to_frame, yolo_rois
from visualize_corner_dataset import draw_corners, make_heatmap_panel


EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def train(*_args, **_kwargs):
    # Compatibility shim for checkpoints that stored argparse's func=train.
    pass


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def bbox_from_points(pts):
    pts = np.asarray(pts, dtype=np.float32)[:, :2]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def face_metrics(corners):
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    front = pts[[0, 1, 3, 2]]
    back = pts[[4, 5, 7, 6]]
    x1, y1, x2, y2 = bbox_from_points(pts)
    bw, bh = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    return {
        "bbox_w": bw,
        "bbox_h": bh,
        "bbox_aspect": bw / bh,
        "front_area": polygon_area(front),
        "back_area": polygon_area(back),
        "front_back_area_ratio": polygon_area(front) / max(1e-6, polygon_area(back)),
        "diag_03": float(np.linalg.norm(pts[0] - pts[3])),
        "diag_47": float(np.linalg.norm(pts[4] - pts[7])),
    }


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


def make_contact_sheet(paths, out_path, thumb=(420, 420), cols=2):
    imgs = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append((Path(path).name, img))
    if not imgs:
        return
    label_h = 30
    tw, th = thumb
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.full((rows * (th + label_h), cols * tw, 3), 245, np.uint8)
    for i, (name, img) in enumerate(imgs):
        row, col = divmod(i, cols)
        x = col * tw
        y = row * (th + label_h)
        tile = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        sheet[y:y + label_h, x:x + tw] = 32
        cv2.putText(sheet, name[:36], (x + 8, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        sheet[y + label_h:y + label_h + th, x:x + tw] = tile
    cv2.imwrite(str(out_path), sheet)


def check_corner_chain(data_root, out_dir, count):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    root = Path(data_root)
    label_paths = sorted((root / "labels").glob("*.json"))[:count]
    overlays = []
    heatmaps = []
    metrics = []
    ds = CornerDataset(root, image_size=(256, 256), heatmap_size=(64, 64), sigma=1.8, roi=False, train=False, aug=False)
    for idx, lp in enumerate(label_paths):
        rec = json.loads(lp.read_text(encoding="utf-8"))
        img = cv2.imread(str(root / rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        corners = np.asarray(rec["corners_2d"], dtype=np.float32)
        overlay = draw_corners(img, corners, title=f"json order {lp.stem}")
        hm_panel = make_heatmap_panel(corners, w, h, hm_size=64, sigma=1.8, scale=4)
        img_t, hm_t, pts_t = ds[idx]
        decoded, _ = decode_heatmaps(hm_t.unsqueeze(0))
        err = np.linalg.norm(decoded[0] - pts_t.numpy(), axis=1)
        metrics.append({
            "label": str(lp),
            "max_heatmap_decode_err_hm_px": float(err.max()),
            "mean_heatmap_decode_err_hm_px": float(err.mean()),
            "face_metrics": face_metrics(corners),
        })
        ov_path = out_dir / f"{lp.stem}_json_order.png"
        hm_path = out_dir / f"{lp.stem}_heatmaps.png"
        cv2.imwrite(str(ov_path), overlay)
        cv2.imwrite(str(hm_path), hm_panel)
        overlays.append(ov_path)
        heatmaps.append(hm_path)
    make_contact_sheet(overlays, out_dir / "corner_order_contact_sheet.png", thumb=(420, 420), cols=2)
    make_contact_sheet(heatmaps, out_dir / "heatmap_contact_sheet.png", thumb=(640, 320), cols=1)
    return metrics


def collect_synth_stats(data_root, max_items):
    root = Path(data_root)
    stats = []
    for lp in sorted((root / "labels").glob("*.json"))[:max_items]:
        rec = json.loads(lp.read_text(encoding="utf-8"))
        img = cv2.imread(str(root / rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        pts = np.asarray(rec["corners_2d"], dtype=np.float32)
        x1, y1, x2, y2 = bbox_from_points(pts)
        item = {
            "name": lp.stem,
            "w_frac": (x2 - x1) / w,
            "h_frac": (y2 - y1) / h,
            "cx_frac": ((x1 + x2) * 0.5) / w,
            "cy_frac": ((y1 + y2) * 0.5) / h,
            "aspect": (x2 - x1) / max(1e-6, (y2 - y1)),
        }
        if "object_rotation_euler" in rec:
            item.update({
                "roll_like_x": float(rec["object_rotation_euler"][0]),
                "pitch_like_y": float(rec["object_rotation_euler"][1]),
                "yaw_like_z": float(rec["object_rotation_euler"][2]),
            })
        item.update({f"geom_{k}": v for k, v in face_metrics(pts).items()})
        stats.append(item)
    return stats


def collect_real_yolo_stats(video, detector_path, out_dir, max_frames, stride, conf, iou, roi_pad, max_rois):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    detector = YOLO(detector_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    stats = []
    debug_paths = []
    frame_idx = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        h, w = frame.shape[:2]
        raw_results = detector.predict(frame, conf=conf, iou=iou, max_det=max_rois, verbose=False)
        raw_boxes = []
        raw_scores = []
        if raw_results and raw_results[0].boxes is not None and len(raw_results[0].boxes) > 0:
            raw_boxes = raw_results[0].boxes.xyxy.detach().cpu().numpy()
            raw_scores = raw_results[0].boxes.conf.detach().cpu().numpy()
            raw_order = np.argsort(-raw_scores)[:max_rois]
            raw_boxes = raw_boxes[raw_order]
            raw_scores = raw_scores[raw_order]
        rois = yolo_rois(detector, frame, conf, iou, max_rois, roi_pad, roi_shrink=0.0, square_roi=True)
        vis = frame.copy()
        for roi_id, (score, x1, y1, x2, y2) in enumerate(rois):
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            item = {
                "frame_idx": frame_idx,
                "roi_id": roi_id,
                "score": float(score),
                "w_frac": bw / w,
                "h_frac": bh / h,
                "cx_frac": ((x1 + x2) * 0.5) / w,
                "cy_frac": ((y1 + y2) * 0.5) / h,
                "aspect": bw / max(1e-6, bh),
            }
            if roi_id < len(raw_boxes):
                rx1, ry1, rx2, ry2 = raw_boxes[roi_id]
                item.update({
                    "object_in_roi_w_frac": float((rx2 - rx1) / max(1e-6, bw)),
                    "object_in_roi_h_frac": float((ry2 - ry1) / max(1e-6, bh)),
                    "object_in_roi_cx_frac": float((((rx1 + rx2) * 0.5) - x1) / max(1e-6, bw)),
                    "object_in_roi_cy_frac": float((((ry1 + ry2) * 0.5) - y1) / max(1e-6, bh)),
                    "raw_box_aspect": float((rx2 - rx1) / max(1e-6, (ry2 - ry1))),
                })
            stats.append(item)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis, f"{roi_id}:{score:.2f}", (x1, max(24, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if written < 24:
            path = out_dir / f"real_yolo_frame_{written:03d}.png"
            cv2.imwrite(str(path), vis)
            debug_paths.append(path)
        written += 1
        frame_idx += 1
        if written >= max_frames:
            break
    cap.release()
    make_contact_sheet(debug_paths, out_dir / "real_yolo_rois_contact_sheet.png", thumb=(420, 300), cols=2)
    return stats


def summarize_stats(synth_stats, real_stats):
    keys = ["w_frac", "h_frac", "cx_frac", "cy_frac", "aspect"]
    out = {"synth": {}, "real_yolo": {}, "delta_real_minus_synth": {}}
    for key in keys:
        sv = [x[key] for x in synth_stats if key in x]
        rv = [x[key] for x in real_stats if key in x]
        out["synth"][key] = summarize(sv)
        out["real_yolo"][key] = summarize(rv)
        if sv and rv:
            out["delta_real_minus_synth"][key] = float(np.mean(rv) - np.mean(sv))
    rot_keys = ["roll_like_x", "pitch_like_y", "yaw_like_z"]
    out["synth_rotation_euler"] = {key: summarize([x[key] for x in synth_stats if key in x]) for key in rot_keys}
    out["real_object_inside_square_roi"] = {
        key: summarize([x[key] for x in real_stats if key in x])
        for key in ["object_in_roi_w_frac", "object_in_roi_h_frac", "object_in_roi_cx_frac", "object_in_roi_cy_frac", "raw_box_aspect"]
    }
    geom_keys = sorted([k for k in synth_stats[0].keys() if k.startswith("geom_")]) if synth_stats else []
    out["synth_geometry_2d"] = {key: summarize([x[key] for x in synth_stats if key in x]) for key in geom_keys}
    return out


def run_model_debug(data_root, ckpt_path, out_dir, count):
    if not ckpt_path or not Path(ckpt_path).exists():
        return []
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetCornerNet(backbone="resnet18", pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    root = Path(data_root)
    label_paths = sorted((root / "labels").glob("*.json"))[:count]
    debug_paths = []
    rows = []
    with torch.no_grad():
        for lp in label_paths:
            rec = json.loads(lp.read_text(encoding="utf-8"))
            img = cv2.imread(str(root / rec["image"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            inp = preprocess_frame(img, image_size=(256, 256), keep_aspect=False).to(device)
            logits = model(inp)
            pred_hm, conf = decode_heatmaps(logits)
            pred = pred_hm[0].copy()
            pred[:, 0] = pred[:, 0] / 64.0 * w
            pred[:, 1] = pred[:, 1] / 64.0 * h
            gt = np.asarray(rec["corners_2d"], dtype=np.float32)[:, :2]
            err = np.linalg.norm(pred - gt, axis=1)
            vis = img.copy()
            for i, (gx, gy) in enumerate(gt):
                cv2.circle(vis, (int(round(gx)), int(round(gy))), 8, (0, 255, 0), 2)
                cv2.putText(vis, f"g{i}", (int(gx) + 6, int(gy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            for i, (px, py) in enumerate(pred):
                cv2.circle(vis, (int(round(px)), int(round(py))), 5, (0, 255, 255), -1)
                cv2.putText(vis, f"p{i}", (int(px) + 6, int(py) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.line(vis, (int(round(gt[i, 0])), int(round(gt[i, 1]))), (int(round(px)), int(round(py))), (0, 0, 255), 2)
            path = out_dir / f"{lp.stem}_gt_pred.png"
            cv2.imwrite(str(path), vis)
            debug_paths.append(path)
            rows.append({"label": str(lp), "mean_err_px": float(err.mean()), "max_err_px": float(err.max()), "mean_conf": float(np.mean(conf[0]))})
    make_contact_sheet(debug_paths, out_dir / "synthetic_gt_pred_contact_sheet.png", thumb=(420, 420), cols=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-root", default="datasets/dji_action4_defaults_locked_check_12_mytry4")
    ap.add_argument("--video", default="/root/autodl-fs/head_left_rgb_raw.mp4")
    ap.add_argument("--detector", default="runs_pre/detect/runs/dji_action4_yolo_real_full_ft/weights/best.pt")
    ap.add_argument("--ckpt", default="runs/corner_resnet18_mytry4_400/best.pt")
    ap.add_argument("--out-dir", default="runs/diagnostics_synth_real_gap")
    ap.add_argument("--synth-count", type=int, default=64)
    ap.add_argument("--real-frames", type=int, default=120)
    ap.add_argument("--real-stride", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out_dir)
    ensure_dir(out)
    corner_metrics = check_corner_chain(args.synth_root, out / "corner_chain", count=min(16, args.synth_count))
    synth_stats = collect_synth_stats(args.synth_root, max_items=args.synth_count)
    real_stats = collect_real_yolo_stats(args.video, args.detector, out / "real_yolo", args.real_frames, args.real_stride, 0.25, 0.5, 0.08, 2)
    model_rows = run_model_debug(args.synth_root, args.ckpt, out / "model_synthetic_gt_pred", count=min(16, args.synth_count))
    report = {
        "corner_chain": corner_metrics,
        "distribution_summary": summarize_stats(synth_stats, real_stats),
        "synthetic_model_debug": model_rows,
        "notes": {
            "corner_edges": EDGES,
            "training_default": "corner_pose_resnet.py train currently uses full synthetic crop unless --roi is passed.",
            "inference_default": "infer_video_yolo_resnet_pose.py uses YOLO square ROI and letterbox unless --stretch-roi is passed.",
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "report": str(out / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
