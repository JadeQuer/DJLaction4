import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from corner_pose_resnet import bbox_from_points, crop_and_resize


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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


def image_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 60, 160)
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "hsv_s_mean": float(hsv[:, :, 1].mean()),
        "hsv_v_mean": float(hsv[:, :, 2].mean()),
        "lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_frac": float((edges > 0).mean()),
    }


def make_contact_sheet(items, out_path, thumb=(256, 256), cols=6):
    if not items:
        return
    tw, th = thumb
    label_h = 28
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * (th + label_h), cols * tw, 3), 238, dtype=np.uint8)
    for i, (name, img) in enumerate(items):
        row, col = divmod(i, cols)
        x = col * tw
        y = row * (th + label_h)
        tile = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        sheet[y:y + label_h, x:x + tw] = 28
        cv2.putText(sheet, name[:28], (x + 6, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        sheet[y + label_h:y + label_h + th, x:x + tw] = tile
    cv2.imwrite(str(out_path), sheet)


def collect_synth(root, count, image_size, roi_pad, roi_jitter, start_index=0, sample_step=1):
    root = Path(root)
    rows = []
    tiles = []
    label_paths = sorted((root / "labels").glob("*.json"))
    selected = label_paths[start_index::max(1, sample_step)][:count]
    for lp in selected:
        rec = json.loads(lp.read_text(encoding="utf-8"))
        img = cv2.imread(str(root / rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        corners = np.asarray(rec["corners_2d"], dtype=np.float32)
        bbox = bbox_from_points(corners[:, :2], w, h, pad_ratio=roi_pad, jitter=roi_jitter, square=True)
        crop, bbox, meta = crop_and_resize(img, bbox, (image_size, image_size), keep_aspect=True)
        x1, y1, x2, y2 = bbox
        xs = corners[:, 0]
        ys = corners[:, 1]
        row = {
            "name": lp.stem,
            "roi_w_frac_in_frame": float((x2 - x1 + 1.0) / w),
            "roi_h_frac_in_frame": float((y2 - y1 + 1.0) / h),
            "object_w_frac_in_roi": float((xs.max() - xs.min()) / max(1e-6, x2 - x1 + 1.0)),
            "object_h_frac_in_roi": float((ys.max() - ys.min()) / max(1e-6, y2 - y1 + 1.0)),
            "object_aspect": float((xs.max() - xs.min()) / max(1e-6, ys.max() - ys.min())),
        }
        row.update(image_metrics(crop))
        rows.append(row)
        if len(tiles) < 36:
            tiles.append((f"s_{lp.stem}", crop))
    return rows, tiles


def collect_real(video, detector_path, frame_count, stride, image_size, conf, iou, roi_pad, max_rois, start_frame=0):
    detector = YOLO(detector_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    rows = []
    tiles = []
    start_frame = max(0, int(start_frame))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        h, w = frame.shape[:2]
        results = detector.predict(frame, conf=conf, iou=iou, max_det=max_rois, verbose=False)
        dets = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.detach().cpu().numpy()
            scores = results[0].boxes.conf.detach().cpu().numpy()
            order = np.argsort(-scores)[:max_rois]
            for idx in order:
                rx1, ry1, rx2, ry2 = [float(v) for v in boxes[idx]]
                score = float(scores[idx])
                bw, bh = rx2 - rx1, ry2 - ry1
                cx, cy = (rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5
                side = max(bw, bh) * (1.0 + 2.0 * roi_pad)
                side = min(side, float(w - 1), float(h - 1))
                x1 = cx - side * 0.5
                x2 = cx + side * 0.5
                y1 = cy - side * 0.5
                y2 = cy + side * 0.5
                if x1 < 0:
                    x2 -= x1
                    x1 = 0.0
                if y1 < 0:
                    y2 -= y1
                    y1 = 0.0
                if x2 > w - 1:
                    x1 -= x2 - (w - 1)
                    x2 = float(w - 1)
                if y2 > h - 1:
                    y1 -= y2 - (h - 1)
                    y2 = float(h - 1)
                x1 = int(max(0, np.floor(x1)))
                y1 = int(max(0, np.floor(y1)))
                x2 = int(min(w - 1, np.ceil(x2)))
                y2 = int(min(h - 1, np.ceil(y2)))
                if x2 > x1 and y2 > y1:
                    dets.append((score, x1, y1, x2, y2, rx1, ry1, rx2, ry2))
        dets.sort(key=lambda x: (x[1] + x[3]) * 0.5)
        for roi_id, (score, x1, y1, x2, y2, rx1, ry1, rx2, ry2) in enumerate(dets):
            crop_raw = frame[y1:y2 + 1, x1:x2 + 1]
            if crop_raw.size == 0:
                continue
            roi_w = max(1e-6, x2 - x1 + 1.0)
            roi_h = max(1e-6, y2 - y1 + 1.0)
            crop, _bbox, _meta = crop_and_resize(crop_raw, [0, 0, crop_raw.shape[1] - 1, crop_raw.shape[0] - 1], (image_size, image_size), keep_aspect=True)
            row = {
                "frame_idx": frame_idx,
                "roi_id": roi_id,
                "score": float(score),
                "roi_w_frac_in_frame": float(roi_w / w),
                "roi_h_frac_in_frame": float(roi_h / h),
                "object_w_frac_in_roi": float((rx2 - rx1) / roi_w),
                "object_h_frac_in_roi": float((ry2 - ry1) / roi_h),
                "object_cx_frac_in_roi": float((((rx1 + rx2) * 0.5) - x1) / roi_w),
                "object_cy_frac_in_roi": float((((ry1 + ry2) * 0.5) - y1) / roi_h),
                "object_aspect": float((rx2 - rx1) / max(1e-6, ry2 - ry1)),
            }
            row.update(image_metrics(crop))
            rows.append(row)
            if len(tiles) < 36:
                tiles.append((f"r_{frame_idx}_{roi_id}", crop))
        written += 1
        frame_idx += 1
        if written >= frame_count:
            break
    cap.release()
    return rows, tiles


def summarize_rows(rows):
    keys = sorted({k for row in rows for k in row if isinstance(row[k], (int, float))})
    return {key: summarize([row[key] for row in rows if key in row]) for key in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-root", required=True)
    ap.add_argument("--video", default="/root/autodl-fs/head_left_rgb_raw.mp4")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synth-count", type=int, default=400)
    ap.add_argument("--synth-start-index", type=int, default=0)
    ap.add_argument("--synth-sample-step", type=int, default=1)
    ap.add_argument("--real-frames", type=int, default=160)
    ap.add_argument("--real-stride", type=int, default=3)
    ap.add_argument("--real-start-frame", type=int, default=0)
    ap.add_argument("--real-start-sec", type=float)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--roi-pad", type=float, default=0.08)
    ap.add_argument("--roi-jitter", type=float, default=0.0)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.5)
    ap.add_argument("--max-rois", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    real_start_frame = args.real_start_frame
    if args.real_start_sec is not None:
        cap = cv2.VideoCapture(str(args.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        real_start_frame = int(round(args.real_start_sec * fps))
    synth_rows, synth_tiles = collect_synth(
        args.synth_root,
        args.synth_count,
        args.image_size,
        args.roi_pad,
        args.roi_jitter,
        args.synth_start_index,
        args.synth_sample_step,
    )
    real_rows, real_tiles = collect_real(
        args.video,
        args.detector,
        args.real_frames,
        args.real_stride,
        args.image_size,
        args.det_conf,
        args.det_iou,
        args.roi_pad,
        args.max_rois,
        real_start_frame,
    )
    make_contact_sheet(synth_tiles, out_dir / "synth_train_roi_contact_sheet.png")
    make_contact_sheet(real_tiles, out_dir / "real_yolo_roi_contact_sheet.png")
    report = {
        "synth_root": args.synth_root,
        "video": args.video,
        "detector": args.detector,
        "roi_pad": args.roi_pad,
        "synth_start_index": args.synth_start_index,
        "synth_sample_step": args.synth_sample_step,
        "real_start_frame": real_start_frame,
        "synth_summary": summarize_rows(synth_rows),
        "real_summary": summarize_rows(real_rows),
        "mean_delta_real_minus_synth": {
            key: float(real_summary["mean"] - synth_summary["mean"])
            for key, synth_summary in summarize_rows(synth_rows).items()
            if key in summarize_rows(real_rows) and "mean" in synth_summary and "mean" in summarize_rows(real_rows)[key]
            for real_summary in [summarize_rows(real_rows)[key]]
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "synth_rows.json").write_text(json.dumps(synth_rows, indent=2), encoding="utf-8")
    (out_dir / "real_rows.json").write_text(json.dumps(real_rows, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "report": str(out_dir / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
