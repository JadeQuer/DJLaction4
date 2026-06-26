import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def square_roi(x1, y1, x2, y2, w, h, pad=0.0):
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = min(max(bw, bh) * (1.0 + 2.0 * pad), float(w - 1), float(h - 1))
    x1, x2 = cx - side * 0.5, cx + side * 0.5
    y1, y2 = cy - side * 0.5, cy + side * 0.5
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
    return (
        int(max(0, np.floor(x1))),
        int(max(0, np.floor(y1))),
        int(min(w - 1, np.ceil(x2))),
        int(min(h - 1, np.ceil(y2))),
    )


def letterbox(img, size=256):
    h, w = img.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def make_sheet(items, path, thumb=192, cols=8):
    rows = int(np.ceil(len(items) / cols))
    label_h = 26
    canvas = np.full((rows * (thumb + label_h), cols * thumb, 3), 235, np.uint8)
    for i, (img, label) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * thumb, r * (thumb + label_h)
        tile = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)
        canvas[y:y + label_h, x:x + thumb] = 28
        cv2.putText(canvas, label[:24], (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[y + label_h:y + label_h + thumb, x:x + thumb] = tile
    cv2.imwrite(str(path), canvas)


def main():
    video = Path("/root/autodl-fs/head_left_rgb_raw.mp4")
    detector_path = Path("runs_pre/detect/runs/dji_action4_yolo_real_full_ft/weights/best.pt")
    out_dir = Path("runs/cropped_test_inputs_for_teacher_full_video")
    raw_dir = out_dir / "raw_crops"
    net_dir = out_dir / "network_inputs_256"
    raw_dir.mkdir(parents=True, exist_ok=True)
    net_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = np.linspace(0, max(0, total - 1), 16).round().astype(int).tolist()
    detector = YOLO(str(detector_path))

    raw_items, net_items, manifest = [], [], []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = detector.predict(frame, conf=0.25, iou=0.5, max_det=2, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            continue
        boxes = res[0].boxes.xyxy.detach().cpu().numpy()
        scores = res[0].boxes.conf.detach().cpu().numpy()
        rois = []
        for bi in np.argsort(-scores)[:2]:
            x1, y1, x2, y2 = boxes[bi]
            sx1, sy1, sx2, sy2 = square_roi(x1, y1, x2, y2, w, h, pad=0.0)
            rois.append((float(scores[bi]), sx1, sy1, sx2, sy2))
        rois.sort(key=lambda r: (r[1] + r[3]) * 0.5)
        for roi_id, (score, x1, y1, x2, y2) in enumerate(rois):
            crop = frame[y1:y2 + 1, x1:x2 + 1]
            if crop.size == 0:
                continue
            t = frame_id / fps
            label = f"{t:05.1f}s_r{roi_id}"
            raw_path = raw_dir / f"frame_{frame_id:06d}_roi_{roi_id}_crop.png"
            net_path = net_dir / f"frame_{frame_id:06d}_roi_{roi_id}_net256.png"
            net = letterbox(crop, 256)
            cv2.imwrite(str(raw_path), crop)
            cv2.imwrite(str(net_path), net)
            raw_items.append((crop, label))
            net_items.append((net, label))
            manifest.append({
                "frame_id": int(frame_id),
                "time_sec": float(t),
                "roi_id": int(roi_id),
                "score": score,
                "raw_crop": str(raw_path),
                "network_input": str(net_path),
                "xyxy": [x1, y1, x2, y2],
            })
    cap.release()
    make_sheet(raw_items, out_dir / "yolo_square_roi_crops_full_video_4x8.png")
    make_sheet(net_items, out_dir / "network_256_letterbox_inputs_full_video_4x8.png")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "raw_crop_sheet": str(out_dir / "yolo_square_roi_crops_full_video_4x8.png"),
        "network_input_sheet": str(out_dir / "network_256_letterbox_inputs_full_video_4x8.png"),
        "manifest": str(out_dir / "manifest.json"),
        "items": len(manifest),
        "frame_ids": frame_ids,
    }, indent=2))


if __name__ == "__main__":
    main()
