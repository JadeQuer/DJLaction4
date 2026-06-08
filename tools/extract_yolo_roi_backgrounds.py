import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def square_box(x1, y1, x2, y2, w, h, pad):
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
    return [int(max(0, math.floor(x1))), int(max(0, math.floor(y1))), int(min(w - 1, math.ceil(x2))), int(min(h - 1, math.ceil(y2)))]


def camera_mask_from_grabcut(crop, camera_box):
    h, w = crop.shape[:2]
    x1, y1, x2, y2 = camera_box
    x1, y1 = max(1, x1), max(1, y1)
    x2, y2 = min(w - 2, x2), min(h - 2, y2)
    rect_w, rect_h = max(2, x2 - x1), max(2, y2 - y1)

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    margin = max(8, int(round(min(w, h) * 0.035)))
    mask[:margin, :] = cv2.GC_BGD
    mask[-margin:, :] = cv2.GC_BGD
    mask[:, :margin] = cv2.GC_BGD
    mask[:, -margin:] = cv2.GC_BGD
    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    # The Action 4 body is mostly dark; use this as a soft foreground hint,
    # but keep bright screens/labels inside the detection box as possible foreground too.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < 95).astype(np.uint8)
    inner = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(inner, (x1, y1), (x2, y2), 1, -1)
    mask[(dark == 1) & (inner == 1)] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, mask, (x1, y1, rect_w, rect_h), bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return (inner * 255).astype(np.uint8)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    fg = cv2.dilate(fg, np.ones((9, 9), np.uint8), iterations=1)
    return fg


def fill_mask_with_neighbor_color(crop, mask, rng):
    h, w = crop.shape[:2]
    mask = (mask > 0).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return crop

    ring = cv2.dilate(mask, np.ones((35, 35), np.uint8), iterations=1)
    ring = cv2.subtract(ring, mask)
    samples = crop[ring > 0]
    if len(samples) == 0:
        samples = crop[mask == 0]
    if len(samples) == 0:
        samples = crop.reshape(-1, 3)
    color = np.median(samples, axis=0).astype(np.float32)

    noise = rng.normal(0, 7, crop.shape).astype(np.float32)
    patch = np.clip(color[None, None, :] + noise, 0, 255)
    patch = cv2.GaussianBlur(patch.astype(np.uint8), (0, 0), 1.8).astype(np.float32)

    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 9.0)
    alpha = np.clip(alpha[..., None], 0.0, 1.0)
    out = crop.astype(np.float32) * (1.0 - alpha) + patch * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def fill_box_with_neighbor_color(crop, camera_box, rng):
    h, w = crop.shape[:2]
    x1, y1, x2, y2 = camera_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return crop

    pad = max(12, int(round(max(x2 - x1, y2 - y1) * 0.08)))
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    ring = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(ring, (ox1, oy1), (ox2, oy2), 255, -1)
    cv2.rectangle(ring, (x1, y1), (x2, y2), 0, -1)
    samples = crop[ring > 0]
    if len(samples) == 0:
        samples = crop.reshape(-1, 3)
    color = np.median(samples, axis=0).astype(np.float32)

    out = crop.copy().astype(np.float32)
    box_h, box_w = y2 - y1 + 1, x2 - x1 + 1
    noise = rng.normal(0, 6, (box_h, box_w, 3)).astype(np.float32)
    patch = np.clip(color[None, None, :] + noise, 0, 255)
    patch = cv2.GaussianBlur(patch.astype(np.uint8), (0, 0), 2.0).astype(np.float32)

    mask = np.zeros((box_h, box_w), dtype=np.float32)
    cv2.rectangle(mask, (0, 0), (box_w - 1, box_h - 1), 1.0, -1)
    feather = max(9, int(round(min(box_w, box_h) * 0.04)))
    mask = cv2.GaussianBlur(mask, (0, 0), feather)
    mask = mask[..., None]
    roi = out[y1 : y2 + 1, x1 : x2 + 1]
    out[y1 : y2 + 1, x1 : x2 + 1] = roi * (1.0 - mask) + patch * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def repair_camera_region(crop, mode, rng, camera_box=None):
    if mode == "none":
        return crop
    if mode == "box-fill":
        if camera_box is None:
            return crop
        return fill_box_with_neighbor_color(crop, camera_box, rng)
    if mode == "grabcut-fill":
        if camera_box is None:
            return crop
        mask = camera_mask_from_grabcut(crop, camera_box)
        return fill_mask_with_neighbor_color(crop, mask, rng)
    if mode == "grabcut-inpaint":
        if camera_box is None:
            return crop
        mask = camera_mask_from_grabcut(crop, camera_box)
        repaired = cv2.inpaint(crop, mask, 7, cv2.INPAINT_TELEA)
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 3.0)[..., None]
        out = crop.astype(np.float32) * (1.0 - alpha) + repaired.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    if mode == "grabcut-light":
        if camera_box is None:
            return crop
        mask = camera_mask_from_grabcut(crop, camera_box)
        repaired = cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA)
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 4.5)[..., None]
        alpha *= 0.45
        out = crop.astype(np.float32) * (1.0 - alpha) + repaired.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    if mode == "box-inpaint":
        if camera_box is None:
            return crop
        h, w = crop.shape[:2]
        x1, y1, x2, y2 = camera_box
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Target the dark camera body while leaving bright screen/background texture usable.
        mask = cv2.inRange(gray, 0, 54)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    if mode == "blur":
        blurred = cv2.GaussianBlur(crop, (41, 41), 0)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        out = crop.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    if mode in {"inpaint", "box-inpaint"}:
        repaired = cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA)
        # Keep some original texture and add mild local noise so repaired
        # regions do not collapse into smooth color patches.
        repaired = cv2.addWeighted(crop, 0.18 if mode == "box-inpaint" else 0.35, repaired, 0.82 if mode == "box-inpaint" else 0.65, 0)
        noise = rng.normal(0, 5 if mode == "box-inpaint" else 7, crop.shape).astype(np.float32)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        out = repaired.astype(np.float32) + noise * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown repair mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/root/autodl-fs/head_left_rgb_raw.mp4")
    ap.add_argument("--detector", default="runs_pre/detect/runs/dji_action4_yolo_real_full_ft/weights/best.pt")
    ap.add_argument("--out-dir", default="assets/yolo_roi_backgrounds_head_left_inpaint")
    ap.add_argument("--num-images", type=int, default=120)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=-1)
    ap.add_argument("--frame-stride", type=int, default=24)
    ap.add_argument("--roi-size", type=int, default=1024)
    ap.add_argument("--roi-pad", type=float, default=0.12)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.5)
    ap.add_argument("--max-rois", type=int, default=2)
    ap.add_argument("--repair-mode", choices=["inpaint", "blur", "none", "box-inpaint", "box-fill", "grabcut-fill", "grabcut-inpaint", "grabcut-light"], default="inpaint")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    det = YOLO(args.detector)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    meta = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    start_frame = max(0, min(args.start_frame, total - 1))
    end_frame = total if args.end_frame < 0 else max(start_frame + 1, min(args.end_frame, total))
    for frame_id in range(start_frame, end_frame, max(1, args.frame_stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = det.predict(frame, conf=args.det_conf, iou=args.det_iou, max_det=args.max_rois, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            continue
        boxes = res[0].boxes.xyxy.detach().cpu().numpy()
        scores = res[0].boxes.conf.detach().cpu().numpy()
        for roi_idx, bi in enumerate(np.argsort(-scores)[: args.max_rois]):
            x1, y1, x2, y2 = boxes[bi]
            sx1, sy1, sx2, sy2 = square_box(x1, y1, x2, y2, w, h, args.roi_pad)
            crop = frame[sy1 : sy2 + 1, sx1 : sx2 + 1]
            if crop.size == 0:
                continue
            crop_h, crop_w = crop.shape[:2]
            cam_x1 = int(round((x1 - sx1) / max(1, crop_w) * args.roi_size))
            cam_y1 = int(round((y1 - sy1) / max(1, crop_h) * args.roi_size))
            cam_x2 = int(round((x2 - sx1) / max(1, crop_w) * args.roi_size))
            cam_y2 = int(round((y2 - sy1) / max(1, crop_h) * args.roi_size))
            camera_box = [
                max(0, min(args.roi_size - 1, cam_x1)),
                max(0, min(args.roi_size - 1, cam_y1)),
                max(0, min(args.roi_size - 1, cam_x2)),
                max(0, min(args.roi_size - 1, cam_y2)),
            ]
            crop = cv2.resize(crop, (args.roi_size, args.roi_size), interpolation=cv2.INTER_AREA)
            crop = repair_camera_region(crop, args.repair_mode, rng, camera_box=camera_box)
            name = f"roi_bg_{written:04d}_frame_{frame_id:06d}_roi_{roi_idx}.jpg"
            cv2.imwrite(str(out / name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            meta.append({"image": name, "frame_id": int(frame_id), "roi_index": int(roi_idx), "score": float(scores[bi]), "bbox": [sx1, sy1, sx2, sy2], "camera_box_roi": camera_box})
            written += 1
            if written >= args.num_images:
                cap.release()
                (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"wrote {written} ROI backgrounds to {out}")
                return
    cap.release()
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {written} ROI backgrounds to {out}")


if __name__ == "__main__":
    main()
