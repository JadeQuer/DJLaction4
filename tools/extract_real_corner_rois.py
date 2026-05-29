import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def sample_frame_ids(total, num):
    if num <= 1:
        return [0]
    return sorted(set(int(round(i * (total - 1) / (num - 1))) for i in range(num)))


def yolo_rois(detector, frame, conf, iou, max_rois, pad_ratio):
    results = detector.predict(frame, conf=conf, iou=iou, max_det=max_rois, verbose=False)
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    h, w = frame.shape[:2]
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    order = np.argsort(-scores)
    rois = []
    for idx in order[:max_rois]:
        x1, y1, x2, y2 = xyxy[idx]
        bw, bh = x2 - x1, y2 - y1
        pad = pad_ratio * max(bw, bh)
        x1 = int(max(0, np.floor(x1 - pad)))
        y1 = int(max(0, np.floor(y1 - pad)))
        x2 = int(min(w - 1, np.ceil(x2 + pad)))
        y2 = int(min(h - 1, np.ceil(y2 + pad)))
        if x2 <= x1 or y2 <= y1:
            continue
        rois.append((float(scores[idx]), x1, y1, x2, y2))
    rois.sort(key=lambda x: (x[1] + x[3]) * 0.5)
    return rois


def draw_box(frame, bbox, label):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    ap.add_argument('--detector', default='runs/detect/runs/dji_action4_yolo_real_full_ft/weights/best.pt')
    ap.add_argument('--out-dir', default='datasets/dji_action4_real_corner_labelset_60')
    ap.add_argument('--num-frames', type=int, default=60)
    ap.add_argument('--max-rois', type=int, default=2)
    ap.add_argument('--det-conf', type=float, default=0.25)
    ap.add_argument('--det-iou', type=float, default=0.5)
    ap.add_argument('--pad-ratio', type=float, default=0.18)
    ap.add_argument('--roi-size', type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out_dir)
    roi_dir = out / 'roi_images'
    frame_dir = out / 'frame_previews'
    ann_dir = out / 'annotations_template'
    for d in [roi_dir, frame_dir, ann_dir]:
        d.mkdir(parents=True, exist_ok=True)

    detector = YOLO(args.detector)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_ids = sample_frame_ids(total, args.num_frames)

    manifest = []
    item_id = 0
    for sample_idx, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        rois = yolo_rois(detector, frame, args.det_conf, args.det_iou, args.max_rois, args.pad_ratio)
        preview = frame.copy()
        for roi_idx, (score, x1, y1, x2, y2) in enumerate(rois):
            draw_box(preview, (x1, y1, x2, y2), f'roi{roi_idx} det={score:.2f}')
            crop = frame[y1:y2 + 1, x1:x2 + 1]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (args.roi_size, args.roi_size), interpolation=cv2.INTER_AREA)
            stem = f'item_{item_id:03d}_frame_{frame_id:06d}_roi_{roi_idx}'
            roi_path = roi_dir / f'{stem}.png'
            cv2.imwrite(str(roi_path), crop_resized)
            template = {
                'id': stem,
                'source_video': args.video,
                'frame_id': frame_id,
                'time_sec': round(frame_id / fps, 3),
                'roi_index': roi_idx,
                'det_score': score,
                'bbox_xyxy_in_frame': [x1, y1, x2, y2],
                'roi_size': [args.roi_size, args.roi_size],
                'corners_2d_in_roi': [],
                'note': 'Fill 8 box corners in ROI pixel coordinates, ordered consistently with synthetic BB8 labels.',
            }
            (ann_dir / f'{stem}.json').write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding='utf-8')
            manifest.append(template)
            item_id += 1
        cv2.imwrite(str(frame_dir / f'frame_{sample_idx:03d}_src_{frame_id:06d}.jpg'), preview)

    cap.release()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'README.txt').write_text(
        '1. Open roi_images/ for annotation.\n'
        '2. For each ROI image, fill the matching annotations_template/*.json with 8 corners in ROI pixel coordinates.\n'
        '3. Keep corner order consistent with synthetic labels.\n',
        encoding='utf-8',
    )
    print(json.dumps({'out': str(out), 'frames_sampled': len(frame_ids), 'roi_items': len(manifest)}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
