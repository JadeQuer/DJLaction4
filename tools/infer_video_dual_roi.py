import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from corner_pose_baseline import TinyCornerNet, decode_heatmaps, preprocess_frame


def find_dark_rois(frame, max_rois=2):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 85)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 2500 or bw < 35 or bh < 25:
            continue
        if area > 0.35 * w * h:
            continue
        aspect = bw / max(1, bh)
        if aspect < 0.5 or aspect > 5.0:
            continue
        pad = int(max(bw, bh) * 0.9)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w - 1, x + bw + pad)
        y2 = min(h - 1, y + bh + pad)
        boxes.append((area, x1, y1, x2, y2))
    boxes.sort(reverse=True)
    picked = []
    for _, x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if all(abs(cx - (a + c) / 2) > 80 or abs(cy - (b + d) / 2) > 80 for a, b, c, d in picked):
            picked.append((x1, y1, x2, y2))
        if len(picked) >= max_rois:
            break
    return picked


def draw_roi_prediction(frame, pts, conf, bbox, roi_id):
    x1, y1, x2, y2 = bbox
    sx = (x2 - x1 + 1) / 160.0
    sy = (y2 - y1 + 1) / 120.0
    full_pts = np.stack([pts[:, 0] * sx + x1, pts[:, 1] * sy + y1], axis=1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    for i, (px, py) in enumerate(full_pts):
        px, py = int(round(px)), int(round(py))
        cv2.circle(frame, (px, py), 8, (0, 255, 255), -1)
        cv2.putText(frame, f'{roi_id}:{i}', (px + 7, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        pa = tuple(np.round(full_pts[a]).astype(int))
        pb = tuple(np.round(full_pts[b]).astype(int))
        cv2.line(frame, pa, pb, (0, 180, 255), 2)
    return full_pts


def infer(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = TinyCornerNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = args.output_width
    out_h = int(round(src_h * out_w / src_w))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps / args.stride, (out_w, out_h))

    stats = []
    frame_idx = written = 0
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue
            vis = frame.copy()
            rois = find_dark_rois(frame, max_rois=args.max_rois)
            per_frame = []
            for roi_id, bbox in enumerate(rois):
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2+1, x1:x2+1]
                if crop.size == 0:
                    continue
                inp = preprocess_frame(crop).to(device)
                logits = model(inp)
                pts, conf = decode_heatmaps(logits)
                draw_roi_prediction(vis, pts[0], conf[0], bbox, roi_id)
                per_frame.append(float(np.mean(conf[0])))
            mean_conf = float(np.mean(per_frame)) if per_frame else 0.0
            cv2.putText(vis, f'rois={len(rois)} mean_conf={mean_conf:.3f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(vis)
            stats.append(mean_conf)
            written += 1
            frame_idx += 1
            if args.max_frames and written >= args.max_frames:
                break
    cap.release()
    writer.release()
    report = {
        'video': args.video,
        'output': args.out,
        'frames_written': written,
        'mean_conf': float(np.mean(stats)) if stats else 0.0,
        'min_conf': float(np.min(stats)) if stats else 0.0,
        'max_conf': float(np.max(stats)) if stats else 0.0,
        'max_rois': args.max_rois,
    }
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/corner_roi_baseline/best.pt')
    ap.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    ap.add_argument('--out', default='runs/corner_roi_baseline/head_left_rgb_raw_dual_roi_pred.mp4')
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--output-width', type=int, default=960)
    ap.add_argument('--max-rois', type=int, default=2)
    ap.add_argument('--cpu', action='store_true')
    infer(ap.parse_args())


if __name__ == '__main__':
    main()
