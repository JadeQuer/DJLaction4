import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from corner_pose_resnet import ResNetCornerNet, decode_heatmaps, preprocess_frame


def train(*_args, **_kwargs):
    # Compatibility shim for checkpoints that stored argparse's func=train.
    pass


def yolo_rois(detector, frame, conf, iou, max_rois):
    results = detector.predict(frame, conf=conf, iou=iou, max_det=max_rois, verbose=False)
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    rois = []
    h, w = frame.shape[:2]
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    order = np.argsort(-scores)
    for idx in order[:max_rois]:
        x1, y1, x2, y2 = xyxy[idx]
        bw, bh = x2 - x1, y2 - y1
        pad = 0.18 * max(bw, bh)
        x1 = int(max(0, np.floor(x1 - pad)))
        y1 = int(max(0, np.floor(y1 - pad)))
        x2 = int(min(w - 1, np.ceil(x2 + pad)))
        y2 = int(min(h - 1, np.ceil(y2 + pad)))
        if x2 <= x1 or y2 <= y1:
            continue
        rois.append((float(scores[idx]), x1, y1, x2, y2))
    rois.sort(key=lambda x: (x[1] + x[3]) * 0.5)
    return rois




def project_heatmap_points_to_frame(pts, bbox, heatmap_size):
    x1, y1, x2, y2 = bbox
    hm_w, hm_h = heatmap_size
    sx = (x2 - x1 + 1) / float(hm_w)
    sy = (y2 - y1 + 1) / float(hm_h)
    return np.stack([pts[:, 0] * sx + x1, pts[:, 1] * sy + y1], axis=1)


def smooth_points(prev_pts, curr_pts, alpha, max_jump):
    if prev_pts is None:
        return curr_pts
    dist = np.linalg.norm(curr_pts - prev_pts, axis=1)
    point_alpha = np.full((curr_pts.shape[0], 1), alpha, dtype=np.float32)
    jump_mask = dist > max_jump
    point_alpha[jump_mask] = min(alpha, 0.18)
    return prev_pts * (1.0 - point_alpha) + curr_pts * point_alpha


def draw_frame_points(frame, full_pts, bbox, roi_id, det_score=None):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    for i, (px, py) in enumerate(full_pts):
        px, py = int(round(px)), int(round(py))
        cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
        cv2.putText(frame, f'{roi_id}:{i}', (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    edges = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]
    for a, b in edges:
        pa = tuple(np.round(full_pts[a]).astype(int))
        pb = tuple(np.round(full_pts[b]).astype(int))
        cv2.line(frame, pa, pb, (0, 180, 255), 2)
    if det_score is not None:
        cv2.putText(frame, f'det={det_score:.2f}', (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

def infer(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    detector = YOLO(args.detector)
    model = ResNetCornerNet(backbone=args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.corner_ckpt, map_location=device)
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

    frame_idx = written = 0
    conf_stats = []
    det_count_stats = []
    prev_points = [None for _ in range(args.max_rois)]
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue
            vis = frame.copy()
            rois = yolo_rois(detector, frame, args.det_conf, args.det_iou, args.max_rois)
            per_frame = []
            for roi_id, (det_score, x1, y1, x2, y2) in enumerate(rois):
                crop = frame[y1:y2 + 1, x1:x2 + 1]
                if crop.size == 0:
                    continue
                inp = preprocess_frame(crop, image_size=(args.image_size, args.image_size)).to(device)
                logits = model(inp)
                pts, kpt_conf = decode_heatmaps(logits)
                full_pts = project_heatmap_points_to_frame(pts[0], (x1, y1, x2, y2), (args.heatmap_size, args.heatmap_size))
                if args.temporal_smooth:
                    full_pts = smooth_points(prev_points[roi_id], full_pts, args.smooth_alpha, args.max_point_jump)
                    prev_points[roi_id] = full_pts.copy()
                draw_frame_points(vis, full_pts, (x1, y1, x2, y2), roi_id, det_score)
                per_frame.append(float(np.mean(kpt_conf[0])))
            mean_conf = float(np.mean(per_frame)) if per_frame else 0.0
            det_count_stats.append(len(rois))
            conf_stats.append(mean_conf)
            cv2.putText(vis, f'yolo_rois={len(rois)} mean_kpt_conf={mean_conf:.3f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(vis)
            written += 1
            frame_idx += 1
            if args.max_frames and written >= args.max_frames:
                break
    cap.release()
    writer.release()
    report = {
        'video': args.video,
        'output': args.out,
        'detector': args.detector,
        'corner_ckpt': args.corner_ckpt,
        'frames_written': written,
        'mean_yolo_rois': float(np.mean(det_count_stats)) if det_count_stats else 0.0,
        'frames_with_detection': int(sum(c > 0 for c in det_count_stats)),
        'mean_kpt_conf': float(np.mean(conf_stats)) if conf_stats else 0.0,
        'min_kpt_conf': float(np.min(conf_stats)) if conf_stats else 0.0,
        'max_kpt_conf': float(np.max(conf_stats)) if conf_stats else 0.0,
        'max_rois': args.max_rois,
        'backbone': args.backbone,
    }
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', default='runs/detect/runs/dji_action4_yolo_real_full_ft/weights/best.pt')
    ap.add_argument('--corner-ckpt', default='runs/corner_resnet18_aug_roi/best.pt')
    ap.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    ap.add_argument('--out', default='runs/corner_resnet18_aug_roi/head_left_rgb_raw_yolo_conf025_pose.mp4')
    ap.add_argument('--det-conf', type=float, default=0.25)
    ap.add_argument('--det-iou', type=float, default=0.5)
    ap.add_argument('--max-rois', type=int, default=2)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--output-width', type=int, default=960)
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--heatmap-size', type=int, default=64)
    ap.add_argument('--temporal-smooth', action='store_true')
    ap.add_argument('--smooth-alpha', type=float, default=0.35)
    ap.add_argument('--max-point-jump', type=float, default=45.0)
    ap.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet34'])
    ap.add_argument('--cpu', action='store_true')
    infer(ap.parse_args())


if __name__ == '__main__':
    main()
