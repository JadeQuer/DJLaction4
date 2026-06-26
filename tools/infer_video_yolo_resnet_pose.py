import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from corner_pose_resnet import ResNetCornerNet, decode_heatmaps, preprocess_frame


FIXED_DJI_ACTION4_DETECTOR = 'runs_pre/detect/runs/dji_action4_yolo_real_full_ft/weights/best.pt'


def train(*_args, **_kwargs):
    # Compatibility shim for checkpoints that stored argparse's func=train.
    pass


def yolo_rois(detector, frame, conf, iou, max_rois, roi_pad, roi_shrink, square_roi=True):
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
        if roi_shrink > 0:
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            bw0, bh0 = (x2 - x1) * (1.0 - roi_shrink), (y2 - y1) * (1.0 - roi_shrink)
            x1, x2 = cx - bw0 * 0.5, cx + bw0 * 0.5
            y1, y2 = cy - bh0 * 0.5, cy + bh0 * 0.5
        bw, bh = x2 - x1, y2 - y1
        if square_roi:
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
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
        else:
            pad = roi_pad * max(bw, bh)
            x1 = x1 - pad
            y1 = y1 - pad
            x2 = x2 + pad
            y2 = y2 + pad
        x1 = int(max(0, np.floor(x1)))
        y1 = int(max(0, np.floor(y1)))
        x2 = int(min(w - 1, np.ceil(x2)))
        y2 = int(min(h - 1, np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        rois.append((float(scores[idx]), x1, y1, x2, y2))
    rois.sort(key=lambda x: (x[1] + x[3]) * 0.5)
    return rois




def project_heatmap_points_to_frame(pts, bbox, heatmap_size, input_size=None, letterbox_meta=None):
    x1, y1, x2, y2 = bbox
    hm_w, hm_h = heatmap_size
    if letterbox_meta is not None and input_size is not None:
        in_w, in_h = input_size
        px = pts[:, 0] / float(hm_w) * float(in_w)
        py = pts[:, 1] / float(hm_h) * float(in_h)
        scale = max(float(letterbox_meta.get('scale', 1.0)), 1e-6)
        cx = (px - float(letterbox_meta.get('pad_x', 0.0))) / scale
        cy = (py - float(letterbox_meta.get('pad_y', 0.0))) / scale
        return np.stack([cx + x1, cy + y1], axis=1)
    sx = (x2 - x1 + 1) / float(hm_w)
    sy = (y2 - y1 + 1) / float(hm_h)
    return np.stack([pts[:, 0] * sx + x1, pts[:, 1] * sy + y1], axis=1)


def draw_frame_points(frame, full_pts, bbox, roi_id, det_score=None):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    for i, (px, py) in enumerate(full_pts):
        px, py = int(round(px)), int(round(py))
        cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
        cv2.putText(frame, f'{roi_id}:{i}', (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        pa = tuple(np.round(full_pts[a]).astype(int))
        pb = tuple(np.round(full_pts[b]).astype(int))
        cv2.line(frame, pa, pb, (0, 180, 255), 2)
    if det_score is not None:
        cv2.putText(frame, f'det={det_score:.2f}', (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

def infer(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    detector_path = FIXED_DJI_ACTION4_DETECTOR
    if not Path(detector_path).exists():
        raise FileNotFoundError(f'Fixed DJI Action 4 detector not found: {detector_path}')
    detector = YOLO(detector_path)
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
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = written = 0
    conf_stats = []
    det_count_stats = []
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue
            vis = frame.copy()
            rois = yolo_rois(detector, frame, args.det_conf, args.det_iou, args.max_rois, args.roi_pad, args.roi_shrink, square_roi=not args.rect_roi)
            per_frame = []
            for roi_id, (det_score, x1, y1, x2, y2) in enumerate(rois):
                crop = frame[y1:y2 + 1, x1:x2 + 1]
                if crop.size == 0:
                    continue
                inp, letterbox_meta = preprocess_frame(
                    crop,
                    image_size=(args.image_size, args.image_size),
                    keep_aspect=not args.stretch_roi,
                    return_meta=True,
                )
                inp = inp.to(device)
                logits = model(inp)
                pts, kpt_conf = decode_heatmaps(logits)
                full_pts = project_heatmap_points_to_frame(
                    pts[0],
                    (x1, y1, x2, y2),
                    (args.heatmap_size, args.heatmap_size),
                    input_size=(args.image_size, args.image_size),
                    letterbox_meta=None if args.stretch_roi else letterbox_meta,
                )
                draw_frame_points(vis, full_pts, (x1, y1, x2, y2), roi_id, det_score)
                per_frame.append(float(np.mean(kpt_conf[0])))
                if debug_dir is not None and written < args.debug_frames:
                    cv2.imwrite(str(debug_dir / f'frame_{written:04d}_roi_{roi_id}_crop.png'), crop)
            mean_conf = float(np.mean(per_frame)) if per_frame else 0.0
            det_count_stats.append(len(rois))
            conf_stats.append(mean_conf)
            cv2.putText(vis, f'yolo_rois={len(rois)} mean_kpt_conf={mean_conf:.3f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            if debug_dir is not None and written < args.debug_frames:
                cv2.imwrite(str(debug_dir / f'frame_{written:04d}_vis.png'), vis)
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
        'detector': detector_path,
        'corner_ckpt': args.corner_ckpt,
        'frames_written': written,
        'mean_yolo_rois': float(np.mean(det_count_stats)) if det_count_stats else 0.0,
        'frames_with_detection': int(sum(c > 0 for c in det_count_stats)),
        'mean_kpt_conf': float(np.mean(conf_stats)) if conf_stats else 0.0,
        'min_kpt_conf': float(np.min(conf_stats)) if conf_stats else 0.0,
        'max_kpt_conf': float(np.max(conf_stats)) if conf_stats else 0.0,
        'max_rois': args.max_rois,
        'roi_pad': args.roi_pad,
        'roi_shrink': args.roi_shrink,
        'stretch_roi': args.stretch_roi,
        'backbone': args.backbone,
    }
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corner-ckpt', default='runs/corner_resnet18_aug_roi/best.pt')
    ap.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    ap.add_argument('--out', default='runs/corner_resnet18_aug_roi/head_left_rgb_raw_yolo_conf025_pose.mp4')
    ap.add_argument('--det-conf', type=float, default=0.25)
    ap.add_argument('--det-iou', type=float, default=0.5)
    ap.add_argument('--roi-pad', type=float, default=0.08)
    ap.add_argument('--roi-shrink', type=float, default=0.0)
    ap.add_argument('--rect-roi', action='store_true')
    ap.add_argument('--max-rois', type=int, default=2)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--output-width', type=int, default=960)
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--heatmap-size', type=int, default=64)
    ap.add_argument('--stretch-roi', action='store_true')
    ap.add_argument('--debug-dir')
    ap.add_argument('--debug-frames', type=int, default=24)
    ap.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet34'])
    ap.add_argument('--cpu', action='store_true')
    infer(ap.parse_args())


if __name__ == '__main__':
    main()
