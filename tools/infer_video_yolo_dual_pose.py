import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from corner_pose_baseline import TinyCornerNet, decode_heatmaps, preprocess_frame
from infer_video_dual_roi import draw_roi_prediction


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


def infer(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    detector = YOLO(args.detector)
    corner_model = TinyCornerNet().to(device)
    ckpt = torch.load(args.corner_ckpt, map_location=device)
    corner_model.load_state_dict(ckpt['model'])
    corner_model.eval()

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
                inp = preprocess_frame(crop).to(device)
                logits = corner_model(inp)
                pts, kpt_conf = decode_heatmaps(logits)
                draw_roi_prediction(vis, pts[0], kpt_conf[0], (x1, y1, x2, y2), roi_id)
                cv2.putText(vis, f'det={det_score:.2f}', (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
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
    }
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', default='runs/detect/runs/dji_action4_yolo_det/weights/best.pt')
    ap.add_argument('--corner-ckpt', default='runs/corner_roi_baseline/best.pt')
    ap.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    ap.add_argument('--out', default='runs/corner_roi_baseline/head_left_rgb_raw_yolo_dual_pose.mp4')
    ap.add_argument('--det-conf', type=float, default=0.15)
    ap.add_argument('--det-iou', type=float, default=0.5)
    ap.add_argument('--max-rois', type=int, default=2)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--output-width', type=int, default=960)
    ap.add_argument('--cpu', action='store_true')
    infer(ap.parse_args())


if __name__ == '__main__':
    main()
