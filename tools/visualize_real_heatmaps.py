import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from corner_pose_resnet import ResNetCornerNet, decode_heatmaps, preprocess_frame
from infer_video_yolo_resnet_pose import yolo_rois


def train(*_args, **_kwargs):
    pass


def make_panel(crop, heatmaps, title):
    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
    tiles = []
    for k in range(8):
        hm = heatmaps[k]
        hm = cv2.resize(hm, (256, 256), interpolation=cv2.INTER_CUBIC)
        hm_u8 = np.clip(hm * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(crop, 0.55, color, 0.45, 0)
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        px = int(round(x / hm.shape[1] * 256))
        py = int(round(y / hm.shape[0] * 256))
        cv2.circle(overlay, (px, py), 5, (255, 255, 255), -1)
        cv2.putText(overlay, f"k{k}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(overlay)
    rows = []
    for r in range(2):
        rows.append(np.concatenate(tiles[r * 4:(r + 1) * 4], axis=1))
    panel = np.concatenate(rows, axis=0)
    bar = np.full((30, panel.shape[1], 3), 28, np.uint8)
    cv2.putText(bar, title[:90], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, panel], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/root/autodl-fs/head_left_rgb_raw.mp4")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.5)
    ap.add_argument("--roi-pad", type=float, default=0.08)
    ap.add_argument("--roi-shrink", type=float, default=0.0)
    ap.add_argument("--max-rois", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(args.detector)
    model = ResNetCornerNet(backbone="resnet18", pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    frame_idx = written = 0
    outputs = []
    with torch.no_grad():
        while written < args.count:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue
            rois = yolo_rois(detector, frame, args.det_conf, args.det_iou, args.max_rois, args.roi_pad, args.roi_shrink, square_roi=True)
            for roi_id, (score, x1, y1, x2, y2) in enumerate(rois):
                crop = frame[y1:y2 + 1, x1:x2 + 1]
                inp, _meta = preprocess_frame(crop, image_size=(args.image_size, args.image_size), keep_aspect=True, return_meta=True)
                logits = model(inp.to(device))
                hm = torch.sigmoid(logits)[0].detach().cpu().numpy()
                pts, conf = decode_heatmaps(logits)
                panel = make_panel(crop, hm, f"frame={frame_idx} roi={roi_id} det={score:.2f} mean_conf={float(np.mean(conf[0])):.3f}")
                path = out_dir / f"frame_{frame_idx:05d}_roi_{roi_id}_heatmaps.png"
                cv2.imwrite(str(path), panel)
                outputs.append(str(path))
                written += 1
                if written >= args.count:
                    break
            frame_idx += 1
    cap.release()
    (out_dir / "outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "count": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
