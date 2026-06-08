import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from corner_pose_resnet import ResNetCornerNet, bbox_from_points, crop_and_resize, decode_heatmaps, preprocess_frame
from visualize_corner_dataset import draw_corners


EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def train(*_args, **_kwargs):
    # Compatibility shim for checkpoints that stored argparse's func=train.
    pass


def transform_corners(corners, bbox, meta):
    x1, y1, _x2, _y2 = bbox
    out = []
    for x, y, z in corners:
        out.append([(x - x1) * meta["scale"] + meta["pad_x"], (y - y1) * meta["scale"] + meta["pad_y"], z])
    return np.asarray(out, dtype=np.float32)


def predict(model, device, img, image_size, heatmap_size):
    inp = preprocess_frame(img, image_size=(image_size, image_size), keep_aspect=True).to(device)
    with torch.no_grad():
        logits = model(inp)
    pts, conf = decode_heatmaps(logits)
    pts_img = pts[0].copy()
    pts_img[:, 0] = pts_img[:, 0] / heatmap_size * image_size
    pts_img[:, 1] = pts_img[:, 1] / heatmap_size * image_size
    return pts_img, conf[0]


def polygon_mask(shape, pts):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.round(pts[:, :2]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def perturbations(img, corners):
    h, w = img.shape[:2]
    pts = corners[:, :2]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bg_color = tuple(int(v) for v in img.reshape(-1, 3).mean(axis=0))
    obj_mask = polygon_mask(img.shape, pts)
    variants = {"original": img.copy()}

    bg_replaced = img.copy()
    bg_replaced[obj_mask == 0] = bg_color
    variants["replace_background"] = bg_replaced

    center = img.copy()
    cx1 = int(round(x1 + (x2 - x1) * 0.25))
    cx2 = int(round(x1 + (x2 - x1) * 0.75))
    cy1 = int(round(y1 + (y2 - y1) * 0.25))
    cy2 = int(round(y1 + (y2 - y1) * 0.75))
    center[cy1:cy2, cx1:cx2] = bg_color
    variants["replace_object_center"] = center

    edge = img.copy()
    k = max(3, int(round(min(w, h) * 0.025)))
    eroded = cv2.erode(obj_mask, np.ones((k, k), np.uint8))
    edge_band = (obj_mask > 0) & (eroded == 0)
    edge[edge_band] = bg_color
    variants["replace_object_edge_band"] = edge

    blur = cv2.GaussianBlur(img, (7, 7), 0)
    variants["blur_all"] = blur

    return variants


def draw_prediction(img, gt, pred, title):
    vis = img.copy()
    draw_corners(vis, gt, title="")
    for i, (x, y) in enumerate(pred):
        cv2.circle(vis, (int(round(x)), int(round(y))), 5, (255, 0, 255), -1)
        cv2.putText(vis, f"p{i}", (int(round(x)) + 4, int(round(y)) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    for a, b in EDGES:
        pa = tuple(np.round(pred[a]).astype(int))
        pb = tuple(np.round(pred[b]).astype(int))
        cv2.line(vis, pa, pb, (255, 0, 255), 1)
    cv2.putText(vis, title[:40], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def make_contact_sheet(items, out_path, thumb=(256, 256), cols=5):
    if not items:
        return
    tw, th = thumb
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * th, cols * tw, 3), 240, dtype=np.uint8)
    for i, img in enumerate(items):
        row, col = divmod(i, cols)
        sheet[row * th:(row + 1) * th, col * tw:(col + 1) * tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--heatmap-size", type=int, default=64)
    ap.add_argument("--roi-pad", type=float, default=0.08)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetCornerNet(backbone="resnet18", pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    tiles = []
    root = Path(args.data)
    for lp in sorted((root / "labels").glob("*.json"))[:args.count]:
        rec = json.loads(lp.read_text(encoding="utf-8"))
        img = cv2.imread(str(root / rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        corners = np.asarray(rec["corners_2d"], dtype=np.float32)
        bbox = bbox_from_points(corners[:, :2], w, h, pad_ratio=args.roi_pad, jitter=0.0, square=True)
        crop, bbox, meta = crop_and_resize(img, bbox, (args.image_size, args.image_size), keep_aspect=True)
        gt = transform_corners(corners, bbox, meta)
        base_pred, base_conf = predict(model, device, crop, args.image_size, args.heatmap_size)
        base_err = np.linalg.norm(base_pred - gt[:, :2], axis=1).mean()
        for name, variant in perturbations(crop, gt).items():
            pred, conf = predict(model, device, variant, args.image_size, args.heatmap_size)
            drift = np.linalg.norm(pred - base_pred, axis=1).mean()
            err = np.linalg.norm(pred - gt[:, :2], axis=1).mean()
            rows.append({
                "label": str(lp),
                "variant": name,
                "base_err_px": float(base_err),
                "err_px": float(err),
                "drift_from_original_px": float(drift),
                "mean_conf": float(np.mean(conf)),
                "base_mean_conf": float(np.mean(base_conf)),
            })
            if len(tiles) < 50:
                tiles.append(draw_prediction(variant, gt, pred, f"{lp.stem} {name} d={drift:.1f}"))
    summary = {}
    for variant in sorted({r["variant"] for r in rows}):
        vals = [r["drift_from_original_px"] for r in rows if r["variant"] == variant]
        errs = [r["err_px"] for r in rows if r["variant"] == variant]
        summary[variant] = {
            "count": len(vals),
            "mean_drift_px": float(np.mean(vals)) if vals else 0.0,
            "p90_drift_px": float(np.percentile(vals, 90)) if vals else 0.0,
            "mean_err_px": float(np.mean(errs)) if errs else 0.0,
        }
    (out_dir / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    make_contact_sheet(tiles, out_dir / "sensitivity_contact_sheet.png")
    print(json.dumps({"out": str(out_dir), "report": str(out_dir / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
