import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from corner_pose_resnet import bbox_from_points, crop_and_resize


def load_records(root):
    root = Path(root)
    records = []
    for lp in sorted((root / "labels").glob("*.json")):
        rec = json.loads(lp.read_text(encoding="utf-8"))
        records.append((lp, rec))
    if not records:
        raise RuntimeError(f"No labels under {root / 'labels'}")
    return records


def collect_real_rois(real_roots):
    paths = []
    for root in real_roots:
        paths.extend(sorted((Path(root) / "rgb").glob("*.png")))
        paths.extend(sorted((Path(root) / "rgb").glob("*.jpg")))
    return paths


def image_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "mean": img.reshape(-1, 3).mean(axis=0).astype(np.float32),
        "std": img.reshape(-1, 3).std(axis=0).astype(np.float32) + 1e-6,
        "sat_mean": float(hsv[:, :, 1].mean()),
        "val_mean": float(hsv[:, :, 2].mean()),
        "lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def match_color(src, ref):
    src_f = src.astype(np.float32)
    s = image_stats(src)
    r = image_stats(ref)
    out = (src_f - s["mean"]) / s["std"] * r["std"] + r["mean"]
    out = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_scale = np.clip(r["sat_mean"] / max(1.0, hsv[:, :, 1].mean()), 0.75, 1.45)
    val_scale = np.clip(r["val_mean"] / max(1.0, hsv[:, :, 2].mean()), 0.80, 1.25)
    hsv[:, :, 1] *= sat_scale
    hsv[:, :, 2] *= val_scale
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def estimate_object_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # The rendered camera is dark. Keep the threshold loose because screens can be bright.
    mask = (gray < 150).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def paste_on_real_background(obj_img, ref_img):
    mask = estimate_object_mask(obj_img)
    ref = cv2.resize(ref_img, (obj_img.shape[1], obj_img.shape[0]), interpolation=cv2.INTER_AREA)
    # Suppress the real camera region without making the background a flat color.
    ref_blur = cv2.GaussianBlur(ref, (31, 31), 0)
    ref = cv2.addWeighted(ref, 0.45, ref_blur, 0.55, 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(obj_img.astype(np.float32) * alpha + ref.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def fit_object_to_real_roi_shape(img, pts, ref_img):
    ref = cv2.resize(ref_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    ref_blur = cv2.GaussianBlur(ref, (31, 31), 0)
    canvas = cv2.addWeighted(ref, 0.50, ref_blur, 0.50, 0)

    p = pts[:, :2].astype(np.float32)
    min_xy = p.min(axis=0)
    max_xy = p.max(axis=0)
    obj_w = max(1.0, float(max_xy[0] - min_xy[0]))
    obj_h = max(1.0, float(max_xy[1] - min_xy[1]))
    cx, cy = (min_xy + max_xy) * 0.5

    # Match the manually labeled real ROI distribution: broad target with more vertical margin.
    target_w_frac = random.uniform(0.80, 0.92)
    target_h_frac = random.uniform(0.60, 0.76)
    target_w = target_w_frac * img.shape[1]
    target_h = target_h_frac * img.shape[0]
    scale = min(target_w / obj_w, target_h / obj_h)

    new_cx = img.shape[1] * random.uniform(0.47, 0.53)
    new_cy = img.shape[0] * random.uniform(0.48, 0.55)
    matrix = np.array(
        [
            [scale, 0.0, new_cx - scale * cx],
            [0.0, scale, new_cy - scale * cy],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = estimate_object_mask(warped)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out = warped.astype(np.float32) * alpha + canvas.astype(np.float32) * (1.0 - alpha)

    pts2 = pts.copy()
    xy = np.concatenate([pts[:, :2], np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    pts2[:, :2] = xy @ matrix.T
    return np.clip(out, 0, 255).astype(np.uint8), pts2


def degrade_to_video(img, ref_img):
    out = match_color(img, ref_img)
    if random.random() < 0.55:
        scale = random.uniform(0.55, 0.90)
        h, w = out.shape[:2]
        small = cv2.resize(out, (max(16, int(w * scale)), max(16, int(h * scale))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if random.random() < 0.35:
        noise = np.random.normal(0.0, random.uniform(2.0, 7.0), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.70:
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(55, 88)])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def crop_to_real_roi_style(img, corners, image_size, roi_pad):
    h, w = img.shape[:2]
    bbox = bbox_from_points(corners[:, :2], w, h, pad_ratio=roi_pad, jitter=0.0, square=True)
    crop, bbox, meta = crop_and_resize(img, bbox, (image_size, image_size), keep_aspect=True)
    x1, y1, _x2, _y2 = bbox
    transformed = []
    for x, y, z in corners:
        transformed.append([
            (x - x1) * meta["scale"] + meta["pad_x"],
            (y - y1) * meta["scale"] + meta["pad_y"],
            z,
        ])
    return crop, np.asarray(transformed, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/dji_action4_current_roll_200")
    ap.add_argument("--out", default="datasets/dji_action4_current_roll_200_realstyle_roi")
    ap.add_argument("--num-images", type=int, default=600)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--roi-pad", type=float, default=0.22)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument(
        "--real-roots",
        nargs="+",
        default=[
            "datasets/dji_action4_real_csv_labeled_85",
            "datasets/dji_action4_real_csv_batch2_labeled",
            "datasets/dji_action4_real_csv_batch3_labeled",
        ],
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    rgb_out = out / "rgb"
    label_out = out / "labels"
    rgb_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    records = load_records(src)
    real_paths = collect_real_rois(args.real_roots)
    if not real_paths:
        raise RuntimeError("No real ROI images found")

    out_records = []
    for i in range(args.num_images):
        lp, rec = records[i % len(records)]
        img = cv2.imread(str(src / rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read {src / rec['image']}")
        corners = np.asarray(rec["corners_2d"], dtype=np.float32)
        crop, pts = crop_to_real_roi_style(img, corners, args.image_size, args.roi_pad)
        ref_path = random.choice(real_paths)
        ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if ref is None:
            ref = crop
        crop, pts = fit_object_to_real_roi_shape(crop, pts, ref)
        if random.random() < 0.30:
            crop = paste_on_real_background(crop, ref)
        crop = degrade_to_video(crop, ref)

        stem = f"{i:06d}"
        cv2.imwrite(str(rgb_out / f"{stem}.png"), crop)
        new_rec = dict(rec)
        new_rec["image"] = f"rgb/{stem}.png"
        new_rec["corners_2d"] = pts.tolist()
        new_rec["camera"] = {"width": args.image_size, "height": args.image_size}
        new_rec["real_style_synth"] = {
            "source_label": str(lp),
            "reference_real_roi": str(ref_path),
            "roi_pad": args.roi_pad,
            "image_size": args.image_size,
            "label_policy": "corners transformed into final YOLO-style ROI crop",
        }
        (label_out / f"{stem}.json").write_text(json.dumps(new_rec, indent=2, ensure_ascii=False), encoding="utf-8")
        out_records.append(new_rec)

    (out / "labels.json").write_text(json.dumps(out_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "images": len(out_records), "real_refs": len(real_paths)}, indent=2))


if __name__ == "__main__":
    main()
