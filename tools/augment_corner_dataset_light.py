import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bbox_from_corners(corners, w, h, pad_ratio=0.55):
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = max(bw, bh) * pad_ratio
    return [
        int(max(0, np.floor(x1 - pad))),
        int(max(0, np.floor(y1 - pad))),
        int(min(w - 1, np.ceil(x2 + pad))),
        int(min(h - 1, np.ceil(y2 + pad))),
    ]


def make_object_mask(img, bbox):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 150).astype(np.uint8) * 255
    x1, y1, x2, y2 = bbox
    roi = np.zeros_like(mask)
    roi[y1:y2 + 1, x1:x2 + 1] = 255
    mask = cv2.bitwise_and(mask, roi)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask


def random_background(paths, size):
    h, w = size
    if not paths:
        v = random.randint(135, 205)
        return np.full((h, w, 3), v, dtype=np.uint8)
    bg = cv2.imread(str(random.choice(paths)), cv2.IMREAD_COLOR)
    if bg is None:
        return np.full((h, w, 3), 175, dtype=np.uint8)
    bh, bw = bg.shape[:2]
    scale = max(w / bw, h / bh)
    resized = cv2.resize(bg, (int(round(bw * scale)), int(round(bh * scale))), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    x = random.randint(0, max(0, rw - w))
    y = random.randint(0, max(0, rh - h))
    bg = resized[y:y + h, x:x + w].copy()
    # Keep real backgrounds present but subdued, so object geometry remains learnable.
    if random.random() < 0.7:
        blur_k = random.choice([9, 13, 17])
        bg = cv2.GaussianBlur(bg, (blur_k, blur_k), 0)
    return bg


def light_color_jitter(img):
    out = img.astype(np.float32)
    out = out * random.uniform(0.88, 1.14) + random.uniform(-12, 12)
    if random.random() < 0.35:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.85, 1.18)
        hsv[:, :, 2] *= random.uniform(0.90, 1.12)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def light_occlusion(img, corners):
    if random.random() > 0.22:
        return img, False
    h, w = img.shape[:2]
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = max(8.0, x2 - x1), max(8.0, y2 - y1)
    ow = int(random.uniform(0.10, 0.22) * bw)
    oh = int(random.uniform(0.10, 0.22) * bh)
    # Bias occluders toward edges instead of covering the central body.
    cx = int(random.choice([random.uniform(x1, x1 + 0.25 * bw), random.uniform(x2 - 0.25 * bw, x2)]))
    cy = int(random.uniform(y1, y2))
    ax1 = max(0, cx - ow // 2)
    ay1 = max(0, cy - oh // 2)
    ax2 = min(w - 1, cx + ow // 2)
    ay2 = min(h - 1, cy + oh // 2)
    if ax2 <= ax1 or ay2 <= ay1:
        return img, False
    color = random.choice([(70, 70, 75), (105, 100, 95), (135, 135, 140)])
    cv2.rectangle(img, (ax1, ay1), (ax2, ay2), color, -1)
    return img, True


def light_degrade(img):
    out = img
    if random.random() < 0.18:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if random.random() < 0.20:
        noise = np.random.normal(0, random.uniform(1.5, 5.0), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.12:
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(78, 94)])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def compose_light(src_img, bg, mask):
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    obj = src_img.astype(np.float32)
    obj = obj * random.uniform(0.92, 1.10) + random.uniform(-5, 8)
    obj = np.clip(obj, 0, 255)
    # Keep object almost fully opaque; only soften mask boundary.
    out = obj * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='datasets/dji_action4_corner_train_500')
    ap.add_argument('--background-dir', default='datasets/dji_action4_real_det_to_label_90/images')
    ap.add_argument('--out', default='datasets/dji_action4_corner_train_light_1500')
    ap.add_argument('--num-images', type=int, default=1500)
    ap.add_argument('--clean-ratio', type=float, default=0.34)
    ap.add_argument('--seed', type=int, default=31)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    rgb_out = out / 'rgb'
    label_out = out / 'labels'
    rgb_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    label_paths = sorted((src / 'labels').glob('*.json'))
    if not label_paths:
        raise RuntimeError(f'No labels found in {src / "labels"}')
    bg_paths = sorted(Path(args.background_dir).glob('*.jpg')) + sorted(Path(args.background_dir).glob('*.png'))

    records = []
    clean_count = min(args.num_images, int(round(args.num_images * args.clean_ratio)))
    for i in range(args.num_images):
        label_path = label_paths[i % len(label_paths)]
        rec = read_json(label_path)
        img = cv2.imread(str(src / rec['image']), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Cannot read {src / rec["image"]}')
        h, w = img.shape[:2]
        mode = 'clean_copy' if i < clean_count else 'light_aug'
        if mode == 'clean_copy':
            aug = img.copy()
            occluded = False
        else:
            bbox = bbox_from_corners(rec['corners_2d'], w, h, pad_ratio=random.uniform(0.45, 0.75))
            mask = make_object_mask(img, bbox)
            bg = random_background(bg_paths, (h, w))
            aug = compose_light(img, bg, mask)
            aug = light_color_jitter(aug)
            aug, occluded = light_occlusion(aug, rec['corners_2d'])
            aug = light_degrade(aug)
        stem = f'{i:06d}'
        cv2.imwrite(str(rgb_out / f'{stem}.png'), aug)
        new_rec = dict(rec)
        new_rec['image'] = f'rgb/{stem}.png'
        new_rec['augmentation'] = {
            'source_label': str(label_path),
            'mode': mode,
            'background_dir': str(args.background_dir),
            'real_background': bool(bg_paths) and mode == 'light_aug',
            'occluded': bool(occluded),
            'policy': 'light augmentation; preserve visible object geometry and keep corners_2d unchanged',
        }
        (label_out / f'{stem}.json').write_text(json.dumps(new_rec, indent=2), encoding='utf-8')
        records.append(new_rec)
    (out / 'labels.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    if (src / 'labels.json').exists():
        shutil.copy2(src / 'labels.json', out / 'source_labels.json')
    print(json.dumps({'out': str(out), 'images': len(records), 'clean_count': clean_count, 'light_aug_count': len(records) - clean_count, 'backgrounds': len(bg_paths)}, indent=2))


if __name__ == '__main__':
    main()
