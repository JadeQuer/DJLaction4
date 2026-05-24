import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bbox_from_corners(corners, w, h, pad_ratio=1.2):
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
    # Existing renders use a very light grey world and a dark DJI model.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 125).astype(np.uint8) * 255
    x1, y1, x2, y2 = bbox
    roi = np.zeros_like(mask)
    roi[y1:y2 + 1, x1:x2 + 1] = 255
    mask = cv2.bitwise_and(mask, roi)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def random_background(paths, size):
    h, w = size
    if not paths:
        base = np.full((h, w, 3), random.randint(120, 210), dtype=np.uint8)
        return base
    bg = cv2.imread(str(random.choice(paths)), cv2.IMREAD_COLOR)
    if bg is None:
        return np.full((h, w, 3), 170, dtype=np.uint8)
    bh, bw = bg.shape[:2]
    scale = max(w / bw, h / bh)
    resized = cv2.resize(bg, (int(round(bw * scale)), int(round(bh * scale))), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    x = random.randint(0, max(0, rw - w))
    y = random.randint(0, max(0, rh - h))
    return resized[y:y + h, x:x + w].copy()


def color_jitter(img):
    out = img.astype(np.float32)
    alpha = random.uniform(0.65, 1.35)
    beta = random.uniform(-35, 35)
    out = out * alpha + beta
    if random.random() < 0.5:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.55, 1.45)
        hsv[:, :, 2] *= random.uniform(0.75, 1.25)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_occlusion(img, corners, max_occ=3):
    h, w = img.shape[:2]
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = max(8.0, x2 - x1), max(8.0, y2 - y1)
    num = random.randint(0, max_occ)
    for _ in range(num):
        ow = int(random.uniform(0.25, 0.85) * bw)
        oh = int(random.uniform(0.25, 0.85) * bh)
        cx = int(random.uniform(x1 - 0.25 * bw, x2 + 0.25 * bw))
        cy = int(random.uniform(y1 - 0.25 * bh, y2 + 0.25 * bh))
        ax1 = max(0, cx - ow // 2)
        ay1 = max(0, cy - oh // 2)
        ax2 = min(w - 1, cx + ow // 2)
        ay2 = min(h - 1, cy + oh // 2)
        if ax2 <= ax1 or ay2 <= ay1:
            continue
        color = random.choice([
            (35, 35, 35), (70, 60, 55), (120, 115, 105), (145, 145, 155), (90, 95, 110)
        ])
        if random.random() < 0.45:
            cv2.rectangle(img, (ax1, ay1), (ax2, ay2), color, -1)
        else:
            cv2.ellipse(img, (cx, cy), (max(1, ow // 2), max(1, oh // 2)), random.uniform(0, 180), 0, 360, color, -1)
    return img


def degrade(img):
    out = img
    if random.random() < 0.45:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if random.random() < 0.25:
        k = random.choice([5, 7, 9])
        kernel = np.zeros((k, k), dtype=np.float32)
        if random.random() < 0.5:
            kernel[k // 2, :] = 1.0
        else:
            kernel[:, k // 2] = 1.0
        kernel /= kernel.sum()
        out = cv2.filter2D(out, -1, kernel)
    if random.random() < 0.45:
        noise = np.random.normal(0, random.uniform(3, 12), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.35:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(45, 85)]
        ok, enc = cv2.imencode('.jpg', out, encode_param)
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def compose(src_img, bg, mask):
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    # Slightly vary rendered object intensity to reduce overfitting to one material.
    obj = src_img.astype(np.float32) * random.uniform(0.75, 1.25) + random.uniform(-12, 12)
    obj = np.clip(obj, 0, 255)
    out = obj * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='datasets/dji_action4_corner_train_500')
    ap.add_argument('--background-dir', default='datasets/dji_action4_real_det_to_label_90/images')
    ap.add_argument('--out', default='datasets/dji_action4_corner_train_aug_3000')
    ap.add_argument('--num-images', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=13)
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
    for i in range(args.num_images):
        label_path = label_paths[i % len(label_paths)]
        rec = read_json(label_path)
        img = cv2.imread(str(src / rec['image']), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Cannot read {src / rec["image"]}')
        h, w = img.shape[:2]
        bbox = bbox_from_corners(rec['corners_2d'], w, h, pad_ratio=random.uniform(0.8, 1.5))
        mask = make_object_mask(img, bbox)
        bg = random_background(bg_paths, (h, w))
        aug = compose(img, bg, mask)
        if random.random() < 0.75:
            aug = add_occlusion(aug, rec['corners_2d'], max_occ=3)
        aug = color_jitter(aug)
        aug = degrade(aug)

        stem = f'{i:06d}'
        cv2.imwrite(str(rgb_out / f'{stem}.png'), aug)
        new_rec = dict(rec)
        new_rec['image'] = f'rgb/{stem}.png'
        new_rec['augmentation'] = {
            'source_label': str(label_path),
            'background_dir': str(args.background_dir),
            'real_background': bool(bg_paths),
            'occlusion': True,
            'color_blur_noise': True,
            'label_policy': 'corners_2d unchanged from Blender projection',
        }
        (label_out / f'{stem}.json').write_text(json.dumps(new_rec, indent=2), encoding='utf-8')
        records.append(new_rec)

    (out / 'labels.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    shutil.copy2(src / 'labels.json', out / 'source_labels.json') if (src / 'labels.json').exists() else None
    print(json.dumps({'out': str(out), 'images': len(records), 'backgrounds': len(bg_paths)}, indent=2))


if __name__ == '__main__':
    main()
