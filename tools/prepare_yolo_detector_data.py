import argparse
import json
import shutil
from pathlib import Path

import cv2


def yolo_box_from_corners(corners, width, height, pad=0.08):
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    x1, x2 = max(0.0, min(xs)), min(float(width - 1), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height - 1), max(ys))
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    grow = max(bw, bh) * pad
    x1 = max(0.0, x1 - grow)
    y1 = max(0.0, y1 - grow)
    x2 = min(float(width - 1), x2 + grow)
    y2 = min(float(height - 1), y2 + grow)
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    nw = (x2 - x1) / width
    nh = (y2 - y1) / height
    return cx, cy, nw, nh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corner-data', default='datasets/dji_action4_corner_train_500')
    ap.add_argument('--out', default='datasets/dji_action4_yolo_det')
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--copy-images', action='store_true')
    args = ap.parse_args()

    src = Path(args.corner_data)
    out = Path(args.out)
    labels = sorted((src / 'labels').glob('*.json'))
    if not labels:
        raise RuntimeError(f'No label json files found in {src / "labels"}')

    for split in ['train', 'val']:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(len(labels) * args.val_ratio))
    val_start = len(labels) - n_val
    counts = {'train': 0, 'val': 0}
    for i, label_path in enumerate(labels):
        split = 'val' if i >= val_start else 'train'
        rec = json.loads(label_path.read_text(encoding='utf-8'))
        img_src = src / rec['image']
        img = cv2.imread(str(img_src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Cannot read image {img_src}')
        h, w = img.shape[:2]
        cx, cy, bw, bh = yolo_box_from_corners(rec['corners_2d'], w, h)
        stem = label_path.stem
        img_dst = out / 'images' / split / f'{stem}.png'
        label_dst = out / 'labels' / split / f'{stem}.txt'
        if args.copy_images:
            shutil.copy2(img_src, img_dst)
        else:
            if img_dst.exists() or img_dst.is_symlink():
                img_dst.unlink()
            img_dst.symlink_to(img_src.resolve())
        label_dst.write_text(f'0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n', encoding='utf-8')
        counts[split] += 1

    yaml_path = out / 'dji_action4.yaml'
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: dji_action4\n",
        encoding='utf-8',
    )
    print(json.dumps({'out': str(out), 'yaml': str(yaml_path), **counts}, indent=2))


if __name__ == '__main__':
    main()
