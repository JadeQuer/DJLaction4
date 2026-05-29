import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0),
    (255, 128, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255),
]


def draw_corners(img, corners, title=None):
    out = img.copy()
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    for a, b in EDGES:
        pa = tuple(np.round(pts[a]).astype(int))
        pb = tuple(np.round(pts[b]).astype(int))
        cv2.line(out, pa, pb, (0, 180, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, 7, COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(out, p, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(i), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[i], 2, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def make_heatmap_panel(corners, src_w, src_h, hm_size=64, sigma=1.8, scale=4):
    hm_w = hm_h = hm_size
    yy, xx = np.mgrid[0:hm_h, 0:hm_w]
    tiles = []
    for i, (x, y, *_rest) in enumerate(corners):
        hx = x / src_w * hm_w
        hy = y / src_h * hm_h
        hm = np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * sigma ** 2))
        hm_u8 = np.clip(hm * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        color = cv2.resize(color, (hm_w * scale, hm_h * scale), interpolation=cv2.INTER_NEAREST)
        cv2.putText(color, f'corner {i}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        tiles.append(color)
    row1 = np.concatenate(tiles[:4], axis=1)
    row2 = np.concatenate(tiles[4:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def resize_to_height(img, h):
    ih, iw = img.shape[:2]
    w = int(round(iw * h / ih))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def visualize_dataset(root, out_dir, count):
    root = Path(root)
    out_dir = Path(out_dir)
    overlay_dir = out_dir / 'overlays'
    heatmap_dir = out_dir / 'heatmaps'
    compare_dir = out_dir / 'compare'
    overlay_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted((root / 'labels').glob('*.json'))[:count]
    records = []
    for lp in labels:
        rec = json.loads(lp.read_text(encoding='utf-8'))
        img = cv2.imread(str(root / rec['image']), cv2.IMREAD_COLOR)
        if img is None:
            print(f'skip unreadable {root / rec["image"]}')
            continue
        h, w = img.shape[:2]
        stem = lp.stem
        overlay = draw_corners(img, rec['corners_2d'], title=f'{root.name}/{stem}')
        heatmaps = make_heatmap_panel(rec['corners_2d'], w, h)
        ov_path = overlay_dir / f'{stem}_corners.png'
        hm_path = heatmap_dir / f'{stem}_heatmaps.png'
        cv2.imwrite(str(ov_path), overlay)
        cv2.imwrite(str(hm_path), heatmaps)

        raw_small = resize_to_height(img, 480)
        overlay_small = resize_to_height(overlay, 480)
        hm_small = resize_to_height(heatmaps, 480)
        compare = np.concatenate([raw_small, overlay_small, hm_small], axis=1)
        cmp_path = compare_dir / f'{stem}_raw_overlay_heatmap.png'
        cv2.imwrite(str(cmp_path), compare)
        records.append({'label': str(lp), 'image': str(root / rec['image']), 'overlay': str(ov_path), 'heatmaps': str(hm_path), 'compare': str(cmp_path)})
    (out_dir / 'index.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'root': str(root), 'out': str(out_dir), 'items': len(records)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--count', type=int, default=24)
    args = ap.parse_args()
    visualize_dataset(args.root, args.out_dir, args.count)


if __name__ == '__main__':
    main()
