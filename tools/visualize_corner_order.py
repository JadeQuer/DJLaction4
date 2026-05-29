import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Corner order in labels:
# 0: x-, y-, z-   1: x-, y-, z+
# 2: x-, y+, z-   3: x-, y+, z+
# 4: x+, y-, z-   5: x+, y-, z+
# 6: x+, y+, z-   7: x+, y+, z+
X_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]
Y_EDGES = [(0, 2), (1, 3), (4, 6), (5, 7)]
Z_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7)]
EDGES = [(X_EDGES, (0, 80, 255), 'x axis edges'), (Y_EDGES, (0, 200, 0), 'y axis edges'), (Z_EDGES, (255, 80, 0), 'z axis edges')]
COLORS = [(0,0,255),(0,100,255),(0,220,255),(0,220,0),(255,120,0),(255,0,0),(255,0,200),(140,0,255)]


def draw_label_box(img, text, pos, color):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    th = 2
    (tw, th_text), base = cv2.getTextSize(text, font, scale, th)
    x2, y2 = x + tw + 10, y + th_text + base + 10
    cv2.rectangle(img, (x, y), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x2, y2), color, 2)
    cv2.putText(img, text, (x + 5, y + th_text + 4), font, scale, color, th, cv2.LINE_AA)


def crop_around_corners(img, corners, pad_ratio=1.6):
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    h, w = img.shape[:2]
    x1, y1 = pts[:,0].min(), pts[:,1].min()
    x2, y2 = pts[:,0].max(), pts[:,1].max()
    bw, bh = max(1, x2-x1), max(1, y2-y1)
    pad = max(bw, bh) * pad_ratio
    x1 = int(max(0, np.floor(x1-pad)))
    y1 = int(max(0, np.floor(y1-pad)))
    x2 = int(min(w-1, np.ceil(x2+pad)))
    y2 = int(min(h-1, np.ceil(y2+pad)))
    crop = img[y1:y2+1, x1:x2+1].copy()
    shifted = pts.copy()
    shifted[:,0] -= x1
    shifted[:,1] -= y1
    return crop, shifted


def draw_order_overlay(img, pts, title):
    out = img.copy()
    scale = 900 / max(out.shape[:2])
    if scale > 1:
        out = cv2.resize(out, (int(out.shape[1]*scale), int(out.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        pts = pts * scale
    for edge_group, color, _ in EDGES:
        for a, b in edge_group:
            pa = tuple(np.round(pts[a]).astype(int))
            pb = tuple(np.round(pts[b]).astype(int))
            cv2.line(out, pa, pb, color, 4, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, 12, COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(out, p, 15, (255,255,255), 3, cv2.LINE_AA)
        draw_label_box(out, str(i), (p[0] + 10, p[1] - 32), COLORS[i])
    draw_label_box(out, title, (18, 18), (0, 255, 0))
    y0 = 70
    for _, color, name in EDGES:
        cv2.line(out, (22, y0), (90, y0), color, 5, cv2.LINE_AA)
        cv2.putText(out, name, (105, y0 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y0 += 32
    return out


def make_canonical_guide(out_path):
    canvas = np.full((900, 1100, 3), 250, dtype=np.uint8)
    # Two projected rectangles: z- and z+ faces.
    pts = np.array([
        [260, 620], [390, 470], [260, 260], [390, 120],
        [720, 620], [850, 470], [720, 260], [850, 120],
    ], dtype=np.float32)
    cv2.putText(canvas, 'Canonical BB8 corner order used by labels', (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20,20,20), 3, cv2.LINE_AA)
    cv2.putText(canvas, 'Each number is fixed by 3D box coordinates, not by image left/right.', (45, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40,40,40), 2, cv2.LINE_AA)
    overlay = draw_order_overlay(canvas, pts, '3D box order')
    rows = [
        '0 = x-, y-, z-    1 = x-, y-, z+',
        '2 = x-, y+, z-    3 = x-, y+, z+',
        '4 = x+, y-, z-    5 = x+, y-, z+',
        '6 = x+, y+, z-    7 = x+, y+, z+',
        '',
        'Orange edges: x direction   Green edges: y direction   Blue edges: z direction',
        'This order is mathematically consistent, but may look unintuitive after projection.',
    ]
    y = 720
    for row in rows:
        cv2.putText(overlay, row, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2, cv2.LINE_AA)
        y += 34
    cv2.imwrite(str(out_path), overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/dji_action4_corner_train_500')
    ap.add_argument('--out-dir', default='runs/vis_corner_order_clear')
    ap.add_argument('--count', type=int, default=24)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir)
    sample_dir = out / 'samples'
    sample_dir.mkdir(parents=True, exist_ok=True)
    make_canonical_guide(out / 'corner_order_guide.png')
    labels = sorted((root / 'labels').glob('*.json'))[:args.count]
    records = []
    for lp in labels:
        rec = json.loads(lp.read_text(encoding='utf-8'))
        img = cv2.imread(str(root / rec['image']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        crop, pts = crop_around_corners(img, rec['corners_2d'])
        overlay = draw_order_overlay(crop, pts, lp.stem)
        out_path = sample_dir / f'{lp.stem}_clear_order.png'
        cv2.imwrite(str(out_path), overlay)
        records.append(str(out_path))
    (out / 'index.txt').write_text('\n'.join(records) + '\n', encoding='utf-8')
    print({'out': str(out), 'samples': len(records)})


if __name__ == '__main__':
    main()
