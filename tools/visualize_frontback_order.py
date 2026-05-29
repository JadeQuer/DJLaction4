import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FRONT = [0, 1, 2, 3]
BACK = [4, 5, 6, 7]
FRONT_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0)]
BACK_EDGES = [(4, 5), (5, 7), (7, 6), (6, 4)]
SIDE_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]
POINT_COLORS = [
    (0, 0, 255), (0, 140, 255), (0, 220, 255), (0, 220, 0),
    (255, 80, 0), (255, 0, 0), (255, 0, 200), (150, 0, 255),
]


def crop_around_corners(img, corners, pad_ratio=1.6):
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    h, w = img.shape[:2]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad = max(bw, bh) * pad_ratio
    x1 = int(max(0, np.floor(x1 - pad)))
    y1 = int(max(0, np.floor(y1 - pad)))
    x2 = int(min(w - 1, np.ceil(x2 + pad)))
    y2 = int(min(h - 1, np.ceil(y2 + pad)))
    crop = img[y1:y2 + 1, x1:x2 + 1].copy()
    shifted = np.asarray(corners, dtype=np.float32).copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    return crop, shifted


def brighten(img):
    out = img.astype(np.float32) * 1.65 + 55
    out = np.clip(out, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def draw_label_box(img, text, pos, color):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.82
    th = 2
    (tw, th_text), base = cv2.getTextSize(text, font, scale, th)
    x2, y2 = x + tw + 10, y + th_text + base + 10
    cv2.rectangle(img, (x, y), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x2, y2), color, 2)
    cv2.putText(img, text, (x + 5, y + th_text + 4), font, scale, color, th, cv2.LINE_AA)


def overlay_poly(img, pts, ids, color, alpha):
    poly = np.round(pts[ids, :2]).astype(np.int32)
    layer = img.copy()
    cv2.fillConvexPoly(layer, poly, color, cv2.LINE_AA)
    img = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
    cv2.polylines(img, [poly], True, color, 4, cv2.LINE_AA)
    return img


def draw_frontback(img, corners, title):
    crop, pts = crop_around_corners(img, corners)
    crop = brighten(crop)
    scale = 980 / max(crop.shape[:2])
    if scale > 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
        pts[:, :2] *= scale
    vis = crop.copy()
    vis = overlay_poly(vis, pts, FRONT, (40, 190, 40), 0.30)
    vis = overlay_poly(vis, pts, BACK, (50, 70, 230), 0.22)
    for a, b in FRONT_EDGES:
        cv2.line(vis, tuple(np.round(pts[a, :2]).astype(int)), tuple(np.round(pts[b, :2]).astype(int)), (40, 190, 40), 5, cv2.LINE_AA)
    for a, b in BACK_EDGES:
        cv2.line(vis, tuple(np.round(pts[a, :2]).astype(int)), tuple(np.round(pts[b, :2]).astype(int)), (50, 70, 230), 4, cv2.LINE_AA)
    for a, b in SIDE_EDGES:
        cv2.line(vis, tuple(np.round(pts[a, :2]).astype(int)), tuple(np.round(pts[b, :2]).astype(int)), (0, 190, 255), 4, cv2.LINE_AA)
    for i, (x, y, *_rest) in enumerate(pts):
        p = (int(round(x)), int(round(y)))
        cv2.circle(vis, p, 12, POINT_COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(vis, p, 15, (255, 255, 255), 3, cv2.LINE_AA)
        draw_label_box(vis, str(i), (p[0] + 10, p[1] - 32), POINT_COLORS[i])
    draw_label_box(vis, title, (18, 18), (0, 0, 0))
    draw_label_box(vis, 'front face: 0-1-2-3', (18, 70), (40, 190, 40))
    draw_label_box(vis, 'back face: 4-5-6-7', (18, 122), (50, 70, 230))
    return vis


def make_guide(path):
    canvas = np.full((900, 1200, 3), 248, dtype=np.uint8)
    pts = np.array([
        [270, 230], [650, 230], [650, 540], [270, 540],
        [450, 110], [830, 110], [830, 420], [450, 420],
    ], dtype=np.float32)
    corners = np.concatenate([pts, np.zeros((8, 1), dtype=np.float32)], axis=1)
    vis = draw_frontback(canvas, corners, 'requested front/back order')
    cv2.putText(vis, 'Requested order:', (55, 760), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    rows = [
        'Front face: 0 = left top, 1 = right top, 2 = right bottom, 3 = left bottom',
        'Back face:  4 = left top, 5 = right top, 6 = right bottom, 7 = left bottom',
        'Yellow lines connect corresponding front/back corners.',
    ]
    y = 800
    for row in rows:
        cv2.putText(vis, row, (55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
        y += 32
    cv2.imwrite(str(path), vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/dji_action4_corner_train_500_frontback')
    ap.add_argument('--out-dir', default='runs/vis_frontback_order')
    ap.add_argument('--count', type=int, default=24)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir)
    samples = out / 'samples'
    samples.mkdir(parents=True, exist_ok=True)
    make_guide(out / 'frontback_order_guide.png')
    records = []
    for lp in sorted((root / 'labels').glob('*.json'))[:args.count]:
        rec = json.loads(lp.read_text(encoding='utf-8'))
        img = cv2.imread(str(root / rec['image']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        vis = draw_frontback(img, np.asarray(rec['corners_2d'], dtype=np.float32), lp.stem)
        out_path = samples / f'{lp.stem}_frontback_order.png'
        cv2.imwrite(str(out_path), vis)
        records.append(str(out_path))
    (out / 'index.txt').write_text('\n'.join(records) + '\n', encoding='utf-8')
    print({'out': str(out), 'samples': len(records)})


if __name__ == '__main__':
    main()
