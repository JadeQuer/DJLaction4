import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FACES = {
    'x-': [0, 1, 3, 2],
    'x+': [4, 5, 7, 6],
    'y-': [0, 1, 5, 4],
    'y+': [2, 3, 7, 6],
    'z-': [0, 2, 6, 4],
    'z+': [1, 3, 7, 5],
}
POINT_COLORS = [(0,0,255),(0,128,255),(0,220,255),(0,220,0),(255,128,0),(255,0,0),(255,0,200),(140,0,255)]


def crop_around_corners(img, corners, pad_ratio=1.7):
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
    shifted = np.asarray(corners, dtype=np.float32).copy()
    shifted[:,0] -= x1
    shifted[:,1] -= y1
    return crop, shifted


def brighten(img):
    out = img.astype(np.float32)
    out = out * 1.65 + 55.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(out)
    l = cv2.equalizeHist(l)
    out = cv2.merge([l, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def draw_label_box(img, text, pos, color):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    th = 2
    (tw, th_text), base = cv2.getTextSize(text, font, scale, th)
    x2, y2 = x + tw + 10, y + th_text + base + 10
    cv2.rectangle(img, (x, y), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x2, y2), color, 2)
    cv2.putText(img, text, (x + 5, y + th_text + 4), font, scale, color, th, cv2.LINE_AA)


def face_depths(corners):
    depths = {}
    for name, ids in FACES.items():
        depths[name] = float(np.mean(corners[ids, 2]))
    return depths


def overlay_face(img, pts, ids, color, alpha=0.30):
    poly = np.round(pts[ids, :2]).astype(np.int32)
    layer = img.copy()
    cv2.fillConvexPoly(layer, poly, color, cv2.LINE_AA)
    out = cv2.addWeighted(layer, alpha, img, 1-alpha, 0)
    cv2.polylines(out, [poly], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/dji_action4_corner_train_500')
    ap.add_argument('--out-dir', default='runs/vis_front_back_faces')
    ap.add_argument('--count', type=int, default=24)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples_dir = out / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    guide = np.full((850, 1200, 3), 248, dtype=np.uint8)
    cv2.putText(guide, 'How to read front/back in synthetic labels', (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20,20,20), 3, cv2.LINE_AA)
    cv2.putText(guide, 'Green = nearest face to camera, Red = farthest face from camera', (45, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30,30,30), 2, cv2.LINE_AA)
    cv2.putText(guide, 'This is only for understanding orientation; point numbers still come from the fixed BB8 order.', (45, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40,40,40), 2, cv2.LINE_AA)
    cv2.rectangle(guide, (60, 240), (170, 320), (40, 180, 40), -1)
    cv2.rectangle(guide, (60, 360), (170, 440), (50, 50, 220), -1)
    cv2.putText(guide, 'Nearest face', (210, 295), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40,180,40), 3, cv2.LINE_AA)
    cv2.putText(guide, 'Farthest face', (210, 415), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,50,220), 3, cv2.LINE_AA)
    cv2.putText(guide, 'Use this guide before looking at the sample images.', (45, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30,30,30), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / 'front_back_guide.png'), guide)

    labels = sorted((root / 'labels').glob('*.json'))[:args.count]
    records = []
    for lp in labels:
        rec = json.loads(lp.read_text(encoding='utf-8'))
        img = cv2.imread(str(root / rec['image']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        crop, corners = crop_around_corners(img, rec['corners_2d'])
        crop = brighten(crop)
        scale = 950 / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        corners[:, :2] *= scale
        depths = face_depths(corners)
        nearest = min(depths, key=depths.get)
        farthest = max(depths, key=depths.get)
        vis = crop.copy()
        vis = overlay_face(vis, corners, FACES[nearest], (40, 180, 40), 0.35)
        vis = overlay_face(vis, corners, FACES[farthest], (50, 50, 220), 0.22)
        for i, (x, y, z) in enumerate(corners):
            p = (int(round(x)), int(round(y)))
            cv2.circle(vis, p, 11, POINT_COLORS[i], -1, cv2.LINE_AA)
            cv2.circle(vis, p, 14, (255,255,255), 3, cv2.LINE_AA)
            draw_label_box(vis, str(i), (p[0] + 10, p[1] - 30), POINT_COLORS[i])
        draw_label_box(vis, f'nearest face: {nearest}', (20, 20), (40,180,40))
        draw_label_box(vis, f'farthest face: {farthest}', (20, 70), (50,50,220))
        out_path = samples_dir / f'{lp.stem}_front_back.png'
        cv2.imwrite(str(out_path), vis)
        records.append(str(out_path))
    (out / 'index.txt').write_text('\n'.join(records) + '\n', encoding='utf-8')
    print({'out': str(out), 'samples': len(records)})


if __name__ == '__main__':
    main()
