import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

EDGES = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]

def square_box(x1, y1, x2, y2, w, h, pad=0.08):
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = min(max(bw, bh) * (1.0 + 2.0 * pad), w - 1, h - 1)
    x1, x2 = cx - side * 0.5, cx + side * 0.5
    y1, y2 = cy - side * 0.5, cy + side * 0.5
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w - 1: x1 -= x2 - (w - 1); x2 = w - 1
    if y2 > h - 1: y1 -= y2 - (h - 1); y2 = h - 1
    return [int(max(0, math.floor(x1))), int(max(0, math.floor(y1))), int(min(w - 1, math.ceil(x2))), int(min(h - 1, math.ceil(y2)))]

def extract_real_rois(args):
    det = YOLO(args.detector)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rois = []
    for frame_id in range(0, total, max(1, args.frame_stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        res = det.predict(frame, conf=args.det_conf, iou=args.det_iou, max_det=args.max_rois, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            continue
        h, w = frame.shape[:2]
        boxes = res[0].boxes.xyxy.detach().cpu().numpy()
        scores = res[0].boxes.conf.detach().cpu().numpy()
        for roi_idx, bi in enumerate(np.argsort(-scores)[:args.max_rois]):
            x1, y1, x2, y2 = boxes[bi]
            sx1, sy1, sx2, sy2 = square_box(x1, y1, x2, y2, w, h, pad=args.roi_pad)
            crop = frame[sy1:sy2+1, sx1:sx2+1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (args.roi_size, args.roi_size), interpolation=cv2.INTER_AREA)
            rois.append({'frame_id': int(frame_id), 'roi_index': int(roi_idx), 'score': float(scores[bi]), 'bbox': [sx1, sy1, sx2, sy2], 'image': crop})
            if len(rois) >= args.num_rois:
                cap.release()
                return rois
    cap.release()
    return rois

def load_synth(args):
    root = Path(args.synth_root)
    labels = sorted((root / 'labels').glob('*.json'))
    if args.max_synth and len(labels) > args.max_synth:
        random.Random(7).shuffle(labels)
        labels = labels[:args.max_synth]
    items = []
    for lp in labels:
        rec = json.loads(lp.read_text(encoding='utf-8'))
        img = cv2.imread(str(root / rec['image']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        pts = np.array([[p[0], p[1]] for p in rec['corners_2d']], dtype=np.float32)
        img = cv2.resize(img, (args.roi_size, args.roi_size), interpolation=cv2.INTER_AREA)
        pts[:, 0] *= args.roi_size / max(1, w)
        pts[:, 1] *= args.roi_size / max(1, h)
        hull = cv2.convexHull(pts).astype(np.int32)
        mask = np.zeros((args.roi_size, args.roi_size), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        items.append({
            'name': lp.stem,
            'image': img,
            'mask': mask,
            'points': pts,
            'aspect': float((x2 - x1) / max(1e-6, y2 - y1)),
            'w_frac': float((x2 - x1) / args.roi_size),
            'h_frac': float((y2 - y1) / args.roi_size),
            'label': str(lp),
        })
    return items

def transform_fg(item, scale, dx, dy, size):
    center = (size * 0.5, size * 0.5)
    M = cv2.getRotationMatrix2D(center, 0.0, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    fg = cv2.warpAffine(item['image'], M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask = cv2.warpAffine(item['mask'], M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pts_h = np.concatenate([item['points'], np.ones((8,1), dtype=np.float32)], axis=1)
    pts = pts_h @ M.T
    return fg, mask, pts

def color_match(fg, target, mask):
    m = mask > 20
    if m.sum() < 100:
        return fg
    out = fg.astype(np.float32)
    src = out[m]
    dst = target.astype(np.float32)[m]
    sm, ss = src.mean(axis=0), src.std(axis=0) + 1e-6
    dm, ds = dst.mean(axis=0), dst.std(axis=0) + 1e-6
    return np.clip((out - sm) / ss * ds + dm, 0, 255).astype(np.uint8)

def composite(real, fg, mask):
    a = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(fg.astype(np.float32) * a + real.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)

def score_pair(real, comp, mask):
    m = mask > 20
    if m.sum() < 100:
        return 1e12
    diff = real.astype(np.float32) - comp.astype(np.float32)
    mse = float(np.mean(diff[m] ** 2))
    er = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY), 40, 120)
    ec = cv2.Canny(cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY), 40, 120)
    edge = float(np.mean((er[m].astype(np.float32) - ec[m].astype(np.float32)) ** 2))
    return mse + 0.08 * edge

def quick_descriptor(img):
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1)
    edges = cv2.Canny(gray, 50, 140)
    desc = np.concatenate([hist_h, hist_s, hist_v, [gray.mean(), gray.std(), (edges > 0).mean() * 4096.0]])
    return desc / (np.linalg.norm(desc) + 1e-6)

def draw_points(img, pts):
    out = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (int(round(x)), int(round(y))), 4, (0,255,255), -1)
        cv2.putText(out, str(i), (int(x)+5, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    for a, b in EDGES:
        cv2.line(out, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), (0,180,255), 1)
    return out

def make_sheet(paths, out_path, thumb_w=256, thumb_h=256):
    cols = 2
    label_h = 26
    rows = (len(paths) + cols - 1) // cols
    sheet = np.full((rows*(thumb_h+label_h), cols*thumb_w, 3), 245, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        r, c = divmod(i, cols)
        thumb = cv2.resize(im, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        y = r*(thumb_h+label_h); x = c*thumb_w
        sheet[y:y+label_h, x:x+thumb_w] = 32
        cv2.putText(sheet, p.name[:34], (x+6, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
        sheet[y+label_h:y+label_h+thumb_h, x:x+thumb_w] = thumb
    cv2.imwrite(str(out_path), sheet)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    ap.add_argument('--detector', default='runs_pre/detect/runs/dji_action4_yolo_real_full_ft/weights/best.pt')
    ap.add_argument('--synth-root', default='datasets/dji_action4_defaults_locked_check_12_mytry4')
    ap.add_argument('--out-dir', default='runs/proto_video_roi_composite')
    ap.add_argument('--num-rois', type=int, default=8)
    ap.add_argument('--frame-stride', type=int, default=36)
    ap.add_argument('--max-synth', type=int, default=160)
    ap.add_argument('--roi-size', type=int, default=256)
    ap.add_argument('--max-rois', type=int, default=2)
    ap.add_argument('--det-conf', type=float, default=0.25)
    ap.add_argument('--det-iou', type=float, default=0.5)
    ap.add_argument('--roi-pad', type=float, default=0.08)
    ap.add_argument('--top-synth', type=int, default=32)
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rois = extract_real_rois(args)
    synth = load_synth(args)
    if not rois or not synth:
        raise RuntimeError(f'Need rois and synth, got {len(rois)} rois, {len(synth)} synth')
    for item in synth:
        item['desc'] = quick_descriptor(item['image'])
    scales = [0.94, 1.0, 1.06] if args.fast else [0.88, 0.94, 1.0, 1.06, 1.12]
    shifts = [-10, 0, 10] if args.fast else [-18, -9, 0, 9, 18]
    report, written = [], []
    for ri, roi in enumerate(rois):
        real = roi['image']
        real_desc = quick_descriptor(real)
        candidates = sorted(synth, key=lambda item: float(np.linalg.norm(real_desc - item['desc'])))[:args.top_synth]
        best = None
        for item in candidates:
            for sc in scales:
                for dx in shifts:
                    for dy in shifts:
                        fg, mask, pts = transform_fg(item, sc, dx, dy, args.roi_size)
                        fgm = color_match(fg, real, mask)
                        comp = composite(real, fgm, mask)
                        score = score_pair(real, comp, mask)
                        if best is None or score < best['score']:
                            best = {'score': score, 'synth': item['name'], 'scale': sc, 'dx': dx, 'dy': dy, 'comp': comp, 'mask': mask, 'pts': pts, 'fg': fgm}
        diff = cv2.absdiff(real, best['comp'])
        top = np.concatenate([real, best['comp'], diff], axis=1)
        pts_vis = draw_points(best['comp'], best['pts'])
        bottom = np.concatenate([pts_vis, cv2.cvtColor(best['mask'], cv2.COLOR_GRAY2BGR), best['fg']], axis=1)
        grid = np.concatenate([top, bottom], axis=0)
        cv2.putText(grid, f"frame={roi['frame_id']} roi={roi['roi_index']} score={best['score']:.1f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        cv2.putText(grid, f"synth={best['synth']} sc={best['scale']} dx={best['dx']} dy={best['dy']}", (8, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        p = out / f'match_{ri:03d}_frame_{roi["frame_id"]}_roi_{roi["roi_index"]}.png'
        cv2.imwrite(str(p), grid)
        written.append(p)
        chosen = next(item for item in synth if item['name'] == best['synth'])
        report.append({
            'frame_id': roi['frame_id'],
            'roi_index': roi['roi_index'],
            'det_score': roi['score'],
            'bbox': roi['bbox'],
            'score': best['score'],
            'synth': best['synth'],
            'scale': best['scale'],
            'dx': best['dx'],
            'dy': best['dy'],
            'synth_aspect': chosen['aspect'],
            'synth_w_frac': chosen['w_frac'],
            'synth_h_frac': chosen['h_frac'],
            'synth_label': chosen['label'],
        })
    make_sheet(written, out / 'contact_sheet.png')
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'rois': len(rois), 'synth': len(synth), 'contact_sheet': str(out/'contact_sheet.png')}, indent=2))

if __name__ == '__main__':
    main()
