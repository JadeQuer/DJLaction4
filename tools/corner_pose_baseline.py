
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


def bbox_from_points(points, w, h, pad_ratio=0.35):
    pts = np.asarray(points, dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = max(bw, bh) * pad_ratio
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(float(w - 1), x2 + pad)
    y2 = min(float(h - 1), y2 + pad)
    return [x1, y1, x2, y2]


def crop_and_resize(img, bbox, out_size):
    x1, y1, x2, y2 = bbox
    x1i, y1i = int(np.floor(x1)), int(np.floor(y1))
    x2i, y2i = int(np.ceil(x2)), int(np.ceil(y2))
    crop = img[y1i:y2i + 1, x1i:x2i + 1]
    if crop.size == 0:
        crop = img
        x1, y1, x2, y2 = 0.0, 0.0, float(img.shape[1] - 1), float(img.shape[0] - 1)
    resized = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
    return resized, [x1, y1, x2, y2]


class CornerDataset(Dataset):
    def __init__(self, root, image_size=(320, 240), heatmap_size=(160, 120), sigma=2.0, roi=False):
        self.root = Path(root)
        self.labels = sorted((self.root / 'labels').glob('*.json'))
        self.image_w, self.image_h = image_size
        self.hm_w, self.hm_h = heatmap_size
        self.sigma = sigma
        self.roi = roi
        if not self.labels:
            raise RuntimeError(f'No labels found under {self.root / "labels"}')

    def __len__(self):
        return len(self.labels)

    def _heatmaps(self, pts, src_w, src_h):
        hms = np.zeros((8, self.hm_h, self.hm_w), dtype=np.float32)
        yy, xx = np.mgrid[0:self.hm_h, 0:self.hm_w]
        for i, (x, y, _) in enumerate(pts):
            hx = x / src_w * self.hm_w
            hy = y / src_h * self.hm_h
            hms[i] = np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * self.sigma ** 2))
        return hms

    def __getitem__(self, idx):
        d = json.loads(self.labels[idx].read_text())
        img = cv2.imread(str(self.root / d['image']), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Failed to read {self.root / d["image"]}')
        src_h, src_w = img.shape[:2]
        corners = np.array(d['corners_2d'], dtype=np.float32)
        if self.roi:
            bbox = bbox_from_points(corners[:, :2], src_w, src_h, pad_ratio=0.60)
            img, bbox = crop_and_resize(img, bbox, (self.image_w, self.image_h))
            x1, y1, x2, y2 = bbox
            crop_w = max(1.0, x2 - x1 + 1.0)
            crop_h = max(1.0, y2 - y1 + 1.0)
            transformed = []
            for x, y, z in corners:
                transformed.append([(x - x1) / crop_w * self.image_w, (y - y1) / crop_h * self.image_h, z])
            corners = np.array(transformed, dtype=np.float32)
            label_w, label_h = self.image_w, self.image_h
        else:
            img = cv2.resize(img, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA)
            label_w, label_h = src_w, src_h
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img.transpose(2, 0, 1)
        hms = self._heatmaps(corners, label_w, label_h)
        pts = np.array([[p[0] / label_w * self.hm_w, p[1] / label_h * self.hm_h] for p in corners], dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(hms), torch.from_numpy(pts)


class TinyCornerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(96, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 8, 1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def decode_heatmaps(logits):
    hms = torch.sigmoid(logits).detach().cpu().numpy()
    pts = []
    conf = []
    for b in range(hms.shape[0]):
        bpts = []
        bconf = []
        for k in range(8):
            y, x = np.unravel_index(np.argmax(hms[b, k]), hms[b, k].shape)
            bpts.append([float(x), float(y)])
            bconf.append(float(hms[b, k, y, x]))
        pts.append(bpts)
        conf.append(bconf)
    return np.array(pts, dtype=np.float32), np.array(conf, dtype=np.float32)


def train(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    ds = CornerDataset(args.data, roi=args.roi)
    n_val = max(1, int(len(ds) * 0.15))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = TinyCornerNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = 1e9
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for img, hm, _ in train_loader:
            img = img.to(device)
            hm = hm.to(device)
            pred = model(img)
            loss = F.mse_loss(torch.sigmoid(pred), hm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_err = 0.0
        count = 0
        with torch.no_grad():
            for img, hm, pts in val_loader:
                img = img.to(device)
                hm = hm.to(device)
                pred = model(img)
                val_loss += F.mse_loss(torch.sigmoid(pred), hm).item() * img.size(0)
                ppts, _ = decode_heatmaps(pred)
                err = np.linalg.norm(ppts - pts.numpy(), axis=-1).mean()
                val_err += err * img.size(0)
                count += img.size(0)
        val_loss /= len(val_ds)
        val_err /= max(1, count)
        print(f'epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_err_hm_px={val_err:.2f}')
        if val_loss < best:
            best = val_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch}, out / 'best.pt')
    torch.save({'model': model.state_dict(), 'epoch': args.epochs}, out / 'last.pt')


def preprocess_frame(frame, image_size=(320, 240)):
    img = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def draw_prediction(frame, pts, conf, color=(0, 255, 255)):
    h, w = frame.shape[:2]
    scale_x = w / 160.0
    scale_y = h / 120.0
    pts_img = []
    for i, (x, y) in enumerate(pts):
        px, py = int(round(x * scale_x)), int(round(y * scale_y))
        pts_img.append((px, py))
        cv2.circle(frame, (px, py), 8, color, -1)
        cv2.putText(frame, str(i), (px + 7, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    # Box edge order for x/y/z binary corner order produced by renderer.
    edges = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]
    for a, b in edges:
        cv2.line(frame, pts_img[a], pts_img[b], (0, 180, 255), 2)
    cv2.putText(frame, f'mean_conf={float(np.mean(conf)):.3f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3)
    return frame


def infer_video(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = TinyCornerNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = args.output_width
    out_h = int(round(src_h * out_w / src_w))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps / args.stride, (out_w, out_h))

    frame_idx = 0
    written = 0
    stats = []
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue
            x = preprocess_frame(frame).to(device)
            logits = model(x)
            pts, conf = decode_heatmaps(logits)
            vis = frame.copy()
            draw_prediction(vis, pts[0], conf[0])
            vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(vis)
            stats.append(float(np.mean(conf[0])))
            written += 1
            frame_idx += 1
            if args.max_frames and written >= args.max_frames:
                break
    cap.release()
    writer.release()
    report = {
        'video': args.video,
        'output': args.out,
        'frames_written': written,
        'mean_conf': float(np.mean(stats)) if stats else 0.0,
        'min_conf': float(np.min(stats)) if stats else 0.0,
        'max_conf': float(np.max(stats)) if stats else 0.0,
    }
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('train')
    p.add_argument('--data', default='datasets/dji_action4_corner_train_100')
    p.add_argument('--out', default='runs/corner_baseline')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--roi', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.set_defaults(func=train)

    p = sub.add_parser('infer-video')
    p.add_argument('--ckpt', default='runs/corner_baseline/best.pt')
    p.add_argument('--video', default='head_left_rgb_raw.mp4')
    p.add_argument('--out', default='runs/corner_baseline/head_left_rgb_raw_pred.mp4')
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--max-frames', type=int, default=300)
    p.add_argument('--output-width', type=int, default=960)
    p.add_argument('--cpu', action='store_true')
    p.set_defaults(func=infer_video)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
