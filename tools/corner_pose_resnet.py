import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models


def bbox_from_points(points, w, h, pad_ratio=0.35, jitter=0.0):
    pts = np.asarray(points, dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = max(bw, bh) * pad_ratio
    x1 = x1 - pad
    y1 = y1 - pad
    x2 = x2 + pad
    y2 = y2 + pad
    if jitter > 0:
        jx = max(bw, bh) * jitter
        jy = max(bw, bh) * jitter
        x1 += np.random.uniform(-jx, jx)
        x2 += np.random.uniform(-jx, jx)
        y1 += np.random.uniform(-jy, jy)
        y2 += np.random.uniform(-jy, jy)
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(w - 1), x2)
    y2 = min(float(h - 1), y2)
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



def apply_image_augment(img):
    out = img.copy()
    if np.random.rand() < 0.8:
        alpha = np.random.uniform(0.7, 1.35)
        beta = np.random.uniform(-28.0, 28.0)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.45:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= np.random.uniform(0.65, 1.45)
        hsv[:, :, 2] *= np.random.uniform(0.75, 1.25)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if np.random.rand() < 0.35:
        k = int(np.random.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if np.random.rand() < 0.20:
        k = int(np.random.choice([5, 7, 9]))
        kernel = np.zeros((k, k), dtype=np.float32)
        if np.random.rand() < 0.5:
            kernel[k // 2, :] = 1.0
        else:
            kernel[:, k // 2] = 1.0
        kernel /= max(kernel.sum(), 1e-6)
        out = cv2.filter2D(out, -1, kernel)
    if np.random.rand() < 0.40:
        noise = np.random.normal(0.0, np.random.uniform(2.0, 10.0), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.25:
        quality = int(np.random.uniform(45, 85))
        ok, enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


class CornerDataset(Dataset):
    def __init__(
        self,
        root,
        image_size=(256, 256),
        heatmap_size=(64, 64),
        sigma=1.8,
        roi=False,
        roi_pad=0.60,
        roi_jitter=0.10,
        train=True,
        aug=True,
    ):
        self.root = Path(root)
        self.labels = sorted((self.root / 'labels').glob('*.json'))
        self.image_w, self.image_h = image_size
        self.hm_w, self.hm_h = heatmap_size
        self.sigma = sigma
        self.roi = roi
        self.roi_pad = roi_pad
        self.roi_jitter = roi_jitter
        self.train = train
        self.aug = aug
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
            bbox = bbox_from_points(
                corners[:, :2],
                src_w,
                src_h,
                pad_ratio=self.roi_pad,
                jitter=self.roi_jitter if self.train else 0.0,
            )
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
        if self.aug and self.train:
            img = apply_image_augment(img)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img.transpose(2, 0, 1)
        hms = self._heatmaps(corners, label_w, label_h)
        pts = np.array([[p[0] / label_w * self.hm_w, p[1] / label_h * self.hm_h] for p in corners], dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(hms), torch.from_numpy(pts)


class ResNetCornerNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, out_channels=8):
        super().__init__()
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            c3, c4, c5 = 128, 256, 512
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            net = models.resnet34(weights=weights)
            c3, c4, c5 = 128, 256, 512
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.lat5 = nn.Conv2d(c5, 256, 1)
        self.lat4 = nn.Conv2d(c4, 256, 1)
        self.lat3 = nn.Conv2d(c3, 256, 1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False) + y

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lat5(c5)
        p4 = self.smooth4(self._upsample_add(p5, self.lat4(c4)))
        p3 = self.smooth3(self._upsample_add(p4, self.lat3(c3)))
        return self.head(p3)


def decode_heatmaps(logits, topk=9):
    hms = torch.sigmoid(logits).detach().cpu().numpy()
    pts = []
    conf = []
    for b in range(hms.shape[0]):
        bpts = []
        bconf = []
        for k in range(8):
            hm = hms[b, k]
            flat = hm.reshape(-1)
            kk = min(topk, flat.shape[0])
            idxs = np.argpartition(flat, -kk)[-kk:]
            scores = flat[idxs]
            ys, xs = np.divmod(idxs, hm.shape[1])
            weights = scores / max(np.sum(scores), 1e-6)
            x = float(np.sum(xs * weights))
            y = float(np.sum(ys * weights))
            bpts.append([x, y])
            bconf.append(float(np.max(scores)))
        pts.append(bpts)
        conf.append(bconf)
    return np.array(pts, dtype=np.float32), np.array(conf, dtype=np.float32)


def preprocess_frame(frame, image_size=(256, 256)):
    img = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def draw_prediction(frame, pts, conf, heatmap_size=(64, 64), color=(0, 255, 255)):
    h, w = frame.shape[:2]
    hm_w, hm_h = heatmap_size
    scale_x = w / float(hm_w)
    scale_y = h / float(hm_h)
    pts_img = []
    for i, (x, y) in enumerate(pts):
        px, py = int(round(x * scale_x)), int(round(y * scale_y))
        pts_img.append((px, py))
        cv2.circle(frame, (px, py), 6, color, -1)
        cv2.putText(frame, str(i), (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        cv2.line(frame, pts_img[a], pts_img[b], (0, 180, 255), 2)
    cv2.putText(frame, f'mean_conf={float(np.mean(conf)):.3f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
    return frame


def train(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    train_base = CornerDataset(
        args.data,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        sigma=args.sigma,
        roi=args.roi,
        roi_pad=args.roi_pad,
        roi_jitter=args.roi_jitter,
        train=True,
        aug=args.augment,
    )
    val_base = CornerDataset(
        args.data,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        sigma=args.sigma,
        roi=args.roi,
        roi_pad=args.roi_pad,
        roi_jitter=args.roi_jitter,
        train=False,
        aug=False,
    )
    total = len(train_base)
    n_val = max(1, int(total * 0.15))
    n_train = total - n_val
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(7)).tolist()
    train_ds = torch.utils.data.Subset(train_base, indices[:n_train])
    val_ds = torch.utils.data.Subset(val_base, indices[n_train:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ResNetCornerNet(backbone=args.backbone, pretrained=not args.no_pretrained).to(device)
    if getattr(args, 'init_ckpt', None):
        ckpt = torch.load(args.init_ckpt, map_location=device)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        print(f'loaded init checkpoint: {args.init_ckpt}')
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
            pred_sigmoid = torch.sigmoid(pred)
            loss_mse = F.mse_loss(pred_sigmoid, hm)
            loss_bce = F.binary_cross_entropy(pred_sigmoid.clamp(1e-5, 1 - 1e-5), hm)
            loss = loss_mse + 0.5 * loss_bce
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
                pred_sigmoid = torch.sigmoid(pred)
                loss_mse = F.mse_loss(pred_sigmoid, hm)
                loss_bce = F.binary_cross_entropy(pred_sigmoid.clamp(1e-5, 1 - 1e-5), hm)
                val_loss += (loss_mse + 0.5 * loss_bce).item() * img.size(0)
                ppts, _ = decode_heatmaps(pred)
                err = np.linalg.norm(ppts - pts.numpy(), axis=-1).mean()
                val_err += err * img.size(0)
                count += img.size(0)
        val_loss /= len(val_ds)
        val_err /= max(1, count)
        print(f'epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_err_hm_px={val_err:.2f}')
        if val_loss < best:
            best = val_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, out / 'best.pt')
    torch.save({'model': model.state_dict(), 'epoch': args.epochs, 'args': vars(args)}, out / 'last.pt')


def infer_video(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = ResNetCornerNet(backbone=args.backbone, pretrained=False).to(device)
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
            x = preprocess_frame(frame, image_size=(args.image_size, args.image_size)).to(device)
            logits = model(x)
            pts, conf = decode_heatmaps(logits)
            vis = frame.copy()
            draw_prediction(vis, pts[0], conf[0], heatmap_size=(args.heatmap_size, args.heatmap_size))
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
        'backbone': args.backbone,
    }
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('train')
    p.add_argument('--data', default='datasets/dji_action4_corner_train_aug_3000')
    p.add_argument('--out', default='runs/corner_resnet18_aug')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet34'])
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--heatmap-size', type=int, default=64)
    p.add_argument('--sigma', type=float, default=1.8)
    p.add_argument('--roi', action='store_true')
    p.add_argument('--roi-pad', type=float, default=0.60)
    p.add_argument('--roi-jitter', type=float, default=0.20)
    p.add_argument('--augment', action='store_true')
    p.add_argument('--init-ckpt')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.set_defaults(func=train)

    p = sub.add_parser('infer-video')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    p.add_argument('--out', default='runs/corner_resnet18_aug/head_left_rgb_raw_pred.mp4')
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--max-frames', type=int, default=300)
    p.add_argument('--output-width', type=int, default=960)
    p.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet34'])
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--heatmap-size', type=int, default=64)
    p.add_argument('--cpu', action='store_true')
    p.set_defaults(func=infer_video)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
