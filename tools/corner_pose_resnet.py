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


BOX_EDGE_INDEX_PAIRS = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def bbox_from_points(points, w, h, pad_ratio=0.35, jitter=0.0, square=False):
    pts = np.asarray(points, dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if square:
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        side = max(bw, bh) * (1.0 + 2.0 * pad_ratio)
        x1 = cx - side * 0.5
        x2 = cx + side * 0.5
        y1 = cy - side * 0.5
        y2 = cy + side * 0.5
    else:
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


def resize_letterbox(img, out_size, fill=(114, 114, 114)):
    out_w, out_h = out_size
    h, w = img.shape[:2]
    scale = min(out_w / max(1, w), out_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_h, out_w, 3), fill, dtype=np.uint8)
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    meta = {
        'scale': float(scale),
        'pad_x': float(pad_x),
        'pad_y': float(pad_y),
        'new_w': int(new_w),
        'new_h': int(new_h),
        'out_w': int(out_w),
        'out_h': int(out_h),
    }
    return canvas, meta


def append_coord_channels(img):
    h, w = img.shape[:2]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    grid_x = np.tile(xs[None, :], (h, 1))
    grid_y = np.tile(ys[:, None], (1, w))
    return np.concatenate([img, grid_x[:, :, None], grid_y[:, :, None]], axis=2)


def apply_lowres_resample(img, lowres_size=0):
    lowres_size = int(lowres_size or 0)
    if lowres_size <= 0:
        return img
    h, w = img.shape[:2]
    lowres_size = max(8, min(lowres_size, max(h, w)))
    scale = lowres_size / float(max(h, w))
    small_w = max(8, int(round(w * scale)))
    small_h = max(8, int(round(h * scale)))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize_bgr_for_network(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return img


def apply_zoom_out_canvas(img, corners, min_scale=1.0, fill=None):
    min_scale = float(min_scale or 1.0)
    if min_scale >= 0.999:
        return img, corners
    h, w = img.shape[:2]
    scale = float(np.random.uniform(max(0.25, min_scale), 1.0))
    if scale >= 0.999:
        return img, corners
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if fill is None:
        border = np.concatenate([
            img[: max(1, h // 12)].reshape(-1, 3),
            img[h - max(1, h // 12):].reshape(-1, 3),
            img[:, : max(1, w // 12)].reshape(-1, 3),
            img[:, w - max(1, w // 12):].reshape(-1, 3),
        ], axis=0)
        fill = tuple(int(v) for v in np.median(border, axis=0))
    canvas = np.full_like(img, fill)
    max_x = max(0, w - new_w)
    max_y = max(0, h - new_h)
    off_x = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
    off_y = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = small
    out_corners = corners.copy()
    out_corners[:, 0] = out_corners[:, 0] * (new_w / float(w)) + off_x
    out_corners[:, 1] = out_corners[:, 1] * (new_h / float(h)) + off_y
    return canvas, out_corners


def crop_and_resize(img, bbox, out_size, keep_aspect=True):
    x1, y1, x2, y2 = bbox
    x1i, y1i = int(np.floor(x1)), int(np.floor(y1))
    x2i, y2i = int(np.ceil(x2)), int(np.ceil(y2))
    crop = img[y1i:y2i + 1, x1i:x2i + 1]
    if crop.size == 0:
        crop = img
        x1, y1, x2, y2 = 0.0, 0.0, float(img.shape[1] - 1), float(img.shape[0] - 1)
    if keep_aspect:
        resized, meta = resize_letterbox(crop, out_size)
    else:
        resized = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
        crop_h, crop_w = crop.shape[:2]
        meta = {
            'scale_x': out_size[0] / max(1.0, float(crop_w)),
            'scale_y': out_size[1] / max(1.0, float(crop_h)),
            'pad_x': 0.0,
            'pad_y': 0.0,
            'out_w': int(out_size[0]),
            'out_h': int(out_size[1]),
        }
    return resized, [x1, y1, x2, y2], meta



def apply_image_augment(img):
    out = img.copy()
    if np.random.rand() < 0.55:
        h, w = out.shape[:2]
        scale = np.random.uniform(0.28, 0.75)
        small_w = max(24, int(round(w * scale)))
        small_h = max(24, int(round(h * scale)))
        interp_down = np.random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
        interp_up = np.random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        small = cv2.resize(out, (small_w, small_h), interpolation=interp_down)
        out = cv2.resize(small, (w, h), interpolation=interp_up)
    if np.random.rand() < 0.8:
        alpha = np.random.uniform(0.55, 1.45)
        beta = np.random.uniform(-42.0, 36.0)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.45:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-8.0, 8.0)) % 180.0
        hsv[:, :, 1] *= np.random.uniform(0.55, 1.55)
        hsv[:, :, 2] *= np.random.uniform(0.65, 1.35)
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


def apply_geometry_augment(img, corners):
    out = img.copy()
    pts = np.asarray(corners, dtype=np.float32)[:, :2]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    fill = tuple(int(v) for v in out.reshape(-1, 3).mean(axis=0))
    if np.random.rand() < 0.45:
        rw = (x2 - x1) * np.random.uniform(0.22, 0.46)
        rh = (y2 - y1) * np.random.uniform(0.18, 0.42)
        cx = np.random.uniform(x1 + rw * 0.5, x2 - rw * 0.5)
        cy = np.random.uniform(y1 + rh * 0.5, y2 - rh * 0.5)
        ax1 = int(np.clip(round(cx - rw * 0.5), 0, out.shape[1] - 1))
        ax2 = int(np.clip(round(cx + rw * 0.5), 0, out.shape[1] - 1))
        ay1 = int(np.clip(round(cy - rh * 0.5), 0, out.shape[0] - 1))
        ay2 = int(np.clip(round(cy + rh * 0.5), 0, out.shape[0] - 1))
        if ax2 > ax1 and ay2 > ay1:
            patch = out[ay1:ay2, ax1:ax2]
            if patch.size:
                blurred = cv2.GaussianBlur(patch, (9, 9), 0)
                mix = np.full_like(patch, fill)
                out[ay1:ay2, ax1:ax2] = cv2.addWeighted(blurred, 0.35, mix, 0.65, 0)
    if np.random.rand() < 0.30:
        mask = np.zeros(out.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(np.round(pts).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        bg = out.copy()
        if np.random.rand() < 0.5:
            bg = cv2.GaussianBlur(bg, (9, 9), 0)
        bg = cv2.convertScaleAbs(bg, alpha=np.random.uniform(0.75, 1.15), beta=np.random.uniform(-20, 20))
        out[mask == 0] = bg[mask == 0]
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
        square_roi=False,
        keep_aspect=True,
        train=True,
        aug=True,
        geometry_aug=False,
        coord_conv=False,
        input_lowres_size=0,
        zoom_out_min_scale=1.0,
    ):
        self.root = Path(root)
        self.labels = sorted((self.root / 'labels').glob('*.json'))
        self.image_w, self.image_h = image_size
        self.hm_w, self.hm_h = heatmap_size
        self.sigma = sigma
        self.roi = roi
        self.roi_pad = roi_pad
        self.roi_jitter = roi_jitter
        self.square_roi = square_roi
        self.keep_aspect = keep_aspect
        self.train = train
        self.aug = aug
        self.geometry_aug = geometry_aug
        self.coord_conv = coord_conv
        self.input_lowres_size = int(input_lowres_size or 0)
        self.zoom_out_min_scale = float(zoom_out_min_scale or 1.0)
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
                square=self.square_roi,
            )
            img, bbox, resize_meta = crop_and_resize(img, bbox, (self.image_w, self.image_h), keep_aspect=self.keep_aspect)
            x1, y1, x2, y2 = bbox
            crop_w = max(1.0, x2 - x1 + 1.0)
            crop_h = max(1.0, y2 - y1 + 1.0)
            transformed = []
            for x, y, z in corners:
                if self.keep_aspect:
                    transformed.append([
                        (x - x1) * resize_meta['scale'] + resize_meta['pad_x'],
                        (y - y1) * resize_meta['scale'] + resize_meta['pad_y'],
                        z,
                    ])
                else:
                    transformed.append([(x - x1) / crop_w * self.image_w, (y - y1) / crop_h * self.image_h, z])
            corners = np.array(transformed, dtype=np.float32)
            label_w, label_h = self.image_w, self.image_h
        else:
            img = cv2.resize(img, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA)
            label_w, label_h = src_w, src_h
        if self.aug and self.train:
            img = apply_image_augment(img)
        if self.geometry_aug and self.train:
            img = apply_geometry_augment(img, corners)
        if self.train and self.zoom_out_min_scale < 0.999:
            img, corners = apply_zoom_out_canvas(img, corners, self.zoom_out_min_scale)
        if self.input_lowres_size > 0:
            img = apply_lowres_resample(img, self.input_lowres_size)
        img = normalize_bgr_for_network(img)
        if self.coord_conv:
            img = append_coord_channels(img)
        img = img.transpose(2, 0, 1)
        hms = self._heatmaps(corners, label_w, label_h)
        pts = np.array([[p[0] / label_w * self.hm_w, p[1] / label_h * self.hm_h] for p in corners], dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(hms), torch.from_numpy(pts)


def make_resnet_layer4_dilated(net):
    """Keep ResNet layer4 at stride 16 instead of stride 32."""
    block0 = net.layer4[0]
    block0.conv1.stride = (1, 1)
    if block0.downsample is not None:
        block0.downsample[0].stride = (1, 1)
    for block in net.layer4:
        block.conv1.dilation = (2, 2)
        block.conv1.padding = (2, 2)
        block.conv2.dilation = (2, 2)
        block.conv2.padding = (2, 2)


def _make_group_norm(num_channels):
    for num_groups in (32, 16, 8, 4, 2, 1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


def build_2d_sincos_pos_embed(height, width, dim, device, dtype):
    if dim % 4 != 0:
        raise ValueError(f'pos embed dim must be divisible by 4, got {dim}')
    pos_dim = dim // 4
    omega = torch.arange(pos_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(pos_dim, 1)))
    grid_y = torch.arange(height, device=device, dtype=torch.float32)
    grid_x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing='ij')
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
    return pos.unsqueeze(0).to(device=device, dtype=dtype)


class SpatialTokenMixer(nn.Module):
    def __init__(self, dim=256, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, feat):
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)
        x = x + build_2d_sincos_pos_embed(h, w, c, feat.device, feat.dtype)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x.transpose(1, 2).reshape(b, c, h, w)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_norm.to(orig_dtype) * self.weight)


class QKNormSelfAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class BETRBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = QKNormSelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BETRHeatmapNet(nn.Module):
    def __init__(
        self,
        out_channels=8,
        heatmap_size=128,
        coord_conv=False,
        dim=256,
        patch_stride=8,
        out_patch_size=4,
        depth=6,
        num_heads=8,
    ):
        super().__init__()
        if heatmap_size != 128:
            raise ValueError('BETRHeatmapNet currently expects heatmap_size=128')
        in_channels = 5 if coord_conv else 3
        self.patch_stride = patch_stride
        self.out_patch_size = out_patch_size
        self.num_keypoints = out_channels
        self.output_wh = (heatmap_size, heatmap_size)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=7, stride=4, padding=3, bias=False),
            _make_group_norm(dim // 2),
            nn.GELU(),
            ResidualConvBlock(dim // 2, dim // 2),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1, bias=False),
            _make_group_norm(dim),
            nn.GELU(),
            ResidualConvBlock(dim, dim),
        )
        self.input_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(dim, dim),
        )
        self.input_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.bbox_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([BETRBlock(dim=dim, num_heads=num_heads) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(dim)
        self.bbox_proj = nn.Linear(dim, out_patch_size * out_patch_size * out_channels)

    def _unpatchify(self, patch_values, grid_h, grid_w):
        b = patch_values.shape[0]
        p = self.out_patch_size
        k = self.num_keypoints
        patch_values = patch_values.view(b, grid_h, grid_w, p, p, k)
        patch_values = patch_values.permute(0, 5, 1, 3, 2, 4)
        return patch_values.reshape(b, k, grid_h * p, grid_w * p)

    def forward(self, x):
        feat = self.patch_embed(x)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.input_norm(self.input_transform(tokens))
        pos = build_2d_sincos_pos_embed(h, w, c, feat.device, feat.dtype)
        tokens = tokens + pos + self.bbox_query.to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.output_norm(tokens)
        patch_logits = self.bbox_proj(tokens)
        heatmaps = self._unpatchify(patch_logits, h, w)
        if heatmaps.shape[-2:] != self.output_wh:
            heatmaps = F.interpolate(heatmaps, size=self.output_wh, mode='bilinear', align_corners=False)
        return heatmaps



class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_group_norm(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _make_group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + residual)


class TeacherHeatmapNet(nn.Module):
    def __init__(self, out_channels=8, heatmap_size=64, coord_conv=False, base_channels=48):
        super().__init__()
        in_channels = 5 if coord_conv else 3
        self.output_wh = (heatmap_size, heatmap_size)
        self.coord_conv = coord_conv
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, base_channels, stride=2),
            ResidualConvBlock(base_channels, base_channels),
        )
        self.stage1 = nn.Sequential(
            ResidualConvBlock(base_channels, base_channels * 2, stride=2),
            ResidualConvBlock(base_channels * 2, base_channels * 2),
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualConvBlock(base_channels * 4, base_channels * 4),
            ResidualConvBlock(base_channels * 4, base_channels * 4),
        )
        self.neck = nn.Sequential(
            ResidualConvBlock(base_channels * 4, base_channels * 4),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1, bias=False),
            _make_group_norm(base_channels * 2),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            _make_group_norm(base_channels * 2),
            nn.GELU(),
            ResidualConvBlock(base_channels * 2, base_channels * 2),
            ResidualConvBlock(base_channels * 2, base_channels * 2),
        )
        self.refine128 = nn.Identity()
        if heatmap_size >= 128:
            self.refine128 = nn.Sequential(
                nn.Upsample(size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=False),
                ResidualConvBlock(base_channels * 2, base_channels * 2),
                ResidualConvBlock(base_channels * 2, base_channels * 2),
            )
        self.head = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        stem = self.stem(x)
        feat64 = self.stage1(stem)
        feat32 = self.stage2(feat64)
        up64 = F.interpolate(self.neck(feat32), size=feat64.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([feat64, up64], dim=1))
        fused = self.refine128(fused)
        heatmaps = self.head(fused)
        if heatmaps.shape[-2:] != self.output_wh:
            heatmaps = F.interpolate(heatmaps, size=self.output_wh, mode='bilinear', align_corners=False)
        return heatmaps


class ResNetCornerNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, out_channels=8, heatmap_size=64, coord_conv=False, token_mixer=False):
        super().__init__()
        if heatmap_size not in (64, 128):
            raise ValueError(f'Unsupported heatmap_size: {heatmap_size}; expected 64 or 128')
        if backbone in ('resnet18', 'resnet18_dilated'):
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            c3, c4, c5 = 128, 256, 512
        elif backbone in ('resnet34', 'resnet34_dilated'):
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            net = models.resnet34(weights=weights)
            c3, c4, c5 = 128, 256, 512
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
        if backbone.endswith('_dilated'):
            make_resnet_layer4_dilated(net)
        self.coord_conv = coord_conv
        self.token_mixer_enabled = token_mixer
        if coord_conv:
            conv1 = nn.Conv2d(5, net.conv1.out_channels, kernel_size=net.conv1.kernel_size, stride=net.conv1.stride, padding=net.conv1.padding, bias=False)
            with torch.no_grad():
                conv1.weight[:, :3].copy_(net.conv1.weight)
                coord_seed = net.conv1.weight.mean(dim=1, keepdim=True)
                conv1.weight[:, 3:4].copy_(coord_seed * 0.5)
                conv1.weight[:, 4:5].copy_(coord_seed * 0.5)
            stem_conv = conv1
        else:
            stem_conv = net.conv1
        self.stem = nn.Sequential(stem_conv, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.lat5 = nn.Conv2d(c5, 256, 1)
        self.lat4 = nn.Conv2d(c4, 256, 1)
        self.lat3 = nn.Conv2d(c3, 256, 1)
        self.lat2 = nn.Conv2d(64, 256, 1)
        self.output_wh = (heatmap_size, heatmap_size)
        self.smooth4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), _make_group_norm(256), nn.ReLU(inplace=True))
        self.smooth3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), _make_group_norm(256), nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), _make_group_norm(256), nn.ReLU(inplace=True))
        self.token_mixer = SpatialTokenMixer(dim=256, num_heads=4, mlp_ratio=2.0, dropout=0.0) if token_mixer else nn.Identity()

        head_layers = [
            nn.Conv2d(256, 128, 3, padding=1),
            _make_group_norm(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            _make_group_norm(64),
            nn.ReLU(inplace=True),
        ]
        head_layers.append(nn.Conv2d(64, out_channels, 1))
        self.head = nn.Sequential(*head_layers)

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
        p3 = self.token_mixer(p3)
        p2 = self.smooth2(self._upsample_add(p3, self.lat2(c2)))
        heatmaps = self.head(p2)
        if heatmaps.shape[-2:] != self.output_wh:
            heatmaps = F.interpolate(heatmaps, size=self.output_wh, mode='bilinear', align_corners=False)
        return heatmaps



def build_pose_model(backbone, pretrained, heatmap_size, coord_conv=False, token_mixer=False):
    if backbone == 'teacher_heatmap':
        return TeacherHeatmapNet(out_channels=8, heatmap_size=heatmap_size, coord_conv=coord_conv, base_channels=48)
    if backbone == 'betr_heatmap':
        return BETRHeatmapNet(out_channels=8, heatmap_size=heatmap_size, coord_conv=coord_conv)
    return ResNetCornerNet(
        backbone=backbone,
        pretrained=pretrained,
        heatmap_size=heatmap_size,
        coord_conv=coord_conv,
        token_mixer=token_mixer,
    )


def softargmax_keypoints_from_logits(logits, temperature=0.05):
    if logits.ndim != 4:
        raise ValueError(f'Expected logits [B, K, H, W], got {tuple(logits.shape)}')
    b, k, h, w = logits.shape
    flat = logits.reshape(b, k, h * w)
    weights = torch.softmax(flat / max(float(temperature), 1e-6), dim=-1)
    xs = torch.arange(w, device=logits.device, dtype=logits.dtype).repeat(h).view(1, 1, -1)
    ys = torch.arange(h, device=logits.device, dtype=logits.dtype).repeat_interleave(w).view(1, 1, -1)
    x = torch.sum(weights * xs, dim=-1)
    y = torch.sum(weights * ys, dim=-1)
    return torch.stack([x, y], dim=-1)


def normalize_keypoints_xy(points, heatmap_w, heatmap_h):
    denom = points.new_tensor([max(float(heatmap_w - 1), 1.0), max(float(heatmap_h - 1), 1.0)])
    return points / denom.view(1, 1, 2)


def coordinate_structure_loss(logits, target_pts, coord_weight=0.0, edge_weight=0.0, temperature=0.05, beta=0.05):
    if coord_weight <= 0.0 and edge_weight <= 0.0:
        return logits.new_zeros(()), logits.new_zeros(())
    pred_pts = softargmax_keypoints_from_logits(logits, temperature=temperature)
    heatmap_h, heatmap_w = logits.shape[-2:]
    pred_norm = normalize_keypoints_xy(pred_pts, heatmap_w, heatmap_h)
    target_norm = normalize_keypoints_xy(target_pts.to(logits.device), heatmap_w, heatmap_h)

    coord_loss = logits.new_zeros(())
    if coord_weight > 0.0:
        coord_loss = F.smooth_l1_loss(pred_norm, target_norm, beta=beta)

    edge_loss = logits.new_zeros(())
    if edge_weight > 0.0:
        pred_edges = []
        target_edges = []
        for a, b in BOX_EDGE_INDEX_PAIRS:
            pred_edges.append(pred_norm[:, b] - pred_norm[:, a])
            target_edges.append(target_norm[:, b] - target_norm[:, a])
        edge_loss = F.smooth_l1_loss(torch.stack(pred_edges, dim=1), torch.stack(target_edges, dim=1), beta=beta)
    return coord_loss, edge_loss


def heatmap_supervision_loss(pred_sigmoid, target, loss_type='mse_bce', positive_weight=0.0, positive_gamma=2.0):
    if loss_type == 'mse_bce':
        loss_mse = F.mse_loss(pred_sigmoid, target)
        loss_bce = F.binary_cross_entropy(pred_sigmoid.clamp(1e-5, 1 - 1e-5), target)
        return loss_mse + 0.5 * loss_bce
    if loss_type == 'mse':
        if positive_weight > 0.0:
            weights = 1.0 + float(positive_weight) * torch.pow(target.clamp(0.0, 1.0), float(positive_gamma))
            return ((pred_sigmoid - target) ** 2 * weights).sum() / weights.sum().clamp_min(1.0)
        return F.mse_loss(pred_sigmoid, target)
    if loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_sigmoid, target, beta=0.05)
    if loss_type == 'focal':
        pred = pred_sigmoid.clamp(1e-4, 1.0 - 1e-4)
        pos_mask = (target >= 0.99).to(dtype=pred.dtype)
        neg_mask = (target < 0.99).to(dtype=pred.dtype)
        neg_weights = torch.pow(1.0 - target.clamp(0.0, 1.0), 4.0)
        pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2.0) * pos_mask
        neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2.0) * neg_weights * neg_mask
        num_pos = pos_mask.sum()
        if num_pos > 0:
            return -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return -neg_loss.sum() / neg_mask.sum().clamp_min(1.0)
    raise ValueError(f'Unsupported heatmap loss: {loss_type}')


def decode_heatmaps(logits, topk=9, mode='topk_mean', temperature=0.05):
    hms = torch.sigmoid(logits).detach().cpu().numpy()
    if mode not in {'topk_mean', 'argmax', 'softargmax'}:
        raise ValueError(f'Unsupported decode mode: {mode}')
    pts = []
    conf = []
    for b in range(hms.shape[0]):
        bpts = []
        bconf = []
        for k in range(8):
            hm = hms[b, k]
            flat = hm.reshape(-1)
            if mode == 'argmax':
                idx = int(np.argmax(flat))
                y, x = divmod(idx, hm.shape[1])
                bpts.append([float(x) + 0.5, float(y) + 0.5])
                bconf.append(float(flat[idx]))
                continue
            if mode == 'softargmax':
                scores = flat.astype(np.float32)
                logits_np = scores / max(float(temperature), 1e-6)
                logits_np = logits_np - float(np.max(logits_np))
                weights = np.exp(logits_np)
                weights = weights / max(float(np.sum(weights)), 1e-12)
                ys, xs = np.divmod(np.arange(flat.shape[0]), hm.shape[1])
                bpts.append([float(np.sum(xs * weights)), float(np.sum(ys * weights))])
                bconf.append(float(np.max(scores)))
                continue
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


def preprocess_frame(frame, image_size=(256, 256), keep_aspect=False, return_meta=False, coord_conv=False, input_lowres_size=0):
    if keep_aspect:
        img_u8, meta = resize_letterbox(frame, image_size)
    else:
        img_u8 = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
        meta = {
            'scale_x': image_size[0] / max(1.0, float(frame.shape[1])),
            'scale_y': image_size[1] / max(1.0, float(frame.shape[0])),
            'pad_x': 0.0,
            'pad_y': 0.0,
            'out_w': int(image_size[0]),
            'out_h': int(image_size[1]),
        }
    img_u8 = apply_lowres_resample(img_u8, input_lowres_size)
    img = normalize_bgr_for_network(img_u8)
    if coord_conv:
        img = append_coord_channels(img)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    if return_meta:
        return tensor, meta
    return tensor


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
    edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
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
        square_roi=args.square_roi,
        keep_aspect=not args.stretch_roi,
        train=True,
        aug=args.augment,
        geometry_aug=args.geometry_augment,
        coord_conv=args.coord_conv,
        input_lowres_size=args.input_lowres_size,
        zoom_out_min_scale=args.zoom_out_min_scale,
    )
    val_base = CornerDataset(
        args.data,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        sigma=args.sigma,
        roi=args.roi,
        roi_pad=args.roi_pad,
        roi_jitter=args.roi_jitter,
        square_roi=args.square_roi,
        keep_aspect=not args.stretch_roi,
        train=False,
        aug=False,
        geometry_aug=False,
        coord_conv=args.coord_conv,
        input_lowres_size=args.input_lowres_size,
        zoom_out_min_scale=1.0,
    )
    total = len(train_base)
    n_val = max(1, int(total * 0.15))
    n_train = total - n_val
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(7)).tolist()
    train_ds = torch.utils.data.Subset(train_base, indices[:n_train])
    val_ds = torch.utils.data.Subset(val_base, indices[n_train:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_pose_model(args.backbone, pretrained=not args.no_pretrained, heatmap_size=args.heatmap_size, coord_conv=args.coord_conv, token_mixer=args.token_mixer).to(device)
    if getattr(args, 'init_ckpt', None):
        ckpt = torch.load(args.init_ckpt, map_location=device)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        print(f'loaded init checkpoint: {args.init_ckpt}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = min(max(0, args.warmup_steps), total_steps - 1)
    scheduler = None
    if args.scheduler == 'cosine':
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            min_ratio = args.min_lr / max(args.lr, 1e-12)
            return float(min_ratio + (1.0 - min_ratio) * cosine)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    best = 1e9
    best_epoch = 0
    stale_epochs = 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_coord_loss = 0.0
        train_edge_loss = 0.0
        for img, hm, pts in train_loader:
            img = img.to(device)
            hm = hm.to(device)
            pred = model(img)
            pred_sigmoid = torch.sigmoid(pred)
            heatmap_loss = heatmap_supervision_loss(
                pred_sigmoid,
                hm,
                loss_type=args.heatmap_loss,
                positive_weight=args.positive_weight,
                positive_gamma=args.positive_gamma,
            )
            coord_loss, edge_loss = coordinate_structure_loss(
                pred,
                pts.to(device),
                coord_weight=args.coord_loss_weight,
                edge_weight=args.edge_loss_weight,
                temperature=args.softargmax_temperature,
            )
            loss = heatmap_loss + args.coord_loss_weight * coord_loss + args.edge_loss_weight * edge_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            train_loss += loss.item() * img.size(0)
            train_coord_loss += coord_loss.item() * img.size(0)
            train_edge_loss += edge_loss.item() * img.size(0)
        train_loss /= len(train_ds)
        train_coord_loss /= len(train_ds)
        train_edge_loss /= len(train_ds)

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
                heatmap_loss = heatmap_supervision_loss(
                    pred_sigmoid,
                    hm,
                    loss_type=args.heatmap_loss,
                    positive_weight=args.positive_weight,
                    positive_gamma=args.positive_gamma,
                )
                coord_loss, edge_loss = coordinate_structure_loss(
                    pred,
                    pts.to(device),
                    coord_weight=args.coord_loss_weight,
                    edge_weight=args.edge_loss_weight,
                    temperature=args.softargmax_temperature,
                )
                total_val_loss = heatmap_loss + args.coord_loss_weight * coord_loss + args.edge_loss_weight * edge_loss
                val_loss += total_val_loss.item() * img.size(0)
                ppts, _ = decode_heatmaps(pred, topk=args.decode_topk, mode=args.decode_mode, temperature=args.softargmax_temperature)
                err = np.linalg.norm(ppts - pts.numpy(), axis=-1).mean()
                val_err += err * img.size(0)
                count += img.size(0)
        val_loss /= len(val_ds)
        val_err /= max(1, count)
        current_lr = opt.param_groups[0]['lr']
        print(f'epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_err_hm_px={val_err:.2f} coord_loss={train_coord_loss:.6f} edge_loss={train_edge_loss:.6f} lr={current_lr:.6g} loss={args.heatmap_loss} decode={args.decode_mode}')
        if val_loss < best - args.early_stop_min_delta:
            best = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'args': vars(args),
                'best_val_loss': float(val_loss),
                'best_val_err_hm_px': float(val_err),
            }, out / 'best.pt')
        else:
            stale_epochs += 1
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            print(f'early stopping at epoch {epoch:03d}; best_epoch={best_epoch:03d} best_val_loss={best:.6f} patience={args.early_stop_patience}')
            break
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, out / 'last.pt')


def infer_video(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = build_pose_model(args.backbone, pretrained=False, heatmap_size=args.heatmap_size, coord_conv=args.coord_conv, token_mixer=args.token_mixer).to(device)
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
            x = preprocess_frame(
                frame,
                image_size=(args.image_size, args.image_size),
                coord_conv=args.coord_conv,
                input_lowres_size=args.input_lowres_size,
            ).to(device)
            logits = model(x)
            pts, conf = decode_heatmaps(logits, topk=args.decode_topk, mode=args.decode_mode, temperature=args.softargmax_temperature)
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
        'decode_mode': args.decode_mode,
        'decode_topk': args.decode_topk,
        'softargmax_temperature': args.softargmax_temperature,
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
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--scheduler', default='none', choices=['none', 'cosine'])
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet34', 'resnet18_dilated', 'resnet34_dilated', 'teacher_heatmap', 'betr_heatmap'])
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--heatmap-size', type=int, default=64)
    p.add_argument('--sigma', type=float, default=1.8)
    p.add_argument('--roi', action='store_true')
    p.add_argument('--roi-pad', type=float, default=0.14)
    p.add_argument('--roi-jitter', type=float, default=0.06)
    p.add_argument('--square-roi', action='store_true')
    p.add_argument('--stretch-roi', action='store_true')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--geometry-augment', action='store_true')
    p.add_argument('--init-ckpt')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--coord-conv', action='store_true')
    p.add_argument('--token-mixer', action='store_true')
    p.add_argument('--input-lowres-size', type=int, default=0, help='Downsample each network input to this max side then upsample back; 0 disables.')
    p.add_argument('--zoom-out-min-scale', type=float, default=1.0, help='Training-only random zoom-out lower bound; 1 disables.')
    p.add_argument('--decode-mode', default='topk_mean', choices=['topk_mean', 'argmax', 'softargmax'])
    p.add_argument('--decode-topk', type=int, default=9)
    p.add_argument('--softargmax-temperature', type=float, default=0.05)
    p.add_argument('--heatmap-loss', default='mse_bce', choices=['mse_bce', 'mse', 'smooth_l1', 'focal'])
    p.add_argument('--positive-weight', type=float, default=0.0)
    p.add_argument('--positive-gamma', type=float, default=2.0)
    p.add_argument('--coord-loss-weight', type=float, default=0.0)
    p.add_argument('--edge-loss-weight', type=float, default=0.0)
    p.add_argument('--early-stop-patience', type=int, default=0)
    p.add_argument('--early-stop-min-delta', type=float, default=0.0)
    p.add_argument('--cpu', action='store_true')
    p.set_defaults(func=train)

    p = sub.add_parser('infer-video')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    p.add_argument('--out', default='runs/corner_resnet18_aug/head_left_rgb_raw_pred.mp4')
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--max-frames', type=int, default=300)
    p.add_argument('--output-width', type=int, default=960)
    p.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet34', 'resnet18_dilated', 'resnet34_dilated', 'teacher_heatmap', 'betr_heatmap'])
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--heatmap-size', type=int, default=64)
    p.add_argument('--coord-conv', action='store_true')
    p.add_argument('--token-mixer', action='store_true')
    p.add_argument('--input-lowres-size', type=int, default=0, help='Downsample each network input to this max side then upsample back; 0 disables.')
    p.add_argument('--decode-mode', default='topk_mean', choices=['topk_mean', 'argmax', 'softargmax'])
    p.add_argument('--decode-topk', type=int, default=9)
    p.add_argument('--softargmax-temperature', type=float, default=0.05)
    p.add_argument('--cpu', action='store_true')
    p.set_defaults(func=infer_video)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
