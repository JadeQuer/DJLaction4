import argparse
from pathlib import Path

import cv2


def build_frame_ids(total, phases):
    ids = []
    for start_r, end_r, count in phases:
        if count <= 1:
            ids.append(round(start_r * (total - 1)))
            continue
        for i in range(count):
            t = i / (count - 1)
            r = start_r * (1 - t) + end_r * t
            ids.append(round(r * (total - 1)))
    ids = sorted(set(max(0, min(total - 1, x)) for x in ids))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    ap.add_argument('--out-dir', default='datasets/dji_action4_real_det_to_label_90/images')
    ap.add_argument('--width', type=int, default=1280)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # More dense around manipulation-heavy middle and late stages.
    phases = [
        (0.00, 0.12, 15),
        (0.12, 0.38, 25),
        (0.38, 0.72, 30),
        (0.72, 1.00, 20),
    ]
    frame_ids = build_frame_ids(total, phases)

    lines = []
    for idx, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if args.width and w > args.width:
            out_h = round(h * args.width / w)
            frame = cv2.resize(frame, (args.width, out_h), interpolation=cv2.INTER_AREA)
        name = f'frame_{idx:03d}_src_{frame_id:06d}.jpg'
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines.append(f'{name}\tframe={frame_id}\ttime={frame_id / fps:.2f}s')
    cap.release()

    manifest = out.parent / 'frames_to_label.txt'
    manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'extracted {len(lines)} frames to {out}')
    print(f'manifest: {manifest}')


if __name__ == '__main__':
    main()
