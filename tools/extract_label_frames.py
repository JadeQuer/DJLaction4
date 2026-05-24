import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='/root/autodl-fs/head_left_rgb_raw.mp4')
    ap.add_argument('--out-dir', default='datasets/dji_action4_real_det_to_label/images')
    ap.add_argument('--num', type=int, default=30)
    ap.add_argument('--width', type=int, default=1280)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Bias samples toward the manipulation period while still covering the opening display.
    ratios = [
        0.00, 0.02, 0.04, 0.06, 0.08,
        0.11, 0.14, 0.17, 0.20, 0.23,
        0.26, 0.29, 0.32, 0.35, 0.38,
        0.41, 0.44, 0.47, 0.50, 0.54,
        0.58, 0.62, 0.66, 0.70, 0.74,
        0.78, 0.82, 0.86, 0.91, 0.96,
    ]
    if args.num != len(ratios):
        ratios = [i / max(1, args.num - 1) for i in range(args.num)]
    frame_ids = sorted(set(min(total - 1, max(0, round(r * (total - 1)))) for r in ratios))

    lines = []
    for idx, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            print(f'skip frame {frame_id}')
            continue
        h, w = frame.shape[:2]
        if args.width and w > args.width:
            out_h = round(h * args.width / w)
            frame = cv2.resize(frame, (args.width, out_h), interpolation=cv2.INTER_AREA)
        name = f'frame_{idx:03d}_src_{frame_id:06d}.jpg'
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        seconds = frame_id / fps
        lines.append(f'{name}\tframe={frame_id}\ttime={seconds:.2f}s')
    cap.release()

    manifest = out.parent / 'frames_to_label.txt'
    manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'extracted {len(lines)} frames to {out}')
    print(f'manifest: {manifest}')


if __name__ == '__main__':
    main()
