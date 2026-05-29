import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--left', required=True)
    ap.add_argument('--right', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--left-label', default='left')
    ap.add_argument('--right-label', default='right')
    args = ap.parse_args()

    cap_l = cv2.VideoCapture(args.left)
    cap_r = cv2.VideoCapture(args.right)
    if not cap_l.isOpened():
        raise RuntimeError(f'Cannot open left video: {args.left}')
    if not cap_r.isOpened():
        raise RuntimeError(f'Cannot open right video: {args.right}')

    fps_l = cap_l.get(cv2.CAP_PROP_FPS) or 10.0
    fps_r = cap_r.get(cv2.CAP_PROP_FPS) or 10.0
    fps = min(fps_l, fps_r)

    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_h = min(h_l, h_r)
    scale_l = target_h / float(h_l)
    scale_r = target_h / float(h_r)
    out_w_l = int(round(w_l * scale_l))
    out_w_r = int(round(w_r * scale_r))
    out_size = (out_w_l + out_w_r, target_h)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)

    frame_idx = 0
    while True:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            break
        if h_l != target_h or w_l != out_w_l:
            frame_l = cv2.resize(frame_l, (out_w_l, target_h), interpolation=cv2.INTER_AREA)
        if h_r != target_h or w_r != out_w_r:
            frame_r = cv2.resize(frame_r, (out_w_r, target_h), interpolation=cv2.INTER_AREA)
        cv2.putText(frame_l, args.left_label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_r, args.right_label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        canvas = np.concatenate([frame_l, frame_r], axis=1)
        writer.write(canvas)
        frame_idx += 1

    cap_l.release()
    cap_r.release()
    writer.release()
    print({'out': str(out_path), 'frames': frame_idx, 'fps': fps, 'size': out_size})


if __name__ == '__main__':
    main()
