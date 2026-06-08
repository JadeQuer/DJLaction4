import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


FACES = {
    "x_min_0-1-3-2": [0, 1, 3, 2],
    "x_max_4-5-7-6": [4, 5, 7, 6],
    "y_min_0-1-5-4": [0, 1, 5, 4],
    "y_max_2-3-7-6": [2, 3, 7, 6],
    "z_min_0-2-6-4": [0, 2, 6, 4],
    "z_max_1-3-7-5": [1, 3, 7, 5],
}

EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def face_summary(points):
    pts = np.asarray(points, dtype=np.float32)
    areas = {name: polygon_area(pts[idxs]) for name, idxs in FACES.items()}
    total = sum(areas.values()) + 1e-6
    weights = {name: val / total for name, val in areas.items()}
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return areas, weights, ranked


def draw_item(img, pts, ranked, title):
    out = img.copy()
    for a, b in EDGES:
        pa = tuple(np.round(pts[a]).astype(int))
        pb = tuple(np.round(pts[b]).astype(int))
        cv2.line(out, pa, pb, (0, 180, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(out, str(i), (p[0] + 5, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, title[:42], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"{ranked[0][0]} {ranked[0][1]:.2f}", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def make_sheet(items, out_path, thumb=(256, 256), cols=5):
    if not items:
        return
    tw, th = thumb
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * th, cols * tw, 3), 235, np.uint8)
    for i, img in enumerate(items):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--labelset", default="datasets/dji_action4_real_corner_yplus_to_label_40")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labelset = Path(args.labelset)

    by_img = defaultdict(list)
    for csv_path in args.csv:
        with Path(csv_path).open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                try:
                    label = int(row[0])
                    x = float(row[1])
                    y = float(row[2])
                except ValueError:
                    continue
                by_img[row[3]].append((label, x, y, csv_path))

    rows = []
    top_counts = Counter()
    second_counts = Counter()
    weighted = Counter()
    tiles = []
    skipped = []
    for img_name, raw_pts in sorted(by_img.items()):
        counts = Counter(p[0] for p in raw_pts)
        if len(raw_pts) != 8 or any(counts[i] != 1 for i in range(8)):
            skipped.append({"image": img_name, "labels": [p[0] for p in raw_pts]})
            continue
        pts = np.asarray([[x, y] for _label, x, y, _csv in sorted(raw_pts, key=lambda p: p[0])], dtype=np.float32)
        areas, weights, ranked = face_summary(pts)
        top_counts[ranked[0][0]] += 1
        second_counts[ranked[1][0]] += 1
        for face, val in weights.items():
            weighted[face] += val
        row = {
            "image": img_name,
            "top_face": ranked[0][0],
            "top_weight": ranked[0][1],
            "second_face": ranked[1][0],
            "second_weight": ranked[1][1],
            "face_weights": weights,
            "face_areas": areas,
        }
        rows.append(row)
        img_path = labelset / "roi_images" / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None and len(tiles) < 50:
            tiles.append(draw_item(img, pts, ranked, img_name))

    n = max(1, len(rows))
    summary = {
        "num_images": len(rows),
        "skipped": skipped,
        "top_face_counts": dict(top_counts),
        "top_face_fraction": {k: v / n for k, v in top_counts.items()},
        "second_face_counts": dict(second_counts),
        "weighted_face_distribution": {k: weighted[k] / n for k in FACES},
    }
    (out_dir / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    make_sheet(tiles, out_dir / "real_csv_face_contact_sheet.png")
    print(json.dumps({"out": str(out_dir), "summary": summary, "contact_sheet": str(out_dir / "real_csv_face_contact_sheet.png")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
