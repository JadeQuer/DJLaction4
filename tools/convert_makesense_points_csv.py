import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--labelset', default='datasets/dji_action4_real_corner_yplus_to_label_40')
    ap.add_argument('--out', default='datasets/dji_action4_real_corner_yplus_labeled')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    labelset = Path(args.labelset)
    out = Path(args.out)
    rgb_out = out / 'rgb'
    label_out = out / 'labels'
    rgb_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    by_img = defaultdict(list)
    with csv_path.open(newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            try:
                label = int(row[0])
                x = float(row[1])
                y = float(row[2])
                img = row[3]
                w = int(float(row[4]))
                h = int(float(row[5]))
            except ValueError:
                continue
            by_img[img].append((label, x, y, w, h))

    records = []
    skipped = []
    for img_name, pts in sorted(by_img.items()):
        counts = Counter(p[0] for p in pts)
        missing = [i for i in range(8) if counts[i] == 0]
        dup = [i for i, c in counts.items() if c > 1]
        if missing or dup or len(pts) != 8:
            skipped.append({'image': img_name, 'points': len(pts), 'missing': missing, 'duplicate': dup})
            continue
        stem = Path(img_name).stem
        src_img = labelset / 'roi_images' / img_name
        if not src_img.exists():
            skipped.append({'image': img_name, 'reason': 'missing source image'})
            continue
        src_template = labelset / 'annotations_template' / f'{stem}.json'
        base = json.loads(src_template.read_text(encoding='utf-8')) if src_template.exists() else {}
        ordered = sorted(pts, key=lambda p: p[0])
        corners = [[float(x), float(y), 1.0] for _, x, y, _, _ in ordered]
        dst_name = f'{stem}.png'
        shutil.copy2(src_img, rgb_out / dst_name)
        rec = {
            'image': f'rgb/{dst_name}',
            'obj_id': 1,
            'corners_2d': corners,
            'camera': {'width': 256, 'height': 256},
            'source': {
                'type': 'real_roi_manual_points_makesense_csv',
                'csv': str(csv_path),
                'source_image': str(src_img),
                'template': str(src_template),
                'frame_id': base.get('frame_id'),
                'roi_index': base.get('roi_index'),
                'bbox_xyxy_in_frame': base.get('bbox_xyxy_in_frame'),
            },
            'corner_order': base.get('corner_order', {
                'scheme': 'front_lens_face_0123_back_face_4567_yplus'
            }),
        }
        (label_out / f'{stem}.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
        records.append(rec)

    (out / 'labels.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    report = {'csv': str(csv_path), 'out': str(out), 'valid_images': len(records), 'skipped': skipped}
    (out / 'conversion_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
