import argparse
import shutil
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='datasets/dji_action4_real_det_to_label_90/images')
    ap.add_argument('--labels-zip', default='datasets/yolo_full.zip')
    ap.add_argument('--out', default='datasets/dji_action4_real_det_full')
    ap.add_argument('--val-every', type=int, default=5)
    args = ap.parse_args()

    images_dir = Path(args.images)
    out = Path(args.out)
    tmp_labels = out / '_uploaded_labels'
    tmp_labels.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.labels_zip) as zf:
        zf.extractall(tmp_labels)

    for split in ['train', 'val']:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    images = sorted(images_dir.glob('*.jpg'))
    train = val = 0
    missing = []
    for idx, img in enumerate(images):
        label = tmp_labels / f'{img.stem}.txt'
        if not label.exists():
            missing.append(label.name)
            continue
        split = 'val' if idx % args.val_every == 0 else 'train'
        shutil.copy2(img, out / 'images' / split / img.name)
        shutil.copy2(label, out / 'labels' / split / label.name)
        if split == 'val':
            val += 1
        else:
            train += 1

    yaml = out / 'dji_action4_real_full.yaml'
    yaml.write_text(
        f'path: {out.resolve()}\n'
        'train: images/train\n'
        'val: images/val\n'
        'names:\n'
        '  0: dji_action4\n',
        encoding='utf-8',
    )
    print({'train': train, 'val': val, 'missing': missing, 'yaml': str(yaml)})


if __name__ == '__main__':
    main()
