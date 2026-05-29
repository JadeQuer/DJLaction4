import argparse
import json
import shutil
from pathlib import Path

# Old order from current labels:
# 0 (-x,-y,-z)
# 1 (-x,-y,+z)
# 2 (-x,+y,-z)
# 3 (-x,+y,+z)
# 4 (+x,-y,-z)
# 5 (+x,-y,+z)
# 6 (+x,+y,-z)
# 7 (+x,+y,+z)
#
# New human-friendly order requested by user, assuming front = z- face, back = z+ face:
# front TL, TR, BR, BL, back TL, TR, BR, BL
# front z- face ids in old order: [2, 6, 4, 0]
# back  z+ face ids in old order: [3, 7, 5, 1]
PERMUTATIONS = {
    'front_zminus_tl_tr_br_bl': [2, 6, 4, 0, 3, 7, 5, 1],
    'front_zplus_tl_tr_br_bl': [3, 7, 5, 1, 2, 6, 4, 0],
    # For Action cameras, front/back is more likely along thickness (y axis), not z axis.
    # Assuming +z is image-up in canonical object coordinates:
    # y- face TL,TR,BR,BL = old [1,5,4,0]
    # y+ face TL,TR,BR,BL = old [3,7,6,2]
    'front_yminus_tl_tr_br_bl': [1, 5, 4, 0, 3, 7, 6, 2],
    'front_yplus_tl_tr_br_bl': [3, 7, 6, 2, 1, 5, 4, 0],
}


def remap_record(rec, perm, scheme_name):
    out = dict(rec)
    out['corners_3d'] = [rec['corners_3d'][i] for i in perm]
    out['corners_2d'] = [rec['corners_2d'][i] for i in perm]
    meta = dict(rec.get('corner_order', {}))
    meta.update({
        'scheme': scheme_name,
        'permutation_from_original': perm,
        'description': 'front TL, TR, BR, BL; back TL, TR, BR, BL',
    })
    out['corner_order'] = meta
    return out


def process_dataset(src_root, out_root, perm, scheme_name):
    src_root = Path(src_root)
    out_root = Path(out_root)
    (out_root / 'rgb').mkdir(parents=True, exist_ok=True)
    (out_root / 'labels').mkdir(parents=True, exist_ok=True)

    label_paths = sorted((src_root / 'labels').glob('*.json'))
    if not label_paths:
        raise RuntimeError(f'No labels found under {src_root / "labels"}')

    records = []
    for lp in label_paths:
        rec = json.loads(lp.read_text(encoding='utf-8'))
        remapped = remap_record(rec, perm, scheme_name)
        img_rel = Path(rec['image'])
        src_img = src_root / img_rel
        dst_img = out_root / img_rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        (out_root / 'labels' / lp.name).write_text(json.dumps(remapped, indent=2, ensure_ascii=False), encoding='utf-8')
        records.append(remapped)

    if (src_root / 'source_labels.json').exists():
        shutil.copy2(src_root / 'source_labels.json', out_root / 'source_labels.json')
    (out_root / 'labels.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = {
        'src': str(src_root),
        'out': str(out_root),
        'scheme': scheme_name,
        'permutation': perm,
        'items': len(records),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scheme', default='front_zminus_tl_tr_br_bl', choices=sorted(PERMUTATIONS.keys()))
    args = ap.parse_args()
    perm = PERMUTATIONS[args.scheme]
    process_dataset(args.src, args.out, perm, args.scheme)


if __name__ == '__main__':
    main()
