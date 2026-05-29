import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ORDER = [
    "x_pos",
    "x_neg",
    "y_pos",
    "y_neg",
    "z_pos",
    "z_neg",
    "diag_pos",
    "diag_neg",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--thumb-width", default=560, type=int)
    parser.add_argument("--thumb-height", default=400, type=int)
    args = parser.parse_args()

    label_h = 34
    pad = 12
    cols = 2
    rows = (len(DEFAULT_ORDER) + cols - 1) // cols
    sheet_w = cols * args.thumb_width + (cols + 1) * pad
    sheet_h = rows * (args.thumb_height + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for i, name in enumerate(DEFAULT_ORDER):
        image_path = args.dir / f"{name}.png"
        if not image_path.exists():
            continue
        im = Image.open(image_path).convert("RGB")
        im.thumbnail((args.thumb_width, args.thumb_height), Image.Resampling.LANCZOS)
        row, col = divmod(i, cols)
        x = pad + col * (args.thumb_width + pad)
        y = pad + row * (args.thumb_height + label_h + pad)
        bg = Image.new("RGB", (args.thumb_width, args.thumb_height), (230, 230, 230))
        bg.paste(im, ((args.thumb_width - im.width) // 2, (args.thumb_height - im.height) // 2))
        sheet.paste(bg, (x, y + label_h))
        draw.rectangle((x, y, x + args.thumb_width, y + label_h), fill=(32, 32, 32))
        draw.text((x + 12, y + 5), f"{name}.png", fill=(255, 255, 255), font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
