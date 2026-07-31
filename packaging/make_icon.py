from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "src" / "localasr" / "assets"
PACKAGING_ASSET_DIR = ROOT / "packaging" / "assets"


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGING_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    icon = draw_icon(1024)
    icon.save(ASSET_DIR / "localasr-icon.png")
    draw_chevron().save(ASSET_DIR / "chevron-down.png")

    if shutil.which("iconutil"):
        iconset = PACKAGING_ASSET_DIR / "localasr.iconset"
        if iconset.exists():
            shutil.rmtree(iconset)
        iconset.mkdir(parents=True, exist_ok=True)
        for size in (16, 32, 128, 256, 512):
            image = icon.resize((size, size), Image.Resampling.LANCZOS)
            image.save(iconset / f"icon_{size}x{size}.png")
            retina = icon.resize((size * 2, size * 2), Image.Resampling.LANCZOS)
            retina.save(iconset / f"icon_{size}x{size}@2x.png")
        result = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(PACKAGING_ASSET_DIR / "localasr.icns")],
            check=False,
        )
        if result.returncode == 0:
            return 0
    icon.save(
        PACKAGING_ASSET_DIR / "localasr.icns",
        format="ICNS",
        sizes=[(16, 16), (32, 32), (128, 128), (256, 256), (512, 512), (1024, 1024)],
    )
    return 0


def draw_icon(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = int(size * 0.075)
    rect = [margin, margin, size - margin, size - margin]

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        [rect[0], rect[1] + int(size * 0.025), rect[2], rect[3] + int(size * 0.025)],
        radius=int(size * 0.22),
        fill=(26, 51, 55, 95),
    )
    image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(int(size * 0.035))))

    base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base_draw = ImageDraw.Draw(base)
    for y in range(rect[1], rect[3]):
        ratio = (y - rect[1]) / max(1, rect[3] - rect[1])
        color = (int(20 + ratio * 10), int(91 + ratio * 36), int(91 + ratio * 28), 255)
        base_draw.line([(rect[0], y), (rect[2], y)], fill=color)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(rect, radius=int(size * 0.22), fill=255)
    base.putalpha(mask)
    image.alpha_composite(base)

    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(rect, radius=int(size * 0.22), outline=(255, 255, 255, 62), width=int(size * 0.018))
    overlay_draw.arc(
        [int(size * 0.08), int(size * 0.05), int(size * 0.92), int(size * 0.9)],
        start=205,
        end=300,
        fill=(244, 185, 94, 95),
        width=int(size * 0.035),
    )

    bubble = [int(size * 0.22), int(size * 0.21), int(size * 0.78), int(size * 0.69)]
    overlay_draw.rounded_rectangle(bubble, radius=int(size * 0.145), fill=(255, 249, 237, 238))
    tail = [
        (int(size * 0.53), int(size * 0.66)),
        (int(size * 0.66), int(size * 0.79)),
        (int(size * 0.63), int(size * 0.62)),
    ]
    overlay_draw.polygon(tail, fill=(255, 249, 237, 238))

    center_y = int(size * 0.45)
    bar_color = (28, 86, 87, 255)
    accent = (203, 111, 54, 255)
    bars = [0.13, 0.22, 0.32, 0.43, 0.3, 0.21, 0.12]
    start_x = int(size * 0.31)
    gap = int(size * 0.055)
    bar_width = int(size * 0.026)
    for index, height_ratio in enumerate(bars):
        x = start_x + index * gap
        half = int(size * height_ratio / 2)
        color = accent if index == 3 else bar_color
        overlay_draw.rounded_rectangle(
            [x, center_y - half, x + bar_width, center_y + half],
            radius=bar_width // 2,
            fill=color,
        )

    image.alpha_composite(overlay)
    return image


def draw_chevron() -> Image.Image:
    image = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.line([(14, 19), (24, 29), (34, 19)], fill=(33, 95, 91, 255), width=5, joint="curve")
    return image


if __name__ == "__main__":
    raise SystemExit(main())
