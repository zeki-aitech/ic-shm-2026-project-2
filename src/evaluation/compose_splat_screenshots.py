"""
Composes two real interactive-splat-viewer screenshots (SuperSplat, https://superspl.at/editor)
of the trained Gaussians - one true-RGB, one colored by predicted semantic class - into a single
side-by-side Figure 6 panel (Section 5.3, optional).

Unlike `plot_splat_pointcloud.py` (a static point-cloud scatter, used as a fallback when no
interactive viewer is available), these are genuine alpha-blended splat renders from the same
viewpoint, exported via `SemanticGaussianModel.export_splat_ply` /
`export_semantic_splat_ply` and manually captured in SuperSplat - this module only lays out and
labels the two screenshots, it does not render anything itself.
"""
import os

from PIL import Image, ImageDraw, ImageFont


def _labeled_panel(image_path: str, label: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    font_size = max(14, img.width // 28)
    label_h = font_size + 24

    panel = Image.new("RGB", (img.width, img.height + label_h), "white")
    panel.paste(img, (0, label_h))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((panel.width - text_w) / 2, (label_h - font_size) / 2 - bbox[1]),
               label, fill="black", font=font)
    return panel


def compose_splat_screenshots(
    rgb_path: str,
    semantic_path: str,
    output_path: str,
    rgb_label: str = "(a) True-color splat render",
    semantic_label: str = "(b) Colored by predicted semantic class",
    gap: int = 12,
) -> str:
    left = _labeled_panel(rgb_path, rgb_label)
    right = _labeled_panel(semantic_path, semantic_label)

    h = max(left.height, right.height)
    combined = Image.new("RGB", (left.width + gap + right.width, h), "white")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + gap, 0))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined.save(output_path)
    return output_path


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(
        description="Compose RGB + semantic SuperSplat screenshots into Figure 6"
    )
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--semantic", required=True)
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "paper", "figures", "fig6_splat_render.png"),
    )
    args = parser.parse_args()

    out = compose_splat_screenshots(args.rgb, args.semantic, args.output)
    print(f"[compose_splat_screenshots] wrote {out}")


if __name__ == "__main__":
    main()
