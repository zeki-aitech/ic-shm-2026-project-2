"""
Exports a trained checkpoint's Gaussians as renderable splat PLYs - both true-color
(`export_splat_ply`) and colored by predicted semantic class (`export_semantic_splat_ply`) - for
viewing in an interactive splat viewer such as SuperSplat (https://superspl.at/editor). Used to
produce Figure 6's source PLYs; kept as a standalone script so re-exporting from a different
checkpoint (e.g. an ablation run) doesn't require an ad-hoc one-off script.
"""
import os
from typing import Tuple


def export_checkpoint_plys(checkpoint_path: str, output_dir: str, prefix: str, device: str = "cuda") -> Tuple[str, str]:
    """Loads `checkpoint_path` and writes `{output_dir}/{prefix}_rgb.ply` (true color) and
    `{output_dir}/{prefix}_semantic.ply` (colored by predicted class). Returns their paths."""
    import torch

    from src.gaussian_splatting.model import SemanticGaussianModel

    os.makedirs(output_dir, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SemanticGaussianModel.from_state_dict(ckpt["params"], device=device)

    rgb_path = model.export_splat_ply(os.path.join(output_dir, f"{prefix}_rgb.ply"))
    sem_path = model.export_semantic_splat_ply(os.path.join(output_dir, f"{prefix}_semantic.ply"))
    return rgb_path, sem_path


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Export a trained checkpoint's Gaussians as renderable splat PLYs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=os.path.join(project_root, "outputs", "renders"))
    parser.add_argument("--prefix", required=True, help="Output filename prefix, e.g. 'bridge_splat_plain_plurality'")
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb_path, sem_path = export_checkpoint_plys(args.checkpoint, args.output_dir, args.prefix, device=device)
    print(f"[export_ply] wrote {rgb_path}")
    print(f"[export_ply] wrote {sem_path}")


if __name__ == "__main__":
    main()
