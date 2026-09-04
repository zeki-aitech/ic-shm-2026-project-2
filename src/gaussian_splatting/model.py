"""
Semantic 3D Gaussian Splatting model.

Each Gaussian carries the standard 3DGS parameters (mean xyz, scale, rotation quaternion,
opacity, RGB color) plus a per-Gaussian semantic logit vector (one per class). Both RGB and
semantic channels are rendered in a *single* fused `gsplat.rasterization()` call by
concatenating them into one `colors` tensor - gsplat's functional API accepts an arbitrary
channel count, so this reuses the same depth-sorted alpha composite for both outputs instead of
running two separate rasterization passes.

Gaussian parameters are held as a plain `Dict[str, nn.Parameter]` (`self.params`), using the
exact key names gsplat's `strategy.DefaultStrategy` expects ("means", "scales", "quats",
"opacities", plus the extra "colors"/"sem_logits" keys it handles generically). This is
deliberate, not incidental: `DefaultStrategy`'s duplicate/split/remove operations resize the
Gaussian count by reassigning `params[key] = new_param`, which only stays visible to the caller
if `params` is a plain mutable dict shared by reference - `nn.Module` attributes would go stale
across a densification step.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gsplat import rasterization

from src.colmap_io.models import Point3D
from src.gaussian_splatting.dataset import GSCamera

NUM_CLASSES = 5
# Uncovered/background pixels default toward RGB mid-gray and a strong "background" (class 0)
# semantic vote, so empty regions of the scene don't render as arbitrary noise.
DEFAULT_BACKGROUND = torch.tensor(
    [0.5, 0.5, 0.5] + [4.0] + [-4.0] * (NUM_CLASSES - 1), dtype=torch.float32
)


def _inv_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x)).astype(np.float32)


class SemanticGaussianModel:
    def __init__(self, params: Dict[str, nn.Parameter]):
        self.params = params

    # -- convenience accessors (read the live dict entry, so they stay valid across
    #    gsplat-strategy densification steps that reassign `self.params[key]`) --
    @property
    def means(self) -> nn.Parameter:
        return self.params["means"]

    @property
    def log_scales(self) -> nn.Parameter:
        return self.params["scales"]

    @property
    def quats(self) -> nn.Parameter:
        return self.params["quats"]

    @property
    def opacity_logits(self) -> nn.Parameter:
        return self.params["opacities"]

    @property
    def color_logits(self) -> nn.Parameter:
        return self.params["colors"]

    @property
    def sem_logits(self) -> nn.Parameter:
        return self.params["sem_logits"]

    @property
    def num_points(self) -> int:
        return self.params["means"].shape[0]

    def parameters(self):
        return self.params.values()

    @classmethod
    def init_from_sparse(
        cls,
        pts3d: Dict[int, Point3D],
        point_classes: Dict[int, int],
        point_colors: Dict[int, np.ndarray],
        num_classes: int = NUM_CLASSES,
        device: str = "cuda",
    ) -> "SemanticGaussianModel":
        """
        Warm-starts Gaussian means/colors from `PycolmapReconstructor`'s triangulated sparse
        cloud and semantic logits from `SemanticProjector`'s per-point voted class, instead of
        random initialization.
        """
        from scipy.spatial import cKDTree

        ids = sorted(pts3d.keys())
        n = len(ids)
        xyz = np.stack([np.asarray(pts3d[i].xyz, dtype=np.float64) for i in ids]).astype(np.float32)
        colors_arr = np.stack(
            [np.asarray(point_colors.get(i, np.array([128, 128, 128])), dtype=np.float32) for i in ids]
        ) / 255.0

        k = min(4, n)
        tree = cKDTree(xyz)
        dists, _ = tree.query(xyz, k=k)
        mean_nn_dist = dists[:, 1:].mean(axis=1) if k > 1 else np.full(n, 0.05, dtype=np.float64)
        mean_nn_dist = np.clip(mean_nn_dist, 1e-4, None)
        log_scales_init = np.log(np.repeat(mean_nn_dist[:, None], 3, axis=1)).astype(np.float32)

        quats_init = np.zeros((n, 4), dtype=np.float32)
        quats_init[:, 0] = 1.0  # identity rotation (w, x, y, z)

        opacity_init = np.full((n,), _inv_sigmoid(np.array(0.3)), dtype=np.float32)

        sem_init = np.full((n, num_classes), -2.0, dtype=np.float32)
        for row, pid in enumerate(ids):
            sem_init[row, point_classes.get(pid, 0)] = 2.0

        color_logits_init = _inv_sigmoid(colors_arr)

        def _param(arr):
            return nn.Parameter(torch.tensor(arr, device=device, dtype=torch.float32))

        params = {
            "means": _param(xyz),
            "scales": _param(log_scales_init),
            "quats": _param(quats_init),
            "opacities": _param(opacity_init),
            "colors": _param(color_logits_init),
            "sem_logits": _param(sem_init),
        }
        return cls(params)

    def get_render_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scales = torch.exp(self.params["scales"])
        quats = torch.nn.functional.normalize(self.params["quats"], dim=-1)
        opacities = torch.sigmoid(self.params["opacities"])
        colors = torch.sigmoid(self.params["colors"])
        fused = torch.cat([colors, self.params["sem_logits"]], dim=-1)  # (N, 3 + num_classes)
        return self.params["means"], quats, scales, opacities, fused

    def render_full(self, camera: GSCamera, packed: bool = True):
        """Full `gsplat.rasterization()` output `(colors, alphas, meta)` - `meta` carries the
        `means2d`/`radii`/`gaussian_ids` info that `gsplat.strategy.DefaultStrategy` needs
        during training; `render()` below is the simple (rgb, semantic_logits) view for
        inference/eval."""
        means, quats, scales, opacities, fused_colors = self.get_render_params()
        device = means.device

        viewmat = torch.eye(4, dtype=torch.float32, device=device)
        viewmat[:3, :3] = torch.tensor(camera.R, dtype=torch.float32, device=device)
        viewmat[:3, 3] = torch.tensor(camera.T, dtype=torch.float32, device=device)
        viewmat = viewmat.unsqueeze(0)
        K = torch.tensor(camera.K, dtype=torch.float32, device=device).unsqueeze(0)
        # gsplat's packed rasterization path collapses the leading camera-batch dim internally,
        # so `backgrounds` must be (channels,) here, not (C, channels).
        background = DEFAULT_BACKGROUND.to(device)

        return rasterization(
            means, quats, scales, opacities, fused_colors, viewmat, K,
            camera.width, camera.height, backgrounds=background, packed=packed,
        )

    def render(self, camera: GSCamera) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (rgb [H,W,3] in [0,1], semantic_logits [H,W,num_classes])."""
        out, _alpha, _meta = self.render_full(camera)
        out = out[0]  # (H, W, 3 + num_classes)
        rgb = out[..., :3].clamp(0.0, 1.0)
        sem_logits = out[..., 3:]
        return rgb, sem_logits

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.params.items()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor], device: str = "cuda") -> "SemanticGaussianModel":
        params = {k: nn.Parameter(v.to(device)) for k, v in state.items()}
        return cls(params)

    def export_ply(self, path: str) -> str:
        """ASCII point-cloud PLY export (x,y,z,rgb,class_id per Gaussian center) - a lightweight
        summary for tabular/analysis inspection, not a renderable splat file."""
        with torch.no_grad():
            xyz = self.params["means"].detach().cpu().numpy()
            colors = torch.sigmoid(self.params["colors"]).detach().cpu().numpy()
            classes = self.params["sem_logits"].detach().cpu().numpy().argmax(axis=1)
        rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(xyz)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("property int class_id\nend_header\n")
            for (x, y, z), (r, g, b), c in zip(xyz, rgb_u8, classes):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {int(c)}\n")
        return path

    def _export_splat_ply_with_colors(self, path: str, colors: torch.Tensor) -> str:
        """Shared writer for `export_splat_ply`/`export_semantic_splat_ply`: standard 3D
        Gaussian Splatting PLY format via `gsplat.export_splats`, viewable in any standard splat
        viewer (e.g. SuperSplat, antimatter15/splat) with proper alpha-blended rendering and
        free camera orbit. `colors`: (N, 3) in [0, 1]."""
        from gsplat import export_splats

        with torch.no_grad():
            means = self.params["means"].detach()
            scales = self.params["scales"].detach()
            quats = torch.nn.functional.normalize(self.params["quats"].detach(), dim=-1)
            opacities = self.params["opacities"].detach()
            sh_c0 = 0.28209479177387814  # standard 3DGS degree-0 SH normalization constant
            sh0 = ((colors - 0.5) / sh_c0).unsqueeze(1)  # (N, 1, 3)
            shN = torch.zeros((means.shape[0], 0, 3), device=means.device, dtype=means.dtype)

            export_splats(means, scales, quats, opacities, sh0, shN, format="ply", save_to=path)
        return path

    def export_splat_ply(self, path: str) -> str:
        """Exports the full splat (position, scale, rotation, opacity, true RGB color) in the
        standard 3D Gaussian Splatting PLY format - unlike `export_ply`, which only summarizes
        Gaussian centers as a plain point cloud, this preserves splat shape/transparency."""
        colors = torch.sigmoid(self.params["colors"].detach())
        return self._export_splat_ply_with_colors(path, colors)

    def export_semantic_splat_ply(self, path: str) -> str:
        """Same as `export_splat_ply`, but colors each Gaussian by its predicted semantic class
        (argmax of `sem_logits`, using the official per-class colors) instead of true RGB - lets
        you visually inspect the 3D semantic segmentation in a standard splat viewer."""
        from src.colmap_io.semantic_voting import CLASS_COLORS

        with torch.no_grad():
            classes = self.params["sem_logits"].detach().argmax(dim=-1).cpu().numpy()
        lut = np.stack([CLASS_COLORS[c] for c in range(NUM_CLASSES)]).astype(np.float32) / 255.0
        colors = torch.from_numpy(lut[classes]).to(
            device=self.params["means"].device, dtype=self.params["means"].dtype
        )
        return self._export_splat_ply_with_colors(path, colors)
