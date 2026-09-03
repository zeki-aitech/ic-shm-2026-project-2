"""
Task A: 2D semantic segmentation dataset.

Loads UAV bridge images + their ground-truth semantic masks (produced by
`src/utils/json_to_mask.py`, stored under `outputs/gt_masks/`, uint8 PNGs with class ids
0-4 matching `src.utils.json_to_mask.CLASS_MAPPING`).
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BridgeSegDataset(Dataset):
    """Pairs `{image_dir}/{id}.png` with `{mask_dir}/{id}.png` for the given `image_ids`."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_ids: List[str],
        image_size: Tuple[int, int] = (512, 384),
        augment: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = list(image_ids)
        self.image_size = image_size  # (width, height)
        self.augment = augment
        self._aug = _build_augmentations() if augment else None

    def __len__(self) -> int:
        return len(self.image_ids)

    def _paths(self, image_id: str) -> Tuple[str, str]:
        return (
            os.path.join(self.image_dir, f"{image_id}.png"),
            os.path.join(self.mask_dir, f"{image_id}.png"),
        )

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        img_path, mask_path = self._paths(image_id)

        image = Image.open(img_path).convert("RGB").resize(self.image_size, Image.BILINEAR)
        mask = Image.open(mask_path).resize(self.image_size, Image.NEAREST)

        image_arr = np.asarray(image, dtype=np.uint8)
        mask_arr = np.asarray(mask, dtype=np.int64)

        if self._aug is not None:
            augmented = self._aug(image=image_arr, mask=mask_arr)
            image_arr, mask_arr = augmented["image"], augmented["mask"]

        image_t = torch.from_numpy(image_arr.copy()).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image_t = (image_t - mean) / std

        mask_t = torch.from_numpy(np.ascontiguousarray(mask_arr)).long()
        return {"pixel_values": image_t, "labels": mask_t, "image_id": image_id}


def _build_augmentations():
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.15, contrast_limit=0.15),
            A.HueSaturationValue(p=0.2, hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8),
        ]
    )


def list_labeled_image_ids(image_dir: str) -> List[str]:
    """Zero-padded numeric ids (e.g. '001') for every `{id}.png` in `image_dir`, sorted."""
    ids = []
    for fname in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() == ".png" and stem.isdigit():
            ids.append(stem)
    return sorted(ids)
