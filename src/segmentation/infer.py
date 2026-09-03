"""
Task A: predict pseudo-masks for the 100 unlabeled UAV images using a trained SegFormer
checkpoint (see `src/segmentation/train.py`). Output masks use the same uint8 class-id
convention (0-4) as `outputs/gt_masks/`, so they can be consumed identically by Task B.

Never run this on the 60 held-out labeled images - they already have real GT masks and must
stay untouched for the final render-based evaluation.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.json_to_mask import CLASS_MAPPING

NUM_CLASSES = len(CLASS_MAPPING)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint_path: str, device: str, pretrained: str = "nvidia/mit-b0"):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    config = SegformerConfig.from_pretrained(pretrained, num_labels=NUM_CLASSES)
    model = SegformerForSemanticSegmentation(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_mask(model, image_path: str, device: str, infer_size=(512, 384)) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized = image.resize(infer_size, Image.BILINEAR)
    arr = (np.asarray(resized, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    logits = model(pixel_values=tensor).logits
    logits = torch.nn.functional.interpolate(
        logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def predict_masks(checkpoint_path: str, image_paths, output_dir: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(checkpoint_path, device)

    written = []
    for image_path in image_paths:
        pred = predict_mask(model, image_path, device)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{stem}.png")
        Image.fromarray(pred, mode="L").save(out_path)
        written.append(out_path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Task A: predict pseudo-masks for unlabeled bridge images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", default=None, help="Directory of images to predict on")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(PROJECT_ROOT, "data", "Contest Dataset"))
    images_dir = args.images or os.path.join(dataset_dir, "unlabeled_Images")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs", "pseudo_masks")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    written = predict_masks(args.checkpoint, image_paths, output_dir)
    print(f"[segmentation] wrote {len(written)} pseudo-masks to {output_dir}")


if __name__ == "__main__":
    main()
