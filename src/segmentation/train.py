"""
Task A: fine-tune SegFormer (mit-b0 backbone) for 2D bridge semantic segmentation.

Trains on the trajectory-interleaved 240-image train split (see
`src.evaluation.metrics.trajectory_interleaved_split`), validates 2D mIoU each epoch on the
60-image holdout split (reusing `src.evaluation.metrics`'s confusion-matrix / mIoU functions),
and saves the best checkpoint by validation mIoU. The 60 holdout images are never trained on -
they are reserved for the final render-based evaluation of Task B (see
`src/evaluation/render_metrics.py`).
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation.dataset import BridgeSegDataset, list_labeled_image_ids
from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_iou_per_class,
    compute_miou,
    trajectory_interleaved_split,
)

NUM_CLASSES = 5


def get_split(labeled_ids: List[str], holdout_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
    """Trajectory-interleaved 240/60 split over sorted labeled image ids."""
    return trajectory_interleaved_split(sorted(labeled_ids), holdout_ratio)


@torch.no_grad()
def _validate(model, loader, device) -> Tuple[float, Dict[int, float]]:
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values=pixel_values).logits
        logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        preds = logits.argmax(dim=1)
        conf += compute_confusion_matrix(
            labels.cpu().numpy().ravel(), preds.cpu().numpy().ravel(), NUM_CLASSES
        )
    ious = compute_iou_per_class(conf)
    miou = compute_miou(ious, include_background=False)
    model.train()
    return miou, ious


def build_model(device, pretrained: str = "nvidia/mit-b0"):
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True, use_safetensors=True
    )
    return model.to(device)


def train(
    images_dir: str,
    gt_masks_dir: str,
    output_dir: str,
    holdout_ratio: float = 0.2,
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 6e-5,
    image_size: Tuple[int, int] = (512, 384),
    num_workers: int = 4,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    labeled_ids = list_labeled_image_ids(images_dir)
    train_ids, val_ids = get_split(labeled_ids, holdout_ratio)
    print(f"[segmentation] {len(labeled_ids)} labeled images -> train={len(train_ids)} val={len(val_ids)}")

    train_ds = BridgeSegDataset(images_dir, gt_masks_dir, train_ids, image_size=image_size, augment=True)
    val_ds = BridgeSegDataset(images_dir, gt_masks_dir, val_ids, image_size=image_size, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2))

    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_miou = -1.0
    best_path = os.path.join(output_dir, "best.pt")
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg_loss = epoch_loss / max(1, len(train_loader))
        val_miou, val_ious = _validate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": avg_loss, "val_miou": val_miou})
        print(f"[segmentation] epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} val_mIoU={val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_miou": val_miou,
                    "val_ious": val_ious,
                    "epoch": epoch,
                    "image_size": image_size,
                    "num_classes": NUM_CLASSES,
                },
                best_path,
            )

    print(f"[segmentation] best val mIoU = {best_miou:.4f} -> {best_path}")
    return best_path, best_miou, history


def main():
    parser = argparse.ArgumentParser(description="Task A: fine-tune SegFormer mit-b0 for bridge 2D segmentation")
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--gt-masks-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    args = parser.parse_args()

    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(PROJECT_ROOT, "data", "Contest Dataset"))
    images_dir = args.images_dir or os.path.join(dataset_dir, "images")
    gt_masks_dir = args.gt_masks_dir or os.path.join(PROJECT_ROOT, "outputs", "gt_masks")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "segformer_mitb0")

    train(
        images_dir,
        gt_masks_dir,
        output_dir,
        holdout_ratio=args.holdout_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
