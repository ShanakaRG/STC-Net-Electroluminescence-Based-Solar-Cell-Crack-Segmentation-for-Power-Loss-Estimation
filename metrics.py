from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def binary_segmentation_stats(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)

    tp = intersection
    fp = preds.sum(dim=(1, 2, 3)) - tp
    fn = targets.sum(dim=(1, 2, 3)) - tp
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


@torch.no_grad()
def multiclass_segmentation_stats(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    ious = []
    dices = []
    precisions = []
    recalls = []
    for c in range(num_classes):
        pred_c = (preds == c)
        tgt_c = (targets == c)
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        pred_sum = pred_c.sum().float()
        tgt_sum = tgt_c.sum().float()
        iou = (inter + 1e-6) / (union + 1e-6)
        dice = (2 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)
        precision = (inter + 1e-6) / (pred_sum + 1e-6)
        recall = (inter + 1e-6) / (tgt_sum + 1e-6)
        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
    return {
        "dice": torch.stack(dices).mean().item(),
        "iou": torch.stack(ious).mean().item(),
        "precision": torch.stack(precisions).mean().item(),
        "recall": torch.stack(recalls).mean().item(),
    }


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, threshold: float = 0.5) -> Dict[str, float]:
    if num_classes <= 1:
        return binary_segmentation_stats(logits, targets, threshold=threshold)
    return multiclass_segmentation_stats(logits, targets, num_classes=num_classes)
