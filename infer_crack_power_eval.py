import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models.build import build_model # noqa: E402


IMAGE_EXTS = set([".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"])


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def normalize_stem(name):
    s = name.lower()
    for token in ["_mask", "-mask", " mask", "_label", "-label", " label", "_gt", "-gt"]:
        s = s.replace(token, "")
    return s


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def build_mask_map(mask_dir):
    masks = list_images(mask_dir)
    if not masks:
        return {}
    mask_map = {}
    for m in masks:
        mask_map[m.stem.lower()] = m
        mask_map[normalize_stem(m.stem)] = m
    return mask_map


class InferenceRunner(object):
    def __init__(self, image_size=512, in_channels=1):
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)

    def read(self, path):
        if self.in_channels == 1:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("Failed to read image: {}".format(path))
            raw = img.copy()
            img = img[..., None]
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to read image: {}".format(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            raw = img.copy()

        h, w = img.shape[:2]
        resized = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # OpenCV drops the singleton channel for grayscale arrays shaped (H, W, 1).
        # Restore it so both grayscale and RGB paths produce CHW tensors.
        if self.in_channels == 1:
            if resized.ndim == 2:
                resized = resized[:, :, None]
            elif resized.ndim == 3 and resized.shape[2] != 1:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[:, :, None]
        else:
            if resized.ndim == 2:
                resized = np.stack([resized, resized, resized], axis=-1)

        resized = resized.astype(np.float32) / 255.0
        resized = (resized - 0.5) / 0.5
        resized = np.transpose(resized, (2, 0, 1))
        tensor = torch.from_numpy(resized).float().unsqueeze(0)
        return tensor, raw, h, w


def save_binary_mask(mask, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), (mask.astype(np.uint8) * 255))


def overlay_mask_on_image(image, mask, alpha=0.35, color=(255, 0, 0)):
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = image.astype(np.float32)
    mask_rgb = np.zeros_like(image)
    mask_rgb[..., 0] = mask.astype(np.uint8) * color[0]
    mask_rgb[..., 1] = mask.astype(np.uint8) * color[1]
    mask_rgb[..., 2] = mask.astype(np.uint8) * color[2]
    overlay = image * (1 - alpha) + mask_rgb * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def read_mask_binary(path, size_hw=None):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("Failed to read mask: {}".format(path))
    if size_hw is not None:
        h, w = size_hw
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def compute_binary_metrics(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())

    iou = float(tp) / float(tp + fp + fn + 1e-6)
    dice = float(2 * tp) / float(2 * tp + fp + fn + 1e-6)
    precision = float(tp) / float(tp + fp + 1e-6)
    recall = float(tp) / float(tp + fn + 1e-6)
    accuracy = float(tp + tn) / float(tp + tn + fp + fn + 1e-6)
    specificity = float(tn) / float(tn + fp + 1e-6)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
    }


def largest_connected_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_idx = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    for idx in range(2, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_idx = idx
    return (labels == largest_idx).astype(np.uint8)


def estimate_cell_mask(raw_image, cell_mode="full_image"):
    if raw_image.ndim == 3:
        gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = raw_image.copy()

    h, w = gray.shape[:2]
    if cell_mode == "full_image":
        return np.ones((h, w), dtype=np.uint8)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_bin = (th > 0).astype(np.uint8)

    if gray[th_bin > 0].mean() < gray[th_bin == 0].mean():
        th_bin = 1 - th_bin

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th_bin = cv2.morphologyEx(th_bin, cv2.MORPH_CLOSE, kernel)
    th_bin = cv2.morphologyEx(th_bin, cv2.MORPH_OPEN, kernel)
    th_bin = largest_connected_component(th_bin)
    return th_bin.astype(np.uint8)


def filter_components_by_min_area(mask, min_area=16):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= int(min_area):
            out[labels == idx] = 1
    return out


def estimate_inactive_mask(raw_image, crack_mask, cell_mask, inactive_percentile=20.0,
                           inactive_dilate_px=11, min_region_px=16):
    """
    Proxy inactive-area estimate for EL imagery.

    The model predicts cracks, not inactive regions directly. This heuristic marks
    dark pixels within the cell that are spatially connected to the predicted crack.
    """
    if raw_image.ndim == 3:
        gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = raw_image.copy()

    crack_mask = crack_mask.astype(np.uint8)
    cell_mask = cell_mask.astype(np.uint8)

    if crack_mask.sum() == 0 or cell_mask.sum() == 0:
        return np.zeros_like(crack_mask, dtype=np.uint8)

    cell_vals = gray[cell_mask > 0]
    if cell_vals.size == 0:
        return np.zeros_like(crack_mask, dtype=np.uint8)

    thr = float(np.percentile(cell_vals, float(inactive_percentile)))
    dark_mask = ((gray <= thr) & (cell_mask > 0)).astype(np.uint8)

    k = max(3, int(inactive_dilate_px))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    crack_zone = cv2.dilate(crack_mask, kernel, iterations=1)

    inactive = ((dark_mask > 0) & (crack_zone > 0)).astype(np.uint8)
    inactive = cv2.morphologyEx(inactive, cv2.MORPH_CLOSE, kernel)
    inactive = filter_components_by_min_area(inactive, min_area=min_region_px)
    inactive = ((inactive > 0) & (cell_mask > 0)).astype(np.uint8)
    inactive = np.maximum(inactive, crack_mask)
    return inactive.astype(np.uint8)


def load_power_map(csv_path):
    if csv_path is None:
        return {}
    power_map = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = ["image", "image_name", "filename", "file", "name"]
            power_keys = ["nominal_power", "nominal_power_w", "p_nominal", "power"]
            image_name = None
            power_val = None
            for k in keys:
                if k in row and row[k]:
                    image_name = row[k]
                    break
            for k in power_keys:
                if k in row and row[k] not in [None, ""]:
                    power_val = float(row[k])
                    break
            if image_name is None or power_val is None:
                continue
            stem = normalize_stem(Path(image_name).stem)
            power_map[stem] = power_val
    return power_map


def write_summary_csv(out_csv, rows, totals, threshold, cell_mode, nominal_power_default):
    with open(out_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["num_images", len(rows)])
        writer.writerow(["threshold", threshold])
        writer.writerow(["cell_mode", cell_mode])
        writer.writerow(["default_nominal_power_w", nominal_power_default])

        if rows:
            def mean_of(key):
                vals = [float(r[key]) for r in rows if r.get(key, "") not in [None, "", "nan"]]
                return float(np.mean(vals)) if vals else float("nan")

            for key in [
                "crack_area_percent",
                "inactive_area_percent",
                "estimated_power_loss_w",
                "estimated_remaining_power_w",
                "iou",
                "dice",
                "precision",
                "recall",
                "accuracy",
                "specificity",
            ]:
                writer.writerow(["mean_" + key, mean_of(key)])

        tp = totals.get("tp", 0)
        tn = totals.get("tn", 0)
        fp = totals.get("fp", 0)
        fn = totals.get("fn", 0)
        writer.writerow(["global_tp", tp])
        writer.writerow(["global_tn", tn])
        writer.writerow(["global_fp", fp])
        writer.writerow(["global_fn", fn])
        writer.writerow(["global_iou", float(tp) / float(tp + fp + fn + 1e-6)])
        writer.writerow(["global_dice", float(2 * tp) / float(2 * tp + fp + fn + 1e-6)])
        writer.writerow(["global_precision", float(tp) / float(tp + fp + 1e-6)])
        writer.writerow(["global_recall", float(tp) / float(tp + fn + 1e-6)])
        writer.writerow(["global_accuracy", float(tp + tn) / float(tp + tn + fp + fn + 1e-6)])
        writer.writerow(["global_specificity", float(tn) / float(tn + fp + 1e-6)])


def main():
    parser = argparse.ArgumentParser(description="Inference + crack area + inactive area proxy + power loss + mask metrics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gt_mask_dir", type=str, default=None,
                        help="Optional directory of ground-truth crack masks for evaluation")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--nominal_power", type=float, default=None,
                        help="Default nominal cell power in watts, used if power CSV is not provided")
    parser.add_argument("--power_csv", type=str, default=None,
                        help="Optional CSV with columns like image, nominal_power")
    parser.add_argument("--cell_mode", type=str, default="full_image", choices=["full_image", "otsu"],
                        help="How to estimate the cell area used in area percentages")
    parser.add_argument("--inactive_percentile", type=float, default=20.0,
                        help="Percentile of EL intensity inside the cell used to define dark inactive candidates")
    parser.add_argument("--inactive_dilate_px", type=int, default=11,
                        help="Dilation kernel size around the predicted crack when building inactive-area proxy")
    parser.add_argument("--min_region_px", type=int, default=16,
                        help="Minimum connected-component area retained in inactive-area proxy")
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_classes = int(cfg["model"].get("num_classes", 1))
    in_channels = int(cfg["model"].get("in_channels", 1))
    image_size = int(cfg["data"].get("image_size", 512))
    threshold = float(args.threshold if args.threshold is not None else cfg["eval"].get("threshold", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print("[INFO] Threshold:", threshold)

    runner = InferenceRunner(image_size=image_size, in_channels=in_channels)
    image_paths = list_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError("No images found in {}".format(args.input_dir))

    gt_mask_map = build_mask_map(args.gt_mask_dir) if args.gt_mask_dir else {}
    power_map = load_power_map(args.power_csv)

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device.type)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = ensure_dir(args.output_dir)
    crack_dir = ensure_dir(out_dir / "masks_crack")
    inactive_dir = ensure_dir(out_dir / "masks_inactive")
    cell_dir = ensure_dir(out_dir / "masks_cell")
    crack_overlay_dir = ensure_dir(out_dir / "overlays_crack")
    inactive_overlay_dir = ensure_dir(out_dir / "overlays_inactive")

    per_image_csv = out_dir / "per_image_results.csv"
    summary_csv = out_dir / "summary_results.csv"

    fieldnames = [
        "image_name",
        "gt_mask_name",
        "cell_pixels",
        "pred_crack_pixels",
        "pred_inactive_pixels",
        "crack_area_percent",
        "inactive_area_percent",
        "nominal_power_w",
        "estimated_power_loss_w",
        "estimated_remaining_power_w",
        "iou",
        "dice",
        "precision",
        "recall",
        "accuracy",
        "specificity",
    ]

    rows = []
    totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    with open(per_image_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for path in tqdm(image_paths, desc="Inference"):
                image, raw, h, w = runner.read(path)
                image = image.to(device)
                outputs = model(image)
                logits = outputs["logits"]

                if num_classes <= 1:
                    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                    crack_mask = (prob >= threshold).astype(np.uint8)
                    crack_mask = cv2.resize(crack_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
                    crack_mask = (pred > 0).astype(np.uint8)
                    crack_mask = cv2.resize(crack_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                cell_mask = estimate_cell_mask(raw, cell_mode=args.cell_mode)
                inactive_mask = estimate_inactive_mask(
                    raw,
                    crack_mask,
                    cell_mask,
                    inactive_percentile=args.inactive_percentile,
                    inactive_dilate_px=args.inactive_dilate_px,
                    min_region_px=args.min_region_px,
                )

                save_binary_mask(crack_mask, crack_dir / (path.stem + "_crack.png"))
                save_binary_mask(inactive_mask, inactive_dir / (path.stem + "_inactive.png"))
                save_binary_mask(cell_mask, cell_dir / (path.stem + "_cell.png"))

                crack_overlay = overlay_mask_on_image(raw, crack_mask, alpha=0.35, color=(255, 0, 0))
                inactive_overlay = overlay_mask_on_image(raw, inactive_mask, alpha=0.35, color=(255, 255, 0))
                cv2.imwrite(str(crack_overlay_dir / (path.stem + "_crack_overlay.png")), cv2.cvtColor(crack_overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(inactive_overlay_dir / (path.stem + "_inactive_overlay.png")), cv2.cvtColor(inactive_overlay, cv2.COLOR_RGB2BGR))

                cell_pixels = int(cell_mask.sum())
                crack_pixels = int(np.logical_and(crack_mask > 0, cell_mask > 0).sum())
                inactive_pixels = int(np.logical_and(inactive_mask > 0, cell_mask > 0).sum())
                crack_area_percent = 100.0 * crack_pixels / float(max(cell_pixels, 1))
                inactive_area_percent = 100.0 * inactive_pixels / float(max(cell_pixels, 1))

                stem = normalize_stem(path.stem)
                nominal_power = power_map.get(stem, args.nominal_power)
                if nominal_power is None:
                    est_power_loss = float("nan")
                    est_remaining_power = float("nan")
                else:
                    nominal_power = float(nominal_power)
                    est_power_loss = nominal_power * (inactive_area_percent / 100.0)
                    est_remaining_power = nominal_power - est_power_loss

                gt_mask_name = ""
                metrics = {
                    "iou": "",
                    "dice": "",
                    "precision": "",
                    "recall": "",
                    "accuracy": "",
                    "specificity": "",
                }
                gt_path = gt_mask_map.get(stem)
                if gt_path is not None:
                    gt_mask = read_mask_binary(gt_path, size_hw=(h, w))
                    gt_mask_name = gt_path.name
                    m = compute_binary_metrics(crack_mask, gt_mask)
                    totals["tp"] += m["tp"]
                    totals["tn"] += m["tn"]
                    totals["fp"] += m["fp"]
                    totals["fn"] += m["fn"]
                    metrics = {
                        "iou": m["iou"],
                        "dice": m["dice"],
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "accuracy": m["accuracy"],
                        "specificity": m["specificity"],
                    }

                row = {
                    "image_name": path.name,
                    "gt_mask_name": gt_mask_name,
                    "cell_pixels": cell_pixels,
                    "pred_crack_pixels": crack_pixels,
                    "pred_inactive_pixels": inactive_pixels,
                    "crack_area_percent": crack_area_percent,
                    "inactive_area_percent": inactive_area_percent,
                    "nominal_power_w": nominal_power if nominal_power is not None else "",
                    "estimated_power_loss_w": est_power_loss,
                    "estimated_remaining_power_w": est_remaining_power,
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "accuracy": metrics["accuracy"],
                    "specificity": metrics["specificity"],
                }
                writer.writerow(row)
                rows.append(row)

    write_summary_csv(summary_csv, rows, totals, threshold, args.cell_mode, args.nominal_power)

    print("[INFO] Per-image CSV saved to {}".format(per_image_csv))
    print("[INFO] Summary CSV saved to {}".format(summary_csv))
    print("[INFO] Crack masks saved to {}".format(crack_dir))
    print("[INFO] Inactive masks saved to {}".format(inactive_dir))


if __name__ == "__main__":
    main()
