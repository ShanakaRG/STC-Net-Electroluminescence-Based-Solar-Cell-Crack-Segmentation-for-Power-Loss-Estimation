import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets import list_images
from models.build import build_model
from utils.common import ensure_dir, flexible_load_model, load_checkpoint, load_config
from utils.visualize import overlay_mask_on_image, save_binary_mask, save_multiclass_mask


class InferenceRunner:
    def __init__(self, image_size: int = 512, in_channels: int = 1, mean=None, std=None):
        self.image_size = image_size
        self.in_channels = in_channels
        self.mean = np.array(mean if mean is not None else [0.5] * in_channels, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(std if std is not None else [0.5] * in_channels, dtype=np.float32).reshape(1, 1, -1)

    def read(self, path: Path):
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
        if resized.ndim == 2:
            resized = resized[..., None]
        resized = resized.astype(np.float32) / 255.0
        resized = (resized - self.mean) / self.std
        resized = np.transpose(resized, (2, 0, 1))
        tensor = torch.from_numpy(resized).float().unsqueeze(0)
        return tensor, raw, h, w


def main():
    parser = argparse.ArgumentParser(description="Inference for solar-cell defect segmentation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save_aux", action="store_true", help="Also save edge and topology probability maps")
    parser.add_argument("--tta", action="store_true", help="Use horizontal and vertical flip TTA")
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_classes = int(cfg["model"].get("num_classes", 1))
    in_channels = int(cfg["model"].get("in_channels", 1))
    image_size = int(cfg["data"].get("image_size", 512))
    threshold = float(args.threshold if args.threshold is not None else cfg["eval"].get("threshold", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = InferenceRunner(
        image_size=image_size,
        in_channels=in_channels,
        mean=cfg.get("data", {}).get("mean", [0.5] * in_channels),
        std=cfg.get("data", {}).get("std", [0.5] * in_channels),
    )
    image_paths = list_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError("No images found in {}".format(args.input_dir))

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device.type)
    flexible_load_model(model, ckpt["model"])
    if args.threshold is None and isinstance(ckpt, dict) and "best_threshold" in ckpt:
        threshold = float(ckpt["best_threshold"])
        print("[INFO] Using checkpoint threshold: {:.3f}".format(threshold))
    model.eval()

    out_dir = ensure_dir(args.output_dir)
    mask_dir = ensure_dir(out_dir / "masks")
    overlay_dir = ensure_dir(out_dir / "overlays")
    if args.save_aux:
        edge_dir = ensure_dir(out_dir / "edge_probs")
        topo_dir = ensure_dir(out_dir / "topo_probs")

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Inference"):
            image, raw, h, w = runner.read(path)
            image = image.to(device)
            if args.tta:
                logits = model(image)["logits"]
                logits_h = torch.flip(model(torch.flip(image, dims=[3]))["logits"], dims=[3])
                logits_v = torch.flip(model(torch.flip(image, dims=[2]))["logits"], dims=[2])
                logits = (logits + logits_h + logits_v) / 3.0
                outputs = model(image)
                outputs["logits"] = logits
            else:
                outputs = model(image)
                logits = outputs["logits"]

            if num_classes <= 1:
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                mask = (prob >= threshold).astype(np.uint8)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                save_binary_mask(mask, mask_dir / "{}_mask.png".format(path.stem))
                overlay = overlay_mask_on_image(raw, mask)
            else:
                pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
                save_multiclass_mask(pred, mask_dir / "{}_mask.png".format(path.stem))
                overlay = overlay_mask_on_image(raw, (pred > 0).astype(np.uint8))

            if raw.ndim == 2:
                raw_rgb = np.stack([raw, raw, raw], axis=-1)
            elif raw.shape[-1] == 1:
                raw_rgb = np.repeat(raw, 3, axis=-1)
            else:
                raw_rgb = raw

            if raw_rgb.ndim == 3 and raw_rgb.shape[-1] == 3:
                cv2.imwrite(str(overlay_dir / "{}_overlay.png".format(path.stem)), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(overlay_dir / "{}_overlay.png".format(path.stem)), overlay)

            if args.save_aux:
                edge_prob = torch.sigmoid(outputs["edge_logits"])[0, 0].cpu().numpy()
                edge_prob = cv2.resize(edge_prob, (w, h), interpolation=cv2.INTER_LINEAR)
                topo_prob = torch.sigmoid(outputs["topo_logits"])[0, 0].cpu().numpy()
                topo_prob = cv2.resize(topo_prob, (w, h), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(edge_dir / "{}_edge.png".format(path.stem)), (edge_prob * 255).astype(np.uint8))
                cv2.imwrite(str(topo_dir / "{}_topo.png".format(path.stem)), (topo_prob * 255).astype(np.uint8))

    print("[INFO] Inference outputs saved to {}".format(out_dir))


if __name__ == "__main__":
    main()
