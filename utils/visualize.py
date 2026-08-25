from pathlib import Path

import cv2
import numpy as np


def tensor_to_image_array(tensor):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    return arr


def save_binary_mask(mask: np.ndarray, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), (mask.astype(np.uint8) * 255))


def save_multiclass_mask(mask: np.ndarray, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask.astype(np.uint8))


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = image.astype(np.float32)
    mask_rgb = np.zeros_like(image)
    mask_rgb[..., 0] = mask * 255  # red overlay
    overlay = image * (1 - alpha) + mask_rgb * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)
