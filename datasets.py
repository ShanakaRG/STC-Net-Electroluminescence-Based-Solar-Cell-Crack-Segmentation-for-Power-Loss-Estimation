import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PairPaths:
    image: Path
    mask: Path


def normalize_stem(name: str) -> str:
    s = name.lower()
    for token in ["_mask", "-mask", " mask", "_label", "-label", " label", "_gt", "-gt"]:
        s = s.replace(token, "")
    return s


def list_images(folder: Union[str, Path]) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def pair_images_and_masks(image_dir: Union[str, Path], mask_dir: Union[str, Path]) -> List[PairPaths]:
    images = list_images(image_dir)
    masks = list_images(mask_dir)
    if not images:
        raise FileNotFoundError("No images found in {}".format(image_dir))
    if not masks:
        raise FileNotFoundError("No masks found in {}".format(mask_dir))

    mask_map = {}
    for m in masks:
        mask_map[m.stem.lower()] = m
        mask_map[normalize_stem(m.stem)] = m

    pairs = []
    missing = []
    for img in images:
        key_exact = img.stem.lower()
        key_norm = normalize_stem(img.stem)
        mask = mask_map.get(key_exact) or mask_map.get(key_norm)
        if mask is None:
            missing.append(img.name)
            continue
        pairs.append(PairPaths(img, mask))

    if not pairs:
        sample_imgs = [p.name for p in images[:5]]
        sample_masks = [p.name for p in masks[:5]]
        raise RuntimeError(
            "Could not pair any image-mask files. Sample images: {}; sample masks: {}".format(
                sample_imgs, sample_masks
            )
        )
    if missing:
        print("[WARN] {} images had no matching mask and were skipped.".format(len(missing)))
    return pairs


class SimpleSegTransform:
    def __init__(self, image_size: int, mean, std, train: bool = True) -> None:
        self.image_size = int(image_size)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        self.train = train

    def _resize(self, image, mask):
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def _random_flip(self, image, mask):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask

    def _random_rotate90(self, image, mask):
        if random.random() < 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        return image, mask

    def _random_affine(self, image, mask):
        if random.random() >= 0.5:
            return image, mask
        h, w = image.shape[:2]
        angle = random.uniform(-12.0, 12.0)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-0.03 * w, 0.03 * w)
        ty = random.uniform(-0.03 * h, 0.03 * h)
        mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty
        image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if image.ndim == 2:
            image = image[..., None]
        return image, mask

    def _random_photo(self, image):
        if random.random() < 0.25:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-12.0, 12.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.15:
            sigma = random.uniform(2.0, 8.0)
            noise = np.random.normal(0, sigma, size=image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
            if image.ndim == 2:
                image = image[..., None]
        return image

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image, mask = self._resize(image, mask)
        if self.train:
            image, mask = self._random_flip(image, mask)
            image, mask = self._random_rotate90(image, mask)
            image, mask = self._random_affine(image, mask)
            image = self._random_photo(image)

        if image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        return image, mask


class SolarDefectSegDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PairPaths],
        transform: Optional[SimpleSegTransform] = None,
        in_channels: int = 1,
        num_classes: int = 1,
        mask_threshold: int = 127,
    ) -> None:
        self.pairs = list(pairs)
        self.transform = transform
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_image(self, path: Path) -> np.ndarray:
        if self.in_channels == 1:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("Failed to read image: {}".format(path))
            img = img[..., None]
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to read image: {}".format(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError("Failed to read mask: {}".format(path))
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if self.num_classes <= 1:
            mask = (mask > self.mask_threshold).astype(np.float32)
        else:
            mask = mask.astype(np.int64)
        return mask

    @staticmethod
    def mask_to_edges(mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(mask_u8, kernel, iterations=1)
        ero = cv2.erode(mask_u8, kernel, iterations=1)
        edge = (dil - ero).astype(np.float32)
        return edge

    @staticmethod
    def mask_to_skeleton(mask: np.ndarray) -> np.ndarray:
        img = ((mask > 0).astype(np.uint8) * 255)
        skel = np.zeros_like(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        working = img.copy()
        while True:
            eroded = cv2.erode(working, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(working, opened)
            skel = cv2.bitwise_or(skel, temp)
            working = eroded.copy()
            if cv2.countNonZero(working) == 0:
                break
        skel = (skel > 0).astype(np.float32)
        if skel.sum() == 0 and mask.sum() > 0:
            skel = mask.astype(np.float32)
        return skel

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        image = self._read_image(pair.image)
        mask = self._read_mask(pair.mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            if image.ndim == 2:
                image = image[..., None]
            image = torch.from_numpy(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))).float()

        pos_mask = mask if mask.ndim == 2 else (mask > 0).astype(np.float32)
        edge = self.mask_to_edges(pos_mask)
        topo = self.mask_to_skeleton(pos_mask)

        if self.num_classes <= 1:
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).long()
        edge = torch.from_numpy(edge).float().unsqueeze(0)
        topo = torch.from_numpy(topo).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "edge": edge,
            "topo": topo,
            "image_path": str(pair.image),
            "mask_path": str(pair.mask),
        }


def build_transforms(cfg: dict, train: bool = True):
    data_cfg = cfg["data"]
    return SimpleSegTransform(
        image_size=int(data_cfg.get("image_size", 512)),
        mean=data_cfg.get("mean", [0.5]),
        std=data_cfg.get("std", [0.5]),
        train=train,
    )


def split_pairs(
    pairs: Sequence[PairPaths],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[PairPaths], List[PairPaths]]:
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs


def make_datasets(cfg: dict):
    data_cfg = cfg["data"]
    train_images = data_cfg.get("train_images")
    train_masks = data_cfg.get("train_masks")
    val_images = data_cfg.get("val_images")
    val_masks = data_cfg.get("val_masks")
    test_images = data_cfg.get("test_images")
    test_masks = data_cfg.get("test_masks")

    in_channels = int(cfg["model"].get("in_channels", 1))
    num_classes = int(cfg["model"].get("num_classes", 1))
    threshold = int(data_cfg.get("mask_threshold", 127))

    train_pairs = pair_images_and_masks(train_images, train_masks)

    if val_images and val_masks:
        val_pairs = pair_images_and_masks(val_images, val_masks)
    else:
        train_pairs, val_pairs = split_pairs(
            train_pairs,
            val_ratio=float(data_cfg.get("val_ratio", 0.2)),
            seed=int(cfg["train"].get("seed", 42)),
        )

    test_pairs = None
    if test_images and test_masks:
        test_pairs = pair_images_and_masks(test_images, test_masks)

    train_ds = SolarDefectSegDataset(
        train_pairs,
        transform=build_transforms(cfg, train=True),
        in_channels=in_channels,
        num_classes=num_classes,
        mask_threshold=threshold,
    )
    val_ds = SolarDefectSegDataset(
        val_pairs,
        transform=build_transforms(cfg, train=False),
        in_channels=in_channels,
        num_classes=num_classes,
        mask_threshold=threshold,
    )
    test_ds = None
    if test_pairs is not None:
        test_ds = SolarDefectSegDataset(
            test_pairs,
            transform=build_transforms(cfg, train=False),
            in_channels=in_channels,
            num_classes=num_classes,
            mask_threshold=threshold,
        )
    return train_ds, val_ds, test_ds
