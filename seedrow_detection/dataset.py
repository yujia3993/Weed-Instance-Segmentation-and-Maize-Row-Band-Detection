# =============================================================================
# Corn Seedling Row Detection -- Dataset
# =============================================================================

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

INPUT_W   = 512
INPUT_H   = 256
PAD_VALUE = 114
LINE_WIDTH = 40


def annotation_to_mask(json_path: str, img_h: int, img_w: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(json_path):
        return mask
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for shape in data.get("shapes", []):
        if shape.get("label") != "crop_line":
            continue
        if shape.get("shape_type") != "linestrip":
            continue
        pts = shape.get("points", [])
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.int32)
        for i in range(len(arr) - 1):
            cv2.line(mask, tuple(arr[i]), tuple(arr[i+1]),
                     color=1, thickness=LINE_WIDTH)
    return mask


def resize_pad(img: np.ndarray, mask, target_w: int, target_h: int):
    """Resize with constant aspect ratio and pad to target size. Returns (img_p, mask_p, scale, pad_top, pad_left)."""
    ih, iw = img.shape[:2]
    scale  = min(target_w / iw, target_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)

    img_r    = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top  = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    pad_bot  = target_h - new_h - pad_top
    pad_right= target_w - new_w - pad_left

    img_p = cv2.copyMakeBorder(
        img_r, pad_top, pad_bot, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))

    mask_p = None
    if mask is not None:
        mask_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_p = cv2.copyMakeBorder(
            mask_r, pad_top, pad_bot, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0)

    return img_p, mask_p, scale, pad_top, pad_left


def online_augment(img: np.ndarray, mask: np.ndarray):
    """Online data augmentation (training set only, applied after resize_pad)."""
    if random.random() < 0.5:
        img  = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()

    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_t = TF.adjust_brightness(img_t, 1.0 + random.uniform(-0.3, 0.3))
    img_t = TF.adjust_contrast(img_t,   1.0 + random.uniform(-0.3, 0.3))
    img_t = TF.adjust_saturation(img_t, 1.0 + random.uniform(-0.3, 0.3))
    img   = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    if random.random() < 0.5:
        h, w   = img.shape[:2]
        scale  = random.uniform(0.75, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0     = random.randint(0, h - ch)
        x0     = random.randint(0, w - cw)
        img    = cv2.resize(img [y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_LINEAR)
        mask   = cv2.resize(mask[y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_NEAREST)

    return img, mask


# ImageNet normalization constants (BGR order)
_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
_STD  = np.array([0.225, 0.224, 0.229], dtype=np.float32)


def to_tensor_normalized(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 BGR → CHW float32 normalized tensor"""
    x = img.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return torch.from_numpy(x).permute(2, 0, 1).float()


class CropRowDataset(Dataset):
    def __init__(self, root: str, split: str = "train", augment: bool = False):
        self.root    = Path(root)
        self.split   = split
        self.augment = augment

        img_dir = self.root / "images"
        exts    = {".jpg", ".jpeg", ".png", ".bmp"}
        self.img_paths = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in exts)
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        lbl_path = self.root / "labels" / (img_path.stem + ".json")

        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Cannot read image: {img_path}")
        ih, iw = img.shape[:2]

        mask = annotation_to_mask(str(lbl_path), ih, iw)
        img, mask, _, _, _ = resize_pad(img, mask, INPUT_W, INPUT_H)

        if self.augment:
            img, mask = online_augment(img, mask)

        img_t  = to_tensor_normalized(img)
        mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t, str(img_path)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    ds   = CropRowDataset(root, split="train", augment=True)
    print(f"Dataset size: {len(ds)}")
    img, mask, name = ds[0]
    print(f"  img shape : {img.shape}   dtype: {img.dtype}")
    print(f"  mask shape: {mask.shape}  unique: {mask.unique().tolist()}")
    fg_ratio = mask.float().mean().item()
    print(f"  fg ratio  : {fg_ratio:.4f}  ({'OK' if fg_ratio > 0 else 'WARNING: all background'})")
