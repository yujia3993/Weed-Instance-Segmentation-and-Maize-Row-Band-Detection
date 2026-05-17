# =============================================================================
# SAM Weed Instance Segmentation -- Dataset Split + Decoder-only Fine-tuning
#
# Strategy: freeze image encoder and prompt encoder, fine-tune mask decoder only.
# Loss: Focal Loss + Dice Loss (each weighted 0.5)
# Optimizer: Adam + CosineAnnealingLR
# Training config: 20 epochs, batch size 4, lr 1e-4
# Dataset split: 8:1:1 (train/val/test)
# =============================================================================

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

CLASSES = [
    'cockspur_grass', 'black_nightshade', 'field_milk_thistle',
    'meadow_grass', 'redroot_amaranth', 'white_goosefoot',
]

# Zero-shot v1 baseline (used for final comparison print)
BASELINE_V1 = {
    "black_nightshade":   {"mean_iou": 0.7666, "ge_0.5": 0.993, "ge_0.9": 0.085},
    "cockspur_grass":     {"mean_iou": 0.5793, "ge_0.5": 0.713, "ge_0.9": 0.004},
    "field_milk_thistle": {"mean_iou": 0.7713, "ge_0.5": 0.981, "ge_0.9": 0.127},
    "meadow_grass":       {"mean_iou": 0.5192, "ge_0.5": 0.609, "ge_0.9": 0.000},
    "redroot_amaranth":   {"mean_iou": 0.8360, "ge_0.5": 0.960, "ge_0.9": 0.339},
    "white_goosefoot":    {"mean_iou": 0.8472, "ge_0.5": 0.979, "ge_0.9": 0.326},
    "overall":            {"mean_iou": 0.6676, "ge_0.5": 0.813, "ge_0.9": 0.094},
}


# =============================================================================
# Dataset Split (8:1:1), generates splits.json
# =============================================================================

def build_samples(base_dir: Path, cls: str) -> list:
    img_dir  = base_dir / cls / 'images'
    json_dir = base_dir / cls / 'json'
    mask_dir = base_dir / cls / 'masks'
    bbox_dir = base_dir / cls / 'bboxes'

    samples = []
    for img_file in sorted(img_dir.glob('*.jpg')):
        stem       = img_file.stem
        json_file  = json_dir / f'{stem}.json'
        bbox_file  = bbox_dir / f'{stem}.json'
        mask_files = sorted(mask_dir.glob(f'{stem}_inst*.png'))

        if not json_file.exists():
            print(f'[WARN] Missing json: {json_file}, skipping')
            continue
        if not mask_files:
            print(f'[WARN] Missing mask: {stem}, skipping')
            continue

        samples.append({
            'class':      cls,
            'image_path': str(Path(cls) / 'images' / img_file.name),
            'json_path':  str(Path(cls) / 'json'   / json_file.name),
            'mask_paths': [str(Path(cls) / 'masks' / m.name) for m in mask_files],
            'bbox_path':  str(Path(cls) / 'bboxes' / bbox_file.name)
                          if bbox_file.exists() else None,
        })
    return samples


def split_samples(samples: list, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    data = samples[:]
    rng.shuffle(data)
    n       = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def prepare_splits(base_dir: str, train_ratio=0.8, val_ratio=0.1, seed=42):
    base = Path(base_dir)
    from collections import defaultdict
    all_splits = defaultdict(list)

    for cls in CLASSES:
        cls_dir = base / cls
        if not cls_dir.exists():
            print(f'Class directory not found: {cls}')
            continue
        samples = build_samples(base, cls)
        train, val, test = split_samples(samples, train_ratio, val_ratio, seed)
        all_splits['train'].extend(train)
        all_splits['val'].extend(val)
        all_splits['test'].extend(test)
        print(f'  {cls:22s}: total={len(samples):4d}  '
              f'train={len(train):3d}  val={len(val):3d}  test={len(test):3d}')

    out_path = base / 'splits.json'
    with open(out_path, 'w') as f:
        json.dump(dict(all_splits), f, indent=2)
    print(f'\n[OK] splits.json written: {out_path}')
    print(f'     train={len(all_splits["train"])}  '
          f'val={len(all_splits["val"])}  '
          f'test={len(all_splits["test"])}')


# =============================================================================
# Dataset
# =============================================================================

def load_gt_masks_and_bboxes(sample: dict, base_dir: Path):
    masks, bboxes = [], []
    for mp in sample["mask_paths"]:
        m  = np.array(Image.open(base_dir / mp).convert("L")) > 127
        bb = _mask_to_bbox(m)
        if bb is None:
            continue
        masks.append(m)
        bboxes.append(bb)
    return masks, bboxes


def _mask_to_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


class WeedSAMDataset(Dataset):
    def __init__(self, samples: list, base_dir: Path,
                 transform: ResizeLongestSide, img_size: int = 1024):
        self.base_dir  = base_dir
        self.transform = transform
        self.img_size  = img_size
        self.items     = []

        for s in samples:
            masks, bboxes = load_gt_masks_and_bboxes(s, base_dir)
            if not masks:
                continue
            for mask, bbox in zip(masks, bboxes):
                self.items.append({
                    "image_path": base_dir / s["image_path"],
                    "bbox":       bbox,
                    "gt_mask":    mask,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item    = self.items[idx]
        image   = np.array(Image.open(item["image_path"]).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        img_resized = self.transform.apply_image(image)
        img_tensor  = torch.as_tensor(img_resized).permute(2, 0, 1).float()
        img_tensor  = F.pad(img_tensor, (0, self.img_size - img_tensor.shape[2],
                                         0, self.img_size - img_tensor.shape[1]))

        bbox            = np.array(item["bbox"], dtype=float)
        bbox_transformed = self.transform.apply_boxes(bbox[None], (orig_h, orig_w))[0]

        gt = item["gt_mask"].astype(np.uint8) * 255
        gt_pil    = Image.fromarray(gt).resize((256, 256), Image.NEAREST)
        gt_tensor = torch.as_tensor(np.array(gt_pil) > 127).float().unsqueeze(0)

        return {
            "image":         img_tensor,
            "bbox":          torch.tensor(bbox_transformed, dtype=torch.float32),
            "gt_mask":       gt_tensor,
            "original_size": (orig_h, orig_w),
        }


# =============================================================================
# Loss
# =============================================================================

def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               alpha=0.25, gamma=2.0) -> torch.Tensor:
    bce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt   = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    prob  = torch.sigmoid(pred)
    inter = (prob * target).sum(dim=(-1, -2))
    denom = prob.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def seg_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return 0.5 * focal_loss(pred_logits, gt) + 0.5 * dice_loss(pred_logits, gt)


# =============================================================================
# Forward Pass (SAM does not support true batch forward; process one at a time)
# =============================================================================

def forward_batch(sam, batch: dict, device: torch.device) -> torch.Tensor:
    images     = batch["image"].to(device)
    bboxes     = batch["bbox"].to(device)
    all_logits = []

    for i in range(images.shape[0]):
        img = images[i:i+1]
        bb  = bboxes[i].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            img_emb = sam.image_encoder(img)

        sparse_emb, dense_emb = sam.prompt_encoder(
            points=None, boxes=bb, masks=None)

        low_res_logits, _ = sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        all_logits.append(low_res_logits)

    return torch.cat(all_logits, dim=0)


# =============================================================================
# Evaluation
# =============================================================================

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return float(inter) / float(union + 1e-6)


@torch.no_grad()
def evaluate(sam, loader: DataLoader, device: torch.device, threshold=0.5) -> dict:
    sam.eval()
    ious = []

    for batch in loader:
        logits     = forward_batch(sam, batch, device)
        pred_masks = (torch.sigmoid(logits) > threshold).squeeze(1).cpu().numpy()
        gt_masks   = batch["gt_mask"].squeeze(1).numpy() > 0.5

        for pred, gt in zip(pred_masks, gt_masks):
            ious.append(compute_iou(pred, gt))

    ious = np.array(ious)
    return {
        "mean_iou": float(ious.mean()),
        "ge_0.5":   float((ious >= 0.5).mean()),
        "ge_0.9":   float((ious >= 0.9).mean()),
        "n":        len(ious),
    }


@torch.no_grad()
def evaluate_per_class(sam, test_samples: list, base_dir: Path,
                       transform: ResizeLongestSide,
                       device: torch.device) -> dict:
    sam.eval()
    class_data = {}

    for s in test_samples:
        cls   = s["class"]
        masks, bboxes = load_gt_masks_and_bboxes(s, base_dir)
        if not masks:
            continue

        image  = np.array(Image.open(base_dir / s["image_path"]).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        img_resized = transform.apply_image(image)
        img_tensor  = torch.as_tensor(img_resized).permute(2, 0, 1).float()
        img_tensor  = F.pad(img_tensor,
                            (0, 1024 - img_tensor.shape[2],
                             0, 1024 - img_tensor.shape[1])).unsqueeze(0).to(device)

        img_emb = sam.image_encoder(img_tensor)

        for mask_gt, bbox in zip(masks, bboxes):
            bb_t = transform.apply_boxes(np.array(bbox, dtype=float)[None],
                                         (orig_h, orig_w))
            bb_tensor = torch.tensor(bb_t, dtype=torch.float32).unsqueeze(0).to(device)

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None, boxes=bb_tensor, masks=None)
            low_res_logits, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            pred = (torch.sigmoid(low_res_logits[0, 0]) > 0.5).cpu().numpy()
            gt_256 = np.array(
                Image.fromarray(mask_gt.astype(np.uint8) * 255).resize(
                    (256, 256), Image.NEAREST)) > 127

            class_data.setdefault(cls, []).append(compute_iou(pred, gt_256))

    results = {}
    for cls, ious in class_data.items():
        a = np.array(ious)
        results[cls] = {
            "mean_iou": float(a.mean()),
            "ge_0.5":   float((a >= 0.5).mean()),
            "ge_0.9":   float((a >= 0.9).mean()),
            "n":        len(a),
        }
    return results


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)

    for param in sam.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in sam.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    base_dir    = Path(args.base_dir)
    splits_path = base_dir / "splits.json"
    assert splits_path.exists(), "Run prepare_splits() first to generate splits.json"

    with open(splits_path) as f:
        splits = json.load(f)

    transform    = ResizeLongestSide(1024)
    train_ds     = WeedSAMDataset(splits["train"], base_dir, transform)
    val_ds       = WeedSAMDataset(splits["val"],   base_dir, transform)
    test_ds      = WeedSAMDataset(splits["test"],  base_dir, transform)
    print(f"Instances -- train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    optimizer = Adam(sam.mask_decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0
    history      = []

    for epoch in range(1, args.epochs + 1):
        sam.train()
        sam.image_encoder.eval()
        sam.prompt_encoder.eval()

        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = forward_batch(sam, batch, device)
            gt     = batch["gt_mask"].to(device)
            loss   = seg_loss(logits, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sam.mask_decoder.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss    = epoch_loss / len(train_loader)
        val_metrics = evaluate(sam, val_loader, device)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  "
              f"val_IoU={val_metrics['mean_iou']:.4f}  "
              f"val>=0.5={val_metrics['ge_0.5']:.3f}  "
              f"val>=0.9={val_metrics['ge_0.9']:.3f}")

        history.append({"epoch": epoch, "loss": avg_loss, **val_metrics})

        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou"]
            torch.save(sam.mask_decoder.state_dict(), out_dir / "best_decoder.pth")
            print(f"  saved best_decoder.pth (val_IoU={best_val_iou:.4f})")

    # Final test
    print("\n===== Test Set Evaluation (best checkpoint) =====")
    sam.mask_decoder.load_state_dict(
        torch.load(out_dir / "best_decoder.pth", map_location=device))
    test_metrics = evaluate(sam, test_loader, device)
    print(f"Test mean IoU : {test_metrics['mean_iou']:.4f}")
    print(f"Test IoU>=0.5 : {test_metrics['ge_0.5']:.3f}")
    print(f"Test IoU>=0.9 : {test_metrics['ge_0.9']:.3f}")

    # Per-class evaluation
    print("\n===== Per-class Test IoU =====")
    class_ious = evaluate_per_class(
        sam, splits["test"], base_dir, transform, device)
    for cls, m in class_ious.items():
        print(f"  {cls:22s}: mean_IoU={m['mean_iou']:.4f}  "
              f"n={m['n']:4d}  >=0.5={m['ge_0.5']:.3f}  >=0.9={m['ge_0.9']:.3f}")

    # Comparison with Zero-shot v1
    print("\n===== Comparison with Zero-shot v1 =====")
    print(f"{'Class':22s}  {'ZS IoU':>8}  {'FT IoU':>8}  {'Delta':>7}")
    for cls, m in class_ious.items():
        zs    = BASELINE_V1.get(cls, {}).get("mean_iou", float("nan"))
        ft    = m["mean_iou"]
        delta = ft - zs
        flag  = "+" if delta > 0 else "-"
        print(f"  {cls:22s}  {zs:8.4f}  {ft:8.4f}  {flag}{abs(delta):.4f}")

    # Save results
    results = {"test_metrics": test_metrics, "class_metrics": class_ious,
               "history": history, "baseline_v1": BASELINE_V1}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_dir / 'results.json'}")


# =============================================================================
# Entry
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",   required=True,
                        help="Root directory of baseline_dataset_AB (containing splits.json)")
    parser.add_argument("--checkpoint", required=True,
                        help="SAM checkpoint path, e.g. sam_vit_l_0b3195.pth")
    parser.add_argument("--model_type", default="vit_l",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--out_dir",    default="sam_finetuned")
    args = parser.parse_args()

    # Generate splits.json if it does not exist
    if not (Path(args.base_dir) / "splits.json").exists():
        prepare_splits(args.base_dir)

    train(args)
