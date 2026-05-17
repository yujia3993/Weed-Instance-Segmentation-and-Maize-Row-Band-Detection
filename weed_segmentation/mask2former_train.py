# =============================================================================
# Mask2Former Weed Instance Segmentation Training (v3: Data Augmentation + Early Stopping)
#
# Model: facebook/mask2former-swin-small-coco-instance (HuggingFace)
# Input: short edge 640, long edge 800, padded to 800×800
# Augmentation: HorizontalFlip / RandomBrightnessContrast / HueSaturationValue / GaussianBlur
# Optimizer: AdamW (lr=5e-5, weight_decay=0.05) + 3-epoch warmup + Cosine Annealing
# Early stopping: patience=5, min_delta=0.3 (monitors val loss)
#
# Install dependencies:
#   pip install transformers==4.40.0 pycocotools albumentations
# =============================================================================

import os
import json
import math
import pathlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import cv2
import albumentations as A
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (Mask2FormerForUniversalSegmentation,
                          Mask2FormerImageProcessor)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_util

# ---------- Path configuration ----------
ROOT_MAP = {
    "D:\\实习\\baseline_dataset_AB"    : "/content/drive/MyDrive/baseline_dataset_AB",
    "D:\\实习\\dataset_sorted\\C_drop" : "/content/C_drop",
    "D:\\实习\\c_cropped"              : "/content/drive/MyDrive/c_cropped",
}
COCO_DIR = "/content/drive/MyDrive/coco_v3"
WORK_DIR = "/content/drive/MyDrive/mask2former_v3"
os.makedirs(WORK_DIR, exist_ok=True)

# ---------- Hyperparameter configuration ----------
IMG_SIZE_SHORT = 640
IMG_SIZE_LONG  = 800
BATCH_SIZE     = 4
LR             = 5e-5
NUM_EPOCHS     = 30
WARMUP_EPOCHS  = 3
ES_PATIENCE    = 5
ES_MIN_DELTA   = 0.3
DEVICE         = "cuda"

ID_MAP  = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
ID2NAME = {0: "cockspur_grass",    1: "black_nightshade",
           2: "field_milk_thistle", 3: "meadow_grass",
           4: "redroot_amaranth",   5: "white_goosefoot"}


# =============================================================================
# Data Augmentation (training set only)
# =============================================================================

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30,
                         val_shift_limit=20, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
], additional_targets={"masks": "masks"})


# =============================================================================
# Path Resolution
# =============================================================================

def resolve_path(root_win: str, file_name: str) -> str:
    colab_root = ROOT_MAP.get(root_win)
    if colab_root is None:
        raise ValueError(f"Unknown root mapping: {root_win}")
    rel = pathlib.PurePosixPath(pathlib.PureWindowsPath(file_name))
    return os.path.join(colab_root, str(rel))


# =============================================================================
# Dataset
# =============================================================================

processor = Mask2FormerImageProcessor(
    ignore_index=255,
    reduce_labels=False,
    do_resize=True,
    size={"shortest_edge": IMG_SIZE_SHORT, "longest_edge": IMG_SIZE_LONG},
    do_pad=True,
    pad_size={"height": IMG_SIZE_LONG, "width": IMG_SIZE_LONG},
)


class WeedDataset(Dataset):
    def __init__(self, ann_path: str, processor, augment: bool = False):
        with open(ann_path) as f:
            data = json.load(f)
        self.processor = processor
        self.augment   = augment
        self.images    = {img['id']: img for img in data['images']}
        self.ann_map   = {}
        for ann in data['annotations']:
            self.ann_map.setdefault(ann['image_id'], []).append(ann)
        self.ids = [iid for iid in self.images if iid in self.ann_map]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_info = self.images[img_id]
        anns     = self.ann_map[img_id]

        img_path = resolve_path(img_info['root'], img_info['file_name'])
        image    = np.array(Image.open(img_path).convert("RGB"))
        H, W     = image.shape[:2]

        masks, labels = [], []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in ID_MAP:
                continue
            seg  = ann['segmentation']
            mask = np.zeros((H, W), dtype=np.uint8)

            if isinstance(seg, dict):
                from pycocotools import mask as coco_mask
                mask = coco_mask.decode(seg).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                m = Image.new('L', (W, H), 0)
                for poly in seg:
                    pts = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                    ImageDraw.Draw(m).polygon(pts, fill=1)
                mask = np.array(m, dtype=np.uint8)

            if mask.sum() == 0:
                continue
            masks.append(mask)
            labels.append(ID_MAP[cat_id])

        if len(masks) == 0:
            masks  = [np.zeros((H, W), dtype=np.uint8)]
            labels = [0]

        if self.augment:
            augmented = train_transform(image=image, masks=masks)
            image     = augmented['image']
            masks     = augmented['masks']

        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        _, _, rH, rW = inputs["pixel_values"].shape

        final_masks = []
        for mask in masks:
            m_resized = np.array(
                Image.fromarray(mask).resize((rW, rH), Image.NEAREST))
            if m_resized.sum() == 0:
                continue
            final_masks.append(torch.from_numpy(m_resized).float())

        if len(final_masks) == 0:
            final_masks = [torch.zeros(rH, rW, dtype=torch.float32)]
            labels      = [0]

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "pixel_mask"  : inputs["pixel_mask"].squeeze(0),
            "mask_labels" : final_masks,
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "img_id"      : img_id,
        }


def collate_fn(batch):
    max_h = max(b["pixel_values"].shape[1] for b in batch)
    max_w = max(b["pixel_values"].shape[2] for b in batch)

    padded_pixels, padded_pm = [], []
    for b in batch:
        c, h, w = b["pixel_values"].shape
        pad = torch.zeros(c, max_h, max_w, dtype=b["pixel_values"].dtype)
        pad[:, :h, :w] = b["pixel_values"]
        padded_pixels.append(pad)

        pm = torch.zeros(max_h, max_w, dtype=b["pixel_mask"].dtype)
        pm[:h, :w] = b["pixel_mask"]
        padded_pm.append(pm)

    stacked_mask_labels = []
    for b in batch:
        masks = b["mask_labels"]
        h, w  = masks[0].shape
        padded = []
        for m in masks:
            pm = torch.zeros(max_h, max_w, dtype=m.dtype)
            pm[:h, :w] = m
            padded.append(pm)
        stacked_mask_labels.append(torch.stack(padded))

    return {
        "pixel_values": torch.stack(padded_pixels),
        "pixel_mask"  : torch.stack(padded_pm),
        "mask_labels" : stacked_mask_labels,
        "class_labels": [b["class_labels"] for b in batch],
        "img_ids"     : [b["img_id"] for b in batch],
    }


# =============================================================================
# LR Schedule (3-epoch warmup + Cosine Annealing)
# =============================================================================

def get_lr(epoch: int, warmup_epochs: int, total_epochs: int,
           base_lr: float, min_lr: float = 1e-6) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# Training
# =============================================================================

def train(model, train_loader: DataLoader, val_loader: DataLoader):
    optimizer     = AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    log_path      = f"{WORK_DIR}/train.log"
    open(log_path, 'w').close()

    best_val_loss = float("inf")
    es_counter    = 0

    for epoch in range(NUM_EPOCHS):
        cur_lr = get_lr(epoch, WARMUP_EPOCHS, NUM_EPOCHS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        # Train
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            outputs = model(
                pixel_values=batch['pixel_values'].to(DEVICE),
                pixel_mask  =batch['pixel_mask'].to(DEVICE),
                mask_labels =[m.to(DEVICE) for m in batch['mask_labels']],
                class_labels=[c.to(DEVICE) for c in batch['class_labels']],
            )
            optimizer.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += outputs.loss.item()

            if step % 20 == 0:
                print(f"  E{epoch+1}/{NUM_EPOCHS} S{step}/{len(train_loader)} "
                      f"loss={outputs.loss.item():.4f} lr={cur_lr:.2e}")

        avg_train = train_loss / len(train_loader)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    pixel_values=batch['pixel_values'].to(DEVICE),
                    pixel_mask  =batch['pixel_mask'].to(DEVICE),
                    mask_labels =[m.to(DEVICE) for m in batch['mask_labels']],
                    class_labels=[c.to(DEVICE) for c in batch['class_labels']],
                )
                val_loss += outputs.loss.item()
        avg_val = val_loss / len(val_loader)

        log = (f"Epoch {epoch+1:3d} | lr={cur_lr:.2e} | "
               f"train={avg_train:.4f} | val={avg_val:.4f}")
        print(log)
        with open(log_path, 'a') as f:
            f.write(log + '\n')

        if avg_val < best_val_loss - ES_MIN_DELTA:
            best_val_loss = avg_val
            es_counter    = 0
            model.save_pretrained(f"{WORK_DIR}/best_model")
        else:
            es_counter += 1
            print(f"  Early stopping counter: {es_counter}/{ES_PATIENCE}")
            if es_counter >= ES_PATIENCE:
                print("Early stopping triggered, halting training")
                model.save_pretrained(f"{WORK_DIR}/checkpoint_epoch{epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            model.save_pretrained(f"{WORK_DIR}/checkpoint_epoch{epoch+1}")

    print("Training complete")


# =============================================================================
# Evaluation (COCO AP@0.5)
# =============================================================================

def evaluate(model, test_ds: WeedDataset, test_loader: DataLoader):
    model.eval()
    results = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img_id   = batch["img_ids"][0]
            img_info = test_ds.images[img_id]
            outputs  = model(
                pixel_values=batch['pixel_values'].to(DEVICE),
                pixel_mask  =batch['pixel_mask'].to(DEVICE),
            )
            target_size = [(img_info['height'], img_info['width'])]
            pred = processor.post_process_instance_segmentation(
                outputs, target_sizes=target_size, threshold=0.5)[0]

            for seg_info in pred['segments_info']:
                mask_np = (pred['segmentation'].cpu().numpy()
                           == seg_info['id']).astype(np.uint8)
                rle = coco_mask_util.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                results.append({
                    "image_id"    : img_id,
                    "category_id" : seg_info['label_id'] + 1,
                    "segmentation": rle,
                    "score"       : seg_info['score'],
                })

    pred_path = f"{WORK_DIR}/test_predictions_v3.json"
    with open(pred_path, 'w') as f:
        json.dump(results, f)

    coco_gt   = COCO(f"{COCO_DIR}/instances_test.json")
    coco_dt   = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\nPer-class AP@0.5:")
    for cat in coco_gt.dataset['categories']:
        ev = COCOeval(coco_gt, coco_dt, 'segm')
        ev.params.iouThrs = np.array([0.5])
        ev.params.catIds  = [cat['id']]
        ev.evaluate()
        ev.accumulate()
        ap = ev.eval['precision'][0, :, 0, 2, 2].mean()
        print(f"  {cat['name']:25s} AP@0.5 = {ap:.3f}")


# =============================================================================
# Dataset statistics (mean instance area per class)
# =============================================================================

def dataset_stats(train_json: str):
    from collections import defaultdict
    with open(train_json) as f:
        data = json.load(f)

    id2name   = {c['id']: c['name'] for c in data['categories']}
    cat_areas = defaultdict(list)
    ann_by_img = defaultdict(list)
    for ann in data['annotations']:
        ann_by_img[ann['image_id']].append(ann)

    for img in data['images']:
        img_area = img['height'] * img['width']
        for ann in ann_by_img[img['id']]:
            cat = ann['category_id']
            seg = ann['segmentation']
            if isinstance(seg, list) and len(seg) > 0:
                poly = seg[0]
                pts  = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                m    = Image.new('L', (img['width'], img['height']), 0)
                ImageDraw.Draw(m).polygon(pts, fill=1)
                area = np.array(m).sum()
                cat_areas[cat].append(area / img_area * 100)

    print("Mean instance area per class (% of image area):")
    for cat_id, areas in sorted(cat_areas.items()):
        print(f"  {id2name[cat_id]:25s}: mean={np.mean(areas):.2f}%  "
              f"median={np.median(areas):.2f}%  min={np.min(areas):.4f}%")


# =============================================================================
# Main Entry
# =============================================================================

if __name__ == '__main__':
    train_ds = WeedDataset(f"{COCO_DIR}/instances_train.json", processor, augment=True)
    val_ds   = WeedDataset(f"{COCO_DIR}/instances_val.json",   processor, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=collate_fn,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False,
                              num_workers=2, collate_fn=collate_fn,
                              pin_memory=True, persistent_workers=True)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance",
        num_labels=6,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Verify forward pass
    batch = next(iter(train_loader))
    with torch.no_grad():
        out = model(
            pixel_values=batch['pixel_values'].to(DEVICE),
            pixel_mask  =batch['pixel_mask'].to(DEVICE),
            mask_labels =[m.to(DEVICE) for m in batch['mask_labels']],
            class_labels=[c.to(DEVICE) for c in batch['class_labels']],
        )
    print(f"forward OK, loss={out.loss.item():.4f}")

    train(model, train_loader, val_loader)

    # Load best_model and evaluate on test set
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        f"{WORK_DIR}/best_model").to(DEVICE)
    test_ds     = WeedDataset(f"{COCO_DIR}/instances_test.json", processor, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=2, collate_fn=collate_fn)
    evaluate(model, test_ds, test_loader)

    # Dataset statistics
    dataset_stats(f"{COCO_DIR}/instances_train.json")
