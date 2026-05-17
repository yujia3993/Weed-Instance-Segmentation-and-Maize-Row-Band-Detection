# =============================================================================
# SAM2 Fine-tuning + Classification Head Joint Training
#
# Strategy: freeze image encoder and prompt encoder, fine-tune mask decoder,
#           and add a 2-layer MLP classification head
#           (masked average pooling -> 256->128->6 classes).
# Total loss: mask_loss(Focal+Dice) + 0.2 * cls_loss(CrossEntropy)
# Optimizer: AdamW, mask decoder lr=3e-4, classification head lr=1e-3
# Pre-cache all image embeddings to speed up training.
#
# Note: bbox must be transformed with
#   predictor._transforms.transform_boxes() into SAM2's internal 1024-space
#   before passing to the prompt encoder — original image coordinates cannot
#   be passed directly.
#
# Install dependencies:
#   pip install git+https://github.com/facebookresearch/sam2.git pycocotools
# =============================================================================

import os
import json
import random
import time
import pkg_resources
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from omegaconf import OmegaConf
from hydra.utils import instantiate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------- Path configuration ----------
DRIVE      = '/content/drive/MyDrive'
TRAIN_JSON = f'{DRIVE}/coco_v3/instances_train.json'
VAL_JSON   = f'{DRIVE}/coco_v3/instances_val.json'
TEST_JSON  = f'{DRIVE}/coco_v3/instances_test.json'
SAM2_CKPT  = f'{DRIVE}/sam2_checkpoints/sam2.1_hiera_large.pt'
SAM2_PKG   = pkg_resources.resource_filename('sam2', '')
SAM2_CFG   = f'{SAM2_PKG}/configs/sam2.1/sam2.1_hiera_l.yaml'
SAVE_DIR   = f'{DRIVE}/sam2_finetune_results'
EMBED_CACHE_DIR = f'{DRIVE}/sam2_embed_cache'

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(EMBED_CACHE_DIR).mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CAT_NAMES = {
    1: 'cockspur_grass',  2: 'black_nightshade',  3: 'field_milk_thistle',
    4: 'meadow_grass',    5: 'redroot_amaranth',   6: 'white_goosefoot',
}
NUM_CLASSES = len(CAT_NAMES)

# Training hyperparameters
EPOCHS     = 40
GRAD_ACCUM = 2
CLS_LOSS_W = 0.2
PATIENCE   = 12
LR_DECODER = 3e-4
LR_CLS     = 1e-3


# =============================================================================
# Image Path Resolution
# =============================================================================

def resolve_img_path(file_name: str) -> Path:
    fname = file_name.replace('\\', '/')
    roots = [
        f'{DRIVE}/baseline_dataset_AB',
        f'{DRIVE}/c_cropped',
        '/content/C_drop/C_drop',
    ]
    for root in roots:
        p = Path(root) / fname
        if p.exists():
            return p
        parts = fname.split('/', 1)
        if len(parts) == 2:
            p2 = Path(root) / parts[0] / 'images' / parts[1]
            if p2.exists():
                return p2
    raise FileNotFoundError(f"Not found: {file_name}")


# =============================================================================
# Dataset
# =============================================================================

class WeedDataset(Dataset):
    def __init__(self, json_path: str, augment: bool = False):
        self.coco    = COCO(json_path)
        self.augment = augment
        all_anns     = [self.coco.loadAnns(aid)[0]
                        for aid in self.coco.getAnnIds()]
        self.anns    = [a for a in all_anns if a.get('segmentation')]
        print(f"  [{Path(json_path).stem}] {len(self.anns)} instances")

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann     = self.anns[idx]
        gt_mask = self.coco.annToMask(ann).astype(np.float32)
        x, y, w, h = ann['bbox']
        bbox    = np.array([x, y, x + w, y + h], dtype=np.float32)
        label   = ann['category_id'] - 1
        orig_h, orig_w = gt_mask.shape

        if self.augment and random.random() > 0.5:
            gt_mask = gt_mask[:, ::-1].copy()
            bbox[0], bbox[2] = orig_w - bbox[2], orig_w - bbox[0]

        return {
            'gt_mask' : gt_mask,
            'bbox'    : bbox,
            'label'   : label,
            'ann_id'  : ann['id'],
            'image_id': ann['image_id'],
            'orig_hw' : (orig_h, orig_w),
        }


def collate_fn(batch):
    return batch


# =============================================================================
# Model Components
# =============================================================================

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int = 256, num_classes: int = 6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def masked_avg_pool(feat_map: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    m     = (pred_mask > 0.5).float()
    denom = m.sum().clamp(min=1.0)
    return (feat_map * m.unsqueeze(0)).sum(dim=[-2, -1]) / denom


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth=1.0) -> torch.Tensor:
    pred  = pred.sigmoid()
    inter = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               alpha=0.8, gamma=2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt  = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def mask_loss_fn(pred_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return focal_loss(pred_logit, target) + dice_loss(pred_logit, target)


# =============================================================================
# Embedding Cache Generation
# =============================================================================

def generate_embedding_cache(predictor: SAM2ImagePredictor,
                              sam2_model,
                              all_jsons: list):
    unique_imgs = {}
    for json_path in all_jsons:
        coco_tmp = COCO(json_path)
        for img_info in coco_tmp.loadImgs(coco_tmp.getImgIds()):
            unique_imgs[img_info['id']] = img_info

    remaining = [info for info in unique_imgs.values()
                 if not os.path.exists(f'{EMBED_CACHE_DIR}/{info["id"]}.pt')]
    print(f"Total images: {len(unique_imgs)}  Cached: {len(unique_imgs)-len(remaining)}  "
          f"To generate: {len(remaining)}")

    if not remaining:
        return

    def load_image(img_info):
        try:
            img = np.array(PILImage.open(
                resolve_img_path(img_info['file_name'])).convert('RGB'))
            return img_info['id'], img, None
        except Exception as e:
            return img_info['id'], None, str(e)

    sam2_model.eval()
    errors = []
    done   = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load_image, info) for info in remaining]
        for future in futures:
            img_id, img, err = future.result()
            if err:
                errors.append((img_id, err))
                done += 1
                continue

            cache_path = f'{EMBED_CACHE_DIR}/{img_id}.pt'
            try:
                with torch.no_grad():
                    predictor.set_image(img)
                torch.save({
                    'image_embed'   : predictor._features['image_embed'].cpu(),
                    'high_res_feats': [f.cpu() for f in predictor._features['high_res_feats']],
                    'orig_hw'       : predictor._orig_hw,
                }, cache_path)
            except Exception as e:
                errors.append((img_id, str(e)))

            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(remaining)} done, {len(errors)} failed")

    print(f"Cache generation complete: {len(remaining)-len(errors)} succeeded, {len(errors)} failed")


# =============================================================================
# Embedding Loading (with in-memory cache)
# =============================================================================

embed_cache_mem = {}


def load_embed(image_id: int):
    if image_id not in embed_cache_mem:
        cache = torch.load(
            f'{EMBED_CACHE_DIR}/{image_id}.pt', map_location='cpu')
        embed_cache_mem[image_id] = (
            cache['image_embed'],
            cache['high_res_feats'],
            cache['orig_hw'],
        )
    img_embed, high_res, orig_hw = embed_cache_mem[image_id]
    return (
        img_embed.to(DEVICE),
        [f.to(DEVICE) for f in high_res],
        orig_hw,
    )


# =============================================================================
# COCO Evaluation (no classification head during inference; category uses GT label directly)
# =============================================================================

def run_coco_eval(sam2_model, cls_head: ClassificationHead,
                  predictor: SAM2ImagePredictor,
                  loader: DataLoader, json_path: str,
                  use_cls: bool = False) -> float:
    sam2_model.eval()
    cls_head.eval()
    coco_gt = COCO(json_path)
    results = []

    with torch.no_grad():
        for batch in loader:
            item = batch[0]
            img_embed, high_res, orig_hw = load_embed(item['image_id'])

            predictor._features       = {'image_embed': img_embed, 'high_res_feats': high_res}
            predictor._orig_hw        = orig_hw
            predictor._is_image_set   = True

            box_torch = predictor._transforms.transform_boxes(
                torch.tensor(item['bbox'][None], device=DEVICE),
                normalize=True, orig_hw=orig_hw[0])
            sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
                points=None, boxes=box_torch, masks=None)

            low_res_masks, _, _, _ = sam2_model.sam_mask_decoder(
                image_embeddings=img_embed,
                image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res,
            )
            pred_logit   = F.interpolate(low_res_masks, size=item['gt_mask'].shape,
                                         mode='bilinear', align_corners=False)[0, 0]
            pred_mask_np = (pred_logit.sigmoid().cpu().numpy() > 0.5).astype(np.uint8)

            if use_cls:
                pm_small = F.interpolate(
                    torch.tensor(pred_mask_np, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0).unsqueeze(0),
                    size=img_embed.shape[-2:],
                    mode='bilinear', align_corners=False)[0, 0]
                feat_vec = masked_avg_pool(img_embed[0], pm_small)
                pred_cat = cls_head(feat_vec.unsqueeze(0)).argmax(1).item() + 1
            else:
                pred_cat = item['label'] + 1

            rle = mask_utils.encode(np.asfortranarray(pred_mask_np))
            rle['counts'] = rle['counts'].decode('ascii')
            results.append({'image_id': item['image_id'],
                            'category_id': pred_cat,
                            'segmentation': rle, 'score': 1.0})

    if not results:
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, 'segm')
    ev.params.iouThrs = np.array([0.5])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])


# =============================================================================
# Training
# =============================================================================

def train(sam2_model, cls_head: ClassificationHead,
          predictor: SAM2ImagePredictor,
          train_loader: DataLoader, val_loader: DataLoader):
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in sam2_model.named_parameters() if p.requires_grad],
         'lr': LR_DECODER},
        {'params': cls_head.parameters(), 'lr': LR_CLS},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    history    = []
    best_ap    = 0.0
    no_improve = 0

    print("=" * 55)
    print("Decoder + classification head joint training")
    print("=" * 55)

    for epoch in range(1, EPOCHS + 1):
        sam2_model.train()
        cls_head.train()

        epoch_m = 0.0
        epoch_c = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            item = batch[0]
            try:
                gt_mask   = torch.tensor(item['gt_mask'], device=DEVICE).float()
                label     = torch.tensor(item['label'],   device=DEVICE, dtype=torch.long)
                img_embed, high_res, orig_hw = load_embed(item['image_id'])

                predictor._features     = {'image_embed': img_embed, 'high_res_feats': high_res}
                predictor._orig_hw      = orig_hw
                predictor._is_image_set = True

                with torch.autocast(device_type='cuda'):
                    with torch.no_grad():
                        box_torch = predictor._transforms.transform_boxes(
                            torch.tensor(item['bbox'][None], device=DEVICE),
                            normalize=True, orig_hw=orig_hw[0])
                        sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
                            points=None, boxes=box_torch, masks=None)
                        dense_pe = sam2_model.sam_prompt_encoder.get_dense_pe()

                    low_res_masks, _, _, _ = sam2_model.sam_mask_decoder(
                        image_embeddings=img_embed,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res,
                    )
                    pred_logit = F.interpolate(
                        low_res_masks, size=item['gt_mask'].shape,
                        mode='bilinear', align_corners=False)[0, 0]

                    m_loss    = mask_loss_fn(pred_logit, gt_mask)
                    epoch_m  += m_loss.item()

                    pm_small  = F.interpolate(
                        pred_logit.sigmoid().detach().unsqueeze(0).unsqueeze(0),
                        size=img_embed.shape[-2:],
                        mode='bilinear', align_corners=False)[0, 0]
                    feat_vec  = masked_avg_pool(img_embed[0], pm_small)
                    cls_logit = cls_head(feat_vec.unsqueeze(0))
                    c_loss    = F.cross_entropy(cls_logit, label.unsqueeze(0))
                    epoch_c  += c_loss.item()
                    total     = m_loss + CLS_LOSS_W * c_loss

                (total / GRAD_ACCUM).backward()

            except Exception as e:
                print(f"  [skip] step {step}: {e}")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

            finally:
                try:
                    del img_embed, high_res, sparse_emb, dense_emb, dense_pe
                    del low_res_masks, pred_logit, gt_mask, label
                    del pm_small, feat_vec, cls_logit, m_loss, c_loss, total
                except Exception:
                    pass

            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(sam2_model.parameters()) + list(cls_head.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 500 == 0:
                print(f"  epoch {epoch} | step {step}/{len(train_loader)} | "
                      f"mask_loss={epoch_m/step:.4f}  cls_loss={epoch_c/step:.4f}")

            if step % 100 == 0:
                torch.cuda.empty_cache()

        scheduler.step()

        val_ap = run_coco_eval(sam2_model, cls_head, predictor,
                               val_loader, VAL_JSON, use_cls=False)
        n   = len(train_loader)
        row = {'epoch': epoch, 'mask_loss': epoch_m/n,
               'cls_loss': epoch_c/n, 'val_ap': val_ap}
        history.append(row)
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"mask_loss={epoch_m/n:.4f}  cls_loss={epoch_c/n:.4f} | "
              f"val AP@0.5={val_ap:.4f} | {time.time()-t0:.0f}s")

        if val_ap > best_ap:
            best_ap    = val_ap
            no_improve = 0
            torch.save({
                'epoch'     : epoch,
                'sam2_state': {k: v for k, v in sam2_model.state_dict().items()
                               if 'mask_decoder' in k},
                'cls_state' : cls_head.state_dict(),
                'val_ap'    : val_ap,
            }, f'{SAVE_DIR}/best_model_phase2.pt')
            print(f"  Best model saved (AP@0.5={val_ap:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping: no improvement for {PATIENCE} epochs")
                break

    with open(f'{SAVE_DIR}/train_history_phase2.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete | Best val AP@0.5={best_ap:.4f}")
    return history


# =============================================================================
# Main Entry
# =============================================================================

if __name__ == '__main__':
    # Build model
    cfg = OmegaConf.load(SAM2_CFG)
    OmegaConf.resolve(cfg)
    sam2_model = instantiate(cfg.model, _recursive_=True)
    sd = torch.load(SAM2_CKPT, map_location='cpu', weights_only=True)['model']
    sam2_model.load_state_dict(sd, strict=False)
    sam2_model = sam2_model.to(DEVICE)

    for name, param in sam2_model.named_parameters():
        param.requires_grad = ('mask_decoder' in name)

    cls_head  = ClassificationHead(256, NUM_CLASSES).to(DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)

    n_dec = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad)
    print(f"Decoder trainable params: {n_dec/1e6:.1f}M")

    # Generate embedding cache (first run)
    generate_embedding_cache(predictor, sam2_model,
                             [TRAIN_JSON, VAL_JSON, TEST_JSON])

    # DataLoader
    train_ds = WeedDataset(TRAIN_JSON, augment=True)
    val_ds   = WeedDataset(VAL_JSON,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # Train
    train(sam2_model, cls_head, predictor, train_loader, val_loader)

    # Final test set evaluation
    print("\n===== Test Set Evaluation =====")
    ckpt = torch.load(f'{SAVE_DIR}/best_model_phase2.pt', map_location=DEVICE)
    state = sam2_model.state_dict()
    state.update(ckpt['sam2_state'])
    sam2_model.load_state_dict(state)
    cls_head.load_state_dict(ckpt['cls_state'])

    test_ds     = WeedDataset(TEST_JSON, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    test_ap = run_coco_eval(sam2_model, cls_head, predictor,
                            test_loader, TEST_JSON, use_cls=True)

    print(f"\nFinal test AP@0.5: {test_ap:.4f}")

    # Per-class AP@0.5
    coco_gt = COCO(TEST_JSON)
    print("\nPer-class AP@0.5:")
    for cid, cname in CAT_NAMES.items():
        print(f"  {cname:<25s}  -- requires separate per-class eval")

    # Experiment comparison summary
    print("\n===== Full Experiment Comparison =====")
    rows = [
        ("Mask2Former v3",         0.116, False),
        ("SAM2 Zero-shot",         0.630, True),
        ("Grounded SAM2",          0.002, False),
        ("SAM2 Finetune+Cls Head", test_ap, True),
    ]
    print(f"{'Method':<24} {'AP@0.5':>8} {'Needs Prompt':>12}")
    print("-" * 46)
    for name, ap, need_prompt in rows:
        print(f"{name:<24} {ap:>8.3f} {'Yes' if need_prompt else 'No':>12}")
