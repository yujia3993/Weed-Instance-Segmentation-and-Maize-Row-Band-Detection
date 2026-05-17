# =============================================================================
# Corn Seedling Row Detection -- BiSeNet-V2 Training
#
# Loss: CE(pos_weight=8) + Dice, each weighted 0.5
# Optimizer: SGD (momentum=0.9, nesterov=True) + Poly LR decay
# Mixed precision: AMP (torch.cuda.amp)
# Early stopping: monitors val IoU, patience=15, min_delta=0.002
#
# Usage:
#   python train.py --data_root /content/corn_augmented --out_dir /content/bisenet_ckpt
#   python train.py --data_root /content/corn_augmented --out_dir /content/bisenet_ckpt --resume
# =============================================================================

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import CropRowDataset
from model import BiSeNetV2


# =============================================================================
# Loss
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        tgt   = (targets == 1).float()
        inter = (probs * tgt).sum()
        union = probs.sum() + tgt.sum()
        return 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight: float = 8.0):
        super().__init__()
        pw        = torch.tensor([1.0, pos_weight])
        self.ce   = nn.CrossEntropyLoss(weight=pw)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        self.ce.weight = self.ce.weight.to(logits.device)
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


# =============================================================================
# Evaluation Metric: Foreground IoU
# =============================================================================

class IoUMeter:
    def __init__(self):
        self.inter = 0
        self.union = 0

    def reset(self):
        self.inter = 0
        self.union = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds   = logits.argmax(dim=1)
        pred_fg = (preds   == 1)
        gt_fg   = (targets == 1)
        self.inter += (pred_fg & gt_fg).sum().item()
        self.union += (pred_fg | gt_fg).sum().item()

    def iou(self) -> float:
        return self.inter / self.union if self.union > 0 else 0.0


# =============================================================================
# Poly LR Schedule
# =============================================================================

class PolyLR:
    def __init__(self, optimizer, max_iters: int, power: float = 0.9,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.power     = power
        self.min_lr    = min_lr
        self.init_lrs  = [pg['lr'] for pg in optimizer.param_groups]
        self.cur_iter  = 0

    def step(self):
        factor = (1 - self.cur_iter / self.max_iters) ** self.power
        for pg, init_lr in zip(self.optimizer.param_groups, self.init_lrs):
            pg['lr'] = max(init_lr * factor, self.min_lr)
        self.cur_iter += 1

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# =============================================================================
# Training / Validation
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, aux_weight=0.4, log_interval=20):
    model.train()
    meter      = IoUMeter()
    total_loss = 0.0
    t0 = time.time()

    for i, (imgs, masks, _) in enumerate(loader):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            out, a3, a4, a5 = model(imgs)
            loss  = criterion(out, masks)
            loss += aux_weight * criterion(a3, masks)
            loss += aux_weight * criterion(a4, masks)
            loss += aux_weight * criterion(a5, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        meter.update(out.detach(), masks)
        total_loss += loss.item()

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(f"  step {i+1:4d}/{len(loader)}  "
                  f"loss={total_loss/(i+1):.4f}  "
                  f"IoU={meter.iou():.4f}  "
                  f"lr={scheduler.get_lr():.2e}  "
                  f"{(i+1)/elapsed:.1f} it/s")

    return total_loss / len(loader), meter.iou()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    meter      = IoUMeter()
    total_loss = 0.0

    for imgs, masks, _ in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast():
            out  = model(imgs)
            loss = criterion(out, masks)
        meter.update(out, masks)
        total_loss += loss.item()

    return total_loss / len(loader), meter.iou()


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(model, optimizer, scaler, path: str, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    print(f"  [ckpt] resumed from epoch {ckpt['epoch']}  best_iou={ckpt['best_iou']:.4f}")
    return ckpt['epoch'], ckpt['best_iou'], ckpt['patience_counter']


# =============================================================================
# Main Training Workflow
# =============================================================================

def main(args):
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path   = out_dir / "best.pth"
    resume_path = out_dir / "latest.pth"

    drive_dir = Path(args.drive_backup) if args.drive_backup else None
    if drive_dir:
        drive_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    train_ds = CropRowDataset(os.path.join(args.data_root, "train"),
                              split="train", augment=True)
    val_ds   = CropRowDataset(os.path.join(args.data_root, "val"),
                              split="val", augment=False)
    test_ds  = CropRowDataset(os.path.join(args.data_root, "test"),
                              split="test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"\nDataset sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    _, sample_mask, _ = train_ds[0]
    fg = sample_mask.float().mean().item()
    print(f"Sample fg ratio: {fg:.4f}  "
          f"{'[OK]' if fg > 0.005 else '[WARNING: very sparse, check LINE_WIDTH in dataset.py]'}")

    model     = BiSeNetV2(num_classes=2).to(device)
    criterion = CombinedLoss(pos_weight=args.pos_weight).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    max_iters = args.epochs * len(train_loader)
    scheduler = PolyLR(optimizer, max_iters=max_iters, power=0.9)
    scaler    = GradScaler()

    start_epoch      = 0
    best_iou         = 0.0
    patience_counter = 0

    if args.resume and resume_path.exists():
        start_epoch, best_iou, patience_counter = load_checkpoint(
            model, optimizer, scaler, str(resume_path), device)
        start_epoch += 1

    print(f"\n{'='*60}")
    print(f"Training: {args.epochs} epochs  bs={args.batch_size}  lr={args.lr}")
    print(f"Early stop: patience={args.patience}  min_delta={args.min_delta}")
    print(f"{'='*60}\n")

    history = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

    for epoch in range(start_epoch, args.epochs):
        ep_t0 = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"(best_iou={best_iou:.4f}  patience={patience_counter}/{args.patience})")

        t_loss, t_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            aux_weight=args.aux_weight, log_interval=args.log_interval)
        v_loss, v_iou = evaluate(model, val_loader, criterion, device)

        ep_time = time.time() - ep_t0
        print(f"  >> train loss={t_loss:.4f}  train IoU={t_iou:.4f}")
        print(f"  >> val   loss={v_loss:.4f}  val   IoU={v_iou:.4f}  ({ep_time:.0f}s)")

        history["train_loss"].append(t_loss)
        history["train_iou"].append(t_iou)
        history["val_loss"].append(v_loss)
        history["val_iou"].append(v_iou)

        ckpt_state = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "best_iou": best_iou, "patience_counter": patience_counter,
            "val_iou": v_iou,
        }

        if v_iou > best_iou + args.min_delta:
            best_iou         = v_iou
            patience_counter = 0
            ckpt_state["best_iou"] = best_iou
            save_checkpoint(ckpt_state, str(best_path))
            if drive_dir:
                shutil.copy(str(best_path), str(drive_dir / "best.pth"))
        else:
            patience_counter += 1
            print(f"  >> No improvement ({patience_counter}/{args.patience})")

        save_checkpoint(ckpt_state, str(resume_path))

        if drive_dir and (epoch + 1) % 5 == 0:
            ep_bak = drive_dir / f"epoch_{epoch+1:03d}.pth"
            shutil.copy(str(resume_path), str(ep_bak))
            print(f"  [backup] → {ep_bak}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}.  Best val IoU = {best_iou:.4f}")
            break
        print()

    # Final test set evaluation
    model.load_state_dict(torch.load(str(best_path), map_location=device)["model"])
    test_loss, test_iou = evaluate(model, test_loader, criterion, device)
    print(f"Test  loss={test_loss:.4f}  Test  IoU={test_iou:.4f}")

    # Inference speed test
    print("\nSpeed test (single image, GPU)...")
    model.eval()
    dummy = torch.randn(1, 3, 256, 512).to(device)
    for _ in range(10):  # warmup
        with torch.no_grad(), autocast():
            _ = model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad(), autocast():
            _ = model(dummy)
    torch.cuda.synchronize()
    ms_per_frame = (time.time() - t0) / 100 * 1000
    fps = 1000 / ms_per_frame
    print(f"  Model only : {ms_per_frame:.1f} ms/frame  ({fps:.1f} fps)")
    print(f"  Status     : {'[OK]' if fps > 20 else '[Needs optimization]'}")

    print(f"\n  Best val IoU  : {best_iou:.4f}  (target >= 0.90)")
    print(f"  Test  IoU     : {test_iou:.4f}")
    print(f"  Peak at epoch : {int(np.argmax(history['val_iou'])) + 1}")


def parse_args():
    p = argparse.ArgumentParser(description="BiSeNet-V2 crop row detection training")
    p.add_argument("--data_root",    type=str, default="/content/corn_augmented")
    p.add_argument("--out_dir",      type=str, default="/content/bisenet_ckpt")
    p.add_argument("--drive_backup", type=str, default=None,
                   help="Google Drive backup path (optional)")
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--pos_weight",   type=float, default=8.0)
    p.add_argument("--aux_weight",   type=float, default=0.4)
    p.add_argument("--patience",     type=int,   default=15)
    p.add_argument("--min_delta",    type=float, default=0.002)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--log_interval", type=int,   default=20)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
