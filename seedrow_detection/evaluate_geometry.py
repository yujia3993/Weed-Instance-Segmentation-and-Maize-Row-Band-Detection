# =============================================================================
# Corn Seedling Row Detection -- Geometry Post-Processing Evaluation
#
# Pipeline: BiSeNet-V2 inference → binary mask → column projection peaks
#           → per-row centroid → RANSAC line fitting
#           → adjacent line merging → restore original image coordinates
#           → match with GT lines → angle/distance errors
#
# Usage:
#   python evaluate_geometry.py
# =============================================================================

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor

from dataset import CropRowDataset, resize_pad, INPUT_W, INPUT_H
from model import BiSeNetV2

# ---------- Path configuration ----------
CKPT_PATH = "/content/bisenet_ckpt_lw40/best.pth"
TEST_ROOT = "/content/test"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIS_DIR   = Path("/content/geo_vis")
VIS_DIR.mkdir(exist_ok=True)

# ---------- Post-processing parameters ----------
PEAK_MIN_DIST    = 50
PEAK_PROMINENCE  = 5
PEAK_HEIGHT      = 3
RANSAC_RESIDUAL  = 8
MIN_POINTS       = 5
MATCH_DIST_THRESH = 50  # px (original image size)

# ---------- Normalization constants (consistent with dataset.py) ----------
_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
_STD  = np.array([0.225, 0.224, 0.229], dtype=np.float32)


# =============================================================================
# GT Parsing
# =============================================================================

def load_gt_lines(json_path: str) -> list:
    if not os.path.exists(json_path):
        return []
    with open(json_path) as f:
        data = json.load(f)

    lines = []
    for shape in data.get("shapes", []):
        if shape.get("label") != "crop_line":
            continue
        pts = np.array(shape["points"], dtype=np.float32)
        if len(pts) < 2:
            continue

        if len(pts) == 2:
            dy = pts[1][1] - pts[0][1]
            dx = pts[1][0] - pts[0][0]
            k  = dx / dy if abs(dy) > 1e-6 else 0.0
            b  = pts[0][0] - k * pts[0][1]
        else:
            Y = pts[:, 1].reshape(-1, 1)
            X = pts[:, 0]
            try:
                reg = RANSACRegressor(residual_threshold=15, random_state=0)
                reg.fit(Y, X)
                k = float(reg.estimator_.coef_[0])
                b = float(reg.estimator_.intercept_)
            except Exception:
                k = 0.0
                b = float(pts[:, 0].mean())

        mid_y = float(pts[:, 1].mean())
        mid_x = k * mid_y + b
        lines.append({"k": k, "b": b, "pts": pts, "mid": (mid_x, mid_y)})

    return lines


# =============================================================================
# Geometry Post-Processing: mask → list of crop row lines
# =============================================================================

def mask_to_lines(mask: np.ndarray) -> list:
    H, W = mask.shape
    col_proj = mask.sum(axis=0).astype(np.float32)

    peaks, _ = find_peaks(col_proj, distance=PEAK_MIN_DIST,
                          prominence=PEAK_PROMINENCE, height=PEAK_HEIGHT)
    if len(peaks) == 0:
        return []

    boundaries = [0]
    for i in range(len(peaks) - 1):
        boundaries.append((peaks[i] + peaks[i+1]) // 2)
    boundaries.append(W)

    lines = []
    for idx, peak_x in enumerate(peaks):
        x_lo = boundaries[idx]
        x_hi = boundaries[idx + 1]

        row_pts = []
        for y in range(H):
            row_fg = np.where(mask[y, x_lo:x_hi] > 0)[0]
            if len(row_fg) == 0:
                continue
            row_pts.append([float(row_fg.mean()) + x_lo, float(y)])

        if len(row_pts) < MIN_POINTS:
            continue

        pts = np.array(row_pts, dtype=np.float32)
        Y   = pts[:, 1].reshape(-1, 1)
        X   = pts[:, 0]
        try:
            reg = RANSACRegressor(
                residual_threshold=RANSAC_RESIDUAL,
                min_samples=max(MIN_POINTS, int(len(pts) * 0.3)),
                random_state=0)
            reg.fit(Y, X)
            k = float(reg.estimator_.coef_[0])
            b = float(reg.estimator_.intercept_)
        except Exception:
            k, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
            k, b = float(k), float(b)

        mid_y = float(pts[:, 1].mean())
        mid_x = k * mid_y + b
        lines.append({"k": k, "b": b, "pts": pts, "mid": (mid_x, mid_y)})

    return _merge_close_lines(lines, min_x_gap=40)


def _merge_close_lines(lines: list, min_x_gap: int = 40) -> list:
    if len(lines) <= 1:
        return lines
    lines  = sorted(lines, key=lambda l: l["mid"][0])
    merged = [lines[0]]
    for cur in lines[1:]:
        prev = merged[-1]
        if abs(cur["mid"][0] - prev["mid"][0]) < min_x_gap:
            merged[-1] = {
                "k":   (prev["k"] + cur["k"]) / 2,
                "b":   (prev["b"] + cur["b"]) / 2,
                "mid": ((prev["mid"][0] + cur["mid"][0]) / 2,
                        (prev["mid"][1] + cur["mid"][1]) / 2),
            }
        else:
            merged.append(cur)
    return merged


def scale_line_to_original(line: dict, scale: float,
                            pad_top: int, pad_left: int) -> dict:
    k      = line["k"]
    b_orig = (line["b"] - pad_left + k * pad_top) / scale
    mid_x  = (line["mid"][0] - pad_left) / scale
    mid_y  = (line["mid"][1] - pad_top)  / scale
    return {"k": k, "b": b_orig, "mid": (mid_x, mid_y)}


# =============================================================================
# Evaluation Metrics
# =============================================================================

def line_angle_deg(k: float) -> float:
    return float(np.degrees(np.arctan(abs(k))))


def point_to_line_dist(px, py, k, b) -> float:
    return abs(px - k * py - b) / np.sqrt(1 + k ** 2)


def match_and_evaluate(pred_lines: list, gt_lines: list) -> tuple:
    used_pred    = set()
    detected     = 0
    angle_errors = []
    dist_errors  = []

    for gt in gt_lines:
        gx, gy    = gt["mid"]
        best_dist = float("inf")
        best_idx  = -1

        for i, pred in enumerate(pred_lines):
            if i in used_pred:
                continue
            px, py = pred["mid"]
            d = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        if best_idx >= 0 and best_dist < MATCH_DIST_THRESH:
            detected += 1
            used_pred.add(best_idx)
            pred = pred_lines[best_idx]
            angle_errors.append(abs(line_angle_deg(gt["k"]) - line_angle_deg(pred["k"])))
            dist_errors.append(point_to_line_dist(gx, gy, pred["k"], pred["b"]))

    return detected, angle_errors, dist_errors


# =============================================================================
# Visualization
# =============================================================================

def draw_lines_on_image(img: np.ndarray, lines: list,
                        color, thickness: int = 2) -> np.ndarray:
    H, W = img.shape[:2]
    for line in lines:
        k, b = line["k"], line["b"]
        x0   = int(k * 0 + b)
        x1   = int(k * (H - 1) + b)
        cv2.line(img, (x0, 0), (x1, H - 1), color, thickness)
    return img


# =============================================================================
# Main Evaluation Workflow
# =============================================================================

def main():
    print("Loading model...")
    model = BiSeNetV2(num_classes=2).to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img_dir   = Path(TEST_ROOT) / "images"
    lbl_dir   = Path(TEST_ROOT) / "labels"
    img_paths = sorted(img_dir.glob("*"))
    print(f"Test images: {len(img_paths)}")

    total_gt       = 0
    total_detected = 0
    all_angle_errs = []
    all_dist_errs  = []
    inference_times = []

    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".json")

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue
        ih, iw = img_orig.shape[:2]

        img_resized, _, scale, pad_top, pad_left = resize_pad(
            img_orig, None, INPUT_W, INPUT_H)

        img_t = (img_resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        img_t = (torch.from_numpy(img_t).permute(2, 0, 1)
                 .unsqueeze(0).float().to(DEVICE))

        t0 = time.time()
        with torch.no_grad():
            out = model(img_t)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        inference_times.append((time.time() - t0) * 1000)

        mask = out.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        pred_lines_small = mask_to_lines(mask)
        pred_lines = [
            scale_line_to_original(l, scale, pad_top, pad_left)
            for l in pred_lines_small
        ]
        gt_lines = load_gt_lines(str(lbl_path))

        detected, angle_errs, dist_errs = match_and_evaluate(pred_lines, gt_lines)
        total_gt       += len(gt_lines)
        total_detected += detected
        all_angle_errs.extend(angle_errs)
        all_dist_errs.extend(dist_errs)

        if len(inference_times) <= 10:
            vis = img_orig.copy()
            draw_lines_on_image(vis, gt_lines,   color=(0, 255, 0),   thickness=2)
            draw_lines_on_image(vis, pred_lines, color=(0, 100, 255), thickness=2)
            vis_small = cv2.resize(vis, (960, int(ih * 960 / iw)))
            cv2.imwrite(str(VIS_DIR / f"{img_path.stem}_geo.jpg"), vis_small)

    det_rate   = total_detected / total_gt if total_gt > 0 else 0
    mean_angle = np.mean(all_angle_errs)  if all_angle_errs else float("nan")
    lt5_angle  = np.mean(np.array(all_angle_errs) < 5) if all_angle_errs else 0
    mean_dist  = np.mean(all_dist_errs)   if all_dist_errs  else float("nan")
    mean_inf   = np.mean(inference_times)
    fps        = 1000 / mean_inf

    print("\n" + "=" * 55)
    print("Geometry Post-Processing Evaluation Results (Test Set)")
    print("=" * 55)
    print(f"Total GT crop rows  : {total_gt}")
    print(f"Detected            : {total_detected}")
    print(f"Detection rate      : {det_rate*100:.1f}%")
    print(f"Mean angle error    : {mean_angle:.2f} deg")
    print(f"Angle error < 5 deg : {lt5_angle*100:.1f}%")
    print(f"Mean distance error : {mean_dist:.1f} px (original size)")
    print(f"Inference speed     : {mean_inf:.1f} ms/frame  ({fps:.1f} fps)")

    print("\n" + "-" * 55)
    print("Comparison with Traditional Baseline")
    print("-" * 55)
    print(f"{'Metric':<18} {'Traditional':>14} {'BiSeNet-V2':>14}")
    print(f"{'Detection rate':<18} {'57.6%':>14} {det_rate*100:>13.1f}%")
    print(f"{'Mean angle error':<18} {'18.14 deg':>14} {mean_angle:>10.2f} deg")
    print(f"{'Angle < 5 deg':<18} {'18.4%':>14} {lt5_angle*100:>13.1f}%")
    print(f"{'Mean dist error':<18} {'70.6 px':>14} {mean_dist:>12.1f} px")
    print(f"{'Inference speed':<18} {'5.5 fps':>14} {fps:>12.1f} fps")
    print("=" * 55)


if __name__ == "__main__":
    main()
