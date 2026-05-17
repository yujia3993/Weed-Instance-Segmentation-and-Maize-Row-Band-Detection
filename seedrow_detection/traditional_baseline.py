# =============================================================================
# Corn Seedling Row Detection -- Traditional Method Baseline
#
# Pipeline:
#   1. ExG vegetation extraction → binary mask
#   2. Column projection → peak detection (find crop row x-coordinates)
#   3. Per row: compute centroid → RANSAC line fitting x=k*y+b
#   4. Match with GT polylines, compute angle and distance errors
#
# Evaluation metrics:
#   - Angle error (deg): mean angle between predicted line and GT polyline segments
#   - Mean point distance error (px): mean distance from GT polyline sample points to predicted line
#   - Detection rate: fraction of GT rows successfully matched
#   - Processing speed: ms/frame
# =============================================================================

import os
import json
import time
import cv2
import numpy as np
import warnings
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor

warnings.filterwarnings("ignore")

# ---------- Path configuration ----------
TEST_DIR   = r"D:\Internship\corn_augmented\test"
OUTPUT_DIR = r"D:\Internship\baseline_results"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ---------- Algorithm parameters ----------
EXG_THRESHOLD     = 20
PEAK_MIN_DISTANCE = 60
PEAK_MIN_HEIGHT   = 0.08
MAX_ROWS          = 6
RANSAC_RESIDUAL   = 20
RANSAC_MIN_SAMPLES = 2
MATCH_THRESHOLD_PX = 100

SAVE_VIS        = True
VIS_SAMPLE_COUNT = 20


# =============================================================================
# Utility Functions
# =============================================================================

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_exg(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    exg   = 2.0 * img_f[:, :, 1] - img_f[:, :, 0] - img_f[:, :, 2]
    exg   = np.clip(exg, 0, None)
    max_val = exg.max()
    if max_val > 0:
        exg = (exg / max_val * 255).astype(np.uint8)
    return exg


def exg_mask(img: np.ndarray, threshold: int) -> np.ndarray:
    exg = compute_exg(img)
    _, mask = cv2.threshold(exg, threshold, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def column_projection(mask: np.ndarray) -> np.ndarray:
    proj  = mask.sum(axis=0) / 255.0
    max_v = proj.max()
    if max_v > 0:
        proj = proj / max_v
    return proj


def detect_row_positions(proj: np.ndarray,
                         min_distance: int, min_height: float) -> list:
    peaks, _ = find_peaks(proj, distance=min_distance, height=min_height)
    return peaks.tolist()


def collect_fg_points_near_x(mask: np.ndarray,
                              center_x: int, half_width: int = 20) -> list:
    h, w   = mask.shape
    x_min  = max(0, center_x - half_width)
    x_max  = min(w - 1, center_x + half_width)
    strip  = mask[:, x_min:x_max + 1]
    ys, xs_rel = np.where(strip > 0)
    xs = xs_rel + x_min
    return list(zip(xs.tolist(), ys.tolist()))


def fit_line_ransac(points_xy: list, residual_threshold: float,
                    min_samples: int):
    if len(points_xy) < min_samples:
        return None
    pts = np.array(points_xy)
    X   = pts[:, 1].reshape(-1, 1)  # y as feature
    Y   = pts[:, 0]                 # x as target
    try:
        ransac = RANSACRegressor(residual_threshold=residual_threshold,
                                 min_samples=min_samples, random_state=42)
        ransac.fit(X, Y)
        k = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        return (k, b)
    except Exception:
        return None


def line_angle_deg(k: float) -> float:
    return np.degrees(np.arctan(abs(k)))


def point_to_line_dist(px: float, py: float, k: float, b: float) -> float:
    return abs(k * py - px + b) / np.sqrt(k ** 2 + 1)


def gt_polyline_to_segments(points: list) -> list:
    return [(tuple(points[i]), tuple(points[i+1]))
            for i in range(len(points) - 1)]


def segment_angle_deg(p1, p2) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dy) < 1e-6:
        return 90.0
    return np.degrees(np.arctan(abs(dx / dy)))


def sample_polyline_points(points: list, n_samples: int = 20) -> list:
    if len(points) < 2:
        return [tuple(points[0])] if points else []
    segs      = []
    total_len = 0
    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i+1])
        l  = np.linalg.norm(p2 - p1)
        segs.append((p1, p2, l))
        total_len += l
    if total_len < 1e-6:
        return [tuple(points[0])]

    step    = total_len / (n_samples - 1)
    sampled = []
    seg_idx = 0
    seg_acc = 0
    for s in range(n_samples):
        target = s * step
        while seg_idx < len(segs) - 1 and seg_acc + segs[seg_idx][2] < target:
            seg_acc += segs[seg_idx][2]
            seg_idx += 1
        p1, p2, l = segs[seg_idx]
        if l < 1e-6:
            sampled.append(tuple(p1))
        else:
            t  = np.clip((target - seg_acc) / l, 0, 1)
            pt = p1 + t * (p2 - p1)
            sampled.append(tuple(pt))
    return sampled


def gt_mean_x_at_mid(points: list, h: int) -> float:
    mid_y = h / 2
    pts   = np.array(points)
    for i in range(len(pts) - 1):
        y1, y2 = pts[i][1], pts[i+1][1]
        if min(y1, y2) <= mid_y <= max(y1, y2):
            if abs(y2 - y1) < 1e-6:
                return (pts[i][0] + pts[i+1][0]) / 2
            t = (mid_y - y1) / (y2 - y1)
            return pts[i][0] + t * (pts[i+1][0] - pts[i][0])
    if mid_y < pts[:, 1].min():
        return pts[np.argmin(pts[:, 1])][0]
    return pts[np.argmax(pts[:, 1])][0]


# =============================================================================
# Single Image Processing
# =============================================================================

def process_image(img_path: str, anno: dict) -> dict | None:
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]

    t_start = time.time()

    mask   = exg_mask(img, EXG_THRESHOLD)
    margin = int(w * 0.10)
    mask[:, :margin]      = 0
    mask[:, w - margin:]  = 0

    proj        = column_projection(mask)
    peak_xs_all = detect_row_positions(proj, PEAK_MIN_DISTANCE, PEAK_MIN_HEIGHT)

    if len(peak_xs_all) > MAX_ROWS:
        peak_heights = [proj[x] for x in peak_xs_all]
        top_indices  = np.argsort(peak_heights)[::-1][:MAX_ROWS]
        peak_xs      = [peak_xs_all[i] for i in sorted(top_indices)]
    else:
        peak_xs = peak_xs_all

    pred_lines = []
    for cx in peak_xs:
        pts_raw = collect_fg_points_near_x(mask, cx, half_width=40)
        if len(pts_raw) < 10:
            continue
        pts_arr       = np.array(pts_raw)
        representative = []
        for y_start in range(0, h, 20):
            row_pts = pts_arr[(pts_arr[:, 1] >= y_start) & (pts_arr[:, 1] < y_start + 20)]
            if len(row_pts) >= 2:
                representative.append((row_pts[:, 0].mean(), row_pts[:, 1].mean()))
        result_line = fit_line_ransac(representative, RANSAC_RESIDUAL, RANSAC_MIN_SAMPLES)
        if result_line is not None:
            k, b = result_line
            pred_lines.append((cx, k, b))

    elapsed_ms = (time.time() - t_start) * 1000

    gt_lines   = [s["points"] for s in anno["shapes"]
                  if s["shape_type"] == "linestrip" and len(s["points"]) >= 2]
    gt_mid_xs  = [gt_mean_x_at_mid(pts, h) for pts in gt_lines]

    matched_pairs = []
    used_pred     = set()
    for gi, gt_mx in enumerate(gt_mid_xs):
        best_dist = MATCH_THRESHOLD_PX
        best_pi   = -1
        for pi, (cx, k, b) in enumerate(pred_lines):
            if pi in used_pred:
                continue
            d = abs(cx - gt_mx)
            if d < best_dist:
                best_dist = d
                best_pi   = pi
        if best_pi >= 0:
            matched_pairs.append((pred_lines[best_pi], gt_lines[gi]))
            used_pred.add(best_pi)

    angle_errors, dist_errors = [], []
    for (cx, k, b), gt_pts in matched_pairs:
        pred_angle = line_angle_deg(k)
        segs       = gt_polyline_to_segments(gt_pts)
        gt_angle   = np.mean([segment_angle_deg(p1, p2) for p1, p2 in segs])
        angle_errors.append(abs(pred_angle - gt_angle))

        sampled = sample_polyline_points(gt_pts, n_samples=20)
        dist_errors.append(np.mean([point_to_line_dist(px, py, k, b)
                                    for px, py in sampled]))

    return {
        "img_path"    : img_path,
        "img"         : img,
        "mask"        : mask,
        "proj"        : proj,
        "pred_lines"  : pred_lines,
        "gt_lines"    : gt_lines,
        "matched"     : matched_pairs,
        "angle_errors": angle_errors,
        "dist_errors" : dist_errors,
        "n_gt"        : len(gt_lines),
        "n_matched"   : len(matched_pairs),
        "elapsed_ms"  : elapsed_ms,
        "h": h, "w": w,
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_result(result: dict) -> np.ndarray:
    img = result["img"].copy()
    h, w = result["h"], result["w"]

    for gt_pts in result["gt_lines"]:
        pts = [(int(p[0]), int(p[1])) for p in gt_pts]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i+1], (0, 220, 0), 2)
        for pt in pts:
            cv2.circle(img, pt, 5, (0, 220, 0), -1)

    for cx, k, b in result["pred_lines"]:
        y1, y2 = 0, h - 1
        x1 = int(np.clip(k * y1 + b, 0, w - 1))
        x2 = int(np.clip(k * y2 + b, 0, w - 1))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    ae  = np.mean(result["angle_errors"]) if result["angle_errors"] else float("nan")
    de  = np.mean(result["dist_errors"])  if result["dist_errors"]  else float("nan")
    ms  = result["elapsed_ms"]
    n_gt, n_matched = result["n_gt"], result["n_matched"]

    for i, t in enumerate([
        f"GT:{n_gt}  Matched:{n_matched}  Miss:{n_gt-n_matched}",
        f"Angle err: {ae:.1f} deg",
        f"Dist err:  {de:.1f} px",
        f"Speed: {ms:.1f} ms",
    ]):
        cv2.putText(img, t, (10, 25 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, t, (10, 25 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    return img


# =============================================================================
# Main Entry
# =============================================================================

def main():
    img_dir = os.path.join(TEST_DIR, "images")
    lbl_dir = os.path.join(TEST_DIR, "labels")
    vis_dir = os.path.join(OUTPUT_DIR, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if Path(f).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        json_path = os.path.join(lbl_dir, Path(f).stem + ".json")
        if os.path.exists(json_path):
            pairs.append((os.path.join(img_dir, f), json_path))

    print(f"Test set: {len(pairs)} images\n")

    all_angle_errors, all_dist_errors, all_speeds = [], [], []
    total_gt, total_matched = 0, 0
    vis_count = 0

    for idx, (img_path, json_path) in enumerate(pairs):
        anno   = load_json(json_path)
        result = process_image(img_path, anno)
        if result is None:
            print(f"Failed to read: {img_path}")
            continue

        all_angle_errors.extend(result["angle_errors"])
        all_dist_errors.extend(result["dist_errors"])
        all_speeds.append(result["elapsed_ms"])
        total_gt      += result["n_gt"]
        total_matched += result["n_matched"]

        if SAVE_VIS and vis_count < VIS_SAMPLE_COUNT:
            vis_img  = visualize_result(result)
            out_name = Path(img_path).stem + "_result.jpg"
            cv2.imencode(".jpg", vis_img)[1].tofile(
                os.path.join(vis_dir, out_name))
            vis_count += 1

        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(pairs)}")

    print("\n" + "=" * 50)
    print("Traditional Baseline Evaluation Results")
    print("=" * 50)
    print(f"Test images:          {len(pairs)}")
    print(f"Total GT crop rows:   {total_gt}")
    print(f"Matched crop rows:    {total_matched}")
    print(f"Detection rate:       {total_matched/total_gt*100:.1f}%  ({total_matched}/{total_gt})")
    print(f"Missed:               {total_gt - total_matched}")

    if all_angle_errors:
        ae = np.array(all_angle_errors)
        print(f"\nAngle error mean:     {ae.mean():.2f} deg")
        print(f"Angle error median:   {np.median(ae):.2f} deg")
        print(f"Angle error < 5 deg:  {(ae < 5).mean()*100:.1f}%")
        print(f"Angle error < 10 deg: {(ae < 10).mean()*100:.1f}%")

    if all_dist_errors:
        de = np.array(all_dist_errors)
        print(f"\nDistance error mean:  {de.mean():.1f} px")
        print(f"Distance error median:{np.median(de):.1f} px")

    sp = np.array(all_speeds)
    print(f"\nSpeed mean:           {sp.mean():.1f} ms/frame")
    print(f"Speed median:         {np.median(sp):.1f} ms/frame")
    print(f"\nVisualization saved to: {vis_dir}")
    print("=" * 50)

    result_txt = os.path.join(OUTPUT_DIR, "baseline_results.txt")
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(f"Test images: {len(pairs)}\n")
        f.write(f"Detection rate: {total_matched/total_gt*100:.1f}% ({total_matched}/{total_gt})\n")
        if all_angle_errors:
            ae = np.array(all_angle_errors)
            f.write(f"Angle error mean: {ae.mean():.2f} deg\n")
            f.write(f"Angle error median: {np.median(ae):.2f} deg\n")
            f.write(f"Angle error < 5 deg: {(ae < 5).mean()*100:.1f}%\n")
        if all_dist_errors:
            f.write(f"Distance error mean: {np.mean(all_dist_errors):.1f} px\n")
        f.write(f"Speed mean: {sp.mean():.1f} ms/frame\n")
    print(f"Numeric results saved: {result_txt}")


if __name__ == "__main__":
    main()
