# =============================================================================
# SAM Weed Instance Segmentation -- Zero-shot Inference (v1: pure bbox prompt, v2: bbox+stem)
#
# Install dependencies:
#   pip install git+https://github.com/facebookresearch/segment-anything.git
#   pip install opencv-python pycocotools matplotlib
#   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# =============================================================================

import os
import json
import csv
import math
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from collections import defaultdict

# ---------- Path configuration ----------
CHECKPOINT   = "sam_vit_l_0b3195.pth"
MODEL_TYPE   = "vit_l"
DATASET_DIR  = "/content/baseline_dataset_AB"
OUTPUT_DIR_V1 = "/content/sam_predictions"
OUTPUT_DIR_V2 = "/content/sam_predictions_v2"
ORIG_BBOX_DIR = "/content/drive/MyDrive/bboxes"  # Original CropAndWeed CSV directory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_IDS = {
    'cockspur_grass':     31,
    'black_nightshade':   42,
    'field_milk_thistle': 39,
    'meadow_grass':       48,
    'redroot_amaranth':   32,
    'white_goosefoot':    33,
}
MATCH_THRESH = 5  # bbox coordinate matching tolerance (px)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def load_gt_masks(mask_dir: str, img_name: str):
    gt_masks = []
    for fname in sorted(os.listdir(mask_dir)):
        if fname.startswith(img_name + '_inst') and fname.endswith('.png'):
            inst_id  = int(fname.replace(img_name + '_inst', '').replace('.png', ''))
            mask_arr = np.array(Image.open(os.path.join(mask_dir, fname)))
            gt_masks.append((inst_id, mask_arr > 0))
    return gt_masks


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def dist(c1, c2):
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


# =============================================================================
# Dataset Statistics
# =============================================================================

def check_dataset(base: str):
    for class_name in sorted(os.listdir(base)):
        img_dir  = os.path.join(base, class_name, 'images')
        bbox_dir = os.path.join(base, class_name, 'bboxes')
        mask_dir = os.path.join(base, class_name, 'masks')
        n_img  = len([f for f in os.listdir(img_dir)  if f.endswith('.jpg')]) if os.path.isdir(img_dir)  else 0
        n_bbox = len([f for f in os.listdir(bbox_dir) if f.endswith('.json')]) if os.path.isdir(bbox_dir) else 0
        n_mask = len([f for f in os.listdir(mask_dir) if f.endswith('.png')]) if os.path.isdir(mask_dir)  else 0
        print(f'{class_name:<22} images={n_img} bboxes={n_bbox} masks={n_mask}')


# =============================================================================
# Zero-shot v1: pure bbox prompt
#
# Uses pretrained SAM vit_l directly with GT bbox as the only prompt,
# multimask_output=False.
# Result: overall IoU = 0.668, used as the official baseline.
# =============================================================================

def run_zero_shot_v1(sam, predictor, dataset_dir: str, output_dir: str):
    class_ious  = defaultdict(list)
    class_stats = defaultdict(lambda: {'imgs': 0, 'instances': 0, 'no_gt': 0})

    for class_name in sorted(os.listdir(dataset_dir)):
        img_dir  = os.path.join(dataset_dir, class_name, 'images')
        bbox_dir = os.path.join(dataset_dir, class_name, 'bboxes')
        mask_dir = os.path.join(dataset_dir, class_name, 'masks')
        out_dir  = os.path.join(output_dir,  class_name)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(img_dir):
            continue

        imgs = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))
        print(f'\n{class_name} ({len(imgs)} images)...')

        for fname in imgs:
            img_name  = os.path.splitext(fname)[0]
            img_path  = os.path.join(img_dir,  fname)
            bbox_path = os.path.join(bbox_dir, img_name + '.json')

            if not os.path.exists(bbox_path):
                continue

            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(img_rgb)

            with open(bbox_path) as f:
                bbox_data = json.load(f)

            gt_masks = load_gt_masks(mask_dir, img_name)
            class_stats[class_name]['imgs'] += 1

            for idx, item in enumerate(bbox_data['bboxes'], start=1):
                x1, y1, x2, y2 = item['bbox']
                box = np.array([x1, y1, x2, y2])

                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
                pred_mask = masks[0].astype(bool)

                pred_png = (pred_mask * 255).astype(np.uint8)
                Image.fromarray(pred_png).save(
                    os.path.join(out_dir, f'{img_name}_inst{idx}_pred.png'))

                gt_match = [m for (i, m) in gt_masks if i == idx]
                if not gt_match:
                    class_stats[class_name]['no_gt'] += 1
                    continue

                iou = compute_iou(pred_mask, gt_match[0])
                class_ious[class_name].append(iou)
                class_stats[class_name]['instances'] += 1

        ious = class_ious[class_name]
        mean_iou = np.mean(ious) if ious else 0
        print(f'  mean IoU = {mean_iou:.4f}  ({len(ious)} instances)')

    _print_summary(class_ious)
    _plot_iou_distribution(class_ious, '/content/iou_distribution_v1.png')
    return class_ious


def _print_summary(class_ious: dict):
    print(f"\n{'='*55}")
    print(f"{'Class':<22} {'Instances':>6} {'mean IoU':>10} {'>=0.5 rate':>10} {'>=0.9 rate':>10}")
    print('-' * 55)
    all_ious = []
    for class_name in sorted(class_ious.keys()):
        ious     = class_ious[class_name]
        all_ious.extend(ious)
        mean     = np.mean(ious)
        above_50 = np.mean(np.array(ious) >= 0.5)
        above_90 = np.mean(np.array(ious) >= 0.9)
        print(f'{class_name:<22} {len(ious):>6} {mean:>10.4f} {above_50:>10.1%} {above_90:>10.1%}')
    print('-' * 55)
    print(f"{'Total':<22} {len(all_ious):>6} {np.mean(all_ious):>10.4f} "
          f"{np.mean(np.array(all_ious)>=0.5):>10.1%} "
          f"{np.mean(np.array(all_ious)>=0.9):>10.1%}")


def _plot_iou_distribution(class_ious: dict, save_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, class_name in enumerate(sorted(class_ious.keys())):
        ious = class_ious[class_name]
        axes[i].hist(ious, bins=20, range=(0, 1), color='steelblue', edgecolor='white')
        axes[i].axvline(np.mean(ious), color='red', linestyle='--',
                        label=f'mean={np.mean(ious):.3f}')
        axes[i].set_title(class_name)
        axes[i].set_xlabel('IoU')
        axes[i].set_ylabel('Count')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# =============================================================================
# Data Preprocessing: Write Stem Point Coordinates from Original CSV
#
# Reads stem point coordinates (Stem X, Stem Y) from the original CropAndWeed
# CSV files, matches them to bboxes, and writes them into the dataset JSON.
# =============================================================================

def bbox_match(b1, b2, thresh: int) -> bool:
    return all(abs(b1[i] - b2[i]) <= thresh for i in range(4))


def add_stem_points(baseline_dir: str, orig_bbox_dir: str,
                    label_ids: dict, match_thresh: int = 5):
    stats = {'matched': 0, 'no_stem': 0, 'no_csv': 0}

    for class_name, label_id in label_ids.items():
        bbox_dir = os.path.join(baseline_dir, class_name, 'bboxes')
        if not os.path.isdir(bbox_dir):
            print(f'[Skip] {class_name} bboxes directory not found')
            continue

        updated = 0
        for json_fname in sorted(os.listdir(bbox_dir)):
            if not json_fname.endswith('.json'):
                continue

            img_name  = os.path.splitext(json_fname)[0]
            json_path = os.path.join(bbox_dir, json_fname)
            csv_path  = os.path.join(orig_bbox_dir, img_name + '.csv')

            if not os.path.exists(csv_path):
                stats['no_csv'] += 1
                continue

            csv_bboxes = []
            with open(csv_path) as f:
                for row in csv.reader(f):
                    if len(row) < 7:
                        continue
                    try:
                        if int(row[4]) == label_id:
                            csv_bboxes.append({
                                'bbox': [int(row[0]), int(row[1]), int(row[2]), int(row[3])],
                                'stem': [float(row[5]), float(row[6])]
                            })
                    except Exception:
                        continue

            with open(json_path) as f:
                data = json.load(f)

            for item in data['bboxes']:
                matched = [c for c in csv_bboxes
                           if bbox_match(item['bbox'], c['bbox'], match_thresh)]
                if matched:
                    item['stem'] = matched[0]['stem']
                    stats['matched'] += 1
                else:
                    item['stem'] = None
                    stats['no_stem'] += 1

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            updated += 1

        print(f'{class_name}: {updated} json files updated')

    print(f'\nMatched: {stats["matched"]}  No stem: {stats["no_stem"]}  Missing csv: {stats["no_csv"]}')


# =============================================================================
# Stem Point Completion: Fill Missing Stems from Nearest Neighbor bbox
#
# Some classes like cockspur_grass have one bbox mapping to multiple connected
# components; the CSV only has one stem point, leaving others without stems.
# Strategy: for instances without stems in the same image, inherit the stem
# from the spatially nearest bbox that has one.
# =============================================================================

def complete_stem_points(baseline_dir: str):
    fixed_total = 0

    for class_name in sorted(os.listdir(baseline_dir)):
        bbox_dir = os.path.join(baseline_dir, class_name, 'bboxes')
        if not os.path.isdir(bbox_dir):
            continue

        fixed_class = 0
        for json_fname in sorted(os.listdir(bbox_dir)):
            if not json_fname.endswith('.json'):
                continue
            json_path = os.path.join(bbox_dir, json_fname)
            with open(json_path) as f:
                data = json.load(f)

            stemmed = [item for item in data['bboxes']
                       if item.get('stem') is not None]
            if not stemmed:
                continue

            changed = False
            for item in data['bboxes']:
                if item.get('stem') is not None:
                    continue
                c = bbox_center(item['bbox'])
                nearest = min(stemmed, key=lambda s: dist(c, bbox_center(s['bbox'])))
                item['stem'] = nearest['stem']
                changed      = True
                fixed_class += 1

            if changed:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        print(f'{class_name:<22} stem points filled: {fixed_class}')
        fixed_total += fixed_class

    print(f'\nTotal filled: {fixed_total}')


# =============================================================================
# Zero-shot v2: bbox + stem joint prompt
#
# Conclusion: overall IoU is lower than pure bbox (0.629 vs 0.668); stem points
# cannot compensate for loose bbox localization. Pure bbox prompt (v1) is used
# as the official baseline going forward.
# =============================================================================

def run_zero_shot_v2(sam, predictor, dataset_dir: str, output_dir: str,
                     v1_ious: dict):
    class_ious_v2 = defaultdict(list)
    prompt_stats  = defaultdict(lambda: {'with_stem': 0, 'bbox_only': 0})

    for class_name in sorted(os.listdir(dataset_dir)):
        img_dir  = os.path.join(dataset_dir, class_name, 'images')
        bbox_dir = os.path.join(dataset_dir, class_name, 'bboxes')
        mask_dir = os.path.join(dataset_dir, class_name, 'masks')
        out_dir  = os.path.join(output_dir, class_name)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(img_dir):
            continue

        imgs = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))
        print(f'\n{class_name} ({len(imgs)} images)...')

        for fname in imgs:
            img_name  = os.path.splitext(fname)[0]
            img_path  = os.path.join(img_dir,  fname)
            bbox_path = os.path.join(bbox_dir, img_name + '.json')
            if not os.path.exists(bbox_path):
                continue

            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(img_rgb)

            with open(bbox_path) as f:
                bbox_data = json.load(f)

            gt_masks = load_gt_masks(mask_dir, img_name)

            for idx, item in enumerate(bbox_data['bboxes'], start=1):
                box  = np.array(item['bbox'])
                stem = item.get('stem')

                if stem is not None:
                    point_coords = np.array([[stem[0], stem[1]]])
                    point_labels = np.array([1])  # 1=foreground point
                    masks, scores, _ = predictor.predict(
                        box=box,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False
                    )
                    prompt_stats[class_name]['with_stem'] += 1
                else:
                    masks, scores, _ = predictor.predict(
                        box=box, multimask_output=False)
                    prompt_stats[class_name]['bbox_only'] += 1

                pred_mask = masks[0].astype(bool)
                Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
                    os.path.join(out_dir, f'{img_name}_inst{idx}_pred.png'))

                gt_match = [m for (i, m) in gt_masks if i == idx]
                if not gt_match:
                    continue
                class_ious_v2[class_name].append(compute_iou(pred_mask, gt_match[0]))

        ious = class_ious_v2[class_name]
        print(f'  mean IoU = {np.mean(ious):.4f}  ({len(ious)} instances)'
              f'  with_stem={prompt_stats[class_name]["with_stem"]}'
              f'  bbox_only={prompt_stats[class_name]["bbox_only"]}')

    # v1 vs v2 comparison
    print(f"\n{'='*65}")
    print(f"{'Class':<22} {'Instances':>6} {'v1 IoU':>8} {'v2 IoU':>8} {'Delta':>8} {'>=0.9 rate':>10}")
    print('-' * 65)
    all_ious_v2 = []
    for class_name in sorted(class_ious_v2.keys()):
        ious     = class_ious_v2[class_name]
        all_ious_v2.extend(ious)
        mean     = np.mean(ious)
        delta    = mean - v1_ious.get(class_name, 0)
        above_90 = np.mean(np.array(ious) >= 0.9)
        sign     = '+' if delta >= 0 else ''
        print(f'{class_name:<22} {len(ious):>6} {v1_ious.get(class_name,0):>8.4f} '
              f'{mean:>8.4f} {sign}{delta:>7.4f} {above_90:>10.1%}')
    print('-' * 65)
    v1_total = np.mean(list(v1_ious.values()))
    v2_total = np.mean(all_ious_v2)
    print(f"{'Total':<22} {len(all_ious_v2):>6} {v1_total:>8.4f} "
          f"{v2_total:>8.4f} {'+' if v2_total>=v1_total else ''}{v2_total-v1_total:>7.4f}")

    return class_ious_v2


# =============================================================================
# Main Entry
# =============================================================================

if __name__ == '__main__':
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    print(f'SAM model loaded, device: {DEVICE}')

    # Dataset check
    check_dataset(DATASET_DIR)

    # Step 1: Zero-shot v1 inference
    print('\n===== Zero-shot v1: pure bbox prompt =====')
    v1_ious = run_zero_shot_v1(sam, predictor, DATASET_DIR, OUTPUT_DIR_V1)

    # Step 2: Write stem point coordinates (requires original CSV)
    print('\n===== Write stem point coordinates =====')
    add_stem_points(DATASET_DIR, ORIG_BBOX_DIR, LABEL_IDS, MATCH_THRESH)

    # Step 3: Stem point completion
    print('\n===== Stem point completion =====')
    complete_stem_points(DATASET_DIR)

    # Step 4: Zero-shot v2 inference (bbox + stem)
    # v1 reference values (for comparison print)
    v1_ref = {
        'black_nightshade':   0.7666,
        'cockspur_grass':     0.5793,
        'field_milk_thistle': 0.7713,
        'meadow_grass':       0.5192,
        'redroot_amaranth':   0.8360,
        'white_goosefoot':    0.8472,
    }
    print('\n===== Zero-shot v2: bbox + stem =====')
    run_zero_shot_v2(sam, predictor, DATASET_DIR, OUTPUT_DIR_V2, v1_ref)
