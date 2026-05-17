# =============================================================================
# SAM2 Weed Instance Segmentation -- Zero-shot Inference
#
# Uses SAM2 hiera_large pretrained weights with GT bboxes from the COCO-format
# test set as prompts. No training is performed; evaluation metric is COCO AP@0.5.
#
# Install dependencies:
#   pip install git+https://github.com/facebookresearch/sam2.git
#   pip install pycocotools
#   wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
# =============================================================================

import os
import json
import pathlib
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from collections import defaultdict
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------- Path configuration ----------
CKPT_DIR  = "/content/sam2_checkpoints"
CKPT_PATH = f"{CKPT_DIR}/sam2.1_hiera_large.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
COCO_DIR  = "/content/drive/MyDrive/coco_v3"
WORK_DIR  = "/content/drive/MyDrive/sam2_results"
DEVICE    = "cuda"

# Multi-path lookup (map Windows paths to Colab)
ROOT_MAP = {
    "D:\\实习\\baseline_dataset_AB"    : "/content/drive/MyDrive/baseline_dataset_AB",
    "D:\\实习\\dataset_sorted\\C_drop" : "/content/C_drop",
    "D:\\实习\\c_cropped"              : "/content/drive/MyDrive/c_cropped",
}

os.makedirs(WORK_DIR, exist_ok=True)


# =============================================================================
# Utility Functions
# =============================================================================

def resolve_path(root_win: str, file_name: str) -> str:
    colab_root = ROOT_MAP.get(root_win)
    if colab_root is None:
        raise ValueError(f"Unknown root mapping: {root_win}")
    rel = pathlib.PurePosixPath(pathlib.PureWindowsPath(file_name))
    return os.path.join(colab_root, str(rel))


def seg_to_mask(seg, H: int, W: int) -> np.ndarray:
    if isinstance(seg, dict):
        return coco_mask_util.decode(seg).astype(bool)
    mask = Image.new('L', (W, H), 0)
    for poly in seg:
        pts = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
        ImageDraw.Draw(mask).polygon(pts, fill=1)
    return np.array(mask, dtype=bool)


# =============================================================================
# Inference
# =============================================================================

def run_inference(predictor: SAM2ImagePredictor, coco_dir: str,
                  work_dir: str) -> str:
    with open(f"{coco_dir}/instances_test.json") as f:
        test_data = json.load(f)

    img_map = {img['id']: img for img in test_data['images']}
    ann_map = defaultdict(list)
    for ann in test_data['annotations']:
        ann_map[ann['image_id']].append(ann)

    results = []
    print(f"Total {len(img_map)} test images")

    for img_id, img_info in img_map.items():
        anns = ann_map.get(img_id, [])
        if not anns:
            continue

        img_path = resolve_path(img_info['root'], img_info['file_name'])
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Failed to read: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_info['height'], img_info['width']

        predictor.set_image(img_rgb)

        for ann in anns:
            x, y, w, h = ann['bbox']
            box = np.array([x, y, x + w, y + h], dtype=float)

            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    box=box, multimask_output=True)

            best_idx  = scores.argmax()
            pred_mask = masks[best_idx].astype(np.uint8)

            rle = coco_mask_util.encode(np.asfortranarray(pred_mask))
            rle['counts'] = rle['counts'].decode('utf-8')

            results.append({
                "image_id"    : img_id,
                "category_id" : ann['category_id'],
                "segmentation": rle,
                "score"       : float(scores[best_idx]),
            })

    pred_path = f"{work_dir}/sam2_test_predictions.json"
    with open(pred_path, 'w') as f:
        json.dump(results, f)
    print(f"Inference complete, {len(results)} predictions saved to: {pred_path}")
    return pred_path


# =============================================================================
# COCO Evaluation
# =============================================================================

def evaluate(coco_dir: str, pred_path: str):
    coco_gt   = COCO(f"{coco_dir}/instances_test.json")
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
# Main Entry
# =============================================================================

if __name__ == '__main__':
    sam2_model = build_sam2(SAM2_CFG, CKPT_PATH, device=DEVICE)
    predictor  = SAM2ImagePredictor(sam2_model)

    pred_path = run_inference(predictor, COCO_DIR, WORK_DIR)
    evaluate(COCO_DIR, pred_path)
