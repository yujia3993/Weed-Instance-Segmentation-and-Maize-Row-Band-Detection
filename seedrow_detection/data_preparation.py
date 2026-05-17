# =============================================================================
# Corn Seedling Row Detection -- Dataset Split and Offline Augmentation
#
# Pipeline:
#   1. Read images and X-AnyLabeling JSON annotations from source directory
#   2. Randomly split into train/val/test at 8:1:1
#   3. Generate 6 augmentation variants for training set
#      (±rotation, horizontal flip, perspective transform, and combinations)
#   4. Annotation keypoint coordinates are transformed synchronously;
#      out-of-bounds annotations are filtered automatically
#   5. Blank regions are filled with constant gray value (114, 114, 114)
# =============================================================================

import os
import json
import shutil
import random
import math
import copy
import cv2
import numpy as np
from pathlib import Path

# ---------- Path configuration ----------
DATASET_ROOT  = r"D:\Internship\corn seedlings_raw dataset\bagresult"
OUTPUT_ROOT   = r"D:\Internship\corn_augmented"
IMAGE_DIR     = DATASET_ROOT
LABEL_DIR     = DATASET_ROOT

TRAIN_RATIO          = 0.8
VAL_RATIO            = 0.1
TEST_RATIO           = 0.1
RANDOM_SEED          = 42
ROTATION_ANGLES      = [-20, -15, -10, 10, 15, 20]
PERSPECTIVE_STRENGTH = 0.08

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# =============================================================================
# Utility Functions
# =============================================================================

def find_pairs(image_dir: str, label_dir: str) -> list:
    pairs = []
    for f in sorted(os.listdir(image_dir)):
        stem = Path(f).stem
        ext  = Path(f).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        json_path = os.path.join(label_dir, stem + ".json")
        if os.path.exists(json_path):
            pairs.append((os.path.join(image_dir, f), json_path))
        else:
            print(f"Annotation file not found, skipping: {f}")
    return pairs


def split_dataset(pairs: list, train_r: float, val_r: float, seed: int):
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def transform_points(points: list, M: np.ndarray) -> list:
    if not points:
        return points
    pts   = np.array(points, dtype=np.float64)
    ones  = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    trans = (M @ pts_h.T).T
    w     = trans[:, 2:3]
    return (trans[:, :2] / w).tolist()


def update_json_points(anno: dict, M: np.ndarray,
                       new_image_name: str, new_h: int, new_w: int) -> dict:
    new_anno = copy.deepcopy(anno)
    new_anno["imagePath"]   = new_image_name
    new_anno["imageHeight"] = new_h
    new_anno["imageWidth"]  = new_w
    new_anno["imageData"]   = None
    for shape in new_anno["shapes"]:
        shape["points"] = transform_points(shape["points"], M)
    # Filter out-of-bounds annotations
    new_anno["shapes"] = [
        s for s in new_anno["shapes"]
        if all(0 <= p[0] <= new_w and 0 <= p[1] <= new_h
               for p in s["points"])
    ]
    return new_anno


# =============================================================================
# Augmentation Functions (return aug_img, 3×3 transform matrix, new_h, new_w)
# =============================================================================

def augment_rotate(img: np.ndarray, angle: float):
    h, w  = img.shape[:2]
    cx, cy = w / 2, h / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos_a = abs(math.cos(math.radians(angle)))
    sin_a = abs(math.sin(math.radians(angle)))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M_rot[0, 2] += (new_w - w) / 2
    M_rot[1, 2] += (new_h - h) / 2
    M3 = np.vstack([M_rot, [0, 0, 1]])
    aug_img = cv2.warpAffine(img, M_rot, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(114, 114, 114))
    return aug_img, M3, new_h, new_w


def augment_flip_horizontal(img: np.ndarray):
    h, w   = img.shape[:2]
    aug_img = cv2.flip(img, 1)
    M = np.array([[-1, 0, w - 1],
                  [ 0, 1, 0    ],
                  [ 0, 0, 1    ]], dtype=np.float64)
    return aug_img, M, h, w


def augment_perspective(img: np.ndarray, strength: float):
    h, w = img.shape[:2]
    d    = strength
    rng  = np.random.default_rng()

    def jitter():
        return rng.uniform(-d, d)

    src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst = np.float32([
        [jitter() * w,       jitter() * h      ],
        [(1+jitter()) * w,   jitter() * h      ],
        [(1+jitter()) * w,   (1+jitter()) * h  ],
        [jitter() * w,       (1+jitter()) * h  ],
    ])
    dst[:, 0] = np.clip(dst[:, 0], -w * d, w * (1 + d))
    dst[:, 1] = np.clip(dst[:, 1], -h * d, h * (1 + d))

    M = cv2.getPerspectiveTransform(src, dst)
    corners    = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    new_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    x_min, y_min = new_corners.min(axis=0)
    x_max, y_max = new_corners.max(axis=0)
    new_w = int(x_max - x_min) + 1
    new_h = int(y_max - y_min) + 1

    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1     ]], dtype=np.float64)
    M_full = T @ M
    aug_img = cv2.warpPerspective(img, M_full, (new_w, new_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(114, 114, 114))
    return aug_img, M_full, new_h, new_w


# =============================================================================
# Visualization Utilities
# =============================================================================

def draw_annotation(img: np.ndarray, anno: dict) -> np.ndarray:
    vis    = img.copy()
    colors = [(255,100,100),(100,255,100),(100,100,255),
              (255,255,0),(0,255,255),(255,0,255)]
    for i, shape in enumerate(anno["shapes"]):
        pts   = [(int(p[0]), int(p[1])) for p in shape["points"]]
        color = colors[i % len(colors)]
        for j in range(len(pts) - 1):
            cv2.line(vis, pts[j], pts[j+1], color, 2)
        for pt in pts:
            cv2.circle(vis, pt, 5, color, -1)
    return vis


# =============================================================================
# Dataset Operations
# =============================================================================

def copy_split(pairs: list, split_name: str, out_root: str):
    img_out = os.path.join(out_root, split_name, "images")
    lbl_out = os.path.join(out_root, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img_path, json_path in pairs:
        shutil.copy2(img_path, os.path.join(img_out, os.path.basename(img_path)))
        shutil.copy2(json_path, os.path.join(lbl_out, os.path.basename(json_path)))
    print(f"  {split_name}: {len(pairs)} images copied")


def augment_train(train_pairs: list, out_root: str):
    img_out  = os.path.join(out_root, "train", "images")
    lbl_out  = os.path.join(out_root, "train", "labels")
    verify_out = os.path.join(out_root, "verify")
    os.makedirs(verify_out, exist_ok=True)

    verify_indices  = set(random.sample(range(len(train_pairs)),
                                        min(5, len(train_pairs))))
    total_generated = 0

    for idx, (img_path, json_path) in enumerate(train_pairs):
        img  = cv2.imread(img_path)
        anno = load_json(json_path)
        stem = Path(img_path).stem
        ext  = Path(img_path).suffix

        angle1 = random.choice(ROTATION_ANGLES[:3])
        angle2 = random.choice(ROTATION_ANGLES[3:])
        angle3 = random.choice(ROTATION_ANGLES)

        augmentations = [
            ("rot1",     lambda i: augment_rotate(i, angle1)),
            ("rot2",     lambda i: augment_rotate(i, angle2)),
            ("flip",     lambda i: augment_flip_horizontal(i)),
            ("persp",    lambda i: augment_perspective(i, PERSPECTIVE_STRENGTH)),
            ("rot3",     lambda i: augment_rotate(i, angle3)),
            ("rot_flip", None),
        ]

        do_verify = idx in verify_indices

        for aug_name, aug_fn in augmentations:
            if aug_name == "rot_flip":
                tmp_img, M1, h1, w1 = augment_rotate(img, random.choice(ROTATION_ANGLES))
                aug_img, M2, new_h, new_w = augment_flip_horizontal(tmp_img)
                M = M2 @ M1
            else:
                aug_img, M, new_h, new_w = aug_fn(img)

            new_stem      = f"{stem}_{aug_name}"
            new_img_name  = new_stem + ext
            new_json_name = new_stem + ".json"

            cv2.imwrite(os.path.join(img_out, new_img_name), aug_img)

            new_anno = update_json_points(anno, M, new_img_name, new_h, new_w)
            save_json(new_anno, os.path.join(lbl_out, new_json_name))

            if do_verify:
                vis = draw_annotation(aug_img, new_anno)
                cv2.imwrite(os.path.join(verify_out, f"{new_stem}_verify.jpg"), vis)

            total_generated += 1

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(train_pairs)} original images")

    print(f"Generated {total_generated} augmented images (+{len(train_pairs)} originals)")
    print(f"Training set total: {total_generated + len(train_pairs)} images")


# =============================================================================
# Main Entry
# =============================================================================

def main():
    pairs = find_pairs(IMAGE_DIR, LABEL_DIR)
    print(f"Found {len(pairs)} image-annotation pairs")

    train, val, test = split_dataset(pairs, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    print(f"Split result: train={len(train)}, val={len(val)}, test={len(test)}")

    copy_split(val,   "val",   OUTPUT_ROOT)
    copy_split(test,  "test",  OUTPUT_ROOT)
    copy_split(train, "train", OUTPUT_ROOT)

    augment_train(train, OUTPUT_ROOT)

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT_ROOT, split, "images")
        count   = len([f for f in os.listdir(img_dir)
                       if Path(f).suffix.lower() in IMAGE_EXTENSIONS])
        print(f"  {split}/images: {count} images")


if __name__ == "__main__":
    main()
