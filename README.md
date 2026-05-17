# Precision Agriculture Deep Learning — Weed Segmentation & Corn Row Detection

An 8-week research project applying deep learning to two precision agriculture tasks:

1. **Weed Instance Segmentation** — segment 6 weed species from field images using foundation models (SAM, SAM2) and Mask2Former
2. **Corn Seedling Row Detection** — detect crop rows from overhead imagery using BiSeNet-V2 with geometric post-processing

All experiments are designed to run on **Google Colab** (GPU required).

---

## Project Structure

```
├── weed_segmentation/
│   ├── sam_zero_shot.py       # SAM vit_l zero-shot inference (bbox / bbox+stem prompts)
│   ├── sam_finetune.py        # SAM decoder-only fine-tuning (Focal + Dice loss)
│   ├── sam2_zero_shot.py      # SAM2 hiera_large zero-shot inference (COCO AP)
│   ├── sam2_cls_head.py       # SAM2 fine-tune + 6-class MLP classification head
│   └── mask2former_train.py   # Mask2Former training via HuggingFace Transformers
│
├── seedrow_detection/
│   ├── data_preparation.py    # Dataset split (8:1:1) + offline augmentation (×7)
│   ├── dataset.py             # CropRowDataset — resize/pad + mask generation
│   ├── model.py               # BiSeNet-V2 architecture (from scratch)
│   ├── train.py               # Training pipeline — CE+Dice loss, Poly LR, early stop
│   ├── traditional_baseline.py # ExG → column projection → RANSAC baseline
│   └── evaluate_geometry.py   # BiSeNet-V2 inference + geometric post-processing eval
│
└── datasets/
    ├── baseline_dataset_AB/
    │   └── annotations/       # ✅ COCO JSON annotations (included)
    │       ├── instances_train.json
    │       ├── instances_val.json
    │       └── instances_test.json
    ├── c_cropped/             # ✅ bbox-expanded crop patches (included)
    ├── coco_v3/
    │   └── annotations/       # ✅ COCO JSON annotations (included)
    │       ├── instances_train.json
    │       ├── instances_val.json
    │       └── instances_test.json
    └── corn_augmented/
        ├── train/
        │   ├── images/        # ✅ 348 raw images included; augmented copies excluded by .gitignore
        │   └── labels/        # ✅ 348 raw JSON annotations included; augmented copies excluded
        ├── val/
        │   ├── images/        # ✅ included (43 images, no augmentation)
        │   └── labels/        # ✅ included
        └── test/
            ├── images/        # ✅ included (45 images, no augmentation)
            └── labels/        # ✅ included
```

> **Note**: CropAndWeed source images are not included in this repo due to size and licensing.
> Corn seedling augmented images (~2,000+) are excluded — run `data_preparation.py` to regenerate them.

---

## Dataset Setup

### Task 1 — Weed Segmentation (CropAndWeed)

The COCO-format annotation files are included in `datasets/coco_v3/annotations/`.
You only need to download the source images separately.

**Step 1 — Download CropAndWeed images**

```bash
# Official dataset page: https://github.com/cropandweed/cropandweed-dataset
# Download and unzip to datasets/cropandweed_images/
```

**Step 2 — Filter images for the 6 target classes**

The 6 classes used in this project and their Label IDs in the CropAndWeed CSV annotations:

| Class | Label ID |
|---|---|
| `cockspur_grass` | 31 |
| `redroot_amaranth` | 32 |
| `white_goosefoot` | 33 |
| `field_milk_thistle` | 39 |
| `black_nightshade` | 42 |
| `meadow_grass` | 69 |

CropAndWeed annotation format (one CSV per image in `bboxes/`):
```
Left, Top, Right, Bottom, Label ID, Stem X, Stem Y
```

**Step 3 — Reconstruct dataset directories**

After downloading, your directory layout should look like this before running any scripts:

```
datasets/
├── cropandweed_images/        # downloaded CropAndWeed source images
│   ├── 1of4/
│   ├── 2of4/
│   ├── 3of4/
│   └── 4of4/
├── cropandweed_annotations/   # downloaded CropAndWeed CSV annotations
│   └── bboxes/
├── baseline_dataset_AB/
│   ├── annotations/           # already in repo
│   └── images/                # symlink or copy filtered images here
├── c_cropped/                 # already in repo (self-generated crops)
└── coco_v3/
    ├── annotations/           # already in repo
    └── images/                # symlink or copy filtered images here
        ├── train/
        ├── val/
        └── test/
```

### Task 2 — Corn Row Detection

The raw images (348 train + 43 val + 45 test) and their annotations are included in `datasets/corn_augmented/`.
Augmented copies in `train/` are excluded from the repo via `.gitignore` — regenerate them by running:

```bash
python seedrow_detection/data_preparation.py
# Configure DATASET_ROOT and OUTPUT_ROOT at the top of the file first
```

This expands the 348 training images to ~2,400 using 6 augmentation types
(`_flip`, `_persp`, `_rot1`, `_rot2`, `_rot3`, `_rot_flip`) with synchronized label transforms.

---

## Task 1 — Weed Instance Segmentation

### Models & Results

| Method | Metric | Score | Notes |
|---|---|---|---|
| SAM vit_l zero-shot (bbox only) | mean IoU | 0.668 | Official baseline |
| SAM vit_l zero-shot (bbox + stem) | mean IoU | 0.629 | Stem point prompt hurt performance |
| SAM decoder fine-tune | mean IoU | 0.629 | Worse than zero-shot |
| Mask2Former (swin-small) | AP@0.5 | 11.6% | Overfitting issue |
| **SAM2 hiera_large zero-shot** | **AP@0.5** | **63.0%** | **Best result; broadleaf >90%** |
| Grounded SAM2 | AP@0.5 | 0.2% | DINO detection failed |
| SAM2 fine-tune + cls head | AP@0.5 | 0.1% | Classification head did not converge |

### Scripts

#### SAM Zero-Shot (`sam_zero_shot.py`)

Runs zero-shot inference with SAM `vit_l` using GT bounding boxes as prompts.

- **v1** — bbox-only prompt (`multimask_output=False`)
- **v2** — bbox + stem point prompt (stem coordinates parsed from original CropAndWeed CSV)

```python
# Main entry point
sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
predictor = SamPredictor(sam.to("cuda"))

v1_ious = run_zero_shot_v1(sam, predictor, dataset_dir, output_dir_v1)

add_stem_points(dataset_dir, orig_bbox_dir, LABEL_IDS)  # write stem coords to JSON
complete_stem_points(dataset_dir)                        # fill missing stem coords
run_zero_shot_v2(sam, predictor, dataset_dir, output_dir_v2, v1_ious)
```

**Dependencies:**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

#### SAM Decoder Fine-Tuning (`sam_finetune.py`)

Freezes the image encoder and prompt encoder; fine-tunes only the mask decoder.

- Loss: Focal Loss + Dice Loss (0.5 each)
- Optimizer: Adam + CosineAnnealingLR
- 20 epochs, batch size 4, lr 1e-4

```bash
python sam_finetune.py \
  --base_dir /path/to/baseline_dataset_AB \
  --checkpoint sam_vit_l_0b3195.pth \
  --model_type vit_l \
  --epochs 20 \
  --batch_size 4 \
  --out_dir sam_finetuned
```

#### SAM2 Zero-Shot (`sam2_zero_shot.py`)

Uses SAM2 `hiera_large` with GT bounding boxes as prompts; evaluated with COCO AP@0.5.

```python
sam2_model = build_sam2(SAM2_CFG, CKPT_PATH, device="cuda")
predictor  = SAM2ImagePredictor(sam2_model)

pred_path = run_inference(predictor, coco_dir, work_dir)
evaluate(coco_dir, pred_path)
```

**Dependencies:**
```bash
pip install git+https://github.com/facebookresearch/sam2.git pycocotools
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

#### SAM2 Fine-Tune + Classification Head (`sam2_cls_head.py`)

Jointly trains the SAM2 mask decoder and a 2-layer MLP classification head (256→128→6 classes).

- Pre-caches all image embeddings to speed up training (~10 min/epoch vs ~3 hr/epoch)
- Total loss: `mask_loss (Focal + Dice) + 0.2 × cls_loss (CrossEntropy)`
- Optimizer: AdamW with separate LRs — decoder 3e-4, cls head 1e-3
- 40 epochs, gradient accumulation × 2, early stop patience = 12

> ⚠️ **Known limitation**: the classification head did not converge on 8,687 instances.
> The mask decoder fine-tuning itself works (val mask AP@0.5 reaches 0.10 when evaluated with GT labels).
> See the report for details and the recommended decoupled alternative.

#### Mask2Former (`mask2former_train.py`)

Fine-tunes `facebook/mask2former-swin-small-coco-instance` from HuggingFace.

- Input: shortest edge 640, longest edge 800
- Augmentation: HorizontalFlip, RandomBrightnessContrast, HueSaturationValue, GaussianBlur
- Optimizer: AdamW (lr=5e-5) + 3-epoch warmup + Cosine Annealing

**Dependencies:**
```bash
pip install transformers==4.40.0 pycocotools albumentations
```

---

## Task 2 — Corn Seedling Row Detection

### Models & Results

| Method | Detection Rate | Angle Error | Distance Error | Speed |
|---|---|---|---|---|
| Traditional (ExG + RANSAC) | 57.6% | 18.14° | 70.6 px | ~5.5 fps |
| **BiSeNet-V2 + geometric post-process** | **58.2%** | **2.14°** | **8.6 px** | **56.5 fps** |

### Scripts

#### Step 1 — Data Preparation (`data_preparation.py`)

Splits the raw dataset (8:1:1) and generates 6 offline augmentations per training image.
Polyline keypoints are transformed in sync; out-of-bounds annotations are filtered.

```bash
python seedrow_detection/data_preparation.py
# Configure DATASET_ROOT and OUTPUT_ROOT at top of file
```

#### Step 2 — Training (`train.py`)

```bash
python seedrow_detection/train.py \
  --data_root /content/corn_augmented \
  --out_dir   /content/bisenet_ckpt

# Resume from checkpoint
python seedrow_detection/train.py \
  --data_root /content/corn_augmented \
  --out_dir   /content/bisenet_ckpt \
  --resume
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 60 | Maximum training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 0.01 | Initial learning rate |
| `--pos_weight` | 8.0 | CE positive class weight |
| `--patience` | 15 | Early stopping patience |

#### Step 3 — Geometric Evaluation (`evaluate_geometry.py`)

Pipeline: predicted mask → column projection → RANSAC line fitting → adjacent line merging → coordinate rescaling → GT matching

```bash
python seedrow_detection/evaluate_geometry.py
# Configure CKPT_PATH and TEST_ROOT at top of file
```

#### Traditional Baseline (`traditional_baseline.py`)

```bash
python seedrow_detection/traditional_baseline.py
# Configure TEST_DIR and OUTPUT_DIR at top of file
```

---

## Installation

```bash
pip install torch torchvision
pip install opencv-python numpy scipy scikit-learn matplotlib pillow pycocotools
```

**SAM:**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

**SAM2:**
```bash
pip install git+https://github.com/facebookresearch/sam2.git
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

**Mask2Former:**
```bash
pip install transformers==4.40.0 albumentations
```

---

## Path Configuration

All scripts use hardcoded paths at the top of each file. Update these before running:

| Variable | Example (Colab) |
|---|---|
| `COCO_DIR` | `/content/drive/MyDrive/coco_v3` |
| `CKPT_PATH` | `/content/drive/MyDrive/sam2_checkpoints/sam2.1_hiera_large.pt` |
| `data_root` | `/content/corn_augmented` |
| `DATASET_DIR` | `/content/baseline_dataset_AB` |

---

## Model Architecture — BiSeNet-V2

The implementation in `seedrow_detection/model.py` is built from scratch following the BiSeNet-V2 paper:

- **Detail Branch**: 3-stage strided conv → 1/8 resolution, 128 channels
- **Semantic Branch**: StemBlock + GE layers (S1/S2) + Context Embedding Block → 1/32 resolution
- **BGA Layer**: Bilateral Guided Aggregation fuses the two branches
- **4 Auxiliary Heads**: from semantic stages 3, 4, 5 (training only)

Total parameters: ~3.4 M. Inference speed: **56.5 fps** (input 256×512, single GPU).
