# X-Ray Baggage Security Inspection — Deep Learning Project

Comparative study of deep learning approaches for automatic detection of prohibited objects in X-ray baggage images. The project evaluates three paradigms: a transfer-learning baseline (ResNet-18), fine-tuned state-of-the-art detectors (YOLO family + RT-DETR), and zero-shot semantic retrieval (CLIP). A Streamlit interface integrates all trained models for interactive inspection.

---

## Problem Statement

Manual X-ray baggage inspection is prone to human error due to cognitive fatigue and the visual complexity caused by overlapping dense objects in a single radiographic plane. This project develops and benchmarks automated detection models for five prohibited object categories: **gun, knife, pliers, scissors, wrench**.

---

## Dataset

**X-Ray Baggage Anomaly Detection** — available on [Kaggle](https://www.kaggle.com/datasets/orvile/x-ray-baggage-anomaly-detection)

| Split      | Images | Share |
|------------|--------|-------|
| Train      | 6,181  | 70%   |
| Validation | 1,766  | 20%   |
| Test       | 883    | 10%   |
| **Total**  | **8,830** | — |

- Fixed resolution: 416 × 416 px
- One annotated object per image (YOLO format)
- Average bounding box area: ~1.07% of the image (small object detection regime)
- Average brightness: 233.63 | RMS contrast: 39.61 (high-brightness, low-contrast scanner background)


---

## Methodology Overview

### Preprocessing

Both pipelines share a CLAHE step on the L* channel of CIE Lab space (clip limit 2.0, 8×8 grid) to counteract scanner background saturation.

| Pipeline | Method | Models |
|----------|--------|--------|
| **P1** — Unsharp Mask | `I = 1.5·I_clahe − 0.5·Gauss(I_clahe)` | YOLOv11n, YOLOv11s |
| **P2** — Bilateral + Morphological Gradient | `I = 0.9·I_bilateral + 0.1·G_morph` | YOLOv26n, RT-DETR, YOLOv11s (ablation) |

### Baseline — ResNet-18 with Transfer Learning

ResNet-18 pretrained on ImageNet acts as a frozen feature extractor. Two task heads are trained on top: a classification head (6 classes) and a bounding box regression head. The model evolved across three versions:

| Component | v1 | v2 | v3 |
|-----------|----|----|-----|
| Epochs | 10 | 15 | 15 |
| LR backbone / heads | 1e-3 (global) | 1e-5 / 1e-3 | 1e-5 / 1e-3 |
| Feature representation | GAP 512-d | GAP 512-d | Spatial reduction 2048-d |
| BBox loss | MSE + mask | GIoU + filter | CIoU + SmoothL1 + filter |
| λ | 2.0 | 4.0 | 5.0 |
| Augmentation | None | H-flip | H/V-flip, rotation, jitter |

The key architectural change in v3 is the `spatial_reduction` module (Conv2D 512→128 + BN + ReLU + AdaptiveAvgPool 4×4), which preserves a 4×4 spatial grid before flattening instead of collapsing to a single vector, giving the regression head localization cues.

### Pretrained Detectors — YOLO / RT-DETR

All models fine-tuned from COCO weights for 50 epochs (batch 16, patience 10, GPU NVIDIA T4, imgsz 416).

| Model | Pipeline | Augmentation |
|-------|----------|--------------|
| YOLOv11n | P1 | Full (mosaic p=1.0, HSV, geometric) |
| YOLOv11s | P1 | Full |
| YOLOv11s *(ablation)* | P2 | Reduced |
| YOLOv26n | P2 | Reduced |
| RT-DETR-L | P2 | Reduced |

### CLIP — Zero-Shot Semantic Retrieval

OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) with frozen weights used for two tasks:
- **Text-to-image retrieval**: class embeddings averaged over 5 prompt templates contrasted against a gallery of 7,947 annotated crops (train + val splits) indexed in Qdrant.
- **Free-form operator queries**: natural language descriptions encoded at runtime and matched against the same gallery.

---

## Usage

### Running a notebook

All notebooks are self-contained. Download the dataset automatically via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("orvile/x-ray-baggage-anomaly-detection")
```

Recommended execution order:

1. `baseline-detection-v1.ipynb`, v2, v3 (iterative baseline development)
2. `pretrained-models-detection_run_again_save_models.ipynb` (consolidated pretrained training)
3. `yolo_detection_pt2.ipynb` (YOLOv11s ablation on Pipeline 2)

---

## Front-end App

Interactive Streamlit console for inference across all trained models.

```bash
streamlit run app.py
```

Before launching, place all model checkpoints in a `BEST_MODELS/` directory at the repo root. The app auto-detects `.pt` (YOLO/RT-DETR) and `.pth` (baseline) files and infers the preprocessing pipeline from the filename.

**Filename convention for automatic pipeline detection:**

| Keyword in filename | Assigned pipeline |
|---------------------|-------------------|
| `p1`, `pipeline1`, `clahe_gauss`, `unsharp` | Pipeline 1 |
| `p2`, `pipeline2`, `bilateral`, `morph` | Pipeline 2 |
| `.pth` (baseline) | Pipeline 1 |
| Other `.pt` | Pipeline 2 (default) |

**Tabs:**

- **Pretrained Models** — YOLO / RT-DETR inference with ground truth comparison on the test set.
- **Baseline CNN** — ResNet-18 inference with predicted class, confidence score, and bounding box.
- **CLIP Zero-Shot** — Standard 5-class scoring or free-form text query with transformer attention heatmap.

All tabs share a synchronized image navigator (Prev / Next / Random) and support uploading external images.
