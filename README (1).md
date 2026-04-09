#  Human Embryo Time-Lapse Classification

Multiclass image classification of human embryo developmental stages using transfer learning in PyTorch.

---

## Overview

This project classifies time-lapse microscopy images of human embryos into **16 biological developmental phases** — from second polar body extrusion (`tPB2`) all the way to hatching blastocyst (`tHB`). Five pre-trained CNN architectures are benchmarked, and a custom ordinal-aware loss function is used to penalize predictions that are far apart in developmental order.

---

## Dataset

**Source:** [Kaggle — embryo-dataset](https://www.kaggle.com/datasets/abhishekbuddiga06/embryo-dataset)

| Component | Details |
|---|---|
| Videos | 704 embryo time-lapse folders |
| Annotations | 704 corresponding CSV files (phase, start frame, end frame) |
| Image format | JPEG frames |
| Labels | 16 ordered developmental phases |

### Developmental Phase Order

```
tPB2 → tPNa → tPNf → t2 → t3 → t4 → t5 → t6 → t7 → t8 → t9+ → tM → tSB → tB → tEB → tHB
```

---

## Models

Five CNN architectures are trained and compared using transfer learning (pretrained ImageNet weights, frozen backbone, fine-tuned classifier head):

| Model | Total Params | Trainable Params |
|---|---|---|
| MobileNetV1 | 2,244,368 | 433,296 |
| MobileNetV2 | 2,244,368 | 433,296 |
| InceptionV3 | 24,388,352 | 4,667,152 |
| VGG16 | 134,326,096 | 16,850,960 |
| VGG19 | 139,635,792 | 16,850,960 |

---

## Custom Loss Function

A combined loss is used to exploit the **ordinal structure** of embryo development:

```
L_total = L_WCE + λ · L_ord
```

- **L_WCE** — Weighted Cross-Entropy (handles class imbalance)
- **L_ord** — Ordinal Phase Distance penalty (penalizes predictions that are further away in the developmental timeline)
- **λ = 2.0** (configurable)

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 128 |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Epochs per model | 5 |
| Early stopping patience | 3 |
| Optimizer | Adam |
| Seed | 42 |
| Hardware | Tesla P100-PCIE-16GB (GPU) |

**Frame sampling:** Every 3rd frame is used, capped at 2,000 frames per video, to reduce redundancy in time-lapse sequences.

**Class imbalance:** Handled via `WeightedRandomSampler` during training.

---

## Requirements

```bash
pip install torch torchvision torchsummary
```

### Key Libraries

- `torch`, `torchvision` — Model training
- `torchsummary` — Parameter inspection
- `Pillow` — Image loading (truncated image tolerance enabled)
- `scikit-learn` — Metrics (classification report, confusion matrix)
- `matplotlib`, `seaborn` — Visualization
- `pandas`, `numpy` — Data handling

---

## Project Structure

```
notebook.ipynb              # Main training & evaluation notebook
embryo_dataset/             # Time-lapse image frames (704 video folders)
embryo_dataset_annotations/ # Phase annotation CSVs (704 files)
```

---

## Evaluation

Models are evaluated using:
- Per-class classification report (precision, recall, F1)
- Confusion matrix
- Model parameter comparison chart

---

## Usage

1. Download the dataset from Kaggle and set `IMAGE_DIR` and `ANNOT_DIR` paths in the configuration cell.
2. Run all cells in order.
3. Results and confusion matrices are displayed inline after training completes.

---

## Notes

- `ImageFile.LOAD_TRUNCATED_IMAGES = True` is set to handle any corrupted frames in the dataset gracefully.
- The notebook is designed to run on Kaggle with GPU acceleration enabled.
- All five models can be trained sequentially by setting `MODELS_TO_TRAIN` in the config.
