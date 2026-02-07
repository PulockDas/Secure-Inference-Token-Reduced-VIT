# Secure Inference Token-Reduced ViT – Project Context

## Ultimate Goal

Implement a **secure Vision Transformer inference pipeline** under CKKS homomorphic encryption (TenSEAL). Train a standard teacher ViT in plaintext, then train an HE-friendly student ViT via knowledge distillation. Run plaintext student first, then encrypted forward on a single image end-to-end.

---

## Technical Constraints (Student)

- **HE-friendly**: only + and * ops; no softmax, GELU, LayerNorm (LayerNorm → affine or removed)
- **Token reduction**: early learnable layer to reduce tokens; target K ∈ {197, 97, 49, 25}
- **Student**: smaller than teacher, trained via distillation
- **Logging**: epoch, wall time, loss, accuracy to CSV; best checkpoint; latency metrics for encrypted inference

---

## Tasks Overview

| Phase | Task | Status |
|-------|------|--------|
| 1 | Generic image dataset loader (train/val/test) | Done |
| 2 | Teacher ViT fine-tuning (timm ViT-B/16 @ 224) | Done |
| 3 | Token reduction module (learnable, preserve CLS) | Done |
| 4 | Activation function approximation (HE-friendly) | Pending |
| 5 | Student ViT with token reduction + approx activations | Pending |
| 6 | Student distillation (teacher → student) | Pending |
| 7 | Per-K student evaluation (K ∈ {197, 97, 49, 25}) | Pending |
| 8 | Encrypted forward (TenSEAL CKKS, single image) | Pending |

---

## What Is Already Done

- **Data**: `data/dataset.py` – `ImageDataset`, `get_dataloaders`, `get_lc25000_root` (LC25000 via kagglehub, public). Train/val/test split, 5 classes (lung + colon).
- **Teacher**: `models/teacher.py` – `get_teacher_vit(num_classes)` (ViT-B/16 @ 224 from timm). Fine-tuning in `training/train_teacher.py`; logs to CSV; best checkpoint by val acc.
- **Evaluation**: `load_teacher_checkpoint`, `evaluate_teacher` – load best teacher, run on test set, per-class accuracy.
- **Token reduction**: `models/token_reduction.py` – `TokenReduction(dim, num_output_tokens)`. Input (B,N,D)→output (B,K,D); preserves CLS; K−1 learnable queries cross-attend to patches.
- **Notebook**: `notebooks/main.ipynb` – clone repo, load data, train teacher, evaluate teacher. Designed for Colab; GPU if available; git pull if repo exists.
- **Result layout**: `checkpoints/`, `logs/`, `results/` – for per-K student runs (logs, checkpoints, summary CSV).

---

## Dataset

- **LC25000**: lung + colon cancer histopathological images, 5 classes. 768×768 JPEG; resized to 224×224 for ViT.
- **Classes**: colon_aca, colon_n, lung_aca, lung_scc, lung_n.

---

## Key Choices

- **Teacher**: Save best (val-based) model for distillation and evaluation.
- **Token configs**: K ∈ {197, 97, 49, 25}; 197 = no reduction.
- **Images**: Resize to 224 for ViT; do not use native 768×768.

---

## Repo Layout

```
data/           # dataset, get_lc25000_root
models/         # teacher, token_reduction
training/       # train_teacher, load_teacher_checkpoint, evaluate_teacher
tests/          # test_token_reduction
notebooks/      # main.ipynb (Colab)
checkpoints/    # teacher_best.pt, later student_K97_best.pt, ...
logs/           # teacher.csv, later student_K97.csv, ...
```
