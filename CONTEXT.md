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

High level phases and status:

| Phase | Task | Status |
|-------|------|--------|
| 1 | Generic image dataset loader (train/val/test) | Done |
| 2 | Teacher ViT fine-tuning (timm ViT-B/16 @ 224) | Done |
| 3 | Token reduction module (learnable, preserve CLS) | Done |
| 4 | HE-friendly attention (no softmax/exp/div) | Done |
| 5 | HE-friendly activation approximation (polynomial GELU) | Done |
| 6 | Remove/replace LayerNorm (affine/none) | Done |
| 7 | Student ViT with token reduction + HE attention + poly activations | Done |
| 8 | Knowledge distillation training (teacher → student) with logging/checkpoints | In progress (notebook + loop implemented; runs depend on Colab) |
| 9 | Per-K student evaluation (K ∈ {197, 97, 49, 25}) | Pending |
| 10 | Encrypted forward (TenSEAL CKKS, single image) | Pending |

---

## What Is Already Done

- **Data**: `data/dataset.py` – `ImageDataset`, `get_dataloaders`, `get_lc25000_root` (LC25000 via kagglehub, public). Train/val/test split, 5 classes (lung + colon).
- **Teacher**: `models/teacher.py` – `get_teacher_vit(num_classes)` (ViT-B/16 @ 224 from timm). Fine-tuning in `training/train_teacher.py`; logs to CSV; best checkpoint by val acc.
- **Teacher evaluation**: `load_teacher_checkpoint`, `evaluate_teacher` – load best teacher, run on test set, per-class accuracy and JSON/text summaries under `results/teacher/`.
- **Token reduction**: `models/token_reduction.py` – `TokenReduction(dim, num_output_tokens)`. Input (B,N,D)→output (B,K,D); preserves CLS; K−1 learnable queries cross-attend to patches.
- **HE-friendly attention**: `models/he_attention.py` – `HEAttention(embed_dim, num_heads)`. Linear attention (no softmax/exp/div), residual controlled in the student block.
- **HE-friendly activations**: `models/activations.py` – `PolynomialGELU` (polynomial GELU approximation) and `RunningNorm` (running mean + inv-std normalization, add/mul only at inference).
- **Norm replacement**: `models/norm.py` – `create_norm(mode, dim)` with modes `"none"` (identity) and `"affine"` (learnable scale/shift only) plus `"layernorm"` for plaintext ablation.
- **Student ViT**: `models/student.py` – `StudentViT` and `get_student_vit(...)`:
  - Patch embedding (ViT-style 16×16 patches @ 224),
  - CLS + patches → `TokenReduction` (K ∈ {197, 97, 49, 25}),
  - Stacked blocks with pre-norm, `HEAttention`, `RunningNorm` + `PolynomialGELU` MLP, and residuals,
  - Configurable norm mode (`"none"` or `"affine"`) and smaller embed dim (default 384).
- **Distillation training loop**: `training/train_student.py` – `train_student(...)`, `distillation_loss(...)`, `load_student_checkpoint(...)`:
  - KL distillation with temperature + optional hard-label CE,
  - Full logging to CSV (soft/hard losses, train/val acc) for plotting,
  - Best student checkpoint saved as `checkpoints/student_best.pt`,
  - Run config saved to JSON per K.
- **Teacher training notebook**: `notebooks/fine-tune-teacher.ipynb` – Colab notebook to clone repo, load LC25000, train teacher, evaluate, and push logs/results/checkpoints to GitHub.
- **Student distillation notebook**: `notebooks/distill-student.ipynb` – Colab notebook to:
  - Clone repo and install deps,
  - Mount Google Drive and load `teacher_best.pt` from Drive,
  - Build student, run `train_student(...)` with logging/checkpoints,
  - Optionally evaluate best student and save results under `results/student_K*/`,
  - Copy `student_best.pt` to Drive and push only logs/results to GitHub (no large checkpoints).
- **Result layout**: `checkpoints/`, `logs/`, `results/` – structured for per-K student runs (logs, checkpoints, evaluation summaries) and later plotting/analysis.

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
models/         # teacher, token_reduction, HEAttention, activations, norms, student
training/       # train_teacher, load_teacher_checkpoint, evaluate_teacher, train_student, load_student_checkpoint
tests/          # test_token_reduction, test_he_attention, test_activations, test_norm, test_student
notebooks/      # fine-tune-teacher.ipynb, distill-student.ipynb (Colab)
checkpoints/    # teacher_best.pt, student_best.pt, later student_K*_best.pt, ...
logs/           # teacher.csv, student_K*.csv, distillation configs, ...
results/        # teacher/..., student_K*/...
```
