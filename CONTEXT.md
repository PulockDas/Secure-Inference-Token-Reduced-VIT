# Secure Inference Token-Reduced ViT – Project Context

**For other Cursor agents / collaborators:** Read this file first for full context. It describes what has been done, the current training setup, and what is planned next (HE approximations, encrypted forward, plots for the paper). Use it as the single source of truth when continuing work on this repo.

---

## Ultimate Goal

Secure ViT inference under **CKKS homomorphic encryption (TenSEAL)**: train a plaintext teacher ViT, distill to a **student** (trained with standard LayerNorm and unbounded attention), then run **encrypted forward** for HE inferencing by approximating activation functions and LayerNorm so that HE inference achieves the same accuracy as plaintext, with some extra compute time. Results and plots will support a **paper manuscript**.

---

## Current Training Setup (Student Distillation)

- **No bounded attention**: HE-friendly attention is used without the bounded / clamping setup.
- **No cubic-squash** in the student during training.
- **Normal LayerNorm** for the full training process (not affine-only).
- **Saved student ViT**: already available on Drive with **92%+ accuracy** (trained with the above setup).
- **Token count**: currently using **K = 97**; will vary K (e.g. 197, 97, 49, 25) and collect results for comparison and plots.

---

## Next Step: Encrypted Forward (HE Inferencing)

- **Goal**: Run encrypted forward for HE inferencing (TenSEAL CKKS, single-image setting).
- **Approximations**: Properly approximate **activation functions** and **LayerNorm** so that:
  - Operations are HE-friendly (only + and ×, or low-degree polynomials).
  - Accuracy matches the plaintext student (92%+), with acceptable **extra time** for HE.
- **Output**: Same accuracy class as current student, with timing/plots for the paper.

---

## Technical Constraints (For HE Inference)

- **HE-friendly**: only addition and multiplication (or low-degree polynomial approximations); no softmax, exp, or division in the encrypted circuit.
- **Token reduction**: early layer to reduce tokens; K ∈ {197, 97, 49, 25} — exploring with **K = 97** first, then other K for plots.
- **Student**: smaller than teacher (e.g. embed_dim 384), trained via distillation with LayerNorm; for HE we replace LayerNorm and activations with approximations.

---

## Tasks Overview

| # | Task | Status |
|---|------|--------|
| 1 | Dataset loader (train/val/test), LC25000 | Done |
| 2 | Teacher ViT fine-tuning (timm ViT-B/16 @ 224) | Done |
| 3 | Token reduction (learnable, preserve CLS) | Done |
| 4 | HE-friendly attention (no softmax/div in design) | Done |
| 5 | Student training with LayerNorm, no cubic-squash | Done |
| 6 | Student ViT (patch embed, token red, HE attn, MLP), 92%+ saved on Drive | Done |
| 7 | Knowledge distillation (KL + optional CE, T=4, logits std logging) | Done |
| 8 | **Approximate activations and LayerNorm for HE** | Pending |
| 9 | **Encrypted forward (TenSEAL CKKS, single image)** | Pending |
| 10 | **Per-K results, plots/tables for paper manuscript** | Pending |

---

## What Is Already Done

- **Data**: `data/dataset.py` – `get_lc25000_root`, `get_dataloaders`, 5 classes (lung + colon), 224×224.
- **Teacher**: `models/teacher.py` – `get_teacher_vit(num_classes)`. Train: `training/train_teacher.py`; eval: `load_teacher_checkpoint`, `evaluate_teacher`; results under `results/teacher/`.
- **Token reduction**: `models/token_reduction.py` – `TokenReduction(dim, num_output_tokens)`; (B,N,D)→(B,K,D).
- **HE attention**: `models/he_attention.py` – `HEAttention`. Used without bounded/cubic-squash in current student training.
- **Activations**: `models/activations.py` – e.g. PolynomialGELU; student trained with standard choices (no cubic-squash requirement for training).
- **Norm**: `models/norm.py` – `create_norm("none"|"affine"|"layernorm", dim)`. **Training uses LayerNorm**; HE path will use approximations.
- **Student**: `models/student.py` – `StudentViT`, `get_student_vit`. Patch embed → CLS+patches → token reduction → blocks → head. Trained with LayerNorm; checkpoint (92%+) on Drive.
- **Distillation**: `training/train_student.py` – `train_student`, distillation loss, `load_student_checkpoint`. Notebook: load with `norm_mode="layernorm"` to match saved checkpoint.
- **Notebooks**: `notebooks/fine-tune-teacher.ipynb`, `notebooks/distill-student.ipynb` (Drive, train student, save ckpt, push logs/results).

---

## Dataset

- **LC25000**: lung + colon histopathology, 5 classes. Resize to 224×224 for ViT.

---

## HE Inference (Planned)

- **Plaintext student**: already at 92%+ with LayerNorm and current attention/activations.
- **Encrypted forward**: replace LayerNorm and non-polynomial activations with **HE-suitable approximations** (polynomial or +/× only) so that:
  - CKKS circuit uses only supported ops.
  - Accuracy stays at same level (92%+), with some **extra time** for HE; measure and report for the paper.
- **K sweep**: run with K = 97 first, then other K values; generate **plots and tables** for the manuscript.

---

## Key Choices

- Teacher: best checkpoint by val acc; student: embed_dim 384, **K = 97** for current run; then vary K for results.
- Student trained with **LayerNorm** (no affine-only, no cubic-squash); saved checkpoint on Google Drive (92%+).
- HE path: add **approximations** for LayerNorm and activations; keep same accuracy; report timing and plots.

---

## Repo Layout

```
data/           # dataset, get_lc25000_root, get_dataloaders
models/         # teacher, token_reduction, he_attention, activations, norm, student
training/       # train_teacher, train_student, load_*_checkpoint, evaluate_teacher
tests/          # test_* for token_reduction, he_attention, activations, norm, student
notebooks/      # fine-tune-teacher.ipynb, distill-student.ipynb (Colab)
checkpoints/    # teacher_best.pt, student_best.pt (often on Drive)
logs/           # teacher.csv, student_K*.csv, distill_run_config_K*.json
results/        # teacher/, student_K*/  (for plots and manuscript)
```
