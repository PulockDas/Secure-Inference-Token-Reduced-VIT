# Secure Inference Token-Reduced ViT – Project Context

## Ultimate Goal

Secure ViT inference under **CKKS homomorphic encryption (TenSEAL)**: train a plaintext teacher ViT, distill to an **HE-friendly student** (only + and *), then run plaintext student and finally encrypted forward on a single image.

---

## Technical Constraints (Student)

- **HE-friendly**: only addition and multiplication; no softmax, exp, division, or LayerNorm (use affine or none).
- **Token reduction**: early layer to reduce tokens; K ∈ {197, 97, 49, 25}.
- **Student**: smaller than teacher (e.g. embed_dim 384), trained via distillation.
- **Logging**: CSV (loss, acc, logits std), best checkpoint, run config JSON for plots.

---

## Tasks Overview

| # | Task | Status |
|---|------|--------|
| 1 | Dataset loader (train/val/test), LC25000 | Done |
| 2 | Teacher ViT fine-tuning (timm ViT-B/16 @ 224) | Done |
| 3 | Token reduction (learnable, preserve CLS) | Done |
| 4 | HE-friendly attention (bounded, no softmax/div) | Done |
| 5 | HE-friendly activations (PolynomialGELU, CubicSquash) | Done |
| 6 | Norm: none / affine / layernorm (create_norm) | Done |
| 7 | Student ViT (patch embed, token red, HE attn, MLP with PolyGELU) | Done |
| 8 | Knowledge distillation (KL + optional CE, T=4, grad clip, logits std logging) | Done |
| 9 | Per-K student evaluation, plots/tables | Pending |
| 10 | Encrypted forward (TenSEAL CKKS, single image) | Pending |

---

## What Is Already Done

- **Data**: `data/dataset.py` – `get_lc25000_root`, `get_dataloaders`, 5 classes (lung + colon), 224×224.
- **Teacher**: `models/teacher.py` – `get_teacher_vit(num_classes)`. Train: `training/train_teacher.py`; eval: `load_teacher_checkpoint`, `evaluate_teacher`; results under `results/teacher/`.
- **Token reduction**: `models/token_reduction.py` – `TokenReduction(dim, num_output_tokens)`; (B,N,D)→(B,K,D).
- **HE attention**: `models/he_attention.py` – `HEAttention`. Cubic squashing on Q/K only; V clamped to [-bound,bound]; `output_squash=False` by default for KD stability. Bounded, no softmax/division.
- **Activations**: `models/activations.py` – `PolynomialGELU` (fitted on [-4,4], input clipped), `CubicSquash` (1.5u−0.5u³).
- **Norm**: `models/norm.py` – `create_norm("none"|"affine"|"layernorm", dim)`.
- **Student**: `models/student.py` – `StudentViT`, `get_student_vit`. Patch embed → CLS+patches → token reduction → blocks (pre-norm, HE attention, MLP: Linear→PolynomialGELU→Linear) → head. Norm mode and residual scale configurable.
- **Distillation**: `training/train_student.py` – `train_student`, `distillation_loss`, `load_student_checkpoint`. Temperature-scaled KD (T²), optional CE; grad clip 1.0; CSV includes `student_logits_std`, `teacher_logits_std`, `teacher_std_over_T` (verify student_std ≈ teacher_std/T). No logit clipping.
- **Notebooks**: `notebooks/fine-tune-teacher.ipynb` (train teacher, eval, push); `notebooks/distill-student.ipynb` (mount Drive, load teacher, train student, save ckpt to Drive, push logs/results).

---

## Dataset

- **LC25000**: lung + colon histopathology, 5 classes. Resize to 224×224 for ViT.

---

## Key Choices

- Teacher: best checkpoint by val acc; student: embed_dim 384 (e.g.), K ∈ {197, 97, 49, 25}.
- Teacher checkpoint can live on Google Drive; student checkpoint saved to Drive from Colab (not pushed to repo).

---

## Repo Layout

```
data/           # dataset, get_lc25000_root, get_dataloaders
models/         # teacher, token_reduction, he_attention, activations, norm, student
training/       # train_teacher, train_student, load_*_checkpoint, evaluate_teacher
tests/          # test_* for token_reduction, he_attention, activations, norm, student
notebooks/      # fine-tune-teacher.ipynb, distill-student.ipynb (Colab)
checkpoints/    # teacher_best.pt, student_best.pt (large; often on Drive)
logs/           # teacher.csv, student_K*.csv, distill_run_config_K*.json
results/        # teacher/, student_K*/
```
