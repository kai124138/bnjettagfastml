# BNJetTagKai — NRP GPU Training Report

**Date:** 2026-06-22
**Outcome:** ✅ Training completed successfully end-to-end on the NRP Nautilus cluster. Final ternary (quantized) jet-tagger model trained, evaluated on all test categories, and saved.

---

## TL;DR

- Got `BNJetTagKai` training on a GPU on NRP, from the data already on the `kai-data` volume through to a finished run.
- Hit **two real blockers** and fixed both: a missing `pydot`/`graphviz` dependency that crashed every run, and a **policy-injected 6-hour lifetime limit on bare Pods** that was silently killing the pod mid-run.
- The durable fix was to **run training as a Kubernetes Job** (not a bare Pod). The Job self-installs all dependencies and has no 6-hour cap.
- The full run took **~7h29m on an RTX 3090**. Final model: **test mean AUC 0.658**, TPR@FPR=1e-2 ≈ 0.095.

---

## What was run

| Item | Value |
| --- | --- |
| Cluster | NRP Nautilus (`kubectl` context `nautilus`) |
| Namespace | `cms-ml` |
| Storage | PVC `kai-data` mounted at `/data` (data `/data/bnjet` 5.2 GB, code `/data/BNJetTagKai`) |
| Image | `tensorflow/tensorflow:2.11.1-gpu` (TF 2.11.1, Python 3.8) |
| Workload | Job `kai-train-job` (manifest: `~/Downloads/kai-train-job.yaml`) |
| GPU | NVIDIA RTX 3090 (24 GB) — earlier bare-Pod attempt landed on a 2080 Ti |
| Duration | ~7h29m |
| Entry point | `python train.py` with `/data/bnjet` path overrides |

---

## Findings (the two blockers and their fixes)

### 1. Missing `pydot` + `graphviz` → crash right after the model summary
`train.py:190` calls `tf.keras.utils.plot_model(...)` **unconditionally** to write a model-architecture diagram. The image (and the runbook's documented `pip install` line) don't include `pydot` or the `graphviz` system binary, so every run died with:

```
ImportError: You must install pydot (`pip install pydot`) and install graphviz ... for plot_model to work.
```

…immediately after printing the model summary (all data already loaded — so it looked like it "almost worked").

**Fix:** `apt-get install -y graphviz && pip install pydot`. The system `graphviz` (the `dot` binary) is required, not just the pip package. Now baked into the Job's setup.

### 2. Bare Pods are killed at exactly 6 hours (`DeadlineExceeded`)
The first training pod kept dying mid-run. Root cause was **not** the training or the fix above — the `cms-ml` namespace **injects `activeDeadlineSeconds=21600` (6 h) onto bare Pods** (`kind: Pod`). The pod is SIGKILLed (`exit 137`, `reason=DeadlineExceeded`) at the 6-hour mark regardless of GPU activity.

- The pod was created at 04:06Z and killed at 10:05Z (≈6 h). Training had only launched at ~09:26, so it got ~40 min before the wall.
- Verified that **Job- and Deployment-managed pods are exempt** (no deadline) — e.g. the group's own `zh-dino-train` pods run as Jobs for days/weeks.

**Fix:** run training as a **Job** instead of an interactive Pod (this is the runbook's own "Real fix"). The Job (a) has no 6-hour cap, and (b) puts the entire environment setup in its command, so there is nothing to reinstall by hand when a pod recycles.

### Minor items also handled
- **Data paths:** `train.py` defaults its `--sig-*/--bkg-*/--test-dir` args to Mulder paths that don't exist in the pod; overridden to `/data/bnjet/...`.
- **Python 3.10 → 3.8 syntax:** the code uses `int | None`-style hints; `from __future__ import annotations` is prepended so Python 3.8 accepts them (idempotent; already applied on the volume).
- **`tf.function` retracing warnings:** appear during QAT; cosmetic/minor overhead, not fatal.

---

## Model & training pipeline

**Model:** `bitnet_jet_tagger` — a ternary/quantized transformer jet tagger, **18,401 params**.
Input `(10, 14)` (10 particles × 14 features) → input projection (32) → 2× `BitTransformerBlock` → global average pool → `BitLinear` head → 1 logit.

The run is a multi-stage quantization-aware-training (QAT) recipe:

| Stage | What | Notes |
| --- | --- | --- |
| 1 | FP32 warm-start | 1 epoch, full-precision |
| 2 | Ternary QAT + knowledge distillation | epochs 1–5; teacher = frozen FP32 Stage-1 weights; `kd_weight=0.30`, `kd_temp=2.0` |
| 2.5 | Activation-QAT calibration | 1 epoch → saves `..._preS3.h5` |
| 3 | pAUC 1-way fine-tuning | 2 epochs; `fpr_thresh=0.01`, `focal_w=0.3`, `pauc_w=0.7` → saves final model |
| eval | Test-category evaluation | 9 categories under `/data/bnjet/test_merged` |

`BATCH_SIZE=50`, `EPOCHS=5`, `validation_split=0.20`. ~17,833 steps/epoch (~892k training jets after the val split).

---

## Results

### Stage-by-stage (validation)

| Stage | Key metric |
| --- | --- |
| 1 — FP32 warm-start | `auc=0.716`, `val_auc=0.754` (375 s/epoch) |
| 2 — ternary QAT+KD | ep5 `loss=0.1342`, `val_loss=0.1364` |
| 2.5 — act-QAT calib | `auc=0.734`, `val_auc=0.735` (438 s/epoch) |
| 3 — pAUC fine-tune | **val AUC 0.6651**, TPR@1e-2 **0.0954**, TPR@1e-3 0.0069 |

> Note: AUC drops from the FP32 warm-start (~0.75) to the final ternary model (~0.67). That is expected — ternary quantization costs accuracy, and Stage 3 optimizes partial-AUC at low FPR (`TPR@FPR=1e-2`) rather than global AUC. This model is built to be hardware-friendly (hls4ml / L1 trigger), so small + quantized is the point.

### Test-category evaluation (final model)

| Category | AUC | TPR@1e-2 | TPR@1e-3 |
| --- | --- | --- | --- |
| phi15_bbbb | 0.6587 | 0.1316 | 0.0177 |
| phi15_cccc | 0.6801 | 0.1255 | 0.0095 |
| phi15_uuuu | 0.6908 | 0.1154 | 0.0019 |
| phi30_bbbb | 0.6012 | 0.0489 | 0.0082 |
| phi30_cccc | 0.6744 | 0.1032 | 0.0078 |
| phi30_uuuu | 0.6670 | 0.0783 | 0.0011 |
| phi60_bbbb | 0.5991 | 0.0432 | 0.0068 |
| phi60_cccc | 0.6845 | 0.1257 | 0.0092 |
| phi60_uuuu | 0.6618 | 0.0834 | 0.0015 |
| **mean** | **0.6575** | **0.0950** | **0.0071** |

---

## Artifacts in this folder

```
bnjettag-training-results/
├── REPORT.md                      ← this file
├── models/
│   ├── MODEL.md
│   └── transformer_d32_l2_ffn64_kd/
│       ├── noNorm_train_d32_l2_ffn64_bitnetJetTagModel.h5        ← FINAL trained model (356 KB)
│       ├── noNorm_train_d32_l2_ffn64_bitnetJetTagModel_preS3.h5  ← pre-Stage-3 checkpoint
│       ├── noNorm_train_d32_l2_ffn64_bitnetLoss.{png,pdf}        ← loss curves (FP32 → ternary)
│       ├── noNorm_train_d32_l2_ffn64_auc_finetune.{png,pdf}      ← Stage-3 AUC fine-tuning
│       ├── noNorm_train_d32_l2_ffn64_model.png                   ← model architecture diagram
│       ├── noNorm_train_d32_l2_ffn64_weights.png                 ← weight histogram
│       ├── noNorm_train_d32_l2_ffn64_bitnetWeights.npy           ← raw weights
│       └── noNorm_train_d32_l2_ffn64_ptRange.npy                 ← pT range of sample
└── logs/
    ├── train_job.log              ← full log of the successful Job run
    ├── train.log                  ← partial run killed by the 6h Pod deadline
    └── train.log.0655-plotcrash   ← earliest run, crashed on the missing pydot/graphviz
```

The 5.2 GB of input HDF5 data was intentionally **not** copied (it's source data, not a result, and lives on `kai-data` at `/data/bnjet`).

---

## How to re-run / monitor

The reusable Job manifest is at `~/Downloads/kai-train-job.yaml`.

```bash
# launch (always pass -n cms-ml; default namespace is forbidden)
kubectl apply -f ~/Downloads/kai-train-job.yaml -n cms-ml

# follow progress
kubectl logs -f job/kai-train-job -n cms-ml
# or the on-disk log (survives pod completion):
#   kubectl exec <pod> -n cms-ml -- tail -f /data/outputs/train_job.log

# status / stop
kubectl get job,pods -n cms-ml -l job-name=kai-train-job
kubectl delete job kai-train-job -n cms-ml

# retrieve results later (Job pod is gone once Complete): mount the volume in a throwaway pod and kubectl cp
```

To re-launch under the same name, delete the old Job first (`kubectl delete job kai-train-job -n cms-ml`).

---

## Recommendations

1. **Always train as a Job (or Deployment) here, never a bare Pod** — bare Pods die at 6 h in `cms-ml`.
2. **Add `pydot` + `graphviz` to the documented setup**, or guard the `plot_model` call so it's non-fatal.
3. **GPU was ~98% idle** during training (`BATCH_SIZE=50` → ~17.8k tiny steps/epoch). A larger batch size would use the RTX 3090 far better and cut wall-clock time substantially — but it changes optimization dynamics, so it's a modeling decision, not a free win.
4. **Stage 2 (ternary QAT + KD) dominated runtime** (~5 of the ~7.5 h). If iterating, that's the stage to profile/speed up first.
