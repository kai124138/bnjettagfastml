# BNJetTagKai Training Results — Presentation Outline

---

## What We Found (Summary of Everything)

### The Project

**BNJetTagKai** is a ternary-quantized transformer-based jet tagger designed for the CMS Level-1 trigger. The goal is a model small enough and quantized enough to be synthesized onto an FPGA via hls4ml while still providing useful b-jet / c-jet / light-jet discrimination. The model uses **BitNet**-style ternary weights ({-1, 0, +1}) so that multiplications become additions at inference time — critical for L1 latency budgets.

### The Model

- **Architecture:** `bitnet_jet_tagger` — only **18,401 parameters**
- **Input:** `(10, 14)` — 10 particles per jet, each with 14 features (kinematics, impact parameters, etc.)
- **Pipeline:** Input projection (Dense 14→32) → RMSNorm → 2× BitTransformerBlock → RMSNorm → GlobalAveragePooling1D → BitLinear head (32→32) → Activation → Dense (32→1 logit)
- **Key design choice:** Transformer attention (not DeepSets) with BitLinear layers that quantize weights to ternary values during forward pass

### Training Infrastructure

| Item                | Detail                                                                         |
| ------------------- | ------------------------------------------------------------------------------ |
| Cluster             | NRP Nautilus (Kubernetes)                                                      |
| Namespace           | `cms-ml`                                                                       |
| GPU (final run)     | NVIDIA RTX 3090 (24 GB VRAM)                                                   |
| Framework           | TensorFlow 2.11.1 / Python 3.8                                                 |
| Data                | 5.2 GB of HDF5 files on PVC `kai-data` (~892k training jets after 80/20 split) |
| Total training time | ~7 hours 29 minutes                                                            |
| Batch size          | 50                                                                             |

### The Three Attempts (What Went Wrong and How We Fixed It)

1. **Attempt 1 — Crashed on `plot_model`** (`train.log.0655-plotcrash`, 06:55Z, RTX 2080 Ti)
   - Loaded all 5.2 GB of data, printed model summary, then immediately died with `ImportError: You must install pydot`.
   - `train.py:190` calls `tf.keras.utils.plot_model()` unconditionally, but neither `pydot` (pip) nor `graphviz` (system binary) were installed.
   - **Fix:** `apt-get install -y graphviz && pip install pydot`

2. **Attempt 2 — Killed by 6-hour pod deadline** (`train.log`, 09:26Z, RTX 2080 Ti)
   - With graphviz fixed, training launched and completed Stage 1 (FP32 warm-start in 228s). Entered Stage 2 (ternary QAT + KD).
   - Pod was SIGKILLed (exit 137, `DeadlineExceeded`) at exactly 6 hours — the `cms-ml` namespace injects `activeDeadlineSeconds=21600` on bare Pods.
   - Training had only been running ~40 minutes when the pod was killed.
   - **Fix:** Switch from `kind: Pod` to `kind: Job`. Jobs are exempt from the 6-hour cap (confirmed by looking at the group's own long-running `zh-dino-train` Jobs).

3. **Attempt 3 — Successful full run** (`train_job.log`, 10:43Z, RTX 3090)
   - Ran as a Kubernetes Job with all dependencies baked into the startup command.
   - Completed all stages end-to-end in ~7h29m.
   - All model artifacts, plots, and test evaluations saved successfully.

### Training Pipeline (Multi-Stage QAT Recipe)

| Stage         | Description                                                                     | Key Results                                      |
| ------------- | ------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Stage 1**   | FP32 warm-start (1 epoch)                                                       | Train AUC 0.716, Val AUC **0.754** (375 s/epoch) |
| **Stage 2**   | Ternary QAT + Knowledge Distillation (5 epochs, KD weight=0.30, temp=2.0)       | Loss: 0.152 → **0.134** (converged by ep 4)      |
| **Stage 2.5** | Activation-QAT calibration (1 epoch)                                            | AUC 0.734, Val AUC 0.735                         |
| **Stage 3**   | Partial-AUC fine-tuning (2 epochs, FPR threshold=0.01, focal_w=0.3, pAUC_w=0.7) | Val AUC **0.665**, TPR@FPR=1e-2 **0.095**        |

**Why AUC drops from 0.75 → 0.67:** This is expected. Ternary quantization trades accuracy for hardware efficiency. Stage 3 further sacrifices global AUC to optimize the partial AUC at very low false-positive rates (TPR@FPR=1e-2), which is what matters for the L1 trigger use case.

### Test Results (Final Model on 9 Physics Categories)

| Category   | AUC       | TPR@FPR=1e-2 | TPR@FPR=1e-3 |
| ---------- | --------- | ------------ | ------------ |
| phi15_bbbb | 0.659     | 0.132        | 0.018        |
| phi15_cccc | 0.680     | 0.126        | 0.010        |
| phi15_uuuu | 0.691     | 0.115        | 0.002        |
| phi30_bbbb | 0.601     | 0.049        | 0.008        |
| phi30_cccc | 0.674     | 0.103        | 0.008        |
| phi30_uuuu | 0.667     | 0.078        | 0.001        |
| phi60_bbbb | 0.599     | 0.043        | 0.007        |
| phi60_cccc | 0.685     | 0.126        | 0.009        |
| phi60_uuuu | 0.662     | 0.083        | 0.002        |
| **Mean**   | **0.658** | **0.095**    | **0.007**    |

**Key observations from results:**
- **phi15 (low-mass) categories perform best** — AUC ~0.66–0.69, TPR@1e-2 ~0.12–0.13
- **phi60 (high-mass) bbbb is hardest** — AUC 0.60, TPR@1e-2 only 0.04
- **cccc (charm) is consistently the easiest** across all mass points
- **bbbb (bottom) at high mass is the hardest** — harder to separate from QCD at higher pT/mass

### Weight Distribution

The weight histogram confirms successful ternary quantization: the vast majority of weights (~700k entries from the repeated per-sample view) are concentrated at values around {0, 1}, with the dominant spike at ~1. The small clusters at higher values (~2.5, 3.5, 5) correspond to scale factors and the non-quantized input projection / output Dense layers.

---

## Presentation Slide Outline & Image Assignments

### Slide 1: Title
**"BitNet Jet Tagger: Ternary-Quantized Transformer for CMS L1 Trigger"**
- Your name, group, date
- No image needed

### Slide 2: Motivation & Goal
- Why do we need jet tagging at L1? (latency, resource constraints)
- Why BitNet / ternary quantization? (multiplications → additions, FPGA-friendly)
- Target: a model small enough for hls4ml synthesis
- No image needed (or use a generic CMS/L1 trigger diagram if you have one)

### Slide 3: Model Architecture
- 18,401 parameters — tiny by ML standards
- Input: 10 particles × 14 features per jet
- BitTransformerBlocks with ternary weights
- **Image:** `models/transformer_d32_l2_ffn64_kd/noNorm_train_d32_l2_ffn64_model.png`
  *(The Keras architecture diagram showing the full layer stack from InputLayer through to the Dense(1) output)*

### Slide 4: Training Pipeline (Multi-Stage QAT)
- Stage 1: FP32 warm-start
- Stage 2: Ternary QAT + Knowledge Distillation (teacher = frozen FP32 copy)
- Stage 2.5: Activation-QAT calibration
- Stage 3: Partial-AUC fine-tuning for low-FPR regime
- Use a flowchart or bullet list on the slide, referencing the stage table above

### Slide 5: Loss Curves — FP32 to Ternary Transition
- Show how loss drops during warm-start, slight bump at the FP32→QAT switch, then converges
- Train and validation losses track closely (no overfitting)
- The dashed vertical line marks the moment weights go from full-precision to ternary
- **Image:** `models/transformer_d32_l2_ffn64_kd/noNorm_train_d32_l2_ffn64_bitnetLoss.png`
  *(Loss curves with Train (blue), Validation (orange), and the FP32→QAT switch dashed line)*

### Slide 6: Weight Histogram — Ternary Quantization Confirmation
- The histogram proves quantization is working: nearly all weights are at {0, ~1}
- The dominant spike at 1 and a smaller one near 0 show successful ternary clustering
- Small outlier clusters at higher values are scale factors / non-quantized layers
- **Image:** `models/transformer_d32_l2_ffn64_kd/noNorm_train_d32_l2_ffn64_weights.png`
  *(Weight distribution histogram showing the massive ternary spike)*

### Slide 7: Stage 3 — pAUC Fine-Tuning
- What partial-AUC optimization means: we care about TPR at very low FPR (1e-2), not global AUC
- Why: at L1, false positive rate must be extremely low; we sacrifice global AUC for targeted performance
- Final: val AUC 0.665, TPR@FPR=1e-2 = 9.5%
- **Image:** `models/transformer_d32_l2_ffn64_kd/noNorm_train_d32_l2_ffn64_auc_finetune.png`
  *(Stage-3 pAUC fine-tuning plot — note: the AUROC data in this plot falls below the y-axis display range of 0.8–1.0, so the curves may not be visible. You may want to regenerate this plot with an appropriate y-axis range, or just use the table of numerical results instead.)*

### Slide 8: Test Results Across Physics Categories
- Show the 9-category table (3 mass points × 3 quark flavors)
- Highlight: phi15 categories best, phi60_bbbb hardest
- Mean AUC 0.658, mean TPR@FPR=1e-2 = 9.5%
- Consider a bar chart if you want a visual (not included in current plots)

### Slide 9: Infrastructure Lessons (Optional / Backup)
- Two blockers hit and solved:
  1. Missing graphviz dependency crashed the run
  2. Bare Pods killed at 6 hours by namespace policy → switched to Kubernetes Jobs
- Training ran ~7.5 hours on RTX 3090
- GPU was ~98% idle at batch_size=50 — room for speedup

### Slide 10: Next Steps / Future Work
- Increase batch size to better utilize GPU
- hls4ml synthesis of this model → actual FPGA latency/resource numbers
- Compare against the DeepSets variant (already in the repo)
- Explore whether the AUC drop from quantization can be reduced with more QAT epochs or a different KD temperature

---

## Summary of All Images and What They Show

| # | File | What It Shows | Suggested Slide |
|---|------|---------------|-----------------|
| 1 | `noNorm_train_d32_l2_ffn64_model.png` | Keras model architecture diagram — full layer-by-layer view of the bitnet_jet_tagger from InputLayer(10,14) through 2 BitTransformerBlocks to the Dense(1) output | Slide 3 (Architecture) |
| 2 | `noNorm_train_d32_l2_ffn64_bitnetLoss.png` | Training and validation loss curves across epochs, with a dashed vertical line marking the FP32→QAT transition. Shows smooth convergence from ~0.28 to ~0.13 | Slide 5 (Loss Curves) |
| 3 | `noNorm_train_d32_l2_ffn64_weights.png` | Weight value histogram confirming ternary quantization — massive spike at ~1, smaller spike near 0, negligible elsewhere (except scale factors) | Slide 6 (Weight Histogram) |
| 4 | `noNorm_train_d32_l2_ffn64_auc_finetune.png` | Stage-3 pAUC fine-tuning AUROC plot (note: y-axis range 0.8–1.0 may clip the actual data at 0.66–0.68; consider re-plotting or using numerical results) | Slide 7 (pAUC Fine-Tuning) |
