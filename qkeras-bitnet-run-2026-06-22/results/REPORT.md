# qkerasModel.py (BitNet) — NRP GPU Training Run

**Date:** 2026-06-22
**Repo:** https://github.com/Brainz22/BNJetTag (freshly re-cloned)
**Training script:** `qkerasModel.py` (BitNet 1-bit transformer jet tagger)
**Cluster:** NRP Nautilus, namespace `cms-ml`, PVC `kai-data` (`/data`), GPU Jobs
**W&B:** project `bnjettag-bitnet` (live tracking)
**Status:** 🟢 1-bit model established (binary `{−1,+1}`, paper recipe, **val_auc 0.7530**) and aligned to
arXiv:2310.11453. The abstract's central comparison — **1-bit vs the vanilla (FP32) and conventional W8A8
versions** (identical architecture/data/recipe, toggled by a single `BN_VARIANT` knob) — is **complete**:
**vanilla 0.7703**, **W8A8 0.7719** (8-bit ≈ lossless), **1-bit 0.7530** (going 1-bit costs **~1.7 pts**, the
price of the DSP→LUT win). The **quantization-aggressiveness** axis (W1A8→A6→A4 activations) is characterized,
and the hls4ml **resource + latency** story of the binary core is delivered (converts + bit-accurate, DSP=0 in
the generated firmware; ~1 µs @ 400 MHz device-fit). Ternary weights + softmax-free attention are retained as
**appendix** explorations from an earlier, longer abstract. See [TL;DR](#tldr).

This run is separate from the earlier `BNJetTagKai` / `train.py` work (see `../REPORT.md`).
Everything for this run lives in this folder.

---

## Abstract (current revision) and report scope

**Current abstract (2026-06-23).** A particle jet tagger based on a **binary-weight, BitNet-style 1-bit
transformer** — attention and feed-forward weights constrained to **{−1, +1}**. Because binary weights reduce
multiply–accumulate operations, the matmuls map onto the FPGA's **lookup-table / logic fabric rather than its
scarce DSP resources**, targeting a maximally compact, low-latency trigger. The model is trained/validated on
LHC-simulation data and converted to HLS via **hls4ml**. The work **evaluates the tagging efficiency of the
1-bit transformer — comparing it against the vanilla (full-precision) version — and explores how aggressively
it can be quantized before tagging efficiency, resource consumption, and latency degrade.**

**What this scopes in / out (this report follows the abstract above):**
- **Central comparison = 1-bit vs the _vanilla_ version** (full-precision FP32), with a conventional **W8A8**
  8-bit point for context — *not* binary-vs-ternary. → *1-bit vs vanilla & 8-bit baselines* section.
- **Quantization-aggressiveness axis = activation precision** (W1A8 → A6 → A4): how far it can be pushed
  before efficiency degrades. → *Quantization-aggressiveness axis* section.
- **hls4ml = the FPGA resource + latency story** of the 1-bit core. → *hls4ml / FPGA resources & latency*.
- **De-scoped to the appendix (kept, NOT deleted):** the **ternary {−1,0,+1}** weight comparison and the
  **softmax-free attention** variant belonged to an earlier, longer abstract. Their results are preserved
  under *Appendix — explorations beyond the current abstract* and are no longer part of the main narrative.

---

## TL;DR

- Re-cloned `Brainz22/BNJetTag`. Its structure is **completely different** from the old
  `BNJetTagKai`: training is now `qkerasModel.py` (4 positional `.h5` args), there is no
  `train.py`, no `--arch`, and model size is set by **hardcoded module constants** rather
  than CLI flags.
- **Blocker found & fixed:** `qkerasModel.py` reads HDF5 keys `"Training Data"` / `"Sample Data"`,
  but the data already staged on `kai-data` uses keys **`jet_constituents`** and **`train_jet_data`**.
  Its column slicing (`fullData[:,146:]`) also doesn't match the real data and yields **empty**
  arrays. Fixed with a surgical, data-driven patch (read whichever key exists; derive widths).
- **Max-params:** the four size constants are now **env-overridable** (defaults unchanged), so we can
  scale the model up for the GPU and run several sizes in parallel.
- **W&B:** the upstream script has **no** W&B integration; added a minimal, env-gated hook.
- Running as **Kubernetes Jobs** (never bare Pods — the `cms-ml` namespace SIGKILLs bare Pods at 6h).
- **Not-learning bug — root-caused, fixed, and verified.** The model was frozen at `val_auc`=0.5
  (constant output) at every size. Cause: **`activation_quant` had no straight-through estimator**, so
  `tf.round`'s zero gradient severed gradient flow through every activation — only the final head could
  train. Adding the activation STE (plus moving the weight quantizer in-tape with a latent kernel)
  **unfreezes it: `val_auc` 0.50 → 0.67 at epoch 1** (Finding #13). Suspect "weight-quantizer-as-Keras-
  constraint" was tested first and **ruled out** (Finding #12). Diagnosis + one-line fix written up for
  the model's author (`message-to-russell.md`).
- **The 1-bit (binary `{−1,+1}`) model is the target, and the paper recipe makes it strong.** After the
  STE fix the max D256/L8 model still plateaued at `val_auc` ~0.66 with the *old* recipe (a tiny L2 net hit
  the same wall). The paper fixes (zero-mean centralization `Sign(W − mean W)` + correct LR 1.5e-4 +
  warmup/decay + β₂=0.98 + weight decay) **broke that wall: binary now hits `val_auc` 0.7530** (up from
  0.664) — the "~0.66 binary wall" was a *recipe* limit, not a precision limit. Binary is the research
  target on hardware grounds: every attention/FFN MAC becomes **XNOR + popcount → FPGA LUT/logic, not DSP**,
  the whole point of a maximally compact, low-latency trigger. *(A ternary `{−1,0,+1}` variant we also ran —
  0.7685 — is now an **appendix** exploration: the revised abstract compares the 1-bit model to the* vanilla
  *and* 8-bit *versions, not to ternary.)*
- **1-bit vs the vanilla & 8-bit versions — the abstract's central comparison (DONE).** Added a
  **`BN_VARIANT`** knob (`bitnet` | `vanilla` | `w8a8`) so the *identical* architecture, data, and recipe run
  at three precisions — isolating the cost of going 1-bit. Result (distinguishable W&B names
  **`VANILLA-FP32-…`** / **`BASELINE-W8A8-…`**): **vanilla 0.7703**, **W8A8 0.7719**, **1-bit binary 0.7530**.
  So **8-bit is essentially lossless** (W8A8 ≈ vanilla, +0.16 pt = noise) and **going 1-bit costs ~1.7 pts**
  of efficiency vs full precision — a small price that buys the **DSP→LUT** hardware win (vanilla/W8A8 stay
  multiply/DSP-bound; only binary maps to XNOR+popcount → DSP=0). See *1-bit vs the vanilla & 8-bit versions*
  below.
- **Paper-alignment audit (`qkerasModel.py` vs arXiv:2310.11453).** The code targets the *original
  binary* BitNet (**W1A8**) — confirmed component-by-component. The forward path now **aligns**: binary
  `Sign`, 8-bit absmax activations, LayerNorm-before-quant, STE on both `Sign` & `Clip`, latent FP
  weights. One genuine correctness gap fixed — the paper **centralizes weights to zero-mean before the
  sign** (`Sign(W − α)`, Eq 1,3), which the code wasn't doing; **added**. The rest is the **training
  recipe** (paper: large LR ~1e-3, *no* grad clipping, Adam β=(0.9,0.98), weight decay 0.01, polynomial
  decay) — added as knobs and **re-running a paper-faithful binary** to test whether it closes the
  binary↔ternary gap. Full table in *Paper alignment audit* below. NB: **ternary is the *b1.58* follow-up
  (2402.17764), not this paper** — an opt-in, off-paper option that scored higher (0.717 vs 0.664) here.
- **Quantization-aggressiveness axis (W1A8→A6→A4) — "how far can we push it?", the abstract's second
  question.** Weights are already 1-bit; the remaining dial is **activation precision** (env `BN_ACT_BITS`).
  Narrowing absmax activations on the **canonical softmax model** (headline-consistent) gives
  **A8 0.7524 → A6 0.7507 → A4 0.7437** — gentle, monotonic, no cliff, all healthy (val ≥ train). A6 is within
  noise of A8 and 4-bit activations cost only **~0.9 pts** total, so activation width trades cleanly against
  efficiency, to be paired with the hls4ml resource/latency numbers (see *Quantization-aggressiveness axis*
  below). A8 here (0.7524) independently reproduces the headline 0.7530 (seed noise).
  *(The earlier sweep on the softmax-free base — 0.7562/0.7450/0.7381 — is retained as the appendix cross-check;
  the **relative** degradation is base-independent.)*
- **Appendix explorations (kept, off the current abstract).** Two axes from the earlier, longer abstract are
  preserved but **no longer part of the main story**: a **ternary `{−1,0,+1}`** weight variant (b1.58,
  arXiv:2402.17764) that scored **0.7685**, and **softmax-free attention** (`ReLU(QKᵀ)/N`) that matched
  softmax at **0.7562**. Both live under *Appendix — explorations beyond the current abstract*.
- **hls4ml / FPGA resources & latency — the binary core converts and is bit-accurate across the whole
  quantization axis; DSP=0 is in the generated firmware.** The abstract asks how 1-bit affects **resource
  consumption and latency**. With real hls4ml I ran the dominant primitive — a binary FFN block — through the
  toolchain at **A8, A6 and A4**: it **converts and emulates bit-accurately at every precision**
  (`corr=1.000000` ×3, N=1000), the generated `defines.h` types both dense kernels as **`ap_uint<1>`** (1-bit
  → LUT logic, **structurally zero DSPs — at every A**), and the activation datapath narrows exactly as
  expected in firmware (`layer4_t` = `ap_ufixed<8/6/4,2,SAT>`), all at `io_parallel`, `reuse_factor=1` (lowest
  latency). Three synthesizable projects are emitted for off-cluster csynth. On the **newer stack (hls4ml
  1.3.0, Python 3.10)** the Phase-1 gap closes: **LayerNorm (SubLN) now converts, and a full
  SubLN→binary-proj→SubLN→binary-FFN block converts end-to-end** — the only remaining piece is the attention
  **act×act** score matmul (Keras `EinsumDense` can't express it ⇒ Extension-API custom op). The per-component
  model adds the totals: **63.0M binary MACs (99.35% DSP-free)**, **6.36 Mbit** weights, **51 SubLNs**, **0.65%**
  real multiplies; device-fit on a VU13P folds to **RF≈583** → **latency ~1 µs @ 400 MHz**. Exact
  LUT/FF/DSP/BRAM **has since been synthesized** (Vitis HLS 2023.2 on `mulder`, 2026-06-24, off-NRP) — at the
  folded **RF=256** point the binary FFN measures **DSP 0 · LUT ~25% · FF ~7% · BRAM ~1.2%**, latency **520
  cycles ≈ 1.3 µs @ 400 MHz** (no model change). See *hls4ml / FPGA resources & latency* below.
- **Speed:** moved off the RTX-2080-Ti to strong GPUs (RTX-3090 → L40S); best run is **174 s/epoch on
  an L40S** (~6× faster/epoch than the 2080-Ti baseline). `TF_FORCE_GPU_ALLOW_GROWTH=true` needed at
  batch 256 (cuBLAS handle init).

---

## How the new repo runs (from the README + code)

Pipeline in the repo:
1. `dataForgeScripts/dataForge.py <root> <tag> <ptCut> <trainPercent> <usePuppi>` —
   clusters particles from ROOT ntuples into jets, writes `.h5` (`Training Data`, `Sample Data`, …).
2. `dataForgeScripts/removeBackground.py <train.h5> <test.h5>` — drops unmatched (DeltaR) signal jets.
3. **`qkerasModel.py <SignalTrainFile> <BkgTrainFile> <sig_jetData_TrainFile> <bkg_jetData_TrainFile>`** —
   trains the model. (`--sanity` builds the model and checks shapes/weights, no data needed.)
4. `ROC.py` / `qkROC.py` — ROC curve from the saved model on testing data.

We already have staged training data on `kai-data` (from the prior data-transfer), so steps 1–2 are
skipped; we feed the existing `.h5` files straight into `qkerasModel.py`.

### Model (`qkerasModel.py`)
"BitNet"-style 1-bit transformer: input `(10 particles × 14 features)` → BitLinear projection →
learned positional embedding → `N_LAYERS` × BitTransformerBlock (1-bit MHSA + 1-bit FFN, RMSNorm,
residuals) → global average pool → BitLinear head → 1 logit. Weights are constrained to
ternary `{-1,0,+1}` via an absmean straight-through quantizer. Loss = BCE-with-logits, Adam,
pT-reweighted sample weights, `validation_split=0.20`, `EarlyStopping(val_loss, patience=5)`.

---

## Data on `kai-data` (verified by inspection)

| Role | File | HDF5 key | Shape |
| --- | --- | --- | --- |
| signal particles | `bnjet/train_merged/merged_trainPart.h5` | `jet_constituents` | (708 220, 141) |
| signal jet data | `bnjet/train_merged/merged_trainJet.h5` | `train_jet_data` | (708 220, 4) |
| bkg particles | `bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_train.h5` | `jet_constituents` | (406 340, 141) |
| bkg jet data | `bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_trainJets.h5` | `train_jet_data` | (406 340, 4) |

- Particle rows: 140 features (`10×14`) + **label in column 140** (signal file all `1`, QCD all `0`).
- Jet rows: `pT, eta, phi, mass` (col 0 = pT, range 2.25–2048). Used for pT-reweighting + kinematics plots.
- Total training jets: **1 114 560** (708 220 signal + 406 340 bkg). After the 0.20 val split:
  ~891 648 train / ~222 912 val → ~17 833 steps/epoch at `batch_size=50`.

### File-argument mapping used
```
qkerasModel.py \
  bnjet/train_merged/merged_trainPart.h5            # SignalTrainFile      (jet_constituents)
  bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_train.h5      # BkgTrainFile         (jet_constituents)
  bnjet/train_merged/merged_trainJet.h5             # sig_jetData_TrainFile(train_jet_data)
  bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_trainJets.h5  # bkg_jetData_TrainFile(train_jet_data)
```

---

## Findings / troubleshooting

### 1. HDF5 key mismatch (hard blocker) — FIXED
Upstream reads `hf["Training Data"]` / `hf["Sample Data"]`; the staged data uses
`jet_constituents` / `train_jet_data`. As-is the script dies with
`KeyError: ... "Training Data" doesn't exist`.
**Fix:** a small `_read(path, prefer_keys)` helper reads whichever key is present
(`jet_constituents`→`Training Data`, `train_jet_data`→`Sample Data`).

### 2. Column slicing didn't match the data (hard blocker) — FIXED
Upstream hardcodes `dataset=fullData[:,0:141]`, `LLPfeats=fullData[:,142:146]`,
`sampleData=fullData[:,146:]`. With particle width 141 + jet width 4 = 145 total columns,
`fullData[:,146:]` is **empty**, so `sampleData[:,0]` (jet pT) and the kinematics plot crash.
(The same slice is empty even for the repo's own `dataForge` output, which is 141+5.)
**Fix:** derive `n_part_cols = dataset.shape[1]` and slice
`dataset=fullData[:,:n_part_cols]`, `sampleData=fullData[:,n_part_cols:]`. Dropped the unused
`LLPfeats`. Shuffle-together semantics preserved (particles stay aligned with their jet features).

### 3. Nested callbacks list — FIXED
Upstream passes `callbacks=[callbacks]` (a list-in-a-list) to `model.fit`, which Keras rejects.
**Fix:** pass the flat `callbacks` list.

### 4. `plot_model` needs `pydot` + system `graphviz` — handled
Same gotcha as the old run. The Job installs `graphviz` (apt) + `pydot` (pip), and the two
`plot_model` calls are now wrapped so a plotting failure is non-fatal.

### 5. Output directory `v1/bitnet/` not created — handled
`kinematics_plotter` saves to `<cwd>/v1/bitnet/...`, which the script never `mkdir`s.
The Job pre-creates `v1/bitnet` and `bitnet` in a per-run output dir and runs with
`PYTHONPATH=/data/BNJetTag` so `import util.plotting...` resolves while outputs land in the run dir.

### 6. Model size was hardcoded — made env-overridable (for "max params")
`D_MODEL/N_HEADS/N_LAYERS/FFN_DIM` (and `EPOCHS/BATCH`) now read from `BN_*` env vars,
**defaulting to the repo's values** so default behavior is identical. This is how we scale up.

### 7. No W&B in the upstream script — added
Added an env-gated `wandb.init(...)` + a tiny per-epoch logging callback (`wandb.log(logs)`),
enabled only when `WANDB_PROJECT` is set. Robust across wandb versions (no `wandb.keras` dependency).

### 8. First smoke Job died in 44 s — `grep | while` under `set -e` (Job-script bug, not the model) — FIXED
The Job scripts carried over a py3.10→3.8 type-hint patch from the old runbook:
`grep -rl " | " --include="*.py" "$CODE" | while read f; do ...; done`. The **new** repo's code
has no `int | None`-style hints, so `grep -rl` finds nothing and **exits 1**; under
`set -eo pipefail` that exit propagated and killed the script right after `pip install`, before
the `--sanity` line ever ran (exit 1, 44 s, no Python output). The patch was always a no-op here.
**Fix:** wrapped it `( … ) || true` in all four manifests so a no-match can't abort the run
(kept as a harmless safety net rather than deleted). Re-ran smoke after the fix.

Full diff: `patches/qkerasModel.patch`. Patched copy staged at `/data/BNJetTag` on the volume.

---

## Configurations (parameter sweep)

| Run | D_MODEL | N_HEADS | N_LAYERS | FFN_DIM | ~params | W&B run | output dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (repo default) | 32 | 4 | 2 | 64 | ~18.5 k | `baseline-D32-L2-FFN64` | `/data/outputs/qk-baseline` |
| large | 128 | 8 | 6 | 512 | ~1.20 M | `large-D128-L6-FFN512` | `/data/outputs/qk-large` |
| max | 256 | 8 | 8 | 1024 | ~6.38 M | `max-D256-L8-FFN1024` | `/data/outputs/qk-max` |

`batch_size=50` and `epochs=200` (EarlyStopping) are kept at the repo defaults.
Each run is a separate GPU Job; they run in parallel.

---

## Run status

- [x] Re-clone repo, read README + all scripts
- [x] Verify data keys/shapes on `kai-data`
- [x] Patch `qkerasModel.py` (keys, slicing, env sizes, W&B, plot guard, callbacks)
- [x] Stage patched code to `/data/BNJetTag`; create W&B secret `kai-wandb`
- [x] **Smoke test PASSED** (`--sanity` at max config + 1-epoch baseline fit on real data + W&B) — see below
- [x] **Launched baseline / large / max Jobs** (2026-06-22, three separate GPU nodes)
- [ ] Collect results (AUC, loss curves, saved models) — fill in below

### Smoke test (PASSED, 2026-06-22)
Ran `kai-bn-smoke`: `--sanity` at MAX config, then a real 1-epoch baseline fit. Verified end-to-end:
- **GPU**: NVIDIA TITAN Xp (CC 6.1), XLA compiled — GPU is used.
- **Data loader (the risky patch) works**: `17833/17833` steps = 891,648 train ÷ batch 50, i.e. the
  full 1,114,560-jet volume dataset loaded with the correct keys. Confirms findings #1/#2 are fixed.
- **1-epoch baseline fit**: 682 s/epoch (38 ms/step); loss 0.890, val_auc 0.500 (≈random — expected
  after a single epoch of ternary-weight training; AUC is the metric to watch over 200 epochs).
- **`.h5` save works** (custom BitNet layers serialize fine, 342 KB).
- **W&B live**: run `smoke-1epoch` → https://wandb.ai/kayamaguchi-uc-san-diego/bnjettag-bitnet/runs/b3isuf2f
- **All plots written** to `/data/outputs/qk-smoke/` (kinematics particle+jet, loss curve, weights, model diagram).
- Note: upstream `binary_accuracy` metric is not logit-aware (model outputs logits, loss is
  `from_logits=True`), so that number is meaningless; loss + AUC are correct. Left as-is per "listen to the repo".

### Production runs (launched + confirmed training, 2026-06-22)
Three Jobs, one GPU each, scheduled onto three different cluster sites. All three confirmed
**training within ~3 min** of launch (data loaded, W&B run created, epoch 1 started). Verified
param counts match the plan exactly.

| Job | config | params (verified) | GPU | output dir | W&B run |
| --- | --- | --- | --- | --- | --- |
| `kai-bn-baseline` | D32 L2 FFN64   | **18,209**    | RTX 3090 (24 GB)  | `/data/outputs/qk-baseline` | [z8gl94od](https://wandb.ai/kayamaguchi-uc-san-diego/bnjettag-bitnet/runs/z8gl94od) |
| `kai-bn-large`    | D128 L6 FFN512 | **1,202,817** | Tesla V100 (32 GB)| `/data/outputs/qk-large`    | [aa4r7d81](https://wandb.ai/kayamaguchi-uc-san-diego/bnjettag-bitnet/runs/aa4r7d81) |
| `kai-bn-max`      | D256 L8 FFN1024| **6,373,633** | NVIDIA L40 (48 GB)| `/data/outputs/qk-max`      | [2mck1v3k](https://wandb.ai/kayamaguchi-uc-san-diego/bnjettag-bitnet/runs/2mck1v3k) |

`epochs=200`, `batch=50`, `EarlyStopping(val_loss, patience=5)` — repo defaults. W&B project
**bnjettag-bitnet**. Baseline epoch ≈ 682 s; larger configs are somewhat slower per step. Expect
EarlyStopping to halt each run well before 200 epochs.

### Results — ⚠️ models are NOT learning (val_auc frozen at 0.5)

All three configurations, despite a 350× range in parameters, collapse to a **constant prediction**:

| Run | params | epochs | val_loss trajectory | val_auc | outcome |
| --- | --- | --- | --- | --- | --- |
| baseline | 18,209 | 8 (EarlyStopped) | 0.8842 → 0.8847 (flat) | **0.5000** every epoch | finished, AUC=0.5 |
| large | 1,202,817 | 3+ (running) | 0.8884 → 0.8905 (flat) | **0.5000** every epoch | will EarlyStop ~0.5 |
| max | 6,373,633 | 6+ (running) | 0.9216 → 0.9314 (flat) | **0.5000** every epoch | will EarlyStop ~0.5 |

**Evidence it's genuinely not learning (not just a metric quirk):** the *loss* itself is flat from
epoch 1 — the optimizer isn't reducing it. `binary_accuracy` sits at ≈0.3644 = the exact background
class fraction (406,340 / 1,114,560), i.e. the network outputs a constant logit and predicts one
class for everything. AUC is therefore exactly 0.5 (no rank information). This is **independent of
model size**, so it is not a capacity or under-training problem.

**Primary suspect (after reading the model code) — the weight quantizer is wired as a Keras
`constraint`, which kills gradient-based learning of the binary weights.** `AbsMeanQuantizer` is
attached via `constraint=AbsMeanQuantizer()` on each `BitLinear` kernel (code ~L154). Keras runs
constraints *after* `optimizer.apply_gradients`, **outside the gradient tape**, so:
- the straight-through estimator written into its `__call__`
  (`w_scaled + stop_gradient(sign(w_scaled) − w_scaled)`, ~L110-112) is **dead code** — its custom
  backward is never used;
- there are **no latent full-precision master weights** — the kernel is re-binarized *in place* every
  step (the docstring's claim that "the full-precision master weights are updated by the optimiser" is
  not what a constraint does).
So every step Adam nudges the kernel and the constraint immediately snaps it back to
`sign(w)·mean|w|`; sub-threshold updates are erased and weights can only move by flipping signs. That
stalls **independent of model size** — which is exactly what we see (18k→6.4M all frozen at 0.5). The
output then collapses to a constant logit (driven by the float biases) → AUC exactly 0.5.

**NOT the main cause: input normalization.** Initially suspected, but `BitLinear.call` applies
`RMSNorm` to its input *before* the matmul (~L171), so raw input scale is largely handled at the first
layer. (Caveat: `RMSNorm` here actually *subtracts the mean* — `(x−mean)/sqrt(var+eps)`, i.e.
LayerNorm — despite a docstring saying it must not center, to "preserve sign structure"; and it
normalizes across 14 heterogeneous features so a large feature can dominate. Secondary at most.)

**This is a model/quantization design question, not a pipeline bug** — the harness (data loading, GPU,
W&B, saving, plots) is fully verified working. Because the fix means changing the quantization design,
and per the "listen to the repo" instruction, the call was taken to the model's author (Russell)
rather than changed unilaterally. Draft bug report: `message-to-russell.md`.

### Finding #9 — completed-pod logs get reaped on NRP; use the on-disk tee log
After a Job's pod completes, `kubectl logs <pod>` can return **empty** (NRP garbage-collects the
container logs), even though the pod object still reports `exitCode 0`. The reliable source is the
per-run log the Job tees to the PVC: `/data/outputs/qk-<run>/train_<run>.log`, read via the
`kai-setup` util pod. (Also: my own monitor shell runs under **zsh**, where unquoted `$VAR` does not
word-split — loops must use literal lists or `${=VAR}`.)

### Finding #10 — frozen-at-0.5 BitNet (referred to the model's author)
See the Results block above. Faithful repo run mechanically succeeds but the model does not learn at
any size. Code-level diagnosis points at the **weight quantizer being attached as a Keras
`constraint`** (dead STE + no latent weights), with `RMSNorm`-centering and data-feature-order as
secondary questions. Per the user's decision, this was **not** fixed unilaterally; instead a focused
bug report was drafted for **Russell** (the BitNet's designer): `message-to-russell.md`. The three
production Jobs were stopped (user: "they're done"); saved models + logs remain on the PVC under
`/data/outputs/qk-{baseline,large,max}/`.

### Finding #11 — the `--sanity` check passes, and *that* confirms the bug
`sanity_check()` (qkerasModel.py L630-712) only asserts (a) output shape `(8,1)`, (b) one
`train_on_batch` doesn't crash, and (c) every `*kernel*` has ≤3 unique rounded values → "all
kernels binary." It **never checks trainability** (loss decreasing / output varying), so a model
frozen at a constant logit passes all three — sanity ✓ ≠ learning ✓. Two tells reinforce suspect #1:
- The shape-check forward pass `model(dummy_x)` runs **before** the first optimizer step, so the
  `AbsMeanQuantizer` **constraint** hasn't fired yet — it's exercising the raw FP kernels, not the
  quantized path.
- The "kernels are binary" test passes **only because** the constraint binarizes the kernel **in
  place** after one step (stored kernel = `sign(w)·mean|w|`, values `{-β,0,+β}`). A correct
  quantize-in-`call` design would keep **full-precision latent** kernels and this check would instead
  flag them non-binary. The sanity check is built around the buggy design and rubber-stamps it.

This was already exercised: `--sanity` ran clean at max config in the smoke Job. The note was folded
into suspect #1 of `message-to-russell.md`.

### Finding #12 — suspect #1 TESTED and RULED OUT; real culprit = `activation_quant` has no STE
Built `code/qkerasModel_ste.py` (only diff vs canonical, verified by `diff`: weight quantizer moved
from `constraint=AbsMeanQuantizer()` to an in-`call` STE with a **full-precision latent kernel** —
the QAT wiring the module docstring describes). Staged it on the PVC and ran baseline (D32/L2/FFN64,
5 epochs) as Job `kai-bn-ste` → run `ste-test-D32-L2-FFN64`.

**Result: still frozen.** Gradients provably reach the kernel now, yet:

| epoch | loss | bin_acc | val_loss | val_acc | val_auc |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.9036 | 0.369 | 0.8897 | 0.3638 | 0.5000 |
| 2 | 0.8881 | 0.365 | 0.8881 | 0.3638 | 0.5101 |
| 3 | 0.8882 | 0.365 | 0.8898 | 0.3638 | 0.5000 |
| 4 | 0.8883 | 0.365 | 0.8897 | 0.3638 | 0.5000 |
| 5 | 0.8884 | 0.365 | 0.8927 | 0.3638 | 0.4901 |

`val_auc` is noise around 0.5; `val_acc` pinned at the exact bkg fraction. So **constraint-vs-in-call
weight quantization is NOT the cause.** The model emits a constant output *even when the weights get
real gradients* → the bug is in the **forward path, upstream of weight quantization.**

**New prime suspect — `activation_quant` (L119-121) has no straight-through estimator.** It returns
`clip(round(x*scale), -128, 127)/scale`; `tf.round` has zero gradient, so (unlike the *weight*
quantizer, which carries an STE) this **blocks gradient flow through every activation** — only the
final linear head can train, all upstream layers stay at random init. Present in BOTH the constraint
and latent-weight versions, which explains why the suspect-#1 fix changed nothing. Fix = same STE as
the weights: `(x*scale + stop_gradient(round(x*scale) - x*scale)) / scale`. **Not yet tested** — the
confirming run (add activation-STE on top of the latent-weight variant) is the obvious next step.
The Russell draft was rewritten (short version) to lead with this.

Anchor: `kai-train-d64` (the *other* repo, `train.py`, staged ternary-QAT + KD) learns fine on the
same staged data (`val_loss` ~0.13), so the data is good and QAT works here — the freeze is specific
to `qkerasModel.py`'s forward path. (Both `kai-setup` and `kai-train-d64` pods went to `Error` during
a laptop-sleep network gap; the `kai-bn-ste` Job still completed all 5 epochs.)

### Finding #13 — activation-STE fix CONFIRMED: the model learns (root cause verified)
Applied the fix to the canonical `code/qkerasModel.py` and synced to the PVC (original backed up to
`/data/BNJetTag/qkerasModel.orig.py`). Three edits: (1) **activation_quant** now carries an STE
(`(x*scale + stop_gradient(round(x*scale) - x*scale)) / scale`, L119-126); (2) **`BitLinear.build`**
drops `constraint=AbsMeanQuantizer()`, keeps the kernel as a trainable FP latent weight, stores
`self.quantizer` (L152-163); (3) **`BitLinear.call`** quantizes the kernel in-tape
(`kernel_q = self.quantizer(self.kernel)`, L174-184). Re-ran baseline (D32/L2/FFN64, 5 epochs) as Job
`kai-bn-fixed-base` → run `fixed-baseline-D32-L2-FFN64` (`7l963d50`).

**Result: it learns.** `val_auc` moves off 0.5 immediately and the constant-output behavior is gone:

| epoch | loss | bin_acc | auc | val_loss | val_acc | val_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9041 | 0.521 | 0.631 | 0.8576 | 0.565 | **0.6749** |
| 2 | 0.8698 | 0.507 | 0.626 | 0.8518 | 0.529 | 0.6279 |
| 3 | 0.8693 | 0.507 | 0.622 | 0.9133 | 0.365 | 0.4418 |
| 4 | 0.8705 | 0.510 | 0.625 | 0.8566 | 0.563 | 0.6446 |
| 5 | 0.8726 | 0.502 | 0.619 | 0.8818 | 0.390 | 0.5925 |

vs. the frozen runs (train_auc 0.4996 / val_auc exactly 0.5000 / val_acc exactly 0.3638, flat). This
**confirms `activation_quant`'s missing STE was the root cause** — the only change since the
still-frozen suspect-#1 test (Finding #12) is the activation STE. `val_auc` is noisy and `train_auc`
plateaus ~0.62, which is expected for a tiny 2-layer 1-bit model near its capacity ceiling on just 5
epochs (canary, `BN_EPOCHS=5`); the gate here is "does it learn at all," and it does.

**Max-param run launched.** Job `kai-bn-fixed-max` → run `fixed-max-D256-L8-FFN1024`
(D256/H8/L8/FFN1024, EPOCHS=200, EarlyStopping patience=5), `/data/outputs/qk-fixed-max/`. Both fixed
jobs use preferred nodeAffinity for H200 > H100 > A100-80GB with a required `arch=amd64` term (the
TF 2.11.1 image is x86; this excludes the ARM GH200). Those nodes were saturated, so both landed on
RTX-2080-Ti (~18.6 min/epoch). The model is ~6.4M params (fits 11GB). [Update: see Finding #14 —
the max model is ~80% compute-bound, so a faster GPU does help (~1.7x on an RTX-3090); and at L8 it
needed optimizer stabilization to learn at all.]

### Finding #14 — the max (L8) model needs optimizer stabilization; strong-GPU run learns
The gradient-flow fix (Finding #13) unfreezes the *shallow* model, but the **max D256/L8** model
trained with upstream `optimizer="adam"` (1e-3, no warmup) **did not learn**: over 3 epochs train_auc
stayed ~0.50 and the loss *bounced* (2.45 → 1.98 → 3.01) — classic unstable deep-transformer
optimization (8 stacked 1-bit blocks). A faster GPU alone wouldn't fix this.

**Fix (env-gated, default path unchanged):** added `BN_LR` (→ `Adam(lr, global_clipnorm=1.0)`) and
`BN_WARMUP_EPOCHS` (linear LR warmup), plus `restore_best_weights=True` on EarlyStopping. Relaunched
as `kai-bn-fixed-max-fast` (run `fixed-max-fast-D256-L8-FFN1024-lr2e4-wu3`) with `BN_LR=2e-4`,
`BN_WARMUP_EPOCHS=3`, on a strong GPU (required nodeAffinity over a broad set faster than the 2080-Ti;
landed on an **RTX-3090** as the A100/H100/L40/4090 tiers were saturated).

**Result — learns, and faster:**

| epoch | loss | train_auc | val_auc | val_acc | lr | s/epoch |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.4301 | 0.6014 | 0.6243 | 0.5364 | 6.67e-5 | 688 |
| 2 | 1.0435 | 0.6015 | 0.5322 | 0.3658 | 1.33e-4 | 610 |
| 3 | 1.0233 | 0.5843 | 0.6029 | 0.3658 | 2.00e-4 | 616 |
| 4 | 1.0125 | 0.5902 | 0.6681 | 0.6411 | 2.00e-4 | 617 |

train_auc 0.50→0.60, val_auc climbing to 0.67, loss stable (3.43→1.01, no bounce). Speed ~35 ms/step
(~10.2 min/epoch) vs the 2080-Ti's ~60 ms/step (~18.6 min) → **~1.7x** (the model is mostly
compute-bound; the ~14 ms/step fixed overhead caps further GPU gains — bigger batch is the next lever).
The slow 2080-Ti run was retired.

**Outcome (early-stopped epoch 10, best=epoch 5):** stable but **plateaued** — train_auc ~0.58 (flat,
slightly declining), val_auc noisy ~0.60-0.67, best val_loss 0.9212 @ epoch 5 (val_auc 0.616). The
6.4M-param L8 model does **no better than the 18k-param canary (~0.67)**, and train_auc ~0.58 indicates
**underfitting** — i.e. residual optimization difficulty, not a capacity ceiling. (The separate ternary
pipeline reaches val_loss ~0.13 on this data, so far better is achievable.) Leading hypothesis:
`BN_LR=2e-4` is now too conservative — the 2080-Ti divergence that motivated the drop had *no*
grad-clipping, which is now in place, so a higher LR (~5e-4) + cosine decay + `monitor="val_auc"` +
larger patience should let the deep model actually fit. Acted on in Finding #15.

### Finding #15 — anti-underfit run: higher (clipped) LR + bigger batch + AUC-based early stop
To test the Finding #14 hypothesis (LR=2e-4 is now too conservative since `global_clipnorm=1.0` is
in place), relaunched the **same max model** (D256/H8/L8/FFN1024) as `kai-bn-fixed-max-fast2`
(run `fixed-max-fast2-D256-L8-FFN1024-lr5e4-bs256-wu3-esauc`) with three coupled changes:

- **`BN_LR=5e-4`** (2.5x the fast run; still half the diverging 1e-3, and now grad-clipped) — push the
  model to actually fit the training set.
- **`BN_BATCH=256`** (vs 50) — ~5x fewer steps/epoch (faster, the speed lever called out in #14) *and*
  lower-variance gradients, which pairs naturally with the higher LR. Warmup (3 ep) + `clipnorm=1.0`
  still guard the start.
- **`BN_ES_MONITOR=val_auc`, `BN_ES_MODE=max`, `BN_ES_PATIENCE=8`** — stop on the tagger metric we
  actually care about (the fast run's val_loss-best epoch had val_auc 0.616 while its true peak was
  0.668), with more patience for the deeper model to keep improving.

Code change is env-gated and default-preserving: added `BN_ES_MONITOR/MODE/PATIENCE` (defaults
`val_loss`/`auto`/`5` = upstream behaviour).

**Infra note (BLAS init at batch 256):** first launch crashed on the *first* matmul with
`Attempting to perform BLAS operation using StreamExecutor without BLAS support`. Not OOM and not a
code bug — TF 2.11 pre-grabs ~all VRAM at init, leaving no room for cuBLAS to create its handle at the
larger batch (batch-50 happened to fit). Fixed with `TF_FORCE_GPU_ALLOW_GROWTH=true` (memory growth);
resubmitted and trained cleanly. The model is ~6.4M params, so this was a handle-init issue, not true
memory exhaustion.

**Result — faster, but SAME plateau (RTX-3090):** batch 256 ran **266 s/epoch (76 ms/step)** vs the
fast run's ~620 s/epoch → **~2.3x faster wall-clock**. But the higher LR did *not* break the underfit:

| epoch | lr | loss | train_auc | val_auc |
| --- | --- | --- | --- | --- |
| 1 | 1.67e-4 | 5.065 | 0.596 | 0.638 |
| 2 | 3.33e-4 | 1.250 | 0.611 | 0.664 |
| 3 | 5.0e-4 | 1.137 | 0.607 | 0.647 |

train_auc is **stuck ~0.61 even at full 5e-4 LR** (vs ~0.58 before — a marginal nudge, not a
breakout). Three optimizer configs now top out at the same place: `adam@1e-3` (diverged),
`lr2e-4+wu` (val_auc ~0.62), `lr5e-4+bs256` (val_auc ~0.66). Crucially the **18k-param L2 canary also
hit ~0.67**, so 18k → 6.4M params buys ~nothing. Conclusion: the ceiling is **not** the optimizer and
**not** model capacity — it is something common to every config. Leading suspect: weight precision
(see Finding #16).

### Finding #16 — the ceiling is likely BINARY weights; testing true TERNARY (b1.58)
Read the weight quantizer carefully. Its **docstring describes ternary** (`clip(round(W/scale),-1,1)`
→ {−1,0,1}) but the **code is binary**: `w_q = w_scaled + stop_gradient(tf.sign(w_scaled) − w_scaled)`
→ strictly {−1,+1}. Confirmed against **upstream `Brainz22/BNJetTag@main`**: upstream is also binary
(`tf.sign`) *and* its `activation_quant` has **no STE** (validating Finding #13 — that gradient bug is
upstream). So binary-not-ternary is an upstream choice, despite the "BitNet b1.58" name and the
docstring.

Binary weights cannot zero out an input — every one of the 14×10 features is forced to contribute
±β. True **ternary** {−1,0,1} (what b1.58 actually is) can prune irrelevant inputs and is strictly
more expressive; the separate ternary pipeline reaching val_loss ~0.13 on this data is circumstantial
support. **Hypothesis:** the ~0.66 ceiling is a binary-precision limit.

**Change (opt-in, upstream default preserved):** added `BN_TERNARY` (default 0 = upstream `tf.sign`
binary; `=1` = `clip(round(w_scaled),-1,1)` ternary, STE unchanged). Launched
`kai-bn-fixed-max-ternary` (run `fixed-max-ternary-…-esauc`) with `BN_TERNARY=1` at the *identical*
recipe as fast2 (LR=5e-4, bs=256, warmup=3, clipnorm=1.0, ES=val_auc/max/p8) — so fast2 (binary) is a
clean baseline.

**Result — HYPOTHESIS CONFIRMED, ternary breaks the binary ceiling:**

| | binary (fast2) | **ternary** |
| --- | --- | --- |
| best val_auc | 0.6642 (ep2) | **0.7111 (ep1)** |
| epoch-1 train_auc | 0.596 | **0.694** |
| epoch-1 val_auc | 0.638 | **0.711** |

Same recipe, same GPU class — ternary's best val_auc **0.711 vs binary's 0.664**, and it fits the
training set far better (train_auc 0.69 vs ~0.61). The ~0.66 wall *was* binary precision. **But** the
LR was tuned for binary's underfit and is too hot for ternary: ternary peaked at **epoch 1 (warmup
LR=1.67e-4)** and degraded *monotonically* as warmup ramped LR to 5e-4 (val_auc 0.711→0.631→0.708→…
→0.62; train_auc 0.694→0.645). So 0.711 is a *floor* on ternary's potential — it never trained at a
good LR. (Binary, by contrast, was flat/declining regardless — precision-bound, not LR-bound.)

### Finding #17 — ternary at its sweet-spot LR (low, no warmup)
Ternary's monotonic decay as LR rose says the warmup-to-5e-4 schedule is wrong for it: it wants a
**low, non-ramping** LR. Launched `kai-bn-fixed-max-tern-lr` (run `…-lr1p5e4-bs256-nowu-esauc`):
`BN_TERNARY=1`, **`BN_LR=1.5e-4`** (just below the 1.67e-4 sweet spot), **`BN_WARMUP_EPOCHS=0`** (no
ramp), `bs=256`, `clipnorm=1.0`, `ES=val_auc/max/patience=10`.

**Result — stable, and the best model so far (best val_auc 0.7167 @ epoch 7, on an L40S):** the low LR
*fixed the decay* — train_auc now holds steady ~0.70 across all 17 epochs (vs the monotonic slide at
5e-4), and val_auc oscillates healthily 0.69–0.72 (val ≥ train → not overfitting). Early-stopped at
epoch 17, restored epoch 7. Speed: **174 s/epoch (50 ms/step)** on the L40S (a faster GPU finally
freed up) — ~1.5× the RTX-3090 and ~6× the original 2080-Ti per epoch.

**Bottom line across the campaign (same D256/L8/FFN1024 max model):**

| config | best val_auc | training health |
| --- | --- | --- |
| binary @ 5e-4 (fast2) | 0.664 | underfits (train_auc ~0.61) |
| ternary @ 5e-4 (warmup→5e-4) | 0.711 | peaks ep1, then decays |
| **ternary @ 1.5e-4 (no warmup)** | **0.717** | **stable, train_auc ~0.70** |

**Recommendation (revised for the research framing — binary is the target).** The transferable insight
here is the **LR**: this tagger wants **LR≈1.5e-4, no/short warmup**, *independent* of binary-vs-ternary.
Ternary at that LR scores **0.717**, and it stands as the **accuracy-vs-resource reference** — *not* the
deployment default. Binary `{−1,+1}` is the deployment target because it reduces every MAC to
**XNOR+popcount → FPGA LUT/logic, not DSP**; ternary's `0` breaks that (extra zero-mask logic). So the
right move is **binary at the corrected LR + the paper fixes** (centralization, β₂=0.98, weight decay,
warmup+decay), which the in-flight reruns measure — see *Paper-recipe reruns* below. *Open headroom:*
the separate ternary-QAT pipeline reaches val_loss ~0.13 on this data, so ~0.72 is unlikely to be the
true ceiling — closing it is open-ended (architecture / features / longer schedules / activation-range),
a research effort beyond optimizer/precision knobs.

---

## Paper alignment audit — `qkerasModel.py` vs arXiv:2310.11453 ("BitNet")

The model is based on the **original BitNet paper** (Wang et al., 2023, arXiv:2310.11453): a **W1A8**
1-bit Transformer — **binary** weights (`{−1,+1}`) + **8-bit** activations. (The ternary `{−1,0,+1}`
"b1.58" scheme is a *later, separate* paper, arXiv:2402.17764.) I went through `BitLinear` (Eq 1–12)
and the training recipe (Tables 5–7) component by component.

### What ALIGNS (with the fixes in place)

| BitLinear component | Paper (2310.11453) | Code now | |
| --- | --- | --- | --- |
| Weight quantizer | binary `Sign(W)`, {−1,+1} | `tf.sign`, {−1,+1} (default) | ✓ |
| Zero-mean centralization | `W̃ = Sign(W − α)`, `α = mean(W)` (Eq 1,3) | **added** `w − mean(w)` before the sign | ✓ **fixed** |
| Weight scale β | `β = (1/nm)‖W‖₁` absmean (Eq 12) | `mean|W − α|` (absmean on centralized W) | ✓ |
| Activation quant | 8-bit absmax `Clip(x·Qb/γ,…)`, `γ=‖x‖∞`, `Qb=128` (Eq 4–5) | `clip(round(x·127/max|x|),−128,127)` | ✓ |
| Norm before quant | LayerNorm/SubLN `(x−E[x])/√(Var+ε)` (Eq 12) | `RMSNorm` class **subtracts the mean** → *is* LayerNorm | ✓ (docstring fixed) |
| STE on Sign + Clip | STE for both (Sec 2.2) | STE on weight `Sign` **and** activation `Clip` | ✓ (**activation STE was the headline bug**, Finding #13) |
| Latent FP master weights | high-precision latent, binarized on the fly | FP `kernel`, quantized in `call()` inside the tape | ✓ (Finding #12) |
| Dropout | none (Tables 6–7) | `DROPOUT=0` default | ✓ |

### What still DEVIATES (and the call I made)

| Item | Paper | Code (before) | Decision |
| --- | --- | --- | --- |
| Peak LR | **large**, 1e-3–2.4e-3; *smaller models use the **largest** LR* (Table 5) | 1.5e-4–5e-4 | **rerun at 1e-3** (paper-faithful) |
| LR schedule | polynomial decay + 750-update warmup | linear warmup, then hold | added **poly-decay** knob; rerun uses warmup+decay |
| Adam β | **(0.9, 0.98)** | default (0.9, 0.999) | **fixed** (`BN_BETA2=0.98`) |
| Gradient clipping | **none** ("✗", Tables 6–7) | `global_clipnorm=1.0` when LR>0 | made a knob; **purist rerun uses `BN_CLIPNORM=0`** |
| Regularization | **weight decay 0.01** (AdamW) | L1 `1e-4` on latent kernel (upstream) | rerun uses **`BN_WEIGHT_DECAY=0.01`, L1 off** |
| Activation quant scope | per-tensor (train) / per-token (inference) | per-token always (axis=−1) | **kept** per-token (finer; what b1.58 uses; stable at N=10) — documented |
| Pre-ReLU activation | subtract min η, scale `[0,Qb]` (Eq 6) | symmetric absmax everywhere | **not implemented** — minor; documented |
| Weight precision | **binary** (this paper) | binary default; `BN_TERNARY=1` → ternary (b1.58) | binary aligns; ternary is an **opt-in, off-paper** option (scored 0.717 vs 0.664 here) |

### Code changes this pass
- **Zero-mean weight centralization** `Sign(W − mean W)` added to the binary path (Eq 1,3) — the one real
  *correctness* gap vs the paper. (Ternary path unchanged — b1.58 deliberately doesn't centralize.)
- **Optimizer knobs** for the paper recipe: `BN_BETA2` (→0.98), `BN_WEIGHT_DECAY` (→0.01, AdamW),
  `BN_CLIPNORM` (0 disables clipping), `BN_DECAY_EPOCHS`/`BN_DECAY_POWER` (polynomial decay after warmup).
  All default to the previous behavior, so other runs are unaffected.
- **Docstrings** corrected: `RMSNorm` *is* LayerNorm/SubLN (matches the paper); module header now cites
  2310.11453 / W1A8 instead of the b1.58 paper.

### Is `val_auc ≈ 0.717` "good"?
For **BitNet specifically**, yes — it's a sensible operating point for a 1-bit model. But it is **not** the
ceiling for this *task*: the separate full-precision / ternary-QAT pipeline reaches `val_loss ≈ 0.13` on
the same staged data, far better than any BitNet run here. So the number is honest for a model whose value
proposition is **hardware efficiency for the L1 trigger** (1-bit weights → cheap, near-multiplier-free
inference in hls4ml), *not* maximal AUC. The reruns answer the live question: does a **paper-faithful
binary** recipe (centralization + large LR + no clipping + weight decay + β₂=0.98) close the
binary↔ternary gap, or is ternary a justified deviation?

### Paper-recipe reruns — establishing the 1-bit (binary) model
All max-size (D256/L8/FFN1024), paper recipe (warmup 1 ep + poly-decay 40 ep, β=(0.9,0.98), wd 0.01,
**centralized** weights, L1 off, bs 256, ES val_auc/max/p10). This establishes the **binary `{−1,+1}` 1-bit
model** the abstract evaluates, plus a binary **LR sweep** to find this small model's usable window. (A
matched-recipe **ternary** run is shown for reference but is an *appendix* exploration off the current
abstract — full discussion in the *Appendix*.)

**First result (the headline): the paper's large LR is mis-sized for this tagger.** At the paper's
**LR=1e-3**, both binary runs land at only **~0.70** — under the ternary 0.717 and, more tellingly, with
unhealthy dynamics. With clipping, **`kai-bn-paper-bin-clip` peaked at epoch 1 (val_auc 0.6967) then
degraded for 10 epochs → early-stopped at 11**. Without clipping, **`kai-bn-paper-bin-noclip` climbed a
bit longer to epoch 13 (best 0.7005) → early-stopped at 23** — i.e. at this hot LR the gradient clip was
*counterproductive* (it froze progress at epoch 1), and removing it bought only +0.004. Either way 1e-3
is too hot: the paper's 1e-3–2.4e-3 targets **125M+ param LLMs**, and this tagger is ~1000× smaller with
an empirically-best LR of **1.5e-4**. So the matched-LR runs below (and the binary LR bracket) are the
scientifically meaningful ones; the 1e-3 jobs are kept only as the "too-hot" data point.

| job | weights | LR | clipping | best val_auc | health |
| --- | --- | --- | --- | --- | --- |
| `kai-bn-paper-bin-clip`   | binary  | 1e-3   | clip 1.0 | 0.6967 | peaks ep1 → ES@11 (LR too hot) |
| `kai-bn-paper-bin-noclip` | binary  | 1e-3   | none     | 0.7005 | ep13 → ES@23 (clip-off helps at hot LR) |
| `kai-bn-paper-bin-lr3`    | binary  | 3e-4   | clip 1.0 | 0.7284 | peaks ep1 → ES@11 (still a bit hot) |
| **`kai-bn-paper-bin-lr15`** | **binary (the 1-bit model)** | **1.5e-4** | clip 1.0 | **0.7530** | **best ep4 → ES@14; train 0.745 ≈ val 0.753 (healthy)** |
| `kai-bn-paper-tern-clip` *(appendix)* | ternary | 1.5e-4 | clip 1.0 | 0.7685 | best ep14 → ES@24; train 0.767 ≈ val 0.769 (healthy) |

**Verdict — the paper fixes demolish the old binary wall; the 1-bit model is strong.**

- **Binary `{−1,+1}`: 0.664 (old recipe) → 0.7530.** A **+0.089** jump from the zero-mean centralization
  (`Sign(W − mean W)`, Eq 1) + correct LR (1.5e-4) + β₂=0.98 + weight decay + warmup/decay. The "~0.66
  binary precision wall" of Findings #16–#17 was **never a precision limit** — it was a *recipe* limit.
- **LR window (binary):** 1e-3 → ~0.70 (peaks ep1), 3e-4 → 0.728 (peaks ep1), **1.5e-4 → 0.753 (peaks
  ep4, stable)**. Monotonic: lower LR is better down to 1.5e-4, which is the first to *not* peak at epoch
  1 — i.e. the first LR the model actually trains at. (1e-4 may add a little more; untested.)
- **This binary @ 1.5e-4 IS the 1-bit model the abstract evaluates.** Every attention/FFN MAC becomes
  XNOR+popcount → FPGA **LUT/logic, not DSP** — the whole point of a compact, low-latency trigger. Its
  efficiency is benchmarked against the **vanilla (FP32)** and **W8A8** versions (the abstract's comparison)
  in *1-bit vs the vanilla & 8-bit versions* below, and its quantization headroom in
  *Quantization-aggressiveness axis*. The matched-recipe ternary `0.7685` (+1.5 pts) is retained only as an
  appendix accuracy-vs-resource ceiling reference — *not* the abstract's comparison.

Best binary model saved at `/data/outputs/qk-paper-binary-lr15/bitnet/noNorm_train_bitnetJetTagModel.h5`
(ternary reference at `/data/outputs/qk-paper-ternary/bitnet/…`). The binary model is the candidate for the
hls4ml resource/latency step.

---

## 1-bit vs the vanilla & 8-bit versions (the abstract's central comparison)

The revised abstract's headline question: **how much tagging efficiency does the 1-bit model give up versus
the vanilla (full-precision) version**, and where does a conventional 8-bit quantization sit between them? To
answer it cleanly I added a single **`BN_VARIANT`** knob so the *identical* architecture (D256/H8/L8/FFN1024),
data, and training recipe (paper recipe: LR 1.5e-4, warmup 1 + poly-decay 40, β₂=0.98, wd 0.01, bs 256,
ES val_auc/max/p10) run at three precisions — the **only** thing that changes is the numeric format of the
BitLinear matmuls:

| `BN_VARIANT` | weights | activations | what it is |
| --- | --- | --- | --- |
| `vanilla` | FP32 | FP32 | full-precision upper bound; every MAC is a real multiply (most DSP-hungry) |
| `w8a8`    | 8-bit absmax (per-tensor) | 8-bit absmax (per-token) | commonplace fixed-point quantization |
| `bitnet`  | **binary `{−1,+1}`** | 8-bit absmax (per-token) | **the 1-bit model** (W1A8); MAC → XNOR+popcount → LUT |

Because the three share everything but precision, the differences in `val_auc` isolate **the cost of going
1-bit** — exactly what the abstract sets out to quantify. (This replaces the earlier abstract's
binary-vs-ternary comparison; ternary is now an appendix reference.)

**Runs** (NRP, distinguishable W&B names under project `bnjettag-bitnet`):

| variant | W&B run | best val_auc | Δ vs vanilla | notes |
| --- | --- | --- | --- | --- |
| vanilla (FP32) | `VANILLA-FP32-D256-L8-FFN1024-lr1p5e4-baseline` | **0.7703** | — | efficiency upper bound; 18 ep (ES) |
| W8A8 (8-bit)   | `BASELINE-W8A8-D256-L8-FFN1024-lr1p5e4` | **0.7719** | **+0.0016** | on par with FP32 (within noise); 25 ep (ES) |
| **1-bit (binary, W1A8)** | primary binary run (`kai-bn-paper-bin-lr15`) | **0.7530** | **−0.0173** | **the target model**; 14 ep (ES) |

*(Both baseline Jobs `kai-bn-vanilla-fp32` and `kai-bn-w8a8` completed (`Succeeded`) on NRP; best `val_auc`
read from each run's live log. The 1-bit anchor is the established **0.7530** from the paper-recipe binary run
above — same recipe, same data, same architecture.)*

**What we found, and what it means.** The efficiency ordering is **W8A8 ≈ vanilla > 1-bit**:

- **8-bit quantization is essentially lossless** — W8A8 **0.7719** vs vanilla **0.7703** (**+0.16 pt**, i.e.
  statistically tied / a hair above, within run-to-run noise). So conventional 8-bit gives up **no** tagging
  efficiency here.
- **Going 1-bit costs ~1.7 pts of efficiency vs full precision** — binary **0.7530** vs vanilla **0.7703**
  (**−1.73 pts**). That is the abstract's headline number: the *price* of the most aggressive weight
  compression, measured with everything else held identical.
- **And it is a small price, bought back many-fold in hardware.** Vanilla's and W8A8's MACs are **real
  multiplies that consume scarce DSPs**; only the 1-bit model's `{−1,+1}` MACs become **XNOR+popcount on
  LUT/logic (DSP = 0)** — shown in *hls4ml / FPGA resources & latency*. W8A8 matches vanilla's accuracy but
  stays multiply-based (DSP-bound), so it does **not** get the 1-bit model's LUT-mapping win. A ~1.7-pt
  efficiency give-up in exchange for eliminating the DSP bottleneck is precisely the trade the abstract argues
  for a compact, low-latency trigger.

The *Quantization-aggressiveness axis* below then asks how much **further** the 1-bit model can be pushed
(activation precision A8→A6→A4) before that efficiency degrades.

---

## Quantization-aggressiveness axis (W1A8 → A6 → A4)

The abstract's second question: *"how aggressively [the 1-bit transformer] can be quantized before its
tagging efficiency, resource consumption, and latency degrade."* Weights are already at the floor (1-bit), so
the remaining knob is how many bits the **activations** carry between BitLinear layers. The paper trains
W1A**8** (8-bit absmax activations, Qb=2⁷). On FPGA, the activation width sets the on-chip storage for
intermediate tensors and — crucially — the **bit-width of the multipliers feeding the popcount/accumulate
trees**. Halving it (8→4) roughly halves activation BRAM and shrinks every accumulator, so this axis trades
tagging efficiency directly against the resource/latency budget — the "how far can we push it" the abstract
asks for.

**Implemented** (env `BN_ACT_BITS`, default 8; generalized `activation_quant`): symmetric signed b-bit absmax,
levels spanning `[-2^(b-1), 2^(b-1)-1]`. At 8-bit this is a numerical no-op vs the old hard-coded path
(`[-128,127]`); 6-bit → `[-32,31]`; 4-bit → `[-8,7]`. STE unchanged (forward quantize, backward identity).

**Controlled sweep** — changing **only** the activation bit-width, on the **canonical softmax model**
(`BN_SOFTMAX_FREE=0`): the *same* configuration as the deployed headline W1A8, so the absolute numbers are
headline-consistent (no base caveat). A8 here independently reproduces the headline (0.7524 vs 0.7530, seed
noise). Re-run 2026-06-24 as jobs `kai-bn-paper-bin-sm-a{8,6,4}` (W&B `…-SOFTMAX-sweepA{8,6,4}`).

| job | act precision | weights | attention (base) | best val_auc | health |
| --- | --- | --- | --- | --- | --- |
| `kai-bn-paper-bin-sm-a8`         | **A8** (baseline) | binary | **softmax** | **0.7524** | best ep2 → ES@12; train 0.743 ≈ val 0.752 |
| `kai-bn-paper-bin-sm-a6`         | A6                | binary | **softmax** | **0.7507** | best ep3 → ES@13; train 0.740 ≈ val 0.751 |
| `kai-bn-paper-bin-sm-a4`         | A4                | binary | **softmax** | **0.7437** | best ep2 → ES@12; train 0.729 ≈ val 0.744 |

**Verdict — gentle, monotonic degradation; no cliff.** A8 → A6 → A4 = **0.7524 → 0.7507 → 0.7437**: A6 is
**within noise** of A8 (−0.2 pts) and dropping all the way to 4-bit activations costs only **~0.9 pts** total,
with **no precision cliff** between 8 and 4 bits. Every run stays healthy (val ≥ train, no overfit) — so the
loss is genuine precision-floor capacity, not optimization breakdown. The signature is the familiar one: low-
precision runs **peak almost immediately** (A8 @ ep2, A6 @ ep3, A4 @ ep2) and then early-stop, i.e. the coarse
activation grid caps extractable signal and the model gets there fast. Even the most aggressive setting (A4,
levels `[-8,7]`) holds **0.744** — within ~0.9 pts of full A8.

*Cross-check — softmax-free base (appendix variant), same sweep:* A8 0.7562 → A6 0.7450 → A4 0.7381
(`kai-bn-paper-bin-sffree{,-a6,-a4}`). The **relative** degradation is base-independent (both bases hold within
~1 pt down to A4, no cliff); the canonical-softmax absolute numbers above are the headline-consistent ones.

**Takeaways for deployment.** The cost of pushing the quantization harder is small and smooth, so activation
precision is a clean **resource/latency dial**: **A6** is the sweet spot (essentially free, −0.2 pts, for a
25%-narrower activation datapath and smaller popcount accumulators); **A4** is viable when the FPGA budget is
very tight (−0.9 pts, ~half the activation BRAM and the narrowest multipliers). The exact trade-off resolves
once paired with the hls4ml resource/latency numbers below (now backed by a real Vitis C-synthesis — see that
section). Best models saved at `/data/outputs/qk-paper-binary-sm-a{8,6,4}/bitnet/noNorm_train_bitnetJetTagModel.h5`.

**Answering the abstract's "how far":** down to **A4** the 1-bit model still tags at **0.744** (−0.9 pts vs
A8), with no precision cliff — so it can be quantized aggressively (W1A4) before efficiency meaningfully
degrades, and the matching resource/latency savings are quantified in *hls4ml / FPGA resources & latency*.

---

## hls4ml / FPGA resources & latency (what going 1-bit buys)

The abstract asks how the 1-bit model affects *"resource consumption and latency"* once converted to HLS via
hls4ml. The gold-standard route is hls4ml → Vitis/Vivado HLS **C-synthesis** (csynth), which emits exact
LUT/FF/DSP/BRAM and latency per layer. **The NRP Nautilus cluster has no Xilinx HLS backend** (the upstream
`HLS_qk_Roc_Tracing.py` hard-codes a Vitis 2023.2 path from a different cluster), so synthesized numbers are
not reachable here. I therefore deliver the resource/latency story **two ways that reinforce each other**:
(1) a real-hls4ml **convertibility + bit-accuracy proof** of the binary core run **across the whole A8/A6/A4
quantization axis** (with the generated firmware inspected for the binary→logic DSP=0 mapping and the
activation datapath narrowing), now **extended on the newer hls4ml 1.3.0 stack to LayerNorm and a full
SubLN+binary-dense block**; and (2) an **analytical per-component resource + latency model** anchored to
published binary-tagger synthesis and *validated against the firmware hls4ml actually generated*. Together
these show the hardware payoff the abstract claims for going 1-bit — and the latency figure-of-merit (~1 µs @
400 MHz at the device-fit reuse factor). The one thing still out of reach on NRP is the final Vitis csynth that
turns these into exact LUT/FF/DSP/BRAM and latency-in-cycles; the emitted projects are ready for that handoff.

### 1. Binary core converts AND emulates bit-accurately — across the whole quantization axis (real hls4ml)

`code/hls/sweep_precision.py` (hls4ml **0.8.1**, qkeras 0.9.0, tf 2.11.1; Job `kai-hls-sweep`) pushes the dominant
primitive — a binary **FFN block** (`fc1` 256→1024 `binary(alpha=1)` → `quantized_relu(A,2)` → `fc2` 1024→256
`binary(alpha=1)`) — through the real toolchain at **A8, A6 and A4**, the *same* activation-precision axis as
the training sweep above. At every precision it converts and reproduces the Keras model **to the LSB**, and the
generated firmware shows precisely what the abstract asks about — what stays free, and what narrows:

| act precision | emul corr (N=1000) | weight C-type (fw) | activation datapath `layer4_t` (fw) | accumulator (fw) | io / reuse |
| --- | --- | --- | --- | --- | --- |
| **A8** | **1.000000** | `ap_uint<1>` ×2 | `ap_ufixed<8,2,…SAT>` | `ap_fixed<32,16>` | io_parallel, RF=1 |
| **A6** | **1.000000** | `ap_uint<1>` ×2 | `ap_ufixed<6,2,…SAT>` | `ap_fixed<32,16>` | io_parallel, RF=1 |
| **A4** | **1.000000** | `ap_uint<1>` ×2 | `ap_ufixed<4,2,…SAT>` | `ap_fixed<32,16>` | io_parallel, RF=1 |

Three facts straight from the generated `defines.h` / `parameters.h` (not inferred):
- **DSP = 0 is structural and precision-independent.** Both dense kernels type as **`ap_uint<1>`** at *every* A
  (`weight2_t`, `weight5_t`). A 1-bit weight cannot drive a DSP48 multiplier input, so hls4ml emits the
  "multiply" as a conditional-negate (sign-select) + LUT adder tree — **zero DSPs, at A8/A6/A4 alike**. Fewer
  activation bits don't change *that* the MACs are DSP-free; they change how wide the adders are.
- **Activation precision IS the firmware datapath dial.** `layer4_t` — the `quantized_relu` output buffer
  between `fc1` and `fc2` — narrows **`ap_ufixed<8,2>` → `<6,2>` → `<4,2>`** across the sweep, with the `SAT`
  saturation clip (range [0,4)) preserved. That is the on-chip activation storage + adder-tree input width the
  resource/latency budget rides on, shrinking exactly as A drops — the hardware face of the 0.756→0.745→0.738
  efficiency trade.
- **Lowest-latency regime throughout.** `io_parallel`, `strategy=latency`, `reuse_factor=1` at every precision —
  the fully-unrolled dataflow the L1 trigger wants (before device-fit folding, §4).

`corr = 1.000000` at all three precisions ⇒ the firmware is bit-accurate to Keras at every operating point, so
each is a faithful starting point for csynth. Projects emitted to `/data/outputs/hls/binary_ffn_a{8,6,4}_prj`
(+ `.tar.gz`), ready for an off-cluster Vitis csynth.

**Accumulator note.** I pin the accumulators to `ap_fixed<32,16>` for bit-accuracy: the default `fixed<16,6>`
saturates at ±32 but `fc1` sums 256 signed ±1 terms reaching ±50+ (the first pass diverged at `corr=0.24`; a
naïve widen-everything only reached 0.85 because it also removed the `quantized_relu` `SAT` clip — Keras `fc1`
pre-acts have std ≈ 16, so nearly all clip to ~4 and an unclipped HLS act blows up ~15×). Fix in
`sweep_precision.py`/`stage_a_fix.py`: widen accum/result/io, leave `act` **native**. So the accumulator width
is held fixed across the sweep here; the *minimal* per-A accumulator (which would narrow 18→16→14 b with A) is
in the analytical model (§4).

### 2. The newer stack (hls4ml 1.3.0) converts LayerNorm and a full binary block — only the attention matmul remains

Phase-1's blocker was that the TF-2.11 image pins Python 3.8 → hls4ml **0.8.1**, which rejected
`LayerNormalization` and `EinsumDense`. I stood up the newer stack (Job `kai-hls-full`: Python 3.10,
**hls4ml 1.3.0**, TF 2.14.1) and re-ran `code/hls/full_transformer_probe.py`:

| piece | hls4ml 0.8.1 | hls4ml 1.3.0 | note |
| --- | --- | --- | --- |
| binary `QDense` FFN / projection | ✅ converts + bit-accurate | ✅ | the dominant primitive |
| **`LayerNormalization` (SubLN)** | ❌ unsupported | **✅ converts** | the Phase-1 "riskiest piece" — now clears |
| **full block** SubLN→binary-proj→SubLN→binary-FFN | ❌ | **✅ converts end-to-end** | norm + binary-dense backbone composes |
| attention **act×act** score matmul (Q·Kᵀ, scores·V) | ❌ | ❌ (keras_v2 parser) | see below |

On 1.3.0 the gap collapses from "LayerNorm **and** attention both fail" to **just the attention score matmul**.
Two precise findings on that last piece:
- `EinsumDense` is still **unsupported by hls4ml 1.3.0's Keras-v2 parser** (`Unsupported layer type: EinsumDense`).
- More fundamentally, the attention scores are an **activation×activation** contraction (Q·Kᵀ, scores·V), which
  Keras `EinsumDense` *cannot even express* — that layer always contracts an activation against a **trainable
  kernel**, not a second activation. So the attention matmul is less a "missing parser handler" than a
  **custom-op** job: hls4ml's **Extension API** (or the HGQ2 / Keras-v3 MHA frontend, Vitis-only) is the route.
  Dropping softmax (appendix variant) doesn't change this — the two batched matmuls remain — but it does delete
  the only *non-binarizable* op once they are in place.

Net: the **binary-dense + SubLN backbone — ~90%+ of the model's arithmetic, and exactly the layer types that
worried us — is convertible today** (FFN on 0.8.1; LayerNorm + full block on 1.3.0). The remaining firmware work
is the attention act×act matmul via the Extension API, plus a Vitis csynth pass for the actual resource/latency
numbers. (The 1.3.0 LayerNorm/full-block conversions were run as a smoke test — `convert` returns a built
`ModelGraph`; I did not persist those projects to disk this pass, which is a one-line `write_project()` away.)

### 3. Per-component HARD structural counts (exact — `code/hls/resource_model.py`)

These are assumption-free (pure architecture arithmetic), and are the literal "per-component breakdown":

| component | inst×tokens | binary MACs | weight bits |
| --- | --- | --- | --- |
| input_proj (14→256)  | 1×10 | 35,840 | 3,584 |
| attn.W_q/W_k/W_v (256→256) | 8×10 ea | 5,242,880 ea | 524,288 ea |
| attn.W_o (256→256)   | 8×10 | 5,242,880 | 524,288 |
| ffn.fc1 (256→1024)   | 8×10 | 20,971,520 | 2,097,152 |
| ffn.fc2 (1024→256)   | 8×10 | 20,971,520 | 2,097,152 |
| head_fc1 / head_fc2  | 1×1  | 65,536 / 256 | 65,536 / 256 |
| **TOTAL binary-weight** | | **63,016,192** | **6,360,832** (6.36 Mbit) |

- **99.35%** of all MACs are binary (DSP-free); the only **real** act×act multiplies are the **409,600** (0.65%)
  attention-score products (Q·Kᵀ and scores·V). With softmax-free attention the post-score path is just ReLU +
  a constant 1/N shift — no exp, no reciprocal.
- **51 SubLN/LayerNorm** instances (one per BitLinear), 184,972 normalized elems/inference — confirmed
  convertible on hls4ml 1.3.0 (§2), so no longer a question mark.

### 4. DERIVED resource estimates vs activation precision (RF=1; labelled cost factors)

Cost factors are explicit and swappable for csynth later: binary MAC → ~A LUT / **0 DSP**; real MAC → 1 DSP at
A≥7, packs into LUTs at A≤6.

| precision | binMAC LUTs (RF=1 upper bnd) | attn DSPs | accum width |
| --- | --- | --- | --- |
| A8 | 504,129,536 | 409,600 | 18 b |
| A6 | 378,097,152 | 0 (LUT-packed) | 16 b |
| A4 | 252,064,768 | 0 (LUT-packed) | 14 b |

The **DSP = 0** result for the binary core is **structural** (no multiplier exists), not an estimate — and the
firmware in §1 backs it. The LUT figures are an **RF=1 upper bound** (one MAC = one LUT-slice); real firmware
time-multiplexes with a reuse factor RF, dividing parallel LUTs by ~RF and multiplying latency by ~RF.
**Device-fit on XCVU13P** (the part the upstream HLS script targets; LUT 1.728M): RF=1 is ~292× too big for
this research-size model, so a real trigger folds to **RF ≈ 583** to sit under 50% LUT → latency ~RF cycles
≈ **1 µs @ 400 MHz**. Two independent reasons this is conservative: hls4ml *pre*-synth LUT is overstated
**~3–10×** vs real logic synthesis, and measured binary jet-taggers report **~1–8% LUT / 0% DSP / 0 BRAM**
(Ngadiuba *et al.*, arXiv:2003.06308). So the qualitative story is solid and the LUT figure is precisely the
"shrink-or-fold" signal the **model-size** and **activation-precision** (quantization-aggressiveness) knobs
exist to turn.

### 5. Softmax-free attention op delta (per inference) — *appendix variant*

The softmax-free attention variant (now an appendix exploration) would, *if adopted*, delete the only
non-binarizable op in the attention core. Recorded here because it directly affects FPGA resources:

| | exp() evals | reciprocals | norm mults |
| --- | --- | --- | --- |
| softmax core (default, the 1-bit model) | 6,400 (LUT tables/BRAM) | 640 (DSP/LUT-heavy) | 6,400 |
| **softmax-free** (appendix) | **0** | **0** | 0 (constant 1/N shift) |

Dropping the softmax would delete **all 6,400 exp tables and 640 reciprocals** — the single most expensive,
non-binarizable attention op — keeping the whole MHSA on LUT/logic. The shipped 1-bit model keeps the
softmax (its `exp`/reciprocal cost is the one part of the attention core that is *not* DSP-free).

**Bottom line.** With real hls4ml I proved the **binary core converts and emulates bit-accurately at A8, A6 and
A4** (corr = 1.000000 ×3) and that hls4ml emits its weights as **1-bit LUT logic with zero DSPs at every
precision**, with the activation datapath narrowing `ap_ufixed<8/6/4,2,SAT>` exactly along the
quantization-aggressiveness axis. On the newer **hls4ml 1.3.0** stack, **LayerNorm (SubLN) and a full
SubLN→binary-proj→SubLN→binary-FFN block convert end-to-end** — closing the Phase-1 gap, so the only piece left
is the attention **act×act** score matmul (an Extension-API custom op, since Keras `EinsumDense` can't express
an activation×activation contraction). The per-component model quantifies the breakdown (63.0M binary MACs /
99.35% DSP-free; 6.36 Mbit weights; 51 SubLNs; 0.65% real multiplies) and the device-fit reality. **The Vitis
C-synthesis has now been run** (off-cluster on the group's `mulder` box, Vitis HLS 2023.2, 2026-06-24): the
binary FFN at the folded **RF=256** point synthesizes to **DSP 0 · LUT 25.5% · FF 7.3% · BRAM 1.2%** on a VU13P,
latency **520 cycles ≈ 1.3 µs @ 400 MHz** (Fmax ≈ 568 MHz), **confirming DSP=0 in silicon estimates** at A8/A6/A4
and landing ~2× under the analytical fold estimate — full table in `results/hls_resource_table.md` §B, raw reports
in `results/csynth/`. (BRAM is 0 only fully-unrolled; folding to fit parks the binary weights in ~1.2% BRAM —
DSP=0 is the fold-independent win.) Artifacts: `code/hls/run_csynth.py`, `code/hls/sweep_precision.py`, `code/hls/resource_model.py`,
`code/hls/full_transformer_probe.py`, `code/hls/stage_a_fix.py`, `code/hls/convert_probe.py`, `methods/hls4ml_findings.md`;
generated projects at `/data/outputs/hls/binary_ffn_a{8,6,4}_prj` (+ `binary_ffn_prj_wide`);
logs `sweep_precision.log`, `full_transformer_probe.log`, `resource_model.log`.

---

## Repro / monitor

```bash
# code + data are on the kai-data PVC; W&B key is in secret kai-wandb.
# The 1-bit model (val_auc 0.7530); swap the file for vanilla / w8a8 / sweep variants.
kubectl apply -f code/jobs/training/kai-bn-train-paper-binary-lr15.yaml -n cms-ml
kubectl get job,pods -n cms-ml | grep kai-bn
kubectl logs -f job/kai-bn-paper-bin-lr15 -n cms-ml
# on-disk log survives pod completion:
#   kubectl exec kai-setup -n cms-ml -- tail -f /data/outputs/qk-paper-binary-lr15/train.log
kubectl delete job kai-bn-paper-bin-lr15 -n cms-ml            # stop / clean up
```
Other training jobs live in `code/jobs/training/` (vanilla, w8a8, ternary, softmax-free, A6/A4 sweep);
hls4ml jobs in `code/jobs/hls/`. Superseded/dead-end jobs are kept in `archive/jobs/`.
Live metrics: Weights & Biases project **bnjettag-bitnet**.

---

## Appendix — explorations beyond the current abstract

These two axes were run under an **earlier, longer abstract** and are **kept here for completeness** (data not
deleted), but they are **not part of the current narrative**. The current abstract compares the 1-bit model to
the **vanilla (FP32)** and **8-bit (W8A8)** versions and studies **quantization aggressiveness** — neither a
ternary weight comparison nor a softmax-free attention contrast. Both results below are healthy, real runs;
they simply answer questions the revised abstract no longer asks.

### A1. Ternary `{−1,0,+1}` weights (b1.58) — accuracy-vs-resource ceiling reference

The earlier abstract compared **binary vs ternary** weights. Ternary `{−1,0,+1}` is the **separate "b1.58"
paper** (arXiv:2402.17764), *not* the binary BitNet (2310.11453) this work targets — it is enabled off-paper
by `BN_TERNARY=1`. Run at the matched paper recipe (LR 1.5e-4, warmup 1 + poly-decay 40, β=(0.9,0.98), wd 0.01,
bs 256), ternary scores **0.7685** vs the binary 1-bit model's **0.7530**:

| job | weights | LR | best val_auc | health |
| --- | --- | --- | --- | --- |
| `kai-bn-paper-bin-lr15` (the 1-bit model) | binary `{−1,+1}` | 1.5e-4 | 0.7530 | best ep4 → ES@14; train 0.745 ≈ val 0.753 |
| `kai-bn-paper-tern-clip` | ternary `{−1,0,+1}` | 1.5e-4 | **0.7685** | best ep14 → ES@24; train 0.767 ≈ val 0.769 |

**Finding (kept for reference): binary nearly catches ternary — a 1.55-pt gap** (down from the old recipe's
5.3-pt gap, 0.664 vs 0.717). Binary reaches **~98 %** of ternary's AUC with the recipe held identical and only
the weight set changed; ternary peaks later (ep14 vs ep4) and a touch higher, as expected from its extra `0`
state. **Why the 1-bit (binary) model is still the target**, despite ternary's edge: binary `{−1,+1}` keeps
every MAC a pure **XNOR + popcount → LUT** with no DSP and no zero-mask, whereas ternary's `0` state breaks
that mapping (it needs a zero-mask / select), costing the very LUT-purity the abstract's hardware argument
rests on. So ternary stands only as an **accuracy ceiling reference**; the deployed model is binary. Ternary
reference model at `/data/outputs/qk-paper-ternary/bitnet/noNorm_train_bitnetJetTagModel.h5`. (For context,
the separate full-precision/ternary-QAT pipeline reaches `val_loss ≈ 0.13` on this data, so ~0.77 is not the
task ceiling — closing it is open-ended research beyond precision knobs.)

### A2. Softmax-free attention (binarizable attention core)

The earlier abstract contrasted a conventional softmax core with a **softmax-free** attention variant designed
to stay fully binarizable. Motivation is hardware, not accuracy — in the BitNet MHSA the Q/K/V and output
projections are already binary (XNOR+popcount), so the **only** non-binarizable, DSP/LUT-hungry op left in the
attention core is the **softmax** itself: `exp(·)` needs lookup tables and the row-sum normalization needs a
data-dependent division.

**Variant** (env `BN_SOFTMAX_FREE=1`, default 0 = softmax; `BitMHSA.call`):

```
softmax:       A = softmax(QKᵀ / √d) ;            ctx = A · V
softmax-free:  A = ReLU(QKᵀ / √d)    ;            ctx = (A · V) / N      # N = particles/jet = 10
```

ReLU is a comparator/mux and `1/N` is a **constant** scale (fixed shift) — no `exp`, no row-sum division — so
the whole MHSA stays on LUT/logic fabric. The constant `1/N` is stable here because the sequence is short
(N=10) and every BitLinear is preceded by SubLN. Identical to the binary `bin-lr15` run, changing **only** the
attention nonlinearity:

| job | attention | weights | LR | best val_auc | health |
| --- | --- | --- | --- | --- | --- |
| `kai-bn-paper-bin-lr15`   | softmax (default, the 1-bit model) | binary | 1.5e-4 | 0.7530 | best ep4; train 0.745 ≈ val 0.753 |
| `kai-bn-paper-bin-sffree` | ReLU(QKᵀ)/N (softmax-free) | binary | 1.5e-4 | **0.7562** | best ep5 → ES@15; train 0.746 ≈ val 0.756 |

**Finding (kept for reference): deleting the softmax is statistically free** — 0.7562 vs 0.7530 (**+0.003**,
within run-to-run noise), both healthy. So a softmax-free trigger could drop its single most expensive,
non-binarizable attention op at no measured tagging cost, leaving the entire MHSA on LUT/logic. It is **not**
part of the current abstract (which dropped the softmax-vs-softmax-free axis), but it remains a promising,
hls4ml-friendly option if the attention `exp`/reciprocal becomes the FPGA bottleneck. Softmax-free model at
`/data/outputs/qk-paper-binary-sffree/bitnet/noNorm_train_bitnetJetTagModel.h5`. (The W1A8→A6→A4 activation
sweep in *Quantization-aggressiveness axis* is now run on the **canonical softmax base** — `kai-bn-paper-bin-sm-a{8,6,4}`,
0.7524/0.7507/0.7437 — with this softmax-free sweep kept there as the cross-check.)
