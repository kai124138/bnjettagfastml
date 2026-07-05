# Decisions — BNJetTag

Key choices and their rationale. **Newest on top.**

## 2026-07-04 — HGQ2 rebuild effort: target models, binary pinning, and the r6-small question
**Context.** New directive: rebuild the binary model in HGQ2 (Keras-3 successor of HGQ, PyPI
`HGQ2` 0.1.9, import `hgq`), verify bit-accuracy via the proxy/bit-exact flow, compute native
EBOPs across the quant sweep, and synthesize via hls4ml (1.2+ has native HGQ2 + MHA support —
mainlined by arXiv:2510.24784) on mulder. Everything config-driven, results in a structured
store keyed by config hash (`code/hgq2/`).
**Decision 1 — which model.** The directive names "the era-2 model (10×16=160, **D=32, L2**,
5-class)" — but era-2 round-5 is frozen at `large` D256/H8/L8/FFN1024, and **no era-2 D32/L2
checkpoint exists anywhere** (the only local D32 checkpoint is era-1 2-class; era-1↔era-2
porting is forbidden). Resolution: (a) the verified rebuild path runs on the **round-5 large
checkpoints** (real weights, W&B → `models/r5/`, all 8 fetched 2026-07-04); (b) the pipeline
is config-driven so the D32/L2 spec is one config away; (c) round-6-small training YAMLs
(`kai-bn6s-*.yaml`, PVC-free pattern, D32/H4/L2/FFN64, era-2 data, r5 recipe, 5 variants ×
seed 1) were generated and **server-dry-run validated but NOT launched** — the sandbox's
permission gate correctly flagged new GPU training as needing Kai's explicit approval.
Fire with `variants/launch_r6s_staged.sh` when approved.
**Decision 2 — binary pinning in HGQ2 is verify-first.** Fixed-point sets at width 1 do not
contain {−1,+1} (bipolar ≠ fixed-point). The exact HGQ2 quantizer config that passes a ±1
kernel through unchanged AND reports 1-bit for EBOPs must be established **empirically**
before any port (candidate: frozen kif/kbi at 2-bit repr holding ±1 values, or the
`quantizers` package's `binary_quantize`); whatever wins is logged in the constraints map.
Any config where the effective weight set is not exactly {−β,+β} is a STOP condition.
**Decision 3 — static activation quantizers replace dynamic per-token absmax.** The QKeras
model's activation quant (per-token dynamic absmax) is unimplementable in static fixed-point
hardware. The HGQ2 rebuild substitutes per-tensor static quantizers calibrated on val data —
same substitution full_model_csynth.py made (corr 0.998 static vs 0.99998 dynamic). The
honest verification gates are therefore: (i) HGQ2-Keras ↔ hls4ml bit-exactness (HGQ2's
design guarantee), and (ii) HGQ2 rebuild ↔ trained-QKeras correlation + recovered macro-OvR
AUC vs the verified r5 npz scores. Bit-exactness vs the *trained* model is impossible by
construction and is not claimed.
**Decision 4 — β-scale folding plan.** Effective BitNet weights are {−β,+β} (per-tensor β).
β is folded exactly where LayerNorm/relu invariances allow (Q·K score scale absorbs
β_q·β_k/√d; V's β dies in Wo's input LN; fc1/head_fc1 β dies in the next LN with bias/β
rescale) and applied as an explicit per-tensor affine only where it changes the math
(residual contributors: input_proj, Wo, fc2, head_fc2 = 2L+2 sites). Verified numerically
during the port.

## 2026-07-02 — TOP next experiment: LayerNorm-DSP elimination (chase a fully 0-DSP model)
**Decision.** The next experiment to prioritize (round-6 candidate) is eliminating the
LayerNorm DSP. Rationale: the full-model csynth backfill (see experiment-log 2026-07-02)
proves the binary matmul core is already **0 DSP** and that **100% of the model's 1,049 DSP
is the 51 SubLN normalizers** (variance + inv-sqrt at fixed<32,16>). So a **fully 0-DSP**
BitNet transformer is within reach — remove/replace the norm's DSP and the whole model is
DSP-free.
**Why this is the differentiator.** arXiv:2510.24784 just published **sub-µs FP transformers**
for the same L1-trigger jet-tagging setting — excellent latency, but they spend DSPs. We will
not out-latency them (our composed spatial latency is ~58.5 µs upper bound; folded designs
trade further). The axis where we are **structurally unanswerable is DSP = 0**: a fully 0-DSP
transformer is a claim an FP transformer cannot make. That is the headline to chase, not latency.
**Design — two axes.**
- *Axis A (train/arch, era-2 data, NRP):* find the cheapest norm that holds macro-OvR AUC.
  Use the EXISTING knobs — `BN_NORM_PLACEMENT` {per_linear (51 norms, default), shared_prenorm
  (~3× fewer → ≈350 DSP, and the canonical pre-norm placement anyway)} × `BN_NORM_TYPE`
  {layernorm (mean-subtracting SubLN), rmsnorm (drops mean-centering — but the rsqrt/DSP
  remains), none (0-DSP control/floor)}. If none of these both-holds-AUC and hits ~0 DSP, add a
  genuinely DSP-free norm option: power-of-2 / shift-based scale, or a frozen/precomputed scale
  so the norm collapses to a constant multiply (LUT-only). Metric: era-2 macro-OvR val AUC vs
  the round-5 layernorm baseline.
- *Axis B (HLS/firmware, mulder):* for the winning norm, re-synthesize (reuse
  `code/hls/full_model_csynth.py`) with a **fixed-point inv-sqrt LUT** and/or narrowed LN
  precision, and measure how far DSP actually drops toward 0. `hls_resource_table.md` §B′
  already flags the inv-sqrt LUT as the lever; the csynth note quantifies the tradeoff
  (default-narrow LN C-sim corr ~0.87 vs fixed<32,16> ~0.9998), so the LUT must be sized to keep
  QKeras↔Vitis corr ≥ 0.997.
**Success / deliverable.** A variant with whole-model **DSP ≈ 0** (or a clean DSP-vs-AUC
frontier) at macro-OvR AUC within noise of the layernorm baseline and C-sim corr ≥ 0.997 →
headline claim: *a fully 0-DSP BitNet transformer jet tagger for the CMS L1 trigger.*
**Sequencing.** Gated on round-5 finishing (frees NRP GPUs, and gives the era-2 layernorm
baseline these variants are measured against; also avoids churning job YAMLs mid-sweep).
Implementation step = ml-engineer scaffolds `kai-bn6-norm-*` job variants (env-var combos —
knobs already exist) + the HLS LUT-inv-sqrt path. **Open question** for physics-researcher/
ml-engineer: does this hls4ml `LayerNormalization` support a table-based inv-sqrt, or does
`full_model_csynth.py` need a custom norm primitive?

## 2026-07-02 — Workspace restructured: two living docs, skills, paper-writer, archive
Kai asked for a setup anyone (including the PI) can walk into and understand. Approved
design (AskUserQuestion, full-tidy): **(1)** two living top-level docs — `00-START-HERE.md`
(the map: folders, the 3-machine architecture, the experiment loop, the logging system) and
`RESEARCH.md` (thesis, abstract, both dataset eras, all verified results with metric/era/
status labels, knowledge ledger, per-number source map). `RESEARCH.md` supersedes the stale
top-level `REPORT.md`. **(2)** Frozen deliverables → `reports/` (June-22 REPORT.md, outline,
slides/notes docx, new `strategy-review-2026-07-02.md`); corrected by dated notes only.
**(3)** Clutter → `archive/` (session exports/zips, top-level runbook duplicates of `nrp/`
— verified byte-identical before moving — and June-22 stray logs + `kai-train-job-d64.yaml`).
**(4)** New `.claude/skills/`: `nrp-training-run`, `verify-roc` (era-aware), `hls-mulder`.
**(5)** New agent `paper-writer` (verified-numbers-only prose); new commands `/status`,
`/log-decision`, `/pi-update`; `/update-report` retargeted to `RESEARCH.md`. **(6)** Uploaded
PDFs filed: BitNet → `papers/bitnet-1bit-ternary/2310.11453_….pdf`, Sub-µs transformers →
`papers/jet-tagging-transformers/2510.24784_….pdf`. References updated in CLAUDE.md,
project-context, HOW-TO-USE, results-analyst/ml-engineer. Verification: all doc-referenced
paths exist; era-1 ROC-test AUCs recomputed from `.npz` (A8 0.7986 ✓, FP32 0.8207 ✓,
A4 0.7886 ✓, n=222,912 each) before being quoted in `RESEARCH.md`.

## 2026-07-01 — Input revised to top-10 constituents × ALL 16 features (supersedes top-32)
Advisor guidance (relayed by Kai, same day as the migration): take the **top 10 particles
sorted by pT, high→low** — a small fixed input close to the previous 10-particle setup —
and **defer** the input-size vs AUC/latency/resources question to a later study ("once
things work"); that's exactly what the `BN_N_PART` knob exists for. The advisor sketched
10×14=140, but Kai chose (AskUserQuestion) to keep **all 16 features → 10×16 = 160 inputs**
rather than drop 2 features to hit 140. Propagated everywhere: `qkerasModel.py` +
`make_roc.py` defaults (BN_N_PART=10), all 8 regenerated `kai-bn5-*.yaml` + preflight,
`kai-roc-r5.yaml` (+ re-embedded make_roc), README. The 2026-07-01 migration entry below
still holds except its "top-32" spec, which this supersedes.

## 2026-07-01 — Migrate ALL training to the public HLS4ML LHC Jet dataset (150 particles)
Kai's directive: all training going forward uses the **HLS4ML LHC Jet dataset (150p)**
(Zenodo, Pierini/Duarte/Tran/Freytsis; arXiv:1804.06913 + 1908.05318), already staged on the
kai-data PVC at `/data/hls4ml_lhc_jet/` (`train/train/` ~62 files, `val/val/` ~26 files,
10k jets each). Choices made with Kai (AskUserQuestion, 2026-07-01): **(1) 5-class** target
g/q/W/Z/t (one-hot columns located BY NAME via `jetFeatureNames`) — head becomes BitLinear(5),
softmax-CE from logits, matches published hls4ml baselines; **(2) top-32 constituents by pT**
× 16 features (re-sorted by the `*_pt` column before truncation, not trusting file order;
`BN_N_PART` knob, default 32). Consequences, all deliberate:
- **Every pre-migration AUC (0.7530/0.7672/0.7703/0.7719, A6/A4, ternary…) is an OLD-dataset
  number** — round-5 starts a fresh baseline table; no cross-dataset comparisons in reports.
- **Peak LR 5e-5 is CARRIED OVER as an assumption**, not re-derived on the new data; flag for
  a spot-check if round-5 training looks unstable.
- The two-AUC discipline survives with cleaner semantics: *val AUC* = training monitor
  (macro-OvR, `validation_split=0.20` on train/ only); *ROC-test AUC* = `make_roc.py` on the
  dataset's own **val/ split, a true never-seen held-out set** (better than the old
  fixed-seed-tail workaround). Headline metric = macro one-vs-rest AUC + per-class AUCs.
- Binary sig-vs-bkg flat-pT reweighting **dropped** (balanced 5-class sample; published
  baselines train unweighted). Old-data checkpoints (10×14 input, 1-logit) are
  format-incompatible → the r4 large-lr05 run was removed from `kai-roc-r5.yaml`.
- `large` arch stays FIXED (D256/L8/H8/FFN1024) but the param count shifts slightly
  (16-feat input projection, 5-logit head) — the old "6,373,633" must not be quoted for new
  runs; `preflight_r5.sh` prints the exact new count and now also validates the dataset files.

## 2026-07-01 — Round-5 = the quantization round, WITH LR-tuned FP32/W8A8 baselines
Round-5 (`kai-bn5-*`, 8 jobs, staged ≤3 concurrent) re-measures everything the round-2→4 LR finding
invalidated, on the FIXED `large` arch: seed-confirm large@5e-5, A6/A4 at the tuned recipe (2 seeds
each), and — the new decision — **re-train the FP32 and W8A8 baselines at 5e-5 too**. Rationale: the
binary model gained +0.0142 from LR tuning; quoting "−0.003 vs FP32" against never-tuned baselines is
the same tuned-vs-untuned trap round-4 caught for "medium beats large". Reporting rule: the honest
baseline per variant = **max(original recipe, 5e-5 run)** — baselines may only get stronger, so any
surviving binary-vs-FP32 gap is real. Corollary: NO tuned number reaches a headline until it is
seed-averaged AND has a ROC-test AUC (`kai-roc-r5.yaml`); the lr15 README table stays as the labeled
reproducible anchor until then.

## 2026-06-29 — Fix the architecture at `large`; adopt EBOPs as the hardware-cost metric
Two coupled decisions. **(1) The architecture is now FIXED** at the upstream main config `large`
(D256/L8/H8/FFN1024, **6,373,633 params** — confirmed by the round-4 preflight + HLS doc; the earlier "~3M"
label was WRONG and is corrected across the reports). The tiny→small→medium→large size sweep and the one-knob
structure ablations are **closed** — their verdict (AUC is monotonic in size and governed by the optimizer LR,
not the architectural knobs; keep all knobs at upstream defaults) stands as the *why-this-size* record, not an
open search. Round-5's "10 more architecture variants" is **on hold**: from here we vary *quantization*, not
*shape*, and any reported model deviating from `large` is labeled explicitly. **(2) Adopt EBOPs (Effective
Bit-Operations, HGQ arXiv:2405.00645) as the primary, synthesis-free hardware-cost axis** alongside the HLS
resource table. Rationale: EBOPs is a *static* number (`code/training/ebops.py`, stdlib-only, milliseconds) that
calibrates to silicon as **EBOPs ≈ #LUT + 55·#DSP**; since the binary core is **DSP=0**, EBOPs ≈ #LUT — a
faithful LUT proxy with no Vitis run. It expresses both thesis axes: (a) *quantize* — W1A8 is **7.65× below
W8A8** / 122.5× below FP32 in EBOPs; (b) *grow at equal cost* — at the W8A8 EBOPs budget a binary model has
~7.65× matmul-MAC headroom. We report EBOPs **and ΔEBOPs (the "change")**. The 7.65× (vs a clean 8×) is honest:
attention is act×act (b_a²), so binarization cannot touch its 0.65% of MACs. Verified end-to-end by
results-analyst (2026-06-29) before publishing — see `results/RESULTS.md` §2f.

## 2026-06-27 — Architecture variants are env-driven knobs on ONE model, not forks
All BitLinear-transformer variants live in the single `qkerasModel.py`, selected by env vars
(BN_NORM_TYPE / BN_NORM_PLACEMENT / BN_POS_ENC / BN_POOL / BN_FFN_ACT, plus the existing
BN_D_MODEL/N_HEADS/N_LAYERS/FFN_DIM). Rationale: keeps a single source of truth (follows the
"listen to the repo" rule — minimal, opt-in deviations from upstream Brainz22), every knob
DEFAULTS to upstream behaviour so the lr15 checkpoint + hls4ml path are untouched, and new
checkpoints self-describe via get_config(). A code generator (`variants/gen_variant_jobs.py`)
emits the per-variant NRP yamls so the recipe stays DRY.

## 2026-06-27 — Variant sweep isolates SIZE and STRUCTURE, holding optimizer+precision fixed
The prior sweep already covered precision × weight-type × attention at fixed D256/L8. So the new
sweep deliberately varies ONLY (a) model size (tiny/small/medium vs the lr15 "large" anchor, all
default-arch → clean size axis) and (b) one structural knob at a time anchored on "small"
(true-RMSNorm, shared pre-norm, no-PE, real-learned-PE, CLS-pool, GELU, softmax-free). Every run
keeps binary weights + A8 + the empirically-best lr15 recipe, so each result is attributable to
the single thing changed. Gate every launch behind a CPU `--sanity` build of all variants.

## 2026-06-22 — The two AUC numbers are both correct
Validation AUC (during training; run README headline table) and ROC-test AUC
(`roc-results/roc_auc.md`, n = 222,912) differ **by design** — different data split and
evaluation. Always label which one a figure is. Do **not** "reconcile" them into a single number.

## 2026-06-22 — DSP = 0 is the headline, and it is structural
Binary `{−1,+1}` weights type as `ap_uint<1>`, so multiplications become sign flips → **0 DSP**,
independent of activation precision. The activation sweep (A8 / A6 / A4) trades AUC for
LUT / latency, **not** DSP. Confirmed by real Vitis HLS 2023.2 C-synthesis on `mulder` (2026-06-24).
