# Code change ledger — every file we authored or edited

A single running record of **all code in this run**: what's upstream, what we edited,
what we wrote from scratch, and — bluntly — **what broke and how we fixed it** (§3, which
also answers "were there bugs in the hls4ml work?").

This file is the *index + chronology + bug log*. It complements but does **not** duplicate
[`../methods/code_changes.md`](../methods/code_changes.md), which holds the **line-level**
diff of the trainer (`qkerasModel.py`, Groups G1–G4). When in doubt: **what/why/when** → here;
**exact lines in the trainer** → `methods/code_changes.md`.

Conventions: paths are relative to the run root. Dates are file mtimes on this machine
(there is no git history for this run, so mtimes are the timeline of record).

---

## 1. File inventory

### 1a. Training (`code/training/`)

| file | origin | what it is | status |
| --- | --- | --- | --- |
| `qkerasModel.py` | upstream + **our edits** | THE trainer; all precisions via `BN_VARIANT`. Edits catalogued as G1–G4 in `methods/code_changes.md`. | **edited** (last Jun 23) |
| `qkerasModel.patch` | **ours** | Phase-0 upstream diff (data path / env / W&B) captured as a standalone patch. | new (Jun 22) |
| `ebops.py` | **ours** | STATIC EBOPs/BOPs profiler. Encodes the closed-form per-layer MAC + param profile of `build_bitnet_jet_tagger` (stdlib only, no TF, no training); exposes `bops(macs,b_w,b_a)` and a `ebops()` aggregator. Param totals reconstruct the verified preflight counts exactly (tiny 26,529 / small 153,793 / medium 808,065 / large 6,373,633). Accumulator (HGQ) term is a 0-stub `accumulator_bops()` hook pending the HGQ formula. | new (Jun 29) |
| `HLS_qk_Roc_Tracing.py` | upstream | upstream HLS+ROC tracing; hardcodes a Vitis path from another cluster. Reference only — not run here. | vendored, unmodified |
| `ROC.py` | upstream | upstream ROC plotting. Not invoked inside the training jobs (ROC lives in W&B). | vendored, unmodified |
| `README_upstream.md` | upstream | the upstream project README. | vendored, unmodified |
| `environment.yml` | upstream | upstream conda environment. | vendored, unmodified |
| `dataForgeScripts/dataForge.py` | upstream | data prep (root → tensors). | vendored, unmodified |
| `dataForgeScripts/removeBackground.py` | upstream | data prep helper. | vendored, unmodified |
| `util/plotting/kinematics_plotter.py` | upstream | the kinematics diagnostic plots. | vendored, unmodified |

### 1b. hls4ml / Vitis (`code/hls/`) — all authored by us

| file | what it is | status |
| --- | --- | --- |
| `run_csynth.py` | **The Vitis C-synthesis driver that produced §B** of `results/hls_resource_table.md`. Builds the binary FFN block, runs `hls_model.build(synth=True)` on hls4ml 1.x, unwraps `CSynthesisReport` (+ `csynth.xml` fallback). | new (Jun 24) |
| `full_model_csynth.py` | **The Vitis driver that produced §B′** — the *full trained transformer* end-to-end. Loads `lr15_bitnetJetTagModel.h5`, rebuilds each `BitLinear` as `LayerNormalization→QActivation→QDense(binary)`, ports trained weights via the BitNet `AbsMeanQuantizer`. Three modes (`HLS_MODE`): `fidelity` (rebuild vs trained, corr 0.99998), `convert` (QKeras↔Vitis bit-accuracy 0.9967–0.9999), `csynth` (per-shape synth + 51-instance composition). **Result: binary matmul = 0 DSP; all 1,049 DSP = LayerNorm.** | new (Jun 26) |
| `RUN_CSYNTH_ON_VITIS.md` | Turnkey runbook for a Vitis-2023.2 box (the steps actually used on `mulder`). | new (Jun 24) |
| `sweep_precision.py` | Quantization-aggressiveness × hls4ml **firmware** sweep: QKeras → HLS C++, g++ bit-accurate emulation (corr), inspects `defines.h`/`parameters.h` to confirm binary weights type as `ap_uint<1>` → **0 DSP**. | new (Jun 23) |
| `resource_model.py` | **Analytical** per-component resource model (MACs / weight-bits / LUT·DSP·BRAM via labeled cost factors). The pre-csynth first-order estimate; separates exact structural counts from derived estimates. Superseded for §B by real csynth, kept for the per-component story. | new (Jun 23) |
| `convert_probe.py` | hls4ml **convertibility** probe (0.8.1 era): proves the binary core (Dense + FFN + head) converts and emulates bit-accurately; probes the two hard pieces (LayerNorm, softmax-free EinsumDense attention). | new (Jun 23) |
| `full_transformer_probe.py` | STRETCH: the **whole** BitNet block through hls4ml ≥ 1.2 (Python ≥ 3.10) — closes the Phase-1 gap (SubLN LayerNorm + EinsumDense attention that 0.8.1 could not convert). | new (Jun 23) |
| `stage_a_fix.py` | Stage-A **rerun with a correctly-sized accumulator** — the fix for the bit-accuracy bug (see §3.2). | new (Jun 23) |

### 1c. Plots (`code/plots/`) — authored by us

| file | what it is | status |
| --- | --- | --- |
| `make_results_plots.py` | Generates the 3 publication figures (`results/plots/results_*.png`). AUCs cited from `RESULTS.md`; FPGA resources read **live** from `results/csynth/*.json`. No invented numbers. | new (Jun 24) |

### 1d. NRP Jobs & watchers (`code/jobs/`) — authored by us

| group | files | what |
| --- | --- | --- |
| `jobs/training/` (10) | `kai-bn-train-paper-binary-lr15`, `-paper-binary-sm-a{8,6,4}`, `-paper-binary-sffree{,-a6,-a4}`, `-paper-ternary`, `-vanilla-fp32`, `-w8a8` | the Jobs that produced the headline + sweep + baseline + appendix numbers. |
| `jobs/hls/` (4) | `kai-hls-{csynth,full,inspect,sweep}.yaml` | NRP Jobs wrapping the `code/hls/` scripts (csynth Job retained for any Vitis-equipped cluster). |
| `jobs/` (2) | `watch_act_sweep.sh`, `watch_paper_runs.sh` | log-watchers. |

### 1e. Archived (superseded — kept for history, `archive/`)

| file(s) | what | why archived |
| --- | --- | --- |
| `qkerasModel_ste.py` | pre-STE intermediate trainer snapshot | superseded by `code/training/qkerasModel.py`. |
| `jobs/*.yaml` (16) | earlier job YAMLs (pre-paper-recipe, smoke tests, dev/setup pods) | off the final paper recipe. |
| `upstream_samples/*.root` | original upstream `.root` samples | replaced by the PVC dataset. |

---

## 2. Phase chronology

**Phase 0 — vendor + data/env/W&B patch (Jun 22).** Vendored the upstream files (1a, unmodified)
and captured the minimal data-path / env / W&B wiring as `qkerasModel.patch` (G1).

**Phase 1 — gradient fix + paper-faithful binary trainer (Jun 22 → Jun 23).** STE gradient fix (G2)
and the paper-faithful binary quantizer + training recipe (G3). `archive/qkerasModel_ste.py` is the
intermediate snapshot from this phase; `qkerasModel.py` is the result.

**Phase 2 — baselines + sweep knobs + Jobs (Jun 23 → Jun 24).** Added the `BN_VARIANT` /
`BN_ACT_BITS` / `BN_TERNARY` / `BN_SOFTMAX_FREE` knobs (G4) and wrote the 10 training YAMLs —
binary headline, A8/A6/A4 canonical-softmax sweep, vanilla FP32 + W8A8 baselines, ternary +
softmax-free appendix.

**Phase 3 — hls4ml convert + bit-accurate emulation (Jun 23).** `convert_probe.py`,
`resource_model.py`, `stage_a_fix.py`, then `full_transformer_probe.py` + `sweep_precision.py`.
Result: the binary core converts and emulates **bit-accurately**, with binary weights typing as
`ap_uint<1>` → **0 DSP** (firmware-confirmed on NRP; no Xilinx synthesis backend there).

**Phase 4 — Vitis C-synthesis on `mulder` (Jun 24).** `run_csynth.py` + `kai-hls-csynth.yaml` +
`RUN_CSYNTH_ON_VITIS.md`. Synthesized the binary FFN at A8/A6/A4, RF=256, on Vitis HLS 2023.2 →
**DSP = 0 confirmed in silicon estimates**; LUT/FF/BRAM/latency filled into `results/hls_resource_table.md` §B;
raw reports → `results/csynth/csynth_report_a{8,6,4}_rf256.json`.

**Phase 5 — publication figures (Jun 24).** `make_results_plots.py` → the 3 `results_*.png`.

**Phase 6 — full trained transformer, synthesized end-to-end (Jun 26).** `full_model_csynth.py`: loaded the real
`lr15_bitnetJetTagModel.h5`, rebuilt every custom layer (`BitLinear`/`RMSNorm`/attention projections) from
hls4ml-supported primitives, **ported the trained binary weights**, validated (fidelity corr 0.99998; QKeras↔Vitis
bit-accuracy 0.9967–0.9999), and C-synthesized the 5 distinct layer shapes at A8 (RF=256) on Vitis HLS 2023.2.
**Headline: binary matmul = 0 DSP, re-confirmed on the real model; the entire transformer's 1,049 DSP (8.5 % of a
VU13P) is 100 % LayerNorm.** Filled `results/hls_resource_table.md` **§B′**; raw → `results/csynth/full_model_*_a8_rf256.json`.
A6/A4 sweep launched same night (backfill). This directly answers the "are you just synthesizing an FFN?" critique.

### Docs & results artifacts touched (not code, logged for completeness)
`README.md`, `results/REPORT.md`, `results/RESULTS.md`, `results/hls_resource_table.md`,
`results/plots/README.md` — updated to past tense + figure pointers once csynth landed and figures
were generated. `results/csynth/*.json` — the three measured reports copied back from `mulder`.

---

## 3. The hls4ml / csynth bug & fix log  ← "were there bugs?"

**Short answer: yes, a handful — all in *our* glue code, all found and fixed; none in hls4ml itself.**
The final emulation is bit-accurate (corr = 1.000) and the synthesis is clean. We also corrected one
of *our own* documentation overclaims. Several scary-looking log lines turned out to be expected, not bugs.

### Real bugs we hit and fixed

1. **csynth driver written against the wrong hls4ml API.** `run_csynth.py` was first written for the
   0.8.1-style `build(csynth=…)` with a `DSP48E`/`ap_fixed<…>` report shape. `mulder` runs hls4ml
   **1.4.0**, where C-synthesis is `build(synth=True)`, the report nests under
   `report["CSynthesisReport"]`, the key is `DSP` (not `DSP48E`), and precision strings drop the `ap_`
   prefix (`fixed<32,16>`). **Fix:** rewrote the driver for the 1.x API + added a `csynth.xml` fallback parser.

2. **Accumulator overflow → bit-*in*accuracy.** First FFN emulation diverged (corr ≈ 0.24). Cause: the
   default `fixed<16,6>` accumulator saturates at ±32, but `fc1` sums 256 signed terms reaching ≈ ±50.
   **Fix:** pin `ap_fixed<32,16>` on the dense path (`HLS_WIDE`, documented in `stage_a_fix.py`).

3. **Over-widening the activation undid the fix.** Naively widening *everything* (including the
   `quantized_relu(8,2)` result) removed its `[0,4)` saturation; HLS activations blew up ~15×, `fc2`
   diverged (corr 0.85, max|diff| 1639). **Fix:** widen accum/result/bias on the Dense layers **only**,
   leave `act` native so the clip is preserved. Net result after 2+3: corr = **1.000**.

4. **protobuf / TF import crash on `mulder`.** TensorFlow wouldn't import until
   `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` was set. **Fix:** export it in the runbook/driver env.

### A documentation overclaim we corrected (our claim, not an hls4ml bug)

5. **"DSP = 0 *and* BRAM = 0."** True only for the fully-unrolled **RF=1 / Latency** design. A folded,
   deployable design (**RF=256 / Resource**) parks the binary weights in **~1.2% BRAM**. We reconciled
   every doc to the folded numbers. **DSP = 0 is the fold-independent structural win** (binary weights →
   no multipliers); BRAM = 0 was not.

### Design choices that look like bugs but aren't

6. **RF=1 is intractable**, not broken — fully unrolling the 262144-MAC layers is the issue. We synthesize
   at **RF=256** (II=256), which is tractable *and* deployable.

### Scary log lines that are **expected**, not bugs

- `config_array_partition -maximum_size … not supported` (Vitis HLS 200-642) — benign warning, ignored.
- `Vivado synthesis report not found / Cosim report not found / Timing report not found` — **expected**:
  we ran `synth=True` (C-synthesis) only, not `csim`/`cosim`/`vsynth`/Vivado logic synthesis.
- Keras `HeNormal` unseeded + TF autograph warnings — cosmetic.

### Honest scope boundary (a capability limit, not a bug)

- hls4ml 0.8.1 could **not** convert LayerNorm (SubLN) or EinsumDense attention; hls4ml ≥ 1.2 added that
  support (`full_transformer_probe.py`). LayerNorm is convertible-but-fragile (`io_parallel` only). §B's numbers
  cover the binary FFN block (the dominant primitive); **§B′ (Phase 6) now extends this to the full trained
  transformer** — all 51 BitLinears + 51 SubLN norms + the 4 weighted attention projections, synthesized with
  real trained weights.
- **The one remaining gap (Phase 6):** hls4ml's parser only converts layer *types* with a registered handler, so
  the custom `BitLinear`/`RMSNorm`/`BitMHSA` subclasses can't be ingested from the `.h5` directly — they were
  rebuilt as `LayerNormalization→QActivation→QDense(binary)` with weights ported in. The attention **score core**
  (Q·Kᵀ / softmax / ·V — *weightless*, 0.65 % of MACs) uses `EinsumDense`, which **does not convert** on this
  hls4ml (verified: probe Stages C & D fail). So §B′ covers **100 % of the weights and 99.35 % of the MACs**; the
  weightless score core is handled analytically (`resource_model.py`), not synthesized. This is a real, documented
  boundary — not a silent omission.
