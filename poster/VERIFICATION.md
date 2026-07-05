# Poster verification gate — 2026-07-04

Independent recomputation of every number the FastML26 poster uses, from **raw store
data only** (npz arrays, ebops.json payloads, csynth_report.json, probe firmware).
Summary tables were used only as comparison targets, never as inputs.

Recompute artifacts: `poster/scripts/verify_gate.py` → `poster/data/verification_results.json`
(full-precision values), `poster/scripts/verify_ebops_analytic.py` → `poster/data/ebops_analytic_check.json`.

**Rule applied (the gate):** a number appears in the figures/draft **only** if it
reconciled from raw data below. Everything in §4 is banned from figures and draft.

---

## 1. AUCs — all reconciled ✅

Method: macro-OvR AUC = unweighted mean of the 5 per-class one-vs-rest ROC AUCs,
recomputed with sklearn from the raw `y`/`score` arrays (n = 260,000 each).

Integrity checks that all passed:
- **Column order proven**: recomputed per-column AUCs match each npz's embedded
  per-class dict (g, q, W, Z, t) to < 5×10⁻¹⁰, for all 8 reference files. Order
  g,q,W,Z,t confirmed (matches `CLASS_LABELS = ["j_g","j_q","j_w","j_z","j_t"]`,
  `code/training/qkerasModel.py:84`).
- **Same eval split**: the rebuild `y` arrays are bit-identical to the r5 reference
  `y` arrays (`np.array_equal` true for all three pairs) — rebuild and trained
  reference are scored on the identical 260k jets in identical order (Zenodo 3602260
  `hls4ml_LHCjet_150p_val` split).

| model | source (raw) | recomputed | claimed | verdict |
|---|---|---|---|---|
| FP32 trained | `roc-results/r5/FP32-lr05.npz` | 0.876514 | 0.8765 | ✅ |
| W8A8 trained | `roc-results/r5/W8A8-lr05.npz` | 0.864246 | 0.8642 | ✅ |
| W1A8 trained (s1) | `roc-results/r5/A8-lr05-s1.npz` | 0.855104 | 0.8551 | ✅ |
| W1A6 trained (s1) | `roc-results/r5/A6-lr05-s1.npz` | 0.839434 | 0.8394 | ✅ |
| W1A4 trained (s1) | `roc-results/r5/A4-lr05-s1.npz` | 0.734561 | 0.7346 | ✅ |
| **W1A8 HGQ2 rebuild** | `results/hgq2/runs/b224a8ea/scores.npz` | **0.849328** | 0.8493 | ✅ |
| **W1A6 HGQ2 rebuild** | `results/hgq2/runs/a428e6e2/scores.npz` | **0.822153** | 0.8222 | ✅ |
| **W1A4 HGQ2 rebuild** | `results/hgq2/runs/53b202bc/scores.npz` | **0.711487** | 0.7115 | ✅ |

(Second seeds also reconcile: A8-s2 0.845163, A6-s2 0.822007, A4-s2 0.731156.)

Rebuild deltas and score correlations vs the s1 references, recomputed from raw:

| rebuild | Δ recomputed | Δ claimed | corr recomputed | corr claimed | verdict |
|---|---|---|---|---|---|
| W1A8 | −0.005776 | −0.0058 | 0.958927 | 0.9589 | ✅ |
| W1A6 | −0.017281 | −0.0173 | 0.894436 | 0.8944 | ✅ |
| W1A4 | −0.023074 | −0.0231 | 0.678891 | 0.6789 | ✅ |

## 2. EBOPs — both conventions reconciled ✅, never blended

**Convention A — HGQ2-native** (accumulator term included; produced by the HGQ2
build of the rebuilt model). Recomputed: per-layer sums equal stored totals exactly
(93 layers each):

| model | Σ per_layer (raw ebops.json) | claimed | verdict |
|---|---|---|---|
| W1A8 | 650,360,941 | 650.4M | ✅ |
| W1A6 | 501,183,824 | 501.2M | ✅ |
| W1A4 | 352,214,162 | 352.2M | ✅ |

FP32 / W8A8 have **no HGQ2-native EBOPs** (no HGQ2 build exists for them) — any
table cell for them in this convention must be "—".

**Convention B — analytic HGQ-v1** (`results/ebops.md` §0: EBOPs = Σ bᵢ·bⱼ per
scalar multiply, **no accumulator term**; weight×act costs b_w·b_a, attention
act×act costs b_a²). Re-derived from architecture dimensions alone
(D=256, L=8, FFN=1024, T=10, F=16, C=5 → 63,022,336 w×a MACs + 409,600 attn MACs,
both matching ebops.md's stated MAC profile exactly):

| model | re-derived | ebops.md claims | verdict |
|---|---|---|---|
| FP32 | 64,954,302,464 | 64,954,302,464 | ✅ |
| W8A8 | 4,059,643,904 | 4,059,643,904 | ✅ |
| W1A8 | 530,393,088 | 530,393,088 | ✅ |
| W1A6 | 392,879,616 | 392,879,616 | ✅ |
| W1A4 | 258,642,944 | 258,642,944 | ✅ |

W8A8/W1A8 = 7.654× ✅ (the "7.65×" headline — valid **within convention B only**).

**Blend audit:** RESEARCH.md:57–58 states the separation rule ("never mix") and the
store keeps them apart: tradeoff_table.md carries only native values; ebops.md only
analytic. **One stale variant found**: `results/RESULTS.md:190–198` (§2f) holds a
third, superseded analytic table (W1A8 = 530,343,936, computed on the era-1
6,373,633-param shape) that differs from ebops.md by ~0.01%. It is **banned** here;
poster uses ebops.md values only. Poster convention policy: sweep/tradeoff figures
use HGQ2-native exclusively; the 7.65× cross-model ratio, if quoted, uses analytic
exclusively and is labeled as such. No figure mixes columns across conventions.

## 3. DSP — what reconciles from raw ✅

Raw whole-probe csynth totals (each read from its `csynth_report.json`; part
xcvu13p-flga2577-2-e, Vitis HLS 2023.2 on mulder):

| probe | contents (proven from probe firmware / §B header) | DSP | LUT | latency | II |
|---|---|---|---|---|---|
| Binary FFN block, QKeras path, RF=256, A8 | fc1 256→1024 → ReLU → fc2 1024→256, **no norm** | **0** | 440,882 | 518–520 | 256 |
| — same, A6 | " | **0** | 429,098 | 518–520 | 256 |
| — same, A4 | " | **0** | 415,259 | 518–520 | 256 |
| `probe_subln_rf1` (HGQ2 path, fully parallel II=1) | **SubLN dim-256 only** (sole layer in `myproject.cpp`) | **1,792** | 165,695 | 36 | 1 |
| `probe_bitlinear_head_fc2_rf32` (HGQ2, Latency, RF=32) | SubLN → binary dense 256→5 → CSD-2 affine | **112** (whole chain) | 194,012 | 100 | 32 |
| `probe_bitlinear_rf256` (HGQ2, Resource, RF=256) | SubLN → binary dense 256→256 (Wo) → affine | **270** | 196,871 | 832–833 | 573 |
| `probe_bitlinear_v2_rf256` (same, v2) | " | **270** | 222,686 | 834–835 | 573 |

Probe compositions were verified from the raw firmware inside the probe tarballs
(extracted to scratchpad, store untouched): `probe_subln_rf1/firmware/myproject.cpp`
contains exactly one layer call (`nnet::subln`), so **1,792 DSP = SubLN alone, by
construction — fully reconciled**. The head_fc2 probe contains exactly three calls
(subln → dense → normalize); its dense weight type is `ap_fixed<2,2>` (pure ±1 in
the datapath) and the affine scale `ap_ufixed<4,-3>` (defines.h, raw).

Additional raw facts carried into figures:
- QKeras-path per-shape probes (each = LayerNorm→QAct→binary dense,
  `full_model_shape_*_a{8,6,4}_rf256.json`): DSP **identical across A8/A6/A4** for
  every shape (11/15/15/51/15) while LUT falls with activation width — the DSP
  block is activation-precision-independent; every norm-free binary probe is 0.
- Composed full-model census (raw `full_model_total_a{8,6,4}_rf256.json`):
  DSP = **1,049** at all three precisions = 8.5% of a VU13P's 12,288. Census
  arithmetic re-verified: 11·1 + 15·33 + 15·8 + 51·8 + 15·1 = 1,049 ✅.
- **~516% claim**: composed fully-spatial LUT 8,912,618 / 1,728,000 avail =
  **515.8%** ✅ (LUT; FF 252.1%, composed from per-shape sums × multiplicities —
  explicitly a non-deployable upper bound, per hls_resource_table.md:185–191).
- Composed latency upper bound: 299 + 8×(679+679+679+682) + 679 + 679 =
  **23,409 cycles** ✅ = 58.5 µs at the 2.5 ns target (attention score core excluded
  in that composition).

## 4. NOT reconciled from raw — banned from figures and draft ⛔

1. **Intra-probe per-function DSP attribution** — "binary dense **0** / CSD-2 affine
   **0** / SubLN **112**" inside `probe_bitlinear_head_fc2_rf32`, and "dense 256 +
   SubLN 14" inside `probe_bitlinear_rf256`. Source is the per-instance table of the
   Vitis report **read on mulder** and recorded in `code/hgq2/LEDGER.md:5–12`; the
   per-instance report file itself was never fetched into the store, so it cannot be
   independently re-derived here. All in-store raw evidence is *consistent* with it
   (whole-chain 112 total; SubLN-only probe = 1,792 = 7 DSP/lane × 256 lanes;
   norm-free binary probes = exactly 0; ±1 weights are 2-bit types), but per the
   gate the split itself is **not shown as a measured per-function number**. Figures
   present the raw probe-level dichotomy instead. → GAPS.md #1.
2. **§B′ "probe DSP = 100% LayerNorm" per-instance attribution** (11/15/51 mapped to
   LN widths, hls_resource_table.md:155–162) — same situation: per-instance reports
   on mulder, only probe totals in the store. The raw-backed statement used instead:
   *norm-free binary probes synthesize to 0 DSP; norm-bearing probes carry a DSP
   count that is activation-precision-independent.*
3. **"Russell-confirmed" top-10 truncation** — no such confirmation exists in the
   repo. The documented record is **advisor guidance relayed by Kai on 2026-07-01**
   (decisions.md: top-10 by pT, input-size study deferred by design). The draft
   attributes it to the advisor; the "Russell-confirmed" attribution goes to
   GAPS.md #4 for Kai to confirm or correct.
4. **`results/RESULTS.md` §2f analytic EBOPs variant** (530,343,936 etc.) —
   superseded era-1-shape computation; banned (ebops.md values reconciled instead).
5. **LEDGER.md interim values** — the v3-era "EBOPs A8 = 634,685,224" and v3 AUC
   0.84868 are earlier iterations of the pipeline, superseded by the store artifacts
   verified in §1–2. Not poster material.
6. **Whole-model HGQ2 hls4ml conversion / C-sim bit-exactness** — `[blocked: convert
   not run]` in the store; the attention-core and big-shape Latency csynth probes
   either failed (HLS 200-1715, structural) or had not landed. No full-model
   HGQ2-path synthesis number exists; the draft says so explicitly.

## 5. Provenance of every figure number

| figure | numbers | raw source |
|---|---|---|
| Tradeoff table | 8 AUCs, 3 Δ, 3 native EBOPs, 5 analytic EBOPs | §1 npz recompute; §2 ebops.json sums + architecture re-derivation |
| ROC overlay | per-class curves + AUCs for FP32, W8A8, A8/A6/A4-s1, 3 rebuilds | `poster/data/roc_curves.npz`, computed from the same raw npz in §1 |
| DSP figure | 0/0/0, 112, 270, 1,792 whole-probe DSPs; 12,288 avail; per-shape DSP×precision | §3 csynth_report.json totals + firmware composition proof |
| Sweep | AUC vs bits (trained+rebuild), native EBOPs vs bits, FFN-block LUT vs bits | §1, §2A, §3 raw JSONs |
