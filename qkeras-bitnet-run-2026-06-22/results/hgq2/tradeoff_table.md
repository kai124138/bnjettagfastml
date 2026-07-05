# HGQ2 rebuild — efficiency vs resource vs latency (era-2 large, D256/L8)

All AUCs are **era-2 ROC-test macro-OvR** (held-out val split, n=260,000).
`ref` = the trained QKeras model's verified scores (roc-results/r5, verified 2026-07-03).
HGQ2 rebuild = binary {−1,+1} pinned, static per-channel act quant, fold-aware CSD-2 β̃.

| model | AUC ref [measured/QKeras] | AUC HGQ2 rebuild | Δ | corr(scores) | EBOPs (HGQ2-native) | hls4ml C-sim bit-exact |
|---|---|---|---|---|---|---|
| W1A8 | 0.8551 | 0.8493 | -0.0058 | 0.9589 | 650,360,941 | [blocked: convert not run] |
| W1A6 | 0.8394 | 0.8222 | -0.0173 | 0.8944 | 501,183,824 | [blocked: convert not run] |
| W1A4 | 0.7346 | 0.7115 | -0.0231 | 0.6789 | 352,214,162 | [blocked: convert not run] |

Baselines (same table, source `roc-results/r5/roc_auc.md`, verified 2026-07-03): FP32 0.8765 · W8A8 0.8642 [measured/QKeras, single-run lr05].

## Real synthesis (Vitis HLS 2023.2, VU13P, mulder)

| probe | precision | LUT | FF | DSP | BRAM_18K | latency (cycles) | II | est. clock |
|---|---|---|---|---|---|---|---|---|
| probe_attn_core_rf1 | A8 | 4,271,510 | 9,467,557 | **52000** | 720 | 31–31 | 1 | 1.812 ns |
| probe_bitlinear_head_fc2_rf32 | A8 | 194,012 | 116,346 | **112** | 0 | 100–100 | 32 | 2.025 ns |
| probe_bitlinear_rf256 | A8 | 196,871 | 118,597 | **270** | 32 | 832–833 | 573 | 3.035 ns |
| probe_bitlinear_v2_rf256 | A8 | 222,686 | 135,332 | **270** | 16 | 834–835 | 573 | 3.035 ns |
| probe_subln_rf1 | A8 | 165,695 | 151,297 | **1792** | 0 | 36–36 | 1 | 1.818 ns |

**Per-instance DSP split (from the per-function csynth reports — the numbers that carry the thesis):** probe_bitlinear_head_fc2_rf32 (Latency, pure ±1 + CSD-2 affine): binary dense **0 DSP** · affine **0 DSP** · SubLN 112 DSP = all of the probe's DSPs. probe_bitlinear_rf256/v2 (Resource): dense 256 DSP — the documented Resource-ROM trap, kept as the negative result; SubLN folded = 14 DSP there. See constraints_map.md.

Prior verified per-shape csynth (QKeras path, results/hls_resource_table.md §B′, RF=256): matmul cores 0 DSP at A8/A6/A4; all 1,049 model DSPs in the old fixed<32,16> LayerNorm; composed whole-model latency upper bound 23,409 cycles ≈ 58.5 µs @ 400 MHz (attention score core excluded there — the HGQ2 attn_core probe closes exactly that gap when its csynth lands).
