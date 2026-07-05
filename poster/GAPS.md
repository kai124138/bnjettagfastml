# Poster gaps — numbers/attributions the poster wants but the store cannot give

Nothing below was invented or approximated in the figures/draft; each item is
either absent from the store or failed the raw-data gate (poster/VERIFICATION.md §4).

1. **Per-instance DSP attribution inside the HGQ2 probes.** The store has only
   whole-probe csynth totals; the per-instance Vitis report that attributes the
   head_fc2 probe's 112 DSP to SubLN (binary dense 0, CSD-2 affine 0) and the
   Resource probe's 270 to dense 256 + SubLN 14 lives on **mulder** and is only
   quoted in `code/hgq2/LEDGER.md`. The DSP figure works around it with the raw
   probe-level dichotomy. *To close (live session, not this one): fetch the
   per-instance csynth reports (`csynth.xml` module table) for
   `probe_bitlinear_head_fc2_rf32` and `probe_bitlinear_rf256` into the store —
   that un-bans the sharper per-function split for the poster.*
   > **CLOSED 2026-07-05 (live session).** Raw `csynth.xml` + parsed
   > `csynth_modules.json` now in the store for all four finished probes
   > (`results/hgq2/runs/b224a8ea/<probe>/`). Confirmed per-function split:
   > head_fc2_rf32 → SubLN 112 / binary dense 0 / CSD-2 affine 0 (top 112);
   > rf256 Resource → dense 256 / SubLN 14 (top 270); subln_rf1 → 1792.
   > Parser: `code/hgq2/parse_csynth_modules.py`; fetch script now pulls both
   > files automatically. The per-function split is un-banned.

2. **Attention-core synthesis.** `probe_attn_core_rf1` (QEinsum + QSoftmax score
   core, C-sim corr 1.0) was still synthesizing on mulder at freeze. The poster
   currently has **no measured resource/latency number for the attention core**;
   the composed latency explicitly excludes it. *To close: fetch when it lands.*
   > **CLOSED (rf1) 2026-07-05 (live session).** `probe_attn_core_rf1` landed
   > and is in the store (`runs/b224a8ea/probe_attn_core_rf1/csynth_report.json`
   > + raw xml + modules json): 31 cycles @ est. 1.812 ns, LUT 4,271,510 (247%
   > VU13P), FF 9,467,557, DSP 52,000 (423%), BRAM 720 — fully-spatial extreme.
   > Per-module: each act×act einsum (QKᵀ and ctx·V, 10×10×8×32 = 25,600 MACs)
   > = 25,600 DSP, i.e. exactly 1 DSP per MAC; softmax itself only 10 DSP.
   > This is the measured quantification of why act×act attention cannot be
   > binarized (no weights to binarize → every product a real multiplier).
   > A deployable folded variant `probe_attn_core_rf64` is synthesizing on
   > mulder; fetch when it lands (already in the fetch script's PAIRS).

3. **The deployable single-model table (r6-small).** No era-2 D32/L2 checkpoint
   exists; training YAMLs are validated but unlaunched (needs Kai's approval).
   Blocked poster items: single-FPGA utilization %, single-model latency in µs at
   an L1-defensible operating point, monolith DSP count per precision.

4. **"Russell-confirmed" truncation attribution.** Not documented anywhere in the
   repo: the top-10-by-pT decision is recorded as **advisor guidance relayed by
   Kai (decisions.md, 2026-07-01)**; the only Russell artifact concerns the STE
   bug. The draft says "advisor guidance". *To close: Kai confirms Russell = the
   advisor (or captures the confirmation), then the attribution can be named.*

5. **Whole-model HGQ2 hls4ml conversion / C-sim bit-exactness.** Marked
   `[blocked: convert not run]` in the store for all three rebuilds. The draft
   states rebuild fidelity at score level only.

6. **A4 substitution-cost quantifiers.** The statements "dynamic-mode correlation
   0.987 at A4" and the E2/E4 per-calibration ΔAUC ladder come from gold-model
   experiments on a 20k subset, recorded in LEDGER/RESEARCH but not re-derivable
   from raw arrays in the store. The draft uses the qualitative claim only. *To
   close: re-run the gold-model A4 dynamic-vs-static scoring on the full 260k and
   store the scores npz.*

7. **Large-model head count.** "H8" appears only in an era-1 speaker-notes label;
   the era-2 code default is H4 with env override, and the store does not record
   the era-2 large run's head count explicitly. The draft says "multi-head"
   without a number. *To close: read it off the r5 checkpoint config and cite.*
   > **CLOSED 2026-07-05 (live session).** `"n_heads": 8` read directly from
   > the `model_config` h5 attribute of all three era-2 large checkpoints
   > (`models/r5/large-lr05-{s1,a6-s1,a4-s1}/bitnet/noNorm_train_bitnetJetTagModel.h5`).
   > The era-2 large model is H8 (E=32); the draft may say "8-head".

8. **FastML26 poster logistics.** No poster size/orientation/template requirement
   is documented in the repo; figures were built vector (PDF/SVG) so they rescale
   to any format. *To close: Kai supplies the venue spec.*

9. **Seed spread.** The tradeoff table quotes single-run (s1) trained references,
   matching the store's own convention; second seeds exist and reconcile (A8
   0.8452, A6 0.8220, A4 0.7312) and ebops.md separately quotes a seed-avg 0.8501
   for W1A8. If the poster wants error bars, decide the convention (s1 vs
   seed-avg) once, everywhere — currently the draft/figures consistently use s1
   and say so.
