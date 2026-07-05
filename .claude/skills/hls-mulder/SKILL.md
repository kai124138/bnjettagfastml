---
name: hls-mulder
description: Take a trained BNJetTag checkpoint through hls4ml to real Vitis HLS C-synthesis on the mulder box, and parse the csynth reports into the resource/latency tables. Use for anything about FPGA resources (LUT/FF/DSP/BRAM), latency in cycles, csynth, hls4ml conversion, reuse factor, the VU13P target, or updating results/hls_resource_table.md.
---

# HLS synthesis on mulder — the hardware axis

**Why mulder:** NRP Nautilus has **no Xilinx backend**, so hls4ml's `build` stage (Vitis
C-synthesis — the only stage that yields real LUT/FF/DSP/BRAM + latency-in-cycles) runs on
the group's `mulder` box (Vitis HLS **2023.2**, same install hard-coded in
`code/training/HLS_qk_Roc_Tracing.py`). Everything below is relative to
`qkeras-bitnet-run-2026-06-22/`.

## The three hls4ml stages — know which truth you have

1. `convert` — codegen only. Exact *structural* facts fall out here (weight C-type
   `ap_uint<1>` ⇒ binary MACs **cannot** use DSPs): read from the emitted
   `defines.h`/`parameters.h`.
2. `compile` — g++ bit-accurate emulation. Gives fidelity vs QKeras (require corr ≥ ~0.997
   before trusting synthesis of that config).
3. `build` (**csynth, mulder only**) — measured resources + latency. Only these numbers go
   in a report as "synthesized".

## Scripts (`code/hls/`)

- `run_csynth.py` — the binary-FFN probe (fc1 256→1024 → ReLU → fc2 1024→256), the primitive
  the device-fit numbers come from.
- `full_model_csynth.py` — the full trained transformer: rebuilds all 51 BitLinears + SubLN
  + attention projections from hls4ml-supported primitives, ports trained weights, validates
  fidelity, csynths per distinct layer shape. (Procedure notes: `RUN_CSYNTH_ON_VITIS.md`,
  pending-work notes: `BACKFILL_A6_A4_FULL_MODEL.md`.)
- `sweep_precision.py` — A8/A6/A4 sweep; `convert_probe.py`, `resource_model.py` — helpers.
- `code/jobs/hls/kai-hls-*.yaml` — containerized variants of the same flow (need a Vitis
  box; header comments explain the ConfigMap step).

## Operating point + targets (current convention)

Part **`xcvu13p-flga2577-2-e`** (VU13P), clock target 2.5 ns (400 MHz). Two named points:
**RF=1** (fully unrolled, io_parallel, II=1 — max fabric, min cycles) and **RF=256**
(folded device-fit, II=256 — the reported deployable point: ≈25% LUT / 7% FF / 1.2% BRAM /
0 DSP, 520 cycles ≈ 1.3 µs). Always state the RF with any number.

## After a run

1. Bring back the JSON reports → `results/csynth/` (naming:
   `csynth_report_a{8,6,4}_rf256.json`, full-model per-layer files alongside).
2. Parse JSON → update `results/hls_resource_table.md` (§A structural, §B FFN probe,
   §B′ full model). Percentages = against VU13P totals.
3. **Standard checks:** DSP=0 on every binary matmul (any DSP>0 there = investigate,
   it's the thesis); LayerNorms are the only expected DSP users (1,049 total, 8.5% VU13P);
   Estimated Fmax ≥ 400 MHz; latency cycles → µs at the *achieved* clock; fidelity metrics
   recorded (rebuild corr, C-sim corr).
4. Log to `.claude/memory/experiment-log.md`; `RESEARCH.md` §6 updates only from parsed
   report values (via the numbers gate), never from memory.

## Boundaries

- No synthesis in this folder/environment — prep and parsing only.
- csynth needs no Xilinx license; bitstream place-and-route does (out of scope for now).
- Mulder⇄NRP file movement: `nrp/mulder-to-nrp-data-transfer.md` (nrpcopy).
