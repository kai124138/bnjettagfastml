# HGQ2 rebuild pipeline (`code/hgq2/`)

Config-driven pipeline that takes a trained BitNet QKeras checkpoint through:

```
(a) config ──▶ (b) HGQ2 rebuild (binary PINNED {−1,+1}) ──▶ (c) weight port +
fidelity gate (corr + AUC vs the verified r5 .npz) ──▶ (d) EBOPs ──▶
(e) hls4ml convert (+ csynth on mulder) ──▶ (f) ROC, log-FPR ──▶ (g) results store
```

Every stage is keyed by the **config hash**; results land in
`../../results/hgq2/runs/<hash8>/` with a manifest in `../../results/hgq2/manifest.json`.
A new model = a new JSON in `configs/` through the same stages — nothing re-derived.

## Layout

| Path | Role |
| --- | --- |
| `configs/*.json` | One model+quantization spec per file (arch, checkpoint, act bits, quantizer policy). |
| `bnhgq2/config.py` | Config load/validate/hash. |
| `bnhgq2/extract.py` | h5py-only extraction of latent kernels/biases + the folded PE constant from a QKeras `.h5` (no TF-2.11 needed). |
| `bnhgq2/binarize.py` | AbsMean binarization math (α/β/sign) + the β-fold bookkeeping (which β is folded where, exactly). |
| `bnhgq2/data.py` | numpy/h5py-only era-2 loader replicating `make_roc.py` exactly (sorted glob, per-jet pT re-sort, top-N). |
| `bnhgq2/build.py` | config → HGQ2 model (binary pinned, static act quantizers). |
| `bnhgq2/port.py` | extracted weights → HGQ2 model (+ datalane calibration). |
| `bnhgq2/verify.py` | fidelity gates: rebuild↔trained corr + macro-OvR AUC vs `roc-results/r5/*.npz`; HGQ2↔hls4ml bit-exactness. |
| `bnhgq2/ebops_calc.py` | native HGQ2 EBOPs per config. |
| `bnhgq2/convert.py` | hls4ml (Vitis backend) conversion + project write for mulder csynth. |
| `bnhgq2/store.py` | results store (JSON per stage + manifest). |
| `probe_binary_pinning.py` | the empirical test matrix that established the binary recipe (run once, kept as evidence). |
| `run_stage.py` | CLI entry: `python run_stage.py <stage> --config configs/X.json`. |
| `LEDGER.md` | running change ledger for this effort (dated, newest on top). |

## Environment

Local venv at repo root: `../../../.venv-hgq2` (Python 3.12, hgq2 0.1.9, hls4ml 1.3.0,
keras 3.15, TF 2.21 backend). Synthesis runs on mulder only (see `../hls/RUN_CSYNTH_ON_VITIS.md`).

## Fidelity gates (what "verified" means here)

1. **rebuild ↔ trained**: Pearson corr of softmax scores + recovered macro-OvR AUC on the
   full 260k-jet era-2 val split, against the *stored, verified* `roc-results/r5/*.npz`.
   Bit-exactness vs the trained model is impossible by construction (the QKeras model uses
   dynamic per-token activation scales; hardware needs static) — the substitution is the
   same one `code/hls/full_model_csynth.py` made, and the correlation is reported, not hidden.
2. **HGQ2 ↔ hls4ml**: bit-exactness of the converted model against the HGQ2 forward pass
   (HGQ2's design guarantee; checked on real jets, not random noise).
