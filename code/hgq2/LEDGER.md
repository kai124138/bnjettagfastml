# HGQ2 rebuild — change ledger

Running log of every consequential change in this effort. Dated, newest on top.

## 2026-07-04 — session start: scaffold + infrastructure
- Scouted repo + HGQ2/hls4ml web state (7-agent sweep; findings in
  `.claude/memory/research-log.md` 2026-07-04 entry).
- Local env built: `.venv-hgq2` (py3.12, hgq2 0.1.9, hls4ml 1.3.0, keras 3.15, tf 2.21).
- All 8 round-5 checkpoints fetched from W&B → `models/r5/<key>/bitnet/*.h5` (76.9 MB each).
- Era-2 val split (Zenodo 3602260, 1.14 GB) downloaded → `data/` (project root).
- Confirmed by h5py inspection: r5 `.h5` holds all latent FP32 kernels/biases AND the folded
  positional-encoding constant (serialized in `model_config` → TFOpLambda add `y` kwarg) —
  the port needs no TF-2.11 environment.
- Round-6-small training YAMLs (D32/H4/L2/FFN64, era-2, PVC-free) generated + server-dry-run
  validated: `code/jobs/training/variants/kai-bn6s-*.yaml`. **NOT launched** — permission
  gate requires Kai's approval for new GPU jobs. See decisions.md 2026-07-04.
- Architecture-target decision + binary-pinning verify-first policy + β-fold plan:
  decisions.md 2026-07-04 (Decisions 1–4).
