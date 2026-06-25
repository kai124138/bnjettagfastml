# Training code — what we have, and exactly what we edited

**The trainer is a single file:** `code/training/qkerasModel.py` (forked from upstream `Brainz22/BNJetTag`).
There is **no separate "vanilla" or "baseline" script** — all three precisions (1-bit BitNet, vanilla FP32,
W8A8) are the *same* file selected by one environment knob, **`BN_VARIANT`**. That's the whole point: identical
architecture / data / recipe, only the numeric precision of the matmul changes, so the comparison is clean.

- **The code:** `code/training/qkerasModel.py`
- **The diff vs upstream (Phase-0 data fix):** `code/training/qkerasModel.patch`
- **How each result was produced:** the job files in `code/jobs/training/` set the env knobs (table at the end).

---

## The edits, grouped

### Group 1 — make it *run* on our data (in `qkerasModel.patch`)
Upstream read HDF5 keys `"Training Data"` / `"Sample Data"` and sliced `[:,146:]`; our staged data uses
`jet_constituents` / `train_jet_data`, so the upstream path produced **empty arrays**. Fix reads whichever key
exists and derives column widths from the data:
```python
def _read(path, prefer):
    with h5py.File(path, "r") as hf:
        for k in prefer:
            if k in hf: return hf[k][:]
        return hf[list(hf.keys())[0]][:]          # fall back to the sole dataset
PART_KEYS = ["jet_constituents", "Training Data"]
JET_KEYS  = ["train_jet_data",   "Sample Data"]
...
n_part_cols = dataset.shape[1]                    # 141 = 140 feats + label  (no hard-coded 146)
```
Also in this patch: size constants became **env-overridable** (`BN_D_MODEL/N_HEADS/N_LAYERS/FFN_DIM/EPOCHS/
BATCH`), and a minimal, env-gated **Weights & Biases** hook (`WANDB_PROJECT` / `WANDB_RUN_NAME`) was added.

### Group 2 — make it *learn* (the gradient-flow fix; the big one)
The model was frozen at val_auc = 0.5 because `tf.round` has zero gradient, severing gradient flow through every
activation. Fix = a **straight-through estimator (STE)** on the activation quantizer (`_absmax_quant`):
```python
def _absmax_quant(x, bits, axis):
    """Symmetric signed b-bit absmax quantizer WITH a straight-through estimator."""
    ...
    q = tf.round(x * scale)                        # forward: quantized
    return (x * scale + tf.stop_gradient(q - x * scale)) / scale   # backward: identity (STE)
```
This single change unfroze training: **val_auc 0.50 → 0.67 at epoch 1.** (The full diagnosis for the upstream
author is in `methods/message-to-russell.md`.)

### Group 3 — make it *paper-faithful* (arXiv:2310.11453)
Two correctness items on the **weight** quantizer (`AbsMeanQuantizer`):
1. **Zero-mean centralization before the sign** (`Sign(W − mean W)`, the paper's Eq 1 & 3) — upstream wasn't
   doing it;
2. **STE on the sign**, against a **full-precision latent master weight** (the kernel keeps **no constraint**, so
   Adam updates a real-valued weight while the forward pass uses the binarized one):
```python
else:  # binary BitNet (arXiv:2310.11453, Eq 1-3,12)
    alpha = tf.reduce_mean(w)                      # Eq 3
    w     = w - alpha                              # centralize (Eq 1): W~ = Sign(W - alpha)
    beta  = tf.reduce_mean(tf.abs(w)) + self.eps   # Eq 12 absmean scale
    q     = tf.sign(...)                           # {-1,+1}
w_q = w_scaled + tf.stop_gradient(q - w_scaled)    # STE: identity backward
return w_q * beta
```
```python
# in BitLinear.build():
self.kernel = self.add_weight(...,  # FIX: NO constraint -> kernel stays full-precision latent master weight
                              name="kernel")
```
Plus the paper **training recipe** as knobs (LR ≈ 1.5e-4, Adam β=(0.9,0.98), weight decay 0.01, warmup +
polynomial decay, no grad clipping). Together these broke the "~0.66 binary wall" → **0.7530**.

### Group 4 — the abstract's comparison & sweep knobs
The one knob that produces vanilla / W8A8 / BitNet from the same file (`BitLinear.call`):
```python
def call(self, x):
    x_norm = RMSNorm(name=self.name + "_norm")(x)            # LayerNorm/SubLN (all variants)
    if VARIANT == "vanilla":
        out = tf.matmul(x_norm, self.kernel)                 # full-precision baseline (no quant)
    elif VARIANT == "w8a8":
        out = tf.matmul(_absmax_quant(x_norm, 8, axis=-1),   # conventional 8-bit:
                        _absmax_quant(self.kernel, 8, axis=None))  # 8-bit act x 8-bit weight
    else:  # "bitnet": binary (or ternary) weights x ACT_BITS absmax activations
        out = tf.matmul(activation_quant(x_norm), self.quantizer(self.kernel))
    if self.use_bias: out = out + self.bias
    return out
```
- **`BN_VARIANT`** ∈ `bitnet | vanilla | w8a8` → the 1-bit-vs-baselines comparison.
- **`BN_ACT_BITS`** ∈ `8 | 6 | 4` → the quantization-aggressiveness axis (activation precision).
- **`BN_TERNARY=1`** → ternary `{−1,0,+1}` weights (appendix, off-thesis).
- **`BN_SOFTMAX_FREE=1`** → `ReLU(QKᵀ)/N` attention (appendix).

---

## Which knob produced which result

| result | val_auc | `BN_VARIANT` | other env | job file (`code/jobs/training/`) |
| --- | --- | --- | --- | --- |
| **1-bit BitNet (the model)** | **0.7530** | `bitnet` | `BN_ACT_BITS=8` (default) | `kai-bn-train-paper-binary-lr15.yaml` |
| vanilla FP32 baseline | 0.7703 | `vanilla` | — | `kai-bn-train-vanilla-fp32.yaml` |
| W8A8 baseline | 0.7719 | `w8a8` | — | `kai-bn-train-w8a8.yaml` |
| sweep A8 / A6 / A4 | 0.7562 / 0.7450 / 0.7381 | `bitnet` | `BN_ACT_BITS=8/6/4`, `BN_SOFTMAX_FREE=1` | `kai-bn-train-paper-binary-sffree{,-a6,-a4}.yaml` |
| ternary (appendix) | 0.7685 | `bitnet` | `BN_TERNARY=1` | `kai-bn-train-paper-ternary.yaml` |
| softmax-free (appendix) | 0.7562 | `bitnet` | `BN_SOFTMAX_FREE=1` | `kai-bn-train-paper-binary-sffree.yaml` |

> Note: the patch (`qkerasModel.patch`) captures Group 1 (the Phase-0 data/env/W&B fix). Groups 2–4 are
> model-logic edits made afterward and live in the current `qkerasModel.py` (snippets above). The pre-STE
> intermediate is kept in `archive/qkerasModel_ste.py` for history.
