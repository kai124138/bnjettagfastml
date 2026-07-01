"""
BitNet-style 1-bit Transformer Jet Tagger
==========================================
Drop-in replacement for the QKeras CNN jet tagger.

Matches exactly:
  - Input  shape : (batch, 10, 14)  [N_PART_PER_JET=10, N_FEAT=14]
  - Output shape : (batch, 1)       [single logit, no sigmoid]
  - Loss         : binary_crossentropy
  - Sample weights, pruning callbacks, and training loop are unchanged.

Architecture overview
---------------------
  Input (10×14)
    │
  BitLinear projection  →  (10×D_MODEL)   [1-bit weights]
    │
  Positional encoding   →  (10×D_MODEL)   [learned, lightweight]
    │
  × N_LAYERS of BitTransformerBlock:
      ├─ RMSNorm
      ├─ 1-bit Multi-Head Self-Attention  (Q/K/V projections are binary)
      ├─ residual add
      ├─ RMSNorm
      ├─ 1-bit FFN  (expand → contract, binary weights)
      └─ residual add
    │
  Global average pool   →  (D_MODEL,)
    │
  BitLinear head        →  (1,)            [logit]

BitLinear implementation  (W1A8, per arXiv:2310.11453 "BitNet")
---------------------------------------------------------------
This follows the ORIGINAL BitNet paper (Wang et al., 2023): binary weights
(W~ = Sign(W - alpha), zero-mean-centralized, with absmean scale beta) and
8-bit absmax-quantized activations, with full LayerNorm/SubLN before the
activation quantizer. Straight-through estimators carry gradients through both
the weight Sign and the activation Clip (paper Sec 2.2).

  alpha = mean(W);  W_c = W - alpha
  W_q   = Sign(W_c)               in {-1,+1}        (beta = mean|W_c| scale, STE)
  x~    = Clip(round(x * 127/max|x|), -128, 127)    (8-bit absmax, STE)
  y     = x~ @ W_q * beta

Opt-in BN_TERNARY=1 instead uses the b1.58 follow-up (arXiv:2402.17764)
ternary {-1,0,+1} weight scheme (absmean scale, no centralization).
"""

import h5py
import os
import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer, Dense, GlobalAveragePooling1D, Input, Add, MultiHeadAttention
)
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import MinMaxScaler
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import (
    prune, pruning_callbacks, pruning_schedule
)

# ─────────────────────────────────────────────
# Constants  (must match your data pipeline)
# ─────────────────────────────────────────────
N_FEAT          = 14
N_PART_PER_JET  = 10

# ─────────────────────────────────────────────
# Hyperparameters  (tunable; overridable via env for NRP multi-size runs)
# ─────────────────────────────────────────────
D_MODEL    = int(os.environ.get("BN_D_MODEL",  32))   # embedding dimension
N_HEADS    = int(os.environ.get("BN_N_HEADS",  4))    # attention heads (D_MODEL % N_HEADS == 0)
N_LAYERS   = int(os.environ.get("BN_N_LAYERS", 2))    # transformer blocks
FFN_DIM    = int(os.environ.get("BN_FFN_DIM",  64))   # feed-forward hidden dim (2-4 x D_MODEL)
DROPOUT    = float(os.environ.get("BN_DROPOUT", 0.0)) # set >0 only if overfitting is observed
L1_REG     = float(os.environ.get("BN_L1_REG", 1e-4)) # matches original regularisation
EPOCHS     = int(os.environ.get("BN_EPOCHS", 200))    # max epochs (EarlyStopping may stop sooner)
BATCH_SIZE = int(os.environ.get("BN_BATCH",  50))     # batch size
# Optimizer controls (opt-in via env; defaults reproduce prior behaviour).
# The BitNet paper (arXiv:2310.11453, Tables 5-7) trains 1-bit transformers with a LARGE
# peak LR (~1e-3), Adam beta=(0.9,0.98), polynomial-decay schedule, 750-update warmup,
# weight decay 0.01, and NO gradient clipping. Deep 1-bit stacks can be unstable on this
# jet task, so a mild grad-norm clip is available as a robustness knob (BN_CLIPNORM=0 gives
# the paper-faithful no-clip recipe). See REPORT.md "paper alignment".
LR            = float(os.environ.get("BN_LR", 0.0))          # >0 overrides "adam"; peak LR (paper ~1e-3)
WARMUP_EPOCHS = int(os.environ.get("BN_WARMUP_EPOCHS", 0))   # linear LR warmup over N epochs (paper: 750 updates)
DECAY_EPOCHS  = int(os.environ.get("BN_DECAY_EPOCHS", 0))    # >0: polynomial LR decay over N epochs after warmup
DECAY_POWER   = float(os.environ.get("BN_DECAY_POWER", 1.0)) # polynomial decay power (1.0 = linear)
BETA2         = float(os.environ.get("BN_BETA2", 0.999))     # Adam beta_2 (paper: 0.98)
WEIGHT_DECAY  = float(os.environ.get("BN_WEIGHT_DECAY", 0.0))# AdamW weight decay (paper: 0.01; 0 = off)
CLIPNORM      = float(os.environ.get("BN_CLIPNORM", 1.0))    # global grad-norm clip (paper: OFF; set 0 to disable)
# EarlyStopping controls (defaults reproduce upstream behaviour: monitor val_loss, patience 5).
ES_MONITOR    = os.environ.get("BN_ES_MONITOR", "val_loss") # e.g. "val_auc" to track the tagger metric
ES_MODE       = os.environ.get("BN_ES_MODE", "auto")        # "min"/"max"/"auto" (use "max" with val_auc)
ES_PATIENCE   = int(os.environ.get("BN_ES_PATIENCE", 5))    # epochs w/o improvement before stopping
# Weight precision. Default (0) reproduces upstream BINARY weights (tf.sign -> {-1,+1}).
# BN_TERNARY=1 switches to true BitNet b1.58 TERNARY weights (round/clip -> {-1,0,1}), which
# the docstring describes and which can zero out irrelevant inputs (more expressive). Opt-in
# deviation from upstream, documented in REPORT.md.
TERNARY       = os.environ.get("BN_TERNARY", "0") not in ("0", "", "false", "False", "no")
# Attention core. Default (0) = conventional softmax( QK^T/sqrt(d) ) attention.
# BN_SOFTMAX_FREE=1 swaps softmax for a fully-binarizable, FPGA-friendly score:
# ReLU(QK^T/sqrt(d)) with a CONSTANT 1/N normalization (no exp, no data-dependent
# division). softmax's exp+divide is the one non-binarizable, DSP/LUT-expensive op
# in the attention core; ReLU is a comparator/mux and the 1/N is a fixed shift, so
# this variant keeps the whole MHSA on LUT/logic fabric -- the research abstract's
# "softmax-free attention designed to remain fully binarizable". Opt-in; documented
# in REPORT.md. N (sequence length) = particles-per-jet = 10 here, so attention is
# short-range and the constant normalization is stable.
SOFTMAX_FREE  = os.environ.get("BN_SOFTMAX_FREE", "0") not in ("0", "", "false", "False", "no")
# Activation precision (the research abstract's "varying activation precision" axis).
# Default 8 = paper W1A8 (8-bit absmax activations, Qb=2^7). Lowering this to 6 or 4
# shrinks the activation datapath / on-chip storage and the multiplier bit-width feeding
# the popcount accumulators, at some cost to LLP tagging efficiency -- the sweep maps
# where that efficiency starts to degrade. Symmetric signed b-bit absmax: levels span
# [-2^(b-1), 2^(b-1)-1]. Opt-in; documented in REPORT.md.
ACT_BITS      = int(os.environ.get("BN_ACT_BITS", 8))
assert ACT_BITS >= 2, "BN_ACT_BITS must be >= 2 (need at least sign + 1 magnitude bit)"
# Model variant (the abstract's "1-bit vs the vanilla version" comparison).
#   "bitnet"  (default) : the 1-bit BitNet tagger -- binary {-1,+1} (or ternary if
#                         BN_TERNARY=1) weights + ACT_BITS absmax activations. The target model.
#   "vanilla"           : the FULL-PRECISION baseline -- identical architecture but NO weight
#                         or activation quantization (standard FP32 Dense + LayerNorm + softmax).
#                         The "what did going 1-bit cost us" reference (upper bound on efficiency).
#   "w8a8"              : a CONVENTIONAL 8-bit baseline -- 8-bit absmax weights (per-tensor) and
#                         8-bit absmax activations (per-token). The "1-bit vs ordinary quantization"
#                         reference. Same arch/recipe; only the quantizers change, so the three
#                         runs are a controlled apples-to-apples comparison.
VARIANT       = os.environ.get("BN_VARIANT", "bitnet").strip().lower()
assert VARIANT in ("bitnet", "vanilla", "w8a8"), \
    f"BN_VARIANT must be one of bitnet|vanilla|w8a8 (got {VARIANT!r})"

# ── Architectural-variant knobs (the "a bunch of different transformers" sweep axis) ──
# Every knob DEFAULTS to the current/upstream behaviour, so existing checkpoints and jobs
# are byte-for-byte unchanged; each non-default value is an opt-in architectural ablation
# trained from scratch on GPU. Grounded in the BitNet paper (SubLN, GELU FFN) + Russell's
# original qkerasModel.py. These let one model definition express the whole variant matrix.
#   BN_NORM_TYPE      "layernorm" (default): the paper's mean-subtracting SubLN (Eq 12),
#                       i.e. the existing RMSNorm class. "rmsnorm": TRUE RMSNorm
#                       x*rsqrt(mean(x^2)+eps) with NO mean subtraction (cheaper; drops the
#                       centering term). "none": identity (control: how much does norm buy?).
#   BN_NORM_PLACEMENT "per_linear" (default): a norm inside EVERY BitLinear (upstream).
#                       "shared_prenorm": ONE norm per sub-layer feeding Q/K/V and the FFN
#                       input — the canonical pre-norm transformer placement (~3x fewer norms).
#   BN_POS_ENC        "learned" (default): the upstream Embedding table — NOTE this is applied
#                       to a constant tf.range() so Keras folds it to a FIXED RANDOM offset that
#                       is NOT a tracked/trained weight (kept for back-compat with the lr15 ckpt).
#                       "learned_real": a genuinely trainable (N,D) position table (add_weight).
#                       "none": treat the jet as an unordered set (permutation-invariant).
#                       "sinusoidal": fixed sin/cos (Vaswani).
#   BN_POOL           "gap" (default): GlobalAveragePooling1D. "sum": sum over particles.
#                       "cls": a learnable BERT-style CLS token (sequence 10->11, read row 0).
#   BN_FFN_ACT        "relu" (default) | "gelu" (the BitNet paper's FFN nonlinearity).
NORM_TYPE      = os.environ.get("BN_NORM_TYPE", "layernorm").strip().lower()
NORM_PLACEMENT = os.environ.get("BN_NORM_PLACEMENT", "per_linear").strip().lower()
POS_ENC        = os.environ.get("BN_POS_ENC", "learned").strip().lower()
POOL           = os.environ.get("BN_POOL", "gap").strip().lower()
FFN_ACT        = os.environ.get("BN_FFN_ACT", "relu").strip().lower()
assert NORM_TYPE in ("layernorm", "rmsnorm", "none"), f"bad BN_NORM_TYPE={NORM_TYPE!r}"
assert NORM_PLACEMENT in ("per_linear", "shared_prenorm"), f"bad BN_NORM_PLACEMENT={NORM_PLACEMENT!r}"
assert POS_ENC in ("learned", "learned_real", "none", "sinusoidal"), f"bad BN_POS_ENC={POS_ENC!r}"
assert POOL in ("gap", "sum", "cls"), f"bad BN_POOL={POOL!r}"
assert FFN_ACT in ("relu", "gelu"), f"bad BN_FFN_ACT={FFN_ACT!r}"
_NORM_INSIDE_DEFAULT = (NORM_PLACEMENT == "per_linear")  # BitLinear self-norms iff per_linear


# ══════════════════════════════════════════════════════════════════════════════
# 1-BIT PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

class AbsMeanQuantizer(tf.keras.constraints.Constraint): #inherit from Constraint to apply after each optimizer step
    """
    Straight-through weight quantizer for BitLinear.

    Called INSIDE BitLinear.call (not as a Keras constraint) so the STE runs
    inside the gradient tape against a full-precision latent kernel, per the
    paper's mixed-precision training (latent weights binarized on the fly).

    Default = binary BitNet (arXiv:2310.11453, Eq 1-3,12): centralize to
    zero-mean then Sign, with absmean scale beta:
        W_q = Sign(W - mean(W)) * beta,   beta = mean|W - mean(W)|
    BN_TERNARY=1 = b1.58 (arXiv:2402.17764): absmean-scaled round/clip to
    {-1,0,+1} (no centralization).

    (Still subclasses tf.keras.constraints.Constraint only for backward
    compatibility with older checkpoints; the constraint hook is unused.)
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, w):
        # Straight-through estimator: quantize in forward, identity in backward.
        if TERNARY:
            # b1.58 (arXiv:2402.17764): absmean-scaled round/clip -> {-1,0,+1}.
            # No mean centralization (the 0 level already drops a weight).
            beta = tf.reduce_mean(tf.abs(w)) + self.eps
            w_scaled = w / beta
            q = tf.clip_by_value(tf.round(w_scaled), -1.0, 1.0)
        else:
            # Binary BitNet (arXiv:2310.11453, Eq 1-3,12): centralize weights to
            # zero-mean (alpha) BEFORE the sign, then absmean scale beta.
            alpha = tf.reduce_mean(w)                    # Eq 3
            w = w - alpha                                # centralize (Eq 1): W~ = Sign(W - alpha)
            beta = tf.reduce_mean(tf.abs(w)) + self.eps  # Eq 12 (on centralized W)
            w_scaled = w / beta
            q = tf.sign(w_scaled)                        # Eq 1-2
        w_q = w_scaled + tf.stop_gradient(q - w_scaled)  # stop_gradient => identity backward
        return w_q * beta  # rescale by the absmean factor beta

    def get_config(self):
        return {"eps": self.eps}

## Quant(x) Function
# b-bit symmetric signed absmax levels (set once from BN_ACT_BITS). At 8-bit this is
# exactly the paper's Qb=2^7 range: qpos=+127, qneg=-128 (refactor is a numerical no-op
# vs the old hard-coded 8-bit path). 6-bit -> [-32,+31]; 4-bit -> [-8,+7].
_ACT_QPOS = float(2 ** (ACT_BITS - 1) - 1)   # +127 (b8) / +31 (b6) / +7 (b4)
_ACT_QNEG = float(-(2 ** (ACT_BITS - 1)))    # -128 (b8) / -32 (b6) / -8 (b4)

def _absmax_quant(x, bits, axis):
    """Symmetric signed b-bit absmax quantizer with a straight-through estimator.

    forward = quantize, backward = identity (paper Sec 2.2). tf.round has zero gradient,
    so without the STE this severs gradients through every input -> the model freezes
    (the val_auc=0.5 bug). axis=-1 -> per-token (activations); axis=None -> per-tensor
    (weights). Used for the ACT_BITS activations AND the conventional W8A8 baseline.
    """
    qpos = float(2 ** (bits - 1) - 1)
    qneg = float(-(2 ** (bits - 1)))
    scale = qpos / tf.maximum(tf.reduce_max(tf.abs(x), axis=axis, keepdims=True), 1e-5)
    q = tf.clip_by_value(tf.round(x * scale), qneg, qpos)
    return (x * scale + tf.stop_gradient(q - x * scale)) / scale

def activation_quant(x):
    # b-bit absmax activation quant (paper Eq 4-5, Qb=2^(b-1)), per-token (axis=-1).
    # NOTE: per-token here; the paper uses per-tensor during training and per-token at
    # inference (Sec 2.1). Per-token is finer-grained, is what the b1.58 follow-up uses,
    # and is more stable on this small-sequence (N=10) task.
    return _absmax_quant(x, ACT_BITS, axis=-1)


class BitLinear(Layer):
    """
    A fully-connected layer with binary {-1, +1} weights.

    Replaces tf.keras.layers.Dense for all projections inside the
    transformer.  Bias uses full float32 (bias contributes negligible
    parameter count and is critical for representational capacity at
    small D_MODEL).

    Args:
        units      : output dimensionality
        use_bias   : whether to add a bias term (default True)
        reg        : L1 regularisation strength on the kernel
        name       : layer name

    Calls activation_quant to quantize RMSNorm activation.
    """
    def __init__(self, units, use_bias=True, reg=L1_REG, norm_inside=None, **kwargs):
        super().__init__(**kwargs)
        self.units    = units
        self.use_bias = use_bias
        self.reg      = reg
        # Does this BitLinear apply its own SubLN before quantizing? Defaults to
        # BN_NORM_PLACEMENT (True=per_linear/upstream; False=shared_prenorm, where the
        # enclosing block owns one shared norm). Saved in get_config so new checkpoints are
        # self-describing; old checkpoints (no key) fall back to the per_linear default.
        self.norm_inside = _NORM_INSIDE_DEFAULT if norm_inside is None else bool(norm_inside)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name        = "kernel",
            shape       = (in_dim, self.units),
            initializer = "glorot_uniform",
            regularizer = l1(self.reg),
            trainable   = True,
        )   # FIX: NO constraint -> kernel stays full-precision (latent master weight).
            # Quantization moved into call() so the weight STE runs inside the gradient
            # tape and Adam's sub-threshold updates accumulate (QAT, per the docstring).
        self.quantizer = AbsMeanQuantizer()
        if self.use_bias:
            self.bias = self.add_weight(
                name        = "bias",
                shape       = (self.units,),
                initializer = "zeros",
                regularizer = l1(self.reg),
                trainable   = True,
            )
        self.built = True

    def call(self, x):
        # LayerNorm/SubLN first (shared by all variants), then the variant selects the
        # numeric precision of the matmul. For "bitnet" we quantize the kernel INSIDE the
        # forward pass (STE) against the full-precision latent master weight Adam updates;
        # combined with the activation STE, gradients flow through the whole stack.
        # make_norm() picks the norm type (BN_NORM_TYPE); when norm_inside is False
        # (shared_prenorm) the enclosing block has already normed, so this is a no-op.
        x_norm = make_norm(name=self.name + "_norm")(x) if self.norm_inside else x
        if VARIANT == "vanilla":
            # Full-precision baseline: standard Dense (no weight/activation quantization).
            out = tf.matmul(x_norm, self.kernel)
        elif VARIANT == "w8a8":
            # Conventional 8-bit baseline: 8-bit absmax activations (per-token) x 8-bit
            # absmax weights (per-tensor) -- ordinary fixed-precision quantization.
            out = tf.matmul(_absmax_quant(x_norm, 8, axis=-1),
                            _absmax_quant(self.kernel, 8, axis=None))
        else:  # "bitnet": binary (or ternary) weights x ACT_BITS absmax activations
            out = tf.matmul(activation_quant(x_norm), self.quantizer(self.kernel))
        if self.use_bias:
            out = out + self.bias
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "use_bias": self.use_bias,
                    "reg": self.reg, "norm_inside": self.norm_inside})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(Layer):
    """
    Normalisation applied before activation quantization in each BitLinear.

    NOTE: despite the class name, this subtracts the mean -> it is full
    LayerNorm / SubLN, exactly as the BitNet paper specifies (arXiv:2310.11453,
    Eq 12: LN(x) = (x - E(x)) / sqrt(Var(x) + eps)). The paper introduces LN
    before absmax activation quantization to preserve the output variance
    (Var(y) ~ 1), which stabilises 1-bit training. Kept as LayerNorm to align
    with the paper; the class name is retained for checkpoint/back-compat.

      y = (x - E(x)) / sqrt( Var(x) + eps )
    """
    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        return (x - mean) / tf.sqrt(var + self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg


class TrueRMSNorm(Layer):
    """TRUE RMSNorm (Zhang & Sennrich 2019): y = x * rsqrt(mean(x^2) + eps).

    Unlike the (misnamed) ``RMSNorm`` above — which subtracts the mean and is therefore
    full LayerNorm / SubLN — this drops the mean-centering term entirely. It is the
    ``BN_NORM_TYPE=rmsnorm`` variant: an ablation of "does the BitNet paper's mean
    subtraction actually matter for 1-bit jet tagging?" In hardware it also removes the
    running-mean subtract (one fewer reduction per token), leaving only Sx^2 + inv-sqrt.
    No affine (gamma/beta), to match the parameter-free RMSNorm used elsewhere here.
    """
    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        ms = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return x * tf.math.rsqrt(ms + self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg


class IdentityNorm(Layer):
    """No normalisation (``BN_NORM_TYPE=none``): the lower-bound control. 1-bit stacks
    usually NEED a norm for output-variance control (BitNet Sec 2), so this is expected
    to train worse — it quantifies how much the SubLN actually buys on this task."""
    def call(self, x):
        return x


def make_norm(name=None):
    """Norm-layer factory selected by ``BN_NORM_TYPE``. Default 'layernorm' returns the
    upstream mean-subtracting ``RMSNorm`` with the SAME name pattern, so loading existing
    checkpoints is unaffected. 'rmsnorm' -> TrueRMSNorm; 'none' -> IdentityNorm."""
    if NORM_TYPE == "rmsnorm":
        return TrueRMSNorm(name=name)
    if NORM_TYPE == "none":
        return IdentityNorm(name=name)
    return RMSNorm(name=name)


class CLSToken(Layer):
    """Prepend a single learnable BERT-style classification token (``BN_POOL=cls``).

    Sequence length 10 -> 11; the transformer mixes the CLS row with every particle, and
    the head reads row 0 as the pooled jet representation — an alternative to averaging
    over particles (GlobalAveragePooling1D). The token is a learnable (1,1,D) weight."""
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.cls = self.add_weight(name="cls_vec", shape=(1, 1, d),
                                   initializer="glorot_uniform", trainable=True)
        self.built = True

    def call(self, x):
        B = tf.shape(x)[0]
        tok = tf.tile(self.cls, [B, 1, 1])     # (B,1,D)
        return tf.concat([tok, x], axis=1)     # (B, N+1, D)


def _sinusoidal_pos(n, d):
    """Fixed sinusoidal positional encoding (Vaswani et al.), as a (n,d) tf constant.
    Used for ``BN_POS_ENC=sinusoidal`` — a non-learned alternative to the Embedding table."""
    pos = np.arange(n)[:, None].astype(np.float32)
    i   = np.arange(d)[None, :].astype(np.float32)
    ang = pos / np.power(10000.0, (2.0 * np.floor(i / 2.0)) / float(d))
    pe = np.zeros((n, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(ang[:, 0::2])
    pe[:, 1::2] = np.cos(ang[:, 1::2])
    return tf.constant(pe)


class LearnedPosEnc(Layer):
    """A genuinely TRAINABLE positional encoding (``BN_POS_ENC=learned_real``).

    The upstream default (``BN_POS_ENC=learned``) builds ``Embedding(N,D)(tf.range(N))``;
    because ``tf.range`` is a plain constant (not a KerasTensor wired from the input), Keras
    folds that Embedding into a FIXED RANDOM offset whose weights are never tracked or updated
    — i.e. the "learned" PE never actually learns (verified: it adds 0 trainable params).
    This layer instead allocates the (N,D) table via ``add_weight`` so it IS tracked and
    trained, exactly like the CLS token. Broadcast-added over the batch."""
    def build(self, input_shape):
        n = int(input_shape[-2]); d = int(input_shape[-1])
        self.pos = self.add_weight(name="pos_table", shape=(1, n, d),
                                   initializer="glorot_uniform", trainable=True)
        self.built = True

    def call(self, x):
        return x + self.pos


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class BitMHSA(Layer):
    """
    1-bit Multi-Head Self-Attention.

    Q, K, V projections and the output projection all use BitLinear
    (binary weights).  The attention scores themselves remain in float32 —
    quantising attention logits severely harms performance at small scale.

    Attention core is selectable (env BN_SOFTMAX_FREE):
      * 0 (default): conventional softmax( QK^T / sqrt(d) ).
      * 1: softmax-free ReLU(QK^T / sqrt(d)) with constant 1/N normalization —
        no exp / no row-sum division, so it stays fully binarizable on FPGA
        LUT/logic fabric (the research abstract's binarizable-attention axis).

    Args:
        d_model  : total model dimension
        n_heads  : number of attention heads (d_model % n_heads == 0)
        reg      : L1 regularisation on projection weights
    """
    def __init__(self, d_model, n_heads, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        self.reg      = reg

    def build(self, input_shape):
        _shared = (NORM_PLACEMENT == "shared_prenorm")
        _ni = not _shared   # sub-linears self-norm only in per_linear (upstream) mode
        self.pre_norm = make_norm(name=self.name + "_prenorm") if _shared else None
        self.W_q = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             norm_inside=_ni, name=self.name + "_Wq")
        self.W_k = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             norm_inside=_ni, name=self.name + "_Wk")
        self.W_v = BitLinear(self.d_model, use_bias=False, reg=self.reg,
                             norm_inside=_ni, name=self.name + "_Wv")
        self.W_o = BitLinear(self.d_model, use_bias=True,  reg=self.reg,
                             norm_inside=_ni, name=self.name + "_Wo")
        self.built = True

    def call(self, x, training=False):
        B  = tf.shape(x)[0]
        N  = tf.shape(x)[1]   # sequence length = N_PART_PER_JET = 10 (or 11 with CLS)

        # In shared_prenorm, normalise ONCE here and feed the same normed tensor to Q/K/V
        # (whose BitLinears have norm_inside=False); W_o then operates on the attention
        # context directly. In per_linear (default) pre_norm is None and each BitLinear
        # self-norms, exactly as upstream.
        xqkv = self.pre_norm(x) if self.pre_norm is not None else x

        # Project with binary weights  →  (B, N, d_model)
        Q = self.W_q(xqkv)
        K = self.W_k(xqkv)
        V = self.W_v(xqkv)

        # Split into heads  →  (B, n_heads, N, d_head)
        def split_heads(t):
            t = tf.reshape(t, (B, N, self.n_heads, self.d_head))
            return tf.transpose(t, perm=[0, 2, 1, 3])

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product scores  →  (B, heads, N, N)
        attn_logits = tf.matmul(Q, K, transpose_b=True) / self.scale

        if SOFTMAX_FREE:
            # Softmax-free, fully-binarizable attention: ReLU scores + CONSTANT 1/N
            # normalization. No exp and no row-sum division (the FPGA-expensive,
            # non-binarizable parts of softmax) — only a ReLU (comparator) and a
            # fixed 1/N scale, so the attention core stays on LUT/logic fabric.
            attn_weights = tf.nn.relu(attn_logits)                       # (B, heads, N, N), ≥0
            ctx = tf.matmul(attn_weights, V) / tf.cast(N, tf.float32)    # (B, heads, N, d_head)
        else:
            # Conventional softmax attention (default / paper core).
            attn_weights = tf.nn.softmax(attn_logits, axis=-1)           # (B, heads, N, N)
            ctx = tf.matmul(attn_weights, V)                            # (B, heads, N, d_head)

        # Merge heads  →  (B, N, d_model)
        ctx = tf.transpose(ctx, perm=[0, 2, 1, 3])
        ctx = tf.reshape(ctx, (B, N, self.d_model))

        # Output projection (binary)
        return self.W_o(ctx)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                    "reg": self.reg})
        return cfg


class BitFFN(Layer):
    """
    1-bit Feed-Forward Network.
    Two BitLinear layers with a ReLU in between:
      x → BitLinear(ffn_dim) → ReLU → BitLinear(d_model)
    """
    def __init__(self, d_model, ffn_dim, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.reg     = reg

    def build(self, input_shape):
        _shared = (NORM_PLACEMENT == "shared_prenorm")
        _ni = not _shared
        self.pre_norm = make_norm(name=self.name + "_prenorm") if _shared else None
        self.fc1 = BitLinear(self.ffn_dim, reg=self.reg, norm_inside=_ni, name=self.name+"_fc1")
        self.fc2 = BitLinear(self.d_model, reg=self.reg, norm_inside=_ni, name=self.name+"_fc2")
        self.built = True

    def call(self, x):
        if self.pre_norm is not None:
            x = self.pre_norm(x)          # shared_prenorm: norm once at the FFN input
        x = self.fc1(x)
        x = tf.nn.gelu(x) if FFN_ACT == "gelu" else tf.nn.relu(x)
        x = self.fc2(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "ffn_dim": self.ffn_dim,
                    "reg": self.reg})
        return cfg


class BitTransformerBlock(Layer):
    """
    One transformer block with pre-norm and residual connections:
      x → BitMHSA → + residual
     → BitFFN  → + residual
    """
    def __init__(self, d_model, n_heads, ffn_dim, reg=L1_REG, **kwargs):
        super().__init__(**kwargs)
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.ffn_dim  = ffn_dim
        self.reg      = reg

    def build(self, input_shape):
        self.attn  = BitMHSA(self.d_model, self.n_heads,
                              reg=self.reg, name=self.name + "_attn")
        self.ffn   = BitFFN(self.d_model, self.ffn_dim,
                             reg=self.reg, name=self.name + "_ffn")
        self.built = True

    def call(self, x, training=False):
        # Self-attention sub-layer
        x = x + self.attn(x, training=training)
        # Feed-forward sub-layer
        x = x + self.ffn(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                    "ffn_dim": self.ffn_dim, "reg": self.reg})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_bitnet_jet_tagger(
    n_particles : int   = N_PART_PER_JET,
    n_features  : int   = N_FEAT,
    d_model     : int   = D_MODEL,
    n_heads     : int   = N_HEADS,
    n_layers    : int   = N_LAYERS,
    ffn_dim     : int   = FFN_DIM,
    reg         : float = L1_REG,
) -> Model:
    """
    Build the 1-bit Transformer jet tagger.

    Input  : (batch, n_particles, n_features)  →  same as QKeras CNN
    Output : (batch, 1)                         →  raw logit, no sigmoid

    The model is permutation-equivariant up to the final GlobalAvgPool,
    which makes it a proper Deep-Sets / set-transformer for jet physics.

    Usage
    -----
    model = build_bitnet_jet_tagger()
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["binary_accuracy"])
    """
    if VARIANT == "vanilla":
        _wstr, _astr = "FULL-PRECISION FP32 (vanilla baseline)", "FP32 (no quant)"
    elif VARIANT == "w8a8":
        _wstr, _astr = "INT8 per-tensor (W8 baseline)", "8-bit absmax (A8)"
    else:
        _wstr = "TERNARY {-1,0,+1}" if TERNARY else "BINARY {-1,+1}"
        _astr = f"{ACT_BITS}-bit absmax [{int(_ACT_QNEG)},{int(_ACT_QPOS)}]"
    print(f"[arch] variant={VARIANT.upper()}  weights={_wstr}  act={_astr}  "
          f"attention={'SOFTMAX-FREE ReLU(QK^T)/N (binarizable)' if SOFTMAX_FREE else 'softmax'}  "
          f"d_model={d_model} heads={n_heads} layers={n_layers} ffn={ffn_dim}")
    print(f"[arch] norm={NORM_TYPE} placement={NORM_PLACEMENT} pos_enc={POS_ENC} "
          f"pool={POOL} ffn_act={FFN_ACT}")

    # ── Input ────────────────────────────────────────────────────────────────
    inputs = Input(shape=(n_particles, n_features), name="input_1")

    # ── Input projection: N_FEAT → D_MODEL  (binary BitLinear) ──────────────
    # Mirrors QConv1D(kernel_size=1) which is mathematically identical
    # to a per-particle Dense / BitLinear applied independently.
    x = BitLinear(d_model, reg=reg, name="input_proj")(inputs)
    # shape: (batch, 10, d_model)

    # ── Positional encoding (BN_POS_ENC)  ─────────────────────────────────────
    # Particles are unordered in principle. "learned" (default) gives the model a
    # learnable position token to discover any residual pT-ordering present in the
    # inputs; "sinusoidal" uses a fixed Vaswani table; "none" treats the jet as a pure
    # set (permutation-invariant up to the final pool) — often the right inductive bias.
    if POS_ENC == "learned":
        pos_emb = tf.keras.layers.Embedding(
            input_dim   = n_particles,
            output_dim  = d_model,
            name        = "pos_embedding"
        )(tf.range(n_particles))                 # shape: (10, d_model)
        x = x + pos_emb                          # broadcast over batch
    elif POS_ENC == "learned_real":
        x = LearnedPosEnc(name="pos_enc_learned")(x)   # genuinely trainable (N,D) table
    elif POS_ENC == "sinusoidal":
        x = x + _sinusoidal_pos(n_particles, d_model)
    # else "none": no positional information added.

    # ── Optional learnable CLS token (BN_POOL=cls)  ───────────────────────────
    if POOL == "cls":
        x = CLSToken(name="cls_token")(x)        # sequence n_particles -> n_particles+1

    # ── Transformer blocks  ───────────────────────────────────────────────────
    for i in range(n_layers):
        x = BitTransformerBlock(
            d_model  = d_model,
            n_heads  = n_heads,
            ffn_dim  = ffn_dim,
            reg      = reg,
            name     = f"bit_block_{i}"
        )(x)
    # shape: (batch, 10, d_model)


    # ── Pool sequence → vector (BN_POOL)  ─────────────────────────────────────
    if POOL == "cls":
        # Read the CLS row (BERT-style); the transformer has mixed it with all particles.
        x = tf.keras.layers.Lambda(lambda t: t[:, 0], name="cls_pool")(x)
    elif POOL == "sum":
        x = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="sum_pool")(x)
    else:  # "gap" (default): aggregate over particles
        x = GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    # shape: (batch, d_model)

    # ── Classification head  ──────────────────────────────────────────────────
    # Two BitLinear layers to mirror two QDense layers.
    x = BitLinear(d_model, reg=reg, name="head_fc1")(x)
    x = tf.keras.layers.Activation("relu", name="head_act")(x)

    outputs = BitLinear(1, reg=reg, name="head_fc2")(x)
    # shape: (batch, 1)  — raw logit, no sigmoid  ✓

    return Model(inputs=inputs, outputs=outputs, name=f"{VARIANT}_jet_tagger")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING SCRIPT  (mirrors original train.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    signalTrainFile      = args.SignalTrainFile
    bkgTrainFile         = args.BkgTrainFile
    sig_jetData_TrainFile= args.sig_jetData_TrainFile
    bkg_jetData_TrainFile= args.bkg_jetData_TrainFile

    # ── Optional reproducibility seed (BN_SEED) ──────────────────────────────
    # Unset (default) = upstream behaviour: unseeded random init + data shuffle, so repeated
    # runs sample run-to-run variance. Set BN_SEED=<int> to fix python/numpy/tf RNGs for
    # reproducible A/B comparisons (used by the variant seed-repeat sweep to separate a real
    # architecture effect from training noise).
    _seed = os.environ.get("BN_SEED", "").strip()
    if _seed:
        import random as _random
        _s = int(_seed)
        _random.seed(_s); np.random.seed(_s); tf.random.set_seed(_s)
        print(f"[seed] BN_SEED={_s}  (python/numpy/tf RNGs fixed for reproducibility)")

    print("Reading signal from "          + signalTrainFile)
    print("Reading background from "      + bkgTrainFile)
    print("Reading signal jet data from " + sig_jetData_TrainFile)
    print("Reading background jet data from " + bkg_jetData_TrainFile)

    # ── Load data ────────────────────────────────────────────────────────────
    # NRP volume data uses HDF5 keys 'jet_constituents' (particle inputs,
    # shape [N,141] = 10*14 feats + 1 label col) and 'train_jet_data'
    # (jet features [N,4] = pT,eta,phi,mass). The upstream repo's dataForge.py
    # instead writes 'Training Data' / 'Sample Data', so read whichever exists.
    def _read(path, prefer):
        with h5py.File(path, "r") as hf:
            for k in prefer:
                if k in hf:
                    return hf[k][:]
            return hf[list(hf.keys())[0]][:]   # fall back to the sole dataset

    PART_KEYS = ["jet_constituents", "Training Data"]
    JET_KEYS  = ["train_jet_data",   "Sample Data"]
    dataset       = _read(signalTrainFile,       PART_KEYS)
    datasetQCD    = _read(bkgTrainFile,          PART_KEYS)
    sampleData    = _read(sig_jetData_TrainFile, JET_KEYS)
    sampleDataQCD = _read(bkg_jetData_TrainFile, JET_KEYS)

    dataset    = np.concatenate((dataset, datasetQCD))        # (N, n_part_cols)
    sampleData = np.concatenate((sampleData, sampleDataQCD))  # (N, n_jet_cols)
    n_part_cols = dataset.shape[1]                            # 141 = 140 feats + label

    # Shuffle particle rows together with their matching jet features.
    fullData   = np.concatenate((dataset, sampleData), axis=1)
    np.random.shuffle(fullData)
    dataset    = fullData[:, :n_part_cols]                    # particle inputs + label
    sampleData = fullData[:, n_part_cols:]                    # jet features (pT,eta,phi,mass)

    X = dataset[:, 0 : n_part_cols - 1]
    y = dataset[:, n_part_cols - 1]
    X = X.reshape((X.shape[0], N_PART_PER_JET, N_FEAT))

    # ── Impact parameter normalisation knob  (unchanged) ─────────────────────
    normalizeIPs = False
    if max(X[:, :, 8].ravel()) < 2.0:
        norm_b4 = True
    else:
        print("\nImpact parameter was not normalized beforehand.\n")
        norm_b4 = False

    if norm_b4:
        tag = "bitnet/bitnet_train"
    elif normalizeIPs:
        tag = "bitnet/bitnet_Norm"
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for feat_idx in [8, 9, 10]:
            tmp = scaler.fit_transform([[v] for v in X[:, :, feat_idx].ravel()])
            X[:, :, feat_idx] = tmp.reshape(X[:, :, feat_idx].shape)
    else:
        tag = "bitnet/noNorm_train"

    os.makedirs(os.path.dirname(os.getcwd() + f"/{tag}_model.png"),
                exist_ok=True)

    #plot kinematics
    from util.plotting.kinematics_plotter import kinematics
    kinematics(X, sampleData, y, "v1", tag)

    # ── pT-reweighting  (unchanged) ───────────────────────────────────────────
    thebins    = np.linspace(0,  max(sampleData[:, 0]), 60)
    bkgPts     = sampleData[y == 0][:, 0]
    sigPts     = sampleData[y == 1][:, 0]
    bkg_counts, _ = np.histogram(bkgPts, bins=thebins)
    sig_counts, _ = np.histogram(sigPts, bins=thebins)
    total_bkg  = len(bkgPts)
    total_sig  = len(sigPts)
    weights_pt = np.nan_to_num(sig_counts / bkg_counts,
                               nan=total_sig / total_bkg)

    weights    = np.ones(len(y))
    pt_indices = np.clip(
        np.digitize(sampleData[:, 0], bins=thebins) - 1, 0, len(weights_pt) - 1
    )
    weights[y == 0] = weights_pt[pt_indices][y == 0]

    plt.figure()
    plt.hist(weights, bins=51)
    plt.xlabel("Weights")
    plt.savefig("{}_weights.png".format(tag))

    np.save("{}_bitnetWeights.npy".format(tag),  weights)
    np.save("{}_ptRange.npy".format(tag),        sampleData[:, 0])

    # ── Weights & Biases (optional; enabled when WANDB_PROJECT is set) ─────────
    use_wandb = bool(os.environ.get("WANDB_PROJECT"))
    if use_wandb:
        import wandb
        wandb.init(
            project = os.environ["WANDB_PROJECT"],
            name    = os.environ.get("WANDB_RUN_NAME") or None,
            config  = {"variant": VARIANT, "weights": ("ternary" if TERNARY else "binary")
                       if VARIANT == "bitnet" else ("fp32" if VARIANT == "vanilla" else "int8"),
                       "act_bits": ACT_BITS if VARIANT == "bitnet" else (32 if VARIANT == "vanilla" else 8),
                       "softmax_free": SOFTMAX_FREE,
                       "norm_type": NORM_TYPE, "norm_placement": NORM_PLACEMENT,
                       "pos_enc": POS_ENC, "pool": POOL, "ffn_act": FFN_ACT,
                       "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                       "ffn_dim": FFN_DIM, "l1_reg": L1_REG, "epochs": EPOCHS,
                       "batch_size": BATCH_SIZE, "n_train": int(len(y)),
                       "n_signal": int((y == 1).sum()), "n_bkg": int((y == 0).sum())},
        )

    # ── Build model  ──────────────────────────────────────────────────────────
    model = build_bitnet_jet_tagger()
    model.summary()
    if use_wandb:
        wandb.config.update({"total_params": int(model.count_params())})

    try:
        tf.keras.utils.plot_model(
            model,
            to_file    = os.getcwd() + f"/{tag}_model.png",
            show_shapes= True,
            show_layer_names=True,
        )
    except Exception as e:
        print(f"[warn] plot_model skipped (non-fatal): {e}")

    # ── Pruning  (same schedule as your original) ─────────────────────────────
    # Note: tfmot pruning wraps Dense-like layers. BitLinear is a custom Layer,
    # so we selectively prune only the head Dense equivalents if needed.
    # For the transformer blocks, the binary constraint already achieves ~67%
    # sparsity on average (roughly 1/3 of weights are zero after quantisation).
    # If you want explicit magnitude pruning on top, uncomment the block below.

    # pruning_params = {
    #     "pruning_schedule":
    #         pruning_schedule.ConstantSparsity(0.75, begin_step=2000,
    #                                           frequency=100)
    # }
    # model = prune.prune_low_magnitude(model, **pruning_params)

    # ── Compile  ──────────────────────────────────────────────────────────────
    # Default optimizer is upstream "adam" (1e-3). When BN_LR>0, swap in an explicit
    # Adam(lr=LR) with global grad-norm clipping to stabilise deep 1-bit training.
    if LR > 0:
        opt_kwargs = dict(learning_rate=LR, beta_1=0.9, beta_2=BETA2)
        if CLIPNORM > 0:
            opt_kwargs["global_clipnorm"] = CLIPNORM          # paper: no clipping (set BN_CLIPNORM=0)
        if WEIGHT_DECAY > 0:
            opt_kwargs["weight_decay"] = WEIGHT_DECAY          # paper: 0.01 (AdamW-style)
        optimizer = tf.keras.optimizers.Adam(**opt_kwargs)
        print(f"[opt] Adam(lr={LR}, beta=(0.9,{BETA2}), "
              f"clipnorm={CLIPNORM if CLIPNORM > 0 else 'OFF'}, weight_decay={WEIGHT_DECAY})"
              + (f" + warmup {WARMUP_EPOCHS}ep" if WARMUP_EPOCHS > 0 else "")
              + (f" + poly-decay {DECAY_EPOCHS}ep^{DECAY_POWER}" if DECAY_EPOCHS > 0 else ""))
    else:
        optimizer = "adam"
    model.compile(
        loss      =tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = optimizer,
        metrics   = ["binary_accuracy"],
        weighted_metrics=[tf.keras.metrics.AUC(name="auc")]
    )

    # ── Callbacks  ────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=ES_MONITOR, mode=ES_MODE, verbose=1,
            patience=ES_PATIENCE, restore_best_weights=True
        ),
        # pruning_callbacks.UpdatePruningStep(),  # ← re-enable if pruning above
    ]
    # Optional linear LR warmup (opt-in via BN_LR>0 and BN_WARMUP_EPOCHS>0): ramp LR from
    # LR/WARMUP up to LR over the first WARMUP_EPOCHS epochs, then hold. Stabilises the start
    # of deep 1-bit training; no effect on the default path.
    if LR > 0 and (WARMUP_EPOCHS > 0 or DECAY_EPOCHS > 0):
        def _lr_schedule(epoch, lr):
            # Linear warmup over WARMUP_EPOCHS, then optional polynomial decay to ~0
            # over DECAY_EPOCHS (paper: 750-update warmup + polynomial-decay schedule).
            if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
                return LR * float(epoch + 1) / float(WARMUP_EPOCHS)
            if DECAY_EPOCHS > 0:
                prog = min(1.0, float(epoch - WARMUP_EPOCHS) / float(DECAY_EPOCHS))
                return LR * (1.0 - prog) ** DECAY_POWER
            return LR
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_lr_schedule, verbose=1))
    if use_wandb:
        class _WandbEpoch(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                wandb.log({**(logs or {}), "epoch": epoch})
        callbacks.append(_WandbEpoch())

    # ── Train  ────────────────────────────────────────────────────────────────
    history = model.fit(
        X, y,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        verbose          = 2,
        sample_weight    = np.asarray(weights),
        validation_split = 0.20,
        callbacks        = callbacks,
    )

    # ── Loss curve  ───────────────────────────────────────────────────────────
    plt.figure(figsize=(7, 5), dpi=120)
    plt.plot(history.history["loss"],     label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("BitNet Model Loss", fontsize=25)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/{}_bitnetLoss.pdf".format(tag), dpi=120)

    # ── Save  ─────────────────────────────────────────────────────────────────
    # model = tfmot.sparsity.keras.strip_pruning(model)  # ← re-enable if pruning
    model.save(os.getcwd() + "/{}_bitnetJetTagModel.h5".format(tag))
    print(f"\nModel saved to {tag}_bitnetJetTagModel.h5")

    if use_wandb:
        try:
            wandb.save(os.getcwd() + "/{}_bitnetJetTagModel.h5".format(tag))
        except Exception as e:
            print(f"[warn] wandb.save skipped: {e}")
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK  (no data needed)
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check():
    """
    Verify input/output shapes and that binary constraints are applied.
    Run with:  python bitnet_jet_tagger.py --sanity
    """
    print("=" * 60)
    print("BitNet Jet Tagger — sanity check")
    print("=" * 60)

    model = build_bitnet_jet_tagger()
    model.summary()

    try:
        tf.keras.utils.plot_model(
            model,
            to_file    = os.getcwd() + f"/model.png",
            show_shapes= True,
            show_layer_names=True,
        )
    except Exception as e:
        print(f"[warn] plot_model skipped (non-fatal): {e}")

    # Shape check
    dummy_x = np.random.randn(8, N_PART_PER_JET, N_FEAT).astype(np.float32)
    dummy_y = np.random.randint(0, 2, (8, 1)).astype(np.float32)
    out   = model(dummy_x, training=False)
    assert out.shape == (8, 1), f"Wrong output shape: {out.shape}"

    model.compile(loss="binary_crossentropy", optimizer="adam")
    model.train_on_batch(dummy_x, dummy_y)

    print(f"\n✓  Input  shape : {dummy_x.shape}")
    print(f"✓  Output shape : {out.shape}  (raw logit, no sigmoid)")

    # Check that BitLinear weights are binary after one build
    binary_ok = True
    for layer in model.layers:
        for w in layer.weights:
            if "kernel" in w.name:
                vals = np.unique(np.round(w.numpy(), 4))
                bad  = [v for v in vals if v not in (np.min(vals), 0.0, np.max(vals))]
                if bad:
                    print(f" {w.name}: non-binary values {bad[:5]}, weights.shape {w.numpy().shape}")
                    binary_ok = False

                #vals = np.unique(np.round(w.numpy(), 4))
                good = [v for v in vals]
                if good:
                    print(f" {w.name}: binary values {good[:5]}, , weights.shape {w.numpy().shape}")
    if binary_ok:
        print("  All BitLinear kernels are binary {-1, +1}")

    # Parameter count comparison
    n_params = model.count_params()
    print(f"\nTotal trainable parameters : {n_params:,}")
    print("(Original QKeras CNN est.  : ~2,000–3,000 params)")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BN 1-bit Transformer jet tagger for CMS L1 trigger"
    )
    parser.add_argument("--sanity", action="store_true",
                        help="Run shape/weight sanity check (no data needed)")
    parser.add_argument("SignalTrainFile",       nargs="?", type=str)
    parser.add_argument("BkgTrainFile",          nargs="?", type=str)
    parser.add_argument("sig_jetData_TrainFile", nargs="?", type=str)
    parser.add_argument("bkg_jetData_TrainFile", nargs="?", type=str)
  
    args = parser.parse_args()

    if args.sanity:
        sanity_check()
    else:
        if not all([args.SignalTrainFile, args.BkgTrainFile,
                    args.sig_jetData_TrainFile, args.bkg_jetData_TrainFile]):
            parser.error("Provide all four data file arguments, or use --sanity")
        main(args)
