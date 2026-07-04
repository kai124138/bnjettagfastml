"""config → HGQ2 model, binary weights PINNED, static activation quantizers.

Design (all API facts verified by execution against hgq2 0.1.9 + hls4ml 1.3.0,
see LEDGER 2026-07-04 "source deep-dive"):

* Weight quantizer  = KBI(k0=1, b0=1, i0=1, RND, SAT_SYM, trainable=False,
  heterogeneous_axis=()) — passes ±1 kernels through bit-identically and reports
  1 bit to EBOPs. β handling is FOLD-AWARE (see build_hgq2_model docstring):
  most β's fold away exactly; the 2L+2 residual contributors carry a CSD-2 β̃
  as the quantizer scaler (2-signed-digit constants: DSP-free, bit-exact).
* Activation quant  = KIF(k0=1, i0, f0, RND, SAT, trainable=False) per site, i0
  from gold-model calibration, f0 = act_bits − 1 − i0.
* Attention         = composition of native hgq layers (QEinsumDense for Q/K/V/O
  with head axes in the einsum equations, QEinsum for the two act×act contractions,
  QSoftmax stable with 1/√d_head folded into its exp LUT) — the exact same IR
  primitives hls4ml's QMultiHeadAttention handler decomposes into. Native
  QMultiHeadAttention itself cannot express this model's SubLN-inside-Wo.
* Norm              = PSubLN custom layer (parameter-free LayerNorm; hls4ml side
  registered by bnhgq2.subln.register_subln — keras LayerNormalization is
  unconvertible in this stack, all three paths verified broken).
* PE + input bias   = folded into input_proj's QEinsumDense bias table
  (bias_axes='td' → a quantized (T,D) additive constant).
* Residual adds     = plain float adds (keras Add) — bit_exact carries exact widths.
* GAP               = QGlobalAveragePooling1D (bit-exact; 1/10 costs a wide
  accumulator — accepted, head-only).

Layer names mirror the QKeras checkpoint names so port.py maps 1:1.
"""
from __future__ import annotations

import numpy as np


def _quant_cfgs(act_bits: int):
    from hgq.quantizer import QuantizerConfig

    def binary_kq(scaler=None):
        kw = {} if scaler is None else {"scaler": float(scaler)}
        return QuantizerConfig(
            "kbi", "weight", k0=1, b0=1, i0=1,
            round_mode="RND", overflow_mode="SAT_SYM",
            trainable=False, heterogeneous_axis=(),
            bc=None, ic=None, br=None, ir=None, **kw,
        )

    def act_q(i0: int):
        # RND_CONV = ties-to-even, matching TF/numpy rounding (gold model + the
        # trained QKeras forward); SAT clips like the gold static quantizer.
        # heterogeneous (per-element) i/f variables — initialized at the scalar
        # i0, refined to per-channel calibrated values by port.assign_per_channel_ibits
        # (per-element widths are native to the hls4ml io_parallel flow).
        f0 = act_bits - 1 - int(i0)
        return QuantizerConfig(
            "kif", "datalane", k0=1, i0=int(i0), f0=f0,
            round_mode="RND_CONV", overflow_mode="SAT",
            trainable=False,
            ic=None, fc=None, ir=None, fr=None,
        )

    return binary_kq, act_q


def build_hgq2_model(cfg: dict, binz: dict, calib: dict, enable_ebops: bool = True):
    """binz: binarize_checkpoint() output (beta + fold class per layer).
    calib: {layer_name: i0 int or per-channel array — arrays are reduced to their
    max for the layer-level i0; per-channel refinement is assigned post-build}.

    β handling (fold-aware, measured in gold experiments E3/E4, LEDGER 2026-07-04):
      score_fold (Wq,Wk)      → pure ±1 weights; exact β_q·β_k/√d_head folded into
                                 QSoftmax's input_scaler (exp-LUT — free, exact)
      ln_killed  (Wv)         → pure ±1; β dropped (next LN is scale-invariant)
      bias_fold  (fc1,head_fc1)→ pure ±1; bias ported as b/β (next LN kills scale)
      explicit   (input_proj, Wo, fc2, head_fc2) → kq scaler = CSD-2(β):
                                 2-signed-digit constants, DSP-free by the Vitis
                                 ≤2-digit rule, bit-exact (finite binary fraction)
    """
    import keras
    from hgq.config import LayerConfigScope
    from hgq.layers import QEinsumDense, QDense, QSoftmax, QGlobalAveragePooling1D
    from hgq.layers.ops.einsum import QEinsum

    from .subln import PSubLN
    from .gold import csd2_snap

    A = cfg["arch"]
    T, F, D = A["n_part"], A["n_feat"], A["d_model"]
    H, L, FFN, C = A["n_heads"], A["n_layers"], A["ffn_dim"], A["n_classes"]
    E = D // H
    binary_kq, act_q = _quant_cfgs(cfg["quant"]["act_bits"])

    def kq(name):
        e = binz[name]
        if e["fold"] == "explicit":
            return binary_kq(csd2_snap(e["beta"]))
        return binary_kq(None)

    def i0(name):
        v = calib[name]
        return int(np.max(v))

    with LayerConfigScope(enable_ebops=enable_ebops, beta0=0.0):
        x_in = keras.Input((T, F), name="input_1")

        # ---- input projection: LN -> quant -> ±β̃ einsum-dense with (T,D) bias ----
        h = PSubLN(name="ln_input_proj")(x_in)
        h = QEinsumDense(
            "btf,fd->btd", output_shape=(T, D), bias_axes="td",
            iq_conf=act_q(i0("input_proj")), kq_conf=kq("input_proj"),
            name="input_proj",
        )(h)  # bias table carries input_proj bias + the folded PE constant

        for li in range(L):
            blk = f"bit_block_{li}"
            # ---- attention ----
            ln = PSubLN(name=f"ln_{blk}_attn_qkv")(h)
            q = QEinsumDense("btd,dhe->bthe", output_shape=(T, H, E),
                             iq_conf=act_q(i0(f"{blk}_attn_Wq")),
                             kq_conf=kq(f"{blk}_attn_Wq"),
                             name=f"{blk}_attn_Wq")(ln)
            k = QEinsumDense("btd,dhe->bthe", output_shape=(T, H, E),
                             iq_conf=act_q(i0(f"{blk}_attn_Wk")),
                             kq_conf=kq(f"{blk}_attn_Wk"),
                             name=f"{blk}_attn_Wk")(ln)
            v = QEinsumDense("btd,dhe->bthe", output_shape=(T, H, E),
                             iq_conf=act_q(i0(f"{blk}_attn_Wv")),
                             kq_conf=kq(f"{blk}_attn_Wv"),
                             name=f"{blk}_attn_Wv")(ln)
            scores = QEinsum("bthe,bshe->bhts", name=f"{blk}_attn_scores")([q, k])
            sscale = (binz[f"{blk}_attn_Wq"]["beta"] * binz[f"{blk}_attn_Wk"]["beta"]
                      / float(np.sqrt(E)))
            attn = QSoftmax(axis=-1, stable=True, input_scaler=float(sscale),
                            name=f"{blk}_attn_softmax")(scores)
            ctx = QEinsum("bhts,bshe->bthe", name=f"{blk}_attn_ctx")([attn, v])
            ctx = PSubLN(flatten_axes=2, name=f"ln_{blk}_attn_Wo")(ctx)
            wo = QEinsumDense("bthe,hed->btd", output_shape=(T, D), bias_axes="d",
                              iq_conf=act_q(i0(f"{blk}_attn_Wo")),
                              kq_conf=kq(f"{blk}_attn_Wo"),
                              name=f"{blk}_attn_Wo")(ctx)
            h = keras.layers.Add(name=f"{blk}_add_attn")([h, wo])

            # ---- FFN ----
            ln = PSubLN(name=f"ln_{blk}_ffn_fc1")(h)
            f1 = QEinsumDense("btd,df->btf", output_shape=(T, FFN), bias_axes="f",
                              iq_conf=act_q(i0(f"{blk}_ffn_fc1")),
                              kq_conf=kq(f"{blk}_ffn_fc1"),
                              name=f"{blk}_ffn_fc1")(ln)
            f1 = keras.layers.ReLU(name=f"{blk}_ffn_act")(f1)
            ln = PSubLN(name=f"ln_{blk}_ffn_fc2")(f1)
            f2 = QEinsumDense("btf,fd->btd", output_shape=(T, D), bias_axes="d",
                              iq_conf=act_q(i0(f"{blk}_ffn_fc2")),
                              kq_conf=kq(f"{blk}_ffn_fc2"),
                              name=f"{blk}_ffn_fc2")(ln)
            h = keras.layers.Add(name=f"{blk}_add_ffn")([h, f2])

        # ---- head ----
        h = QGlobalAveragePooling1D(name="gap")(h)
        h = PSubLN(name="ln_head_fc1")(h)
        h = QDense(D, use_bias=True,
                   iq_conf=act_q(i0("head_fc1")), kq_conf=kq("head_fc1"),
                   name="head_fc1")(h)
        h = keras.layers.ReLU(name="head_act")(h)
        h = PSubLN(name="ln_head_fc2")(h)
        out = QDense(C, use_bias=True,
                     iq_conf=act_q(i0("head_fc2")), kq_conf=kq("head_fc2"),
                     name="head_fc2")(h)

        model = keras.Model(x_in, out, name=cfg["name"])
    return model
