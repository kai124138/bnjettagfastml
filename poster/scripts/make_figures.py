#!/usr/bin/env python3
"""FastML26 poster figures — reconciled numbers only (see poster/VERIFICATION.md).

Every constant below was re-derived from raw store data by poster/scripts/verify_gate.py
and verify_ebops_analytic.py on 2026-07-04; the raw source is cited next to each value.
Output: poster/figures/*.pdf + *.svg (vector).

Palette validated with the dataviz six-checks validator (light surface, all PASS):
  A8 #2a78d6 · A6 #199e70 · A4 #e34948 · norm/SubLN #4a3aa7; FP32 ink, W8A8 muted.
Color follows the entity across all figures; trained = solid, HGQ2 rebuild = dashed.
"""
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/kaiyamaguchi/Downloads/bnjettag-training-results/poster"
FIG = f"{ROOT}/figures"

INK = "#0b0b0b"; SEC = "#52514e"; MUT = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"; SURF = "#ffffff"
C_A8 = "#2a78d6"; C_A6 = "#199e70"; C_A4 = "#e34948"; C_NORM = "#4a3aa7"
C_FP32 = INK; C_W8A8 = MUT

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": SEC, "axes.linewidth": 1.0,
    "axes.titlecolor": INK, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUT, "ytick.color": MUT, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

PROVISIONAL = ("Large model (D256/L8), per-shape/per-probe measurements — provisional; "
               "deployable single-model table awaits r6-small")

# ---------------- verified values (raw sources in comments) ----------------
V = json.load(open(f"{ROOT}/data/verification_results.json"))

AUC = {  # recomputed from raw y/score arrays (verify_gate.py); n = 260,000 each
    "FP32": V["refs"]["FP32-lr05"]["recomputed_macro"],
    "W8A8": V["refs"]["W8A8-lr05"]["recomputed_macro"],
    "A8": V["refs"]["A8-lr05-s1"]["recomputed_macro"],
    "A6": V["refs"]["A6-lr05-s1"]["recomputed_macro"],
    "A4": V["refs"]["A4-lr05-s1"]["recomputed_macro"],
    "R8": V["rebuilds"]["W1A8"]["recomputed_macro"],
    "R6": V["rebuilds"]["W1A6"]["recomputed_macro"],
    "R4": V["rebuilds"]["W1A4"]["recomputed_macro"],
}
# HGQ2-native EBOPs: sums of results/hgq2/runs/<hash>/ebops.json per_layer (exact)
EBOPS_NATIVE = {"8": 650_360_941, "6": 501_183_824, "4": 352_214_162}
# Analytic (HGQ arXiv:2405.00645, no accumulator): re-derived from D256/L8/FFN1024/T10/F16/C5
EBOPS_ANALYTIC = {"FP32": 64_954_302_464, "W8A8": 4_059_643_904,
                  "W1A8": 530_393_088, "W1A6": 392_879_616, "W1A4": 258_642_944}
# Binary FFN block (fc1 256->1024 -> ReLU -> fc2 1024->256, no norm), QKeras path,
# RF=256, VU13P — raw results/csynth/csynth_report_a{8,6,4}_rf256.json
FFN = {"8": dict(LUT=440_882, DSP=0), "6": dict(LUT=429_098, DSP=0), "4": dict(LUT=415_259, DSP=0)}
VU13P_LUT = 1_728_000; VU13P_DSP = 12_288
# Whole-probe csynth totals, HGQ2 path — raw results/hgq2/runs/b224a8ea/*/csynth_report.json
PROBES = [  # (label line, detail line, DSP, color, per-function annotation)
    # per-function splits raw-verified from csynth.xml module tables (VERIFICATION.md §5)
    ("Binary FFN block — A8",  "fc1 256-1024, ReLU, fc2 · no norm · RF=256 · era-1-shape build", 0, C_A8, None),
    ("Binary FFN block — A6",  "same block, 6-bit activations",              0,    C_A6,   None),
    ("Binary FFN block — A4",  "same block, 4-bit activations",              0,    C_A4,   None),
    ("SubLN + binary dense (256-5) + affine", "whole chain · Latency, RF=32", 112,  C_NORM,
     "all 112 in SubLN · dense 0 · affine 0"),
    ("SubLN + binary dense (256-256, v3)", "beta in weights · Resource RF=256 — ROM trap", 270, C_NORM,
     "= dense 256 + SubLN 14 (different build, not a strategy flip)"),
    ("SubLN alone, fully unrolled",   "single-layer probe · II=1, dim 256",  1792, C_NORM, None),
]

CLASSES = ["g", "q", "W", "Z", "t"]
CLASS_TITLES = {"g": "gluon", "q": "light quark", "W": "W boson", "Z": "Z boson", "t": "top"}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.svg")
    plt.close(fig)
    print(f"wrote {name}.pdf/.svg")


def footer(fig, source, y=0.012):
    fig.text(0.02, y + 0.030, PROVISIONAL, fontsize=8.2, color=SEC, style="italic")
    fig.text(0.02, y, source, fontsize=7.4, color=MUT)


# ============ Figure 1 — tradeoff table ============
def fig_tradeoff():
    rows = [
        # model, trained AUC, rebuild AUC, delta, native EBOPs, analytic EBOPs
        ("FP32 (vanilla)",      AUC["FP32"], None,      None,     None,             EBOPS_ANALYTIC["FP32"]),
        ("W8A8",                AUC["W8A8"], None,      None,     None,             EBOPS_ANALYTIC["W8A8"]),
        ("W1A8  (binary, 8-bit act)", AUC["A8"], AUC["R8"], AUC["R8"] - AUC["A8"], EBOPS_NATIVE["8"], EBOPS_ANALYTIC["W1A8"]),
        ("W1A6  (binary, 6-bit act)", AUC["A6"], AUC["R6"], AUC["R6"] - AUC["A6"], EBOPS_NATIVE["6"], EBOPS_ANALYTIC["W1A6"]),
        ("W1A4  (binary, 4-bit act)", AUC["A4"], AUC["R4"], AUC["R4"] - AUC["A4"], EBOPS_NATIVE["4"], EBOPS_ANALYTIC["W1A4"]),
    ]
    headers = ["Model", "Trained AUC", "HGQ2 rebuild AUC", "Δ (rebuild − trained)",
               "EBOPs (HGQ2-native)", "EBOPs (analytic)"]
    sub = ["", "ROC-test macro-OvR", "binary-pinned, static act quant", "",
           "accumulator in", "no accumulator"]

    def fm_ebop(v):
        if v is None: return "—"
        return f"{v/1e9:.2f} G" if v >= 1e9 else f"{v/1e6:.1f} M"

    fig = plt.figure(figsize=(11.4, 4.6))
    ax = fig.add_axes([0.02, 0.16, 0.96, 0.70]); ax.axis("off")
    ax.set_title("Tagging efficiency vs compute across the activation-quantization axis",
                 loc="left", pad=26, fontsize=14)
    fig.text(0.02, 0.845, "Era-2 HLS4ML LHC Jet 5-class · held-out split n = 260,000 · trained refs = seed s1 · "
             "the two EBOPs conventions are separate columns, never mixed",
             fontsize=9.5, color=SEC)
    xs = [0.00, 0.25, 0.41, 0.585, 0.74, 0.895]
    y0, dy = 0.86, 0.155
    for j, (h, s) in enumerate(zip(headers, sub)):
        ha = "left" if j == 0 else "right"
        xj = xs[j] if j == 0 else xs[j] + 0.09
        ax.text(xj, y0 + 0.06, h, ha=ha, fontsize=10.5, weight="bold", color=INK, transform=ax.transAxes)
        if s: ax.text(xj, y0 + 0.005, s, ha=ha, fontsize=8, color=MUT, transform=ax.transAxes)
    ax.plot([0, 1], [y0 - 0.035] * 2, color=AXIS, lw=1.2, transform=ax.transAxes, clip_on=False)
    colors = {2: C_A8, 3: C_A6, 4: C_A4}
    for i, (name, tr, rb, d, en, ea) in enumerate(rows):
        y = y0 - 0.115 - i * dy
        if i in colors:
            ax.plot([0.002], [y + 0.012], marker="s", ms=7, color=colors[i],
                    transform=ax.transAxes, clip_on=False)
            ax.text(xs[0] + 0.018, y, name, fontsize=10.5, color=INK, transform=ax.transAxes)
        else:
            ax.text(xs[0] + 0.018, y, name, fontsize=10.5, color=INK, transform=ax.transAxes)
        cells = [f"{tr:.4f}",
                 "—" if rb is None else f"{rb:.4f}",
                 "—" if d is None else f"{d:+.4f}",
                 fm_ebop(en), fm_ebop(ea)]
        for j, c in enumerate(cells, start=1):
            w = "bold" if (j == 2 and rb is not None) else "normal"
            ax.text(xs[j] + 0.09, y, c, ha="right", fontsize=10.5, color=INK,
                    weight=w, transform=ax.transAxes,
                    fontfamily="sans-serif")
        if i < len(rows) - 1:
            ax.plot([0, 1], [y - 0.045] * 2, color=GRID, lw=0.8, transform=ax.transAxes, clip_on=False)
    fig.text(0.02, 0.085,
             "HGQ2-native EBOPs exist only for the three rebuilt binary models (no HGQ2 build of FP32/W8A8). "
             "Within the analytic convention only: W8A8 / W1A8 = 7.65×.",
             fontsize=8.6, color=SEC)
    footer(fig, "Sources: roc-results/r5/*.npz · results/hgq2/runs/*/scores.npz · "
                "results/hgq2/runs/*/ebops.json · results/ebops.md (re-derived) — poster/VERIFICATION.md §1–§2")
    save(fig, "fig1_tradeoff_table")


# ============ Figure 2 — per-class ROC overlay (HEP convention) ============
def interp_curve(fpr, tpr, n=900, lo=8e-4):
    grid = np.logspace(np.log10(lo), 0, n)
    return grid, np.interp(grid, fpr, tpr)


def fig_roc():
    curves = np.load(f"{ROOT}/data/roc_curves.npz")
    series = [  # (stem, label, color, ls, lw)
        ("FP32-lr05",  f"FP32 trained · macro {AUC['FP32']:.4f}", C_FP32, "-", 1.7),
        ("W8A8-lr05",  f"W8A8 trained · {AUC['W8A8']:.4f}",       C_W8A8, "-", 1.7),
        ("A8-lr05-s1", f"W1A8 trained · {AUC['A8']:.4f}",         C_A8,   "-", 2.0),
        ("W1A8",       f"W1A8 HGQ2 rebuild · {AUC['R8']:.4f}",    C_A8,   (0, (4, 2)), 2.0),
        ("A6-lr05-s1", f"W1A6 trained · {AUC['A6']:.4f}",         C_A6,   "-", 2.0),
        ("W1A6",       f"W1A6 HGQ2 rebuild · {AUC['R6']:.4f}",    C_A6,   (0, (4, 2)), 2.0),
        ("A4-lr05-s1", f"W1A4 trained · {AUC['A4']:.4f}",         C_A4,   "-", 2.0),
        ("W1A4",       f"W1A4 HGQ2 rebuild · {AUC['R4']:.4f}",    C_A4,   (0, (4, 2)), 2.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.865, bottom=0.14, hspace=0.24, wspace=0.10)
    fig.suptitle("Per-class ROC — HGQ2 rebuild vs trained vs FP32 (era-2, n = 260,000)",
                 x=0.065, ha="left", fontsize=14, weight="bold", color=INK)
    fig.text(0.065, 0.895, "Signal efficiency vs mistag rate, mistag on log axis · "
             "solid = trained QKeras (seed s1) · dashed = binary-pinned HGQ2 rebuild",
             fontsize=9.5, color=SEC)
    for k, cls in enumerate(CLASSES):
        ax = axes.flat[k]
        for stem, lab, col, ls, lw in series:
            fpr = curves[f"{stem}__{cls}__fpr"]; tpr = curves[f"{stem}__{cls}__tpr"]
            g, t = interp_curve(fpr, tpr)
            ax.plot(t, g, color=col, ls=ls, lw=lw, solid_capstyle="round")
        ax.set_yscale("log"); ax.set_ylim(8e-4, 1.0); ax.set_xlim(0, 1.0)
        ax.grid(True, which="major", axis="both")
        ax.set_title(f"{cls} — {CLASS_TITLES[cls]}", fontsize=11.5, loc="left", weight="bold")
        if k >= 2: ax.set_xlabel("Signal efficiency (TPR)")
        if k % 3 == 0: ax.set_ylabel("Mistag rate (FPR)")
    lax = axes.flat[5]; lax.axis("off")
    handles = [mpl.lines.Line2D([], [], color=c, ls=ls, lw=lw) for _, _, c, ls, lw in series]
    lax.legend(handles, [s[1] for s in series], loc="upper left", fontsize=9.6,
               handlelength=2.6, labelspacing=0.55, borderaxespad=0.0)
    lax.text(0.0, 0.06, "Macro-OvR AUCs recomputed from raw score arrays;\n"
             "identical 260k eval split for every curve (bit-identical labels).",
             fontsize=8.2, color=SEC, transform=lax.transAxes)
    axes.flat[2].set_xlabel("Signal efficiency (TPR)")  # panel above the legend cell
    axes.flat[2].tick_params(labelbottom=True)
    footer(fig, "Sources: roc-results/r5/*.npz · results/hgq2/runs/*/scores.npz — poster/VERIFICATION.md §1")
    save(fig, "fig2_roc_per_class")


# ============ Figure 3 — DSP: the norm is the cost ============
def fig_dsp():
    fig = plt.figure(figsize=(11.4, 5.8))
    ax = fig.add_axes([0.30, 0.155, 0.66, 0.60])
    vals = [p[2] for p in PROBES]; cols = [p[3] for p in PROBES]
    ypos = np.arange(len(PROBES))[::-1] * 1.0
    ypos[3:] -= 0.45  # gap between the norm-free group and the norm-bearing group
    for y, v, c, p in zip(ypos, vals, cols, PROBES):
        if v == 0:
            ax.plot([0], [y], marker="|", ms=16, mew=2.4, color=c)
            ax.text(18, y, "0 DSP", va="center", fontsize=11, weight="bold", color=c)
        else:
            ax.barh(y, v, height=0.55, color=c, edgecolor="none")
            ax.text(v + 18, y, f"{v:,}", va="center", fontsize=11, weight="bold", color=INK)
            if p[4]:
                ax.text(v + 115, y, p[4], va="center", fontsize=8.2, color=SEC)
        ax.text(-30, y + 0.14, p[0], ha="right", va="center", fontsize=9.8, color=INK)
        ax.text(-30, y - 0.20, p[1], ha="right", va="center", fontsize=7.6, color=MUT)
    ax.set_yticks([])
    ax.set_xlim(0, 1900); ax.set_ylim(min(ypos) - 0.7, max(ypos) + 0.75)
    ax.set_xlabel("DSP48 slices used (whole csynth probe, VU13P, Vitis HLS 2023.2)")
    ax.grid(True, axis="x")
    fig.suptitle("Norm-free binary matmul probes synthesize to exactly 0 DSP",
                 x=0.035, y=0.965, ha="left", fontsize=14, weight="bold", color=INK)
    fig.text(0.035, 0.905, "Whole-probe csynth totals, with per-function splits from the raw "
             "per-instance module tables — in the SubLN-bearing chains, the norm carries the DSPs.",
             fontsize=9.5, color=SEC)
    # group captions inside the plot, over each group
    ax.text(30, ypos[0] + 0.62, "norm-free binary probes", fontsize=9.6, weight="bold", color=SEC)
    ax.text(30, ypos[3] + 0.62, "SubLN-bearing probes", fontsize=9.6, weight="bold", color=C_NORM)
    ax.text(900, ypos[1] + 0.45, f"VU13P total: {VU13P_DSP:,} DSP — even the fully-unrolled\n"
            "SubLN uses 14.6%; binary matmuls with compile-time-\n"
            "constant weights use none (folded chain: dense 0 DSP /\n"
            "23,788 LUT · CSD-2 affine 0 DSP / 285 LUT).\n"
            "Per-shape probe DSP is identical at A8/A6/A4\n"
            "(activation-precision-independent).\n\n"
            "Context — the weightless attention act×act core\n"
            "(nothing to binarize, 0.65% of MACs), fully unrolled:\n"
            "each einsum exactly 1 DSP/MAC (2×25,600 of 52,000 total).",
            fontsize=8.8, color=SEC, va="top", linespacing=1.5)
    footer(fig, "Sources: results/csynth/csynth_report_a{8,6,4}_rf256.json · "
                "results/hgq2/runs/b224a8ea/probe_*/{csynth_report.json, csynth.xml module tables} "
                "+ probe firmware — poster/VERIFICATION.md §3, §5",
           y=0.012)
    save(fig, "fig3_dsp_probes")


# ============ Figure 4 — sweep vs activation precision ============
def fig_sweep():
    bits = [8, 6, 4]
    fam = {8: C_A8, 6: C_A6, 4: C_A4}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.8, 4.4))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.20, wspace=0.30)
    fig.suptitle("The activation-quantization axis: efficiency, compute, resources (binary weights throughout)",
                 x=0.06, ha="left", fontsize=14, weight="bold", color=INK)

    # (a) AUC vs bits
    tr = [AUC["A8"], AUC["A6"], AUC["A4"]]; rb = [AUC["R8"], AUC["R6"], AUC["R4"]]
    a.axhline(AUC["FP32"], color=C_FP32, lw=1.2, ls=(0, (1, 2)))
    a.axhline(AUC["W8A8"], color=C_W8A8, lw=1.2, ls=(0, (1, 2)))
    a.text(3.92, AUC["FP32"] + 0.004, f"FP32  {AUC['FP32']:.4f}", fontsize=8.4, color=C_FP32, ha="left")
    a.text(3.92, AUC["W8A8"] - 0.011, f"W8A8  {AUC['W8A8']:.4f}", fontsize=8.4, color=C_W8A8, ha="left")
    a.plot(bits, tr, "-", color=INK, lw=1.8, marker="o", ms=7, mfc=INK, mec=INK, zorder=3)
    a.plot(bits, rb, ls=(0, (4, 2)), color=INK, lw=1.6, marker="o", ms=7, mfc="white", mec=INK, zorder=3)
    for x, t, r in zip(bits, tr, rb):
        a.plot([x, x], [r, t], color=fam[x], lw=2.6, alpha=0.55, zorder=2)
    a.text(8.05, tr[0] + 0.006, "trained", fontsize=9, color=INK)
    a.text(8.05, rb[0] - 0.017, "HGQ2 rebuild", fontsize=9, color=INK)
    a.set_xticks(bits, [f"A{b}" for b in bits]); a.set_xlim(8.9, 3.4)
    a.set_ylim(0.69, 0.90); a.grid(True, axis="y")
    a.set_xlabel("activation bits (weights 1-bit)"); a.set_ylabel("ROC-test macro-OvR AUC")
    a.set_title("Tagging efficiency", loc="left", fontsize=11.5)

    # (b) EBOPs (HGQ2-native) vs bits
    ebv = [EBOPS_NATIVE["8"], EBOPS_NATIVE["6"], EBOPS_NATIVE["4"]]
    xs = np.arange(3)
    b.bar(xs, [v / 1e6 for v in ebv], width=0.56, color=[fam[8], fam[6], fam[4]])
    for x, v in zip(xs, ebv):
        b.text(x, v / 1e6 + 12, f"{v/1e6:.1f} M", ha="center", fontsize=9.6, weight="bold", color=INK)
    b.set_xticks(xs, ["W1A8", "W1A6", "W1A4"]); b.set_ylim(0, 720)
    b.grid(True, axis="y"); b.set_ylabel("EBOPs (millions)")
    b.set_title("Compute — EBOPs (HGQ2-native)", loc="left", fontsize=11.5)
    b.text(0.97, 0.965, "one convention throughout\n(accumulator term in)",
           fontsize=7.8, color=MUT, ha="right", va="top", transform=b.transAxes)

    # (c) FFN-block LUT vs bits, DSP=0
    lut = [FFN["8"]["LUT"], FFN["6"]["LUT"], FFN["4"]["LUT"]]
    c.bar(xs, [v / 1e3 for v in lut], width=0.56, color=[fam[8], fam[6], fam[4]])
    for x, v in zip(xs, lut):
        c.text(x, v / 1e3 + 8, f"{v/1e3:.0f} k", ha="center", fontsize=9.6, weight="bold", color=INK)
        c.text(x, v / 1e3 / 2, f"{v/VU13P_LUT*100:.1f}%\nof VU13P", ha="center", fontsize=8.0,
               color="white", weight="bold")
    c.set_xticks(xs, ["A8", "A6", "A4"]); c.set_ylim(0, 560)
    c.grid(True, axis="y"); c.set_ylabel("LUT (thousands)")
    c.set_title("Binary FFN block — DSP = 0 at all precisions", loc="left", fontsize=11.5)
    c.text(0.97, 0.965, "RF=256 · 520 cycles · II 256 · 400 MHz met\nQKeras path, era-1-shape build (shape era-identical)",
           fontsize=7.8, color=MUT, ha="right", va="top", transform=c.transAxes)

    footer(fig, "Sources: verified AUCs (§1) · ebops.json sums (§2A) · "
                "csynth_report_a{8,6,4}_rf256.json (§3) — poster/VERIFICATION.md", y=0.015)
    save(fig, "fig4_precision_sweep")


fig_tradeoff()
fig_roc()
fig_dsp()
fig_sweep()
print("done")
