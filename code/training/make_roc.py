#!/usr/bin/env python3
"""ROC curves for the trained BitNet jet-tagger runs (multi-run overlay).

Dataset (since 2026-07-01): the public HLS4ML LHC Jet dataset (150 particles)
(Zenodo; arXiv:1804.06913 + 1908.05318) — 5-class one-hot target (g/q/W/Z/t),
constituent lists truncated to the top-N_PART by pT (mirror of
qkerasModel.load_hls4ml_jets). On NRP the held-out evaluation split lives at
/data/hls4ml_lhc_jet/val/val (~26 files × 10k jets).

Two subcommands:
  eval  load ONE saved model + the held-out val/ directory, write {y, score} .npz
        (y one-hot (N,5), score softmax probabilities (N,5))
  plot  aggregate several .npz files into per-class overlaid ROC figures + an
        AUC table (per-class one-vs-rest + macro average)

Why per-model processes: qkerasModel bakes BN_VARIANT / BN_ACT_BITS /
BN_SOFTMAX_FREE into module globals at import, and BitLinear/activation_quant read
them at call time. They are NOT stored in the layer configs, so each model must be
evaluated in a process whose env matches how it was trained. The launcher exports
the right env before each `eval` call.

Unlike the old 2-file format (which had no saved test set and needed a fixed-seed
held-out tail), this dataset ships a dedicated val/ split that training never
touches (training uses validation_split on train/ only) — a true ROC-test set.
Still: NEVER conflate val_auc (the training monitor) with roc_test_auc (this
file's output); they are different data and different evaluations by design
(decisions.md 2026-06-22).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

N_FEAT   = 16
N_PART   = int(os.environ.get("BN_N_PART", 10))   # mirror qkerasModel default
N_CLASSES = 5
CLASS_LABELS = ["j_g", "j_q", "j_w", "j_z", "j_t"]   # one-hot columns, BY NAME
CLASS_NICE   = ["g", "q", "W", "Z", "t"]


def load_eval_set(data_dir, max_n=0, seed=1234):
    """Load every jetImage_*.h5 in data_dir → X (N,N_PART,16), y one-hot (N,5).

    Constituents are re-sorted by constituent pT (descending) before truncation,
    exactly as in qkerasModel.load_hls4ml_jets, so train and eval see the same
    top-N definition. Label columns are located by name via jetFeatureNames.
    """
    import h5py
    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"no .h5 files in {data_dir}")

    def _names(ds):
        return [n.decode() if isinstance(n, bytes) else str(n) for n in ds]

    Xs, Ys = [], []
    for fp in files:
        with h5py.File(fp, "r") as hf:
            const  = hf["jetConstituentList"][:]
            jets   = hf["jets"][:]
            jnames = _names(hf["jetFeatureNames"][:])
            pnames = _names(hf["particleFeatureNames"][:])
        missing = [l for l in CLASS_LABELS if l not in jnames]
        if missing:
            raise KeyError(f"{fp}: labels {missing} not in jetFeatureNames")
        lab_idx = [jnames.index(l) for l in CLASS_LABELS]
        pt_col = next((i for i, n in enumerate(pnames) if n.endswith("_pt")), None)
        if pt_col is not None:
            order = np.argsort(-const[:, :, pt_col], axis=1, kind="stable")
            const = np.take_along_axis(const, order[:, :, None], axis=1)
        Xs.append(const[:, :N_PART, :].astype("float32"))
        Ys.append(jets[:, lab_idx].astype("float32"))
    X, y = np.concatenate(Xs), np.concatenate(Ys)
    if max_n and len(X) > max_n:
        idx = np.random.default_rng(seed).permutation(len(X))[:max_n]
        X, y = X[idx], y[idx]
    return X, y


def find_model(out_dir):
    hits = [h for h in glob.glob(os.path.join(out_dir, "**", "*_bitnetJetTagModel.h5"),
                                 recursive=True)
            if "preS3" not in os.path.basename(h)]   # skip pre-stage3 checkpoints
    if not hits:
        raise FileNotFoundError(f"no *_bitnetJetTagModel.h5 under {out_dir}")
    hits.sort(key=os.path.getmtime)
    return hits[-1]


def load_trained_model(h5):
    import tensorflow as tf
    import qkerasModel as M   # reads BN_VARIANT/BN_ACT_BITS/... globals at import
    co = {n: getattr(M, n) for n in
          ("BitLinear", "RMSNorm", "BitMHSA", "BitFFN", "BitTransformerBlock",
           "AbsMeanQuantizer") if hasattr(M, n)}
    try:
        return tf.keras.models.load_model(h5, custom_objects=co, compile=False), "load_model"
    except Exception as e:                   # robust fallback: rebuild arch from env + load weights
        sys.stderr.write(f"[roc] load_model failed ({e!r}); rebuilding + load_weights\n")
        model = M.build_bitnet_jet_tagger()
        model.load_weights(h5)
        return model, "rebuild+load_weights"


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _runkey(meta):
    if meta["variant"] == "vanilla":
        return "FP32"
    if meta["variant"] == "w8a8":
        return "W8A8"
    return f"A{meta['act_bits']}"            # bitnet -> A8 / A6 / A4


def _wandb_log_roc(meta, y, score, png=None):
    """OPT-IN (--wandb): log ROC-test AUCs + per-class ROC curves to W&B.

    Never lets a W&B failure kill the eval — the .npz on the PVC stays the durable
    source of truth; W&B is a viewing convenience. Requires WANDB_API_KEY in env.
    """
    try:
        import wandb
        from sklearn.metrics import roc_curve
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "bnjettag-bitnet"),
                         name=f"roc-{meta['label']}", job_type="roc-eval",
                         config={k: meta[k] for k in
                                 ("label", "variant", "act_bits", "softmax_free", "n", "model")},
                         reinit=True)
        run.summary["roc_test_auc_macro"] = meta["auc"]
        for c, v in meta["auc_per_class"].items():
            run.summary[f"roc_test_auc_{c}"] = v
        run.summary["n_eval"] = meta["n"]
        for k, c in enumerate(CLASS_NICE):
            fpr, tpr, _ = roc_curve(y[:, k], score[:, k])
            step = max(1, len(fpr) // 500)       # <=~500 points; the curve is smooth
            # HEP convention (matches the matplotlib overlay): x = tagging
            # efficiency (TPR), y = mistag rate (FPR).
            tbl = wandb.Table(data=[[float(t), float(f)] for f, t in
                                    zip(fpr[::step], tpr[::step])],
                              columns=["efficiency", "mistag_rate"])
            run.log({f"roc_curve_{c}": wandb.plot.line(
                tbl, "efficiency", "mistag_rate",
                title=f"ROC {meta['label']} {c} (AUC={meta['auc_per_class'][c]:.4f})")})
        if png and os.path.exists(png):
            run.log({"roc_overlay": wandb.Image(png)})
        run.finish()
        print(f"[roc] wandb: logged roc-{meta['label']} (macro AUC={meta['auc']:.4f})")
    except Exception as e:
        sys.stderr.write(f"[roc] wandb logging skipped: {e!r}\n")


def cmd_eval(a):
    os.environ.setdefault("MPLBACKEND", "Agg")
    X, y = load_eval_set(a.data, a.max_n, a.seed)
    h5 = find_model(a.out_dir)
    model, how = load_trained_model(h5)
    logits = model.predict(X, batch_size=a.batch, verbose=0)
    assert logits.shape[1] == N_CLASSES, \
        f"model outputs {logits.shape[1]} logits, expected {N_CLASSES} — old-format checkpoint?"
    # per-class OvR ROC needs a probability-like per-class score → softmax.
    # (Per-class AUC on softmax probs is the standard hls4ml jet-tagging metric.)
    score = _softmax(logits.astype("float64")).astype("float32")
    from sklearn.metrics import roc_auc_score
    per_class = {c: float(roc_auc_score(y[:, k], score[:, k]))
                 for k, c in enumerate(CLASS_NICE)}
    auc_macro = float(np.mean(list(per_class.values())))
    meta = dict(label=a.label, auc=auc_macro, auc_per_class=per_class,
                n=int(len(y)), model=os.path.basename(h5), how=how,
                variant=os.environ.get("BN_VARIANT", "bitnet"),
                act_bits=int(os.environ.get("BN_ACT_BITS", "8")),
                softmax_free=os.environ.get("BN_SOFTMAX_FREE", "0"))
    os.makedirs(os.path.dirname(os.path.abspath(a.npz)), exist_ok=True)
    np.savez_compressed(a.npz, y=y.astype("float32"), score=score,
                        meta=json.dumps(meta))
    percls = "  ".join(f"{c}={v:.4f}" for c, v in per_class.items())
    print(f"[roc] {a.label:24s} macroAUC={auc_macro:.4f}  [{percls}]  "
          f"n={meta['n']:>8d}  via {how}  ({meta['model']})")
    if a.wandb:
        _wandb_log_roc(meta, y, score)


COLORS = {"A8": "#56b4b0", "A6": "#e6a817", "A4": "#cf3bc4",
          "W8A8": "#d2691e", "FP32": "#e87bd0"}
ORDER = {"A8": 0, "A6": 1, "A4": 2, "W8A8": 3, "FP32": 4}


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as auc_of

    files = sorted(glob.glob(a.glob))
    if not files:
        raise FileNotFoundError(f"no .npz match {a.glob}")
    runs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        m = json.loads(str(d["meta"]))
        y, score = d["y"], d["score"]
        curves, aucs = {}, {}
        for k, c in enumerate(CLASS_NICE):
            fpr, tpr, _ = roc_curve(y[:, k], score[:, k])
            curves[c] = (fpr, tpr)
            aucs[c] = float(auc_of(fpr, tpr))
        m.update(curves=curves, auc_per_class=aucs,
                 auc=float(np.mean(list(aucs.values()))), key=_runkey(m))
        runs.append(m)
    runs.sort(key=lambda m: ORDER.get(m["key"], 99))

    # One panel per class (HEP style: signal eff. vs log-y mistag), runs overlaid.
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6), dpi=150)
    axes = axes.ravel()
    for k, c in enumerate(CLASS_NICE):
        ax = axes[k]
        for m in runs:
            fpr, tpr = m["curves"][c]
            ok = fpr > 0
            ax.plot(tpr[ok], fpr[ok], color=COLORS.get(m["key"]), lw=1.6,
                    label=f'{m["label"]} (AUC={m["auc_per_class"][c]:.4f})')
        ax.set_yscale("log")
        ax.set(xlabel=f"{c} tagging efficiency (TPR)",
               ylabel="Mistag rate (FPR, one-vs-rest)",
               title=f"{c} vs rest", xlim=(0, 1))
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.3, which="both")
    # last cell: macro-AUC summary
    axm = axes[-1]
    axm.axis("off")
    lines = [f'{m["label"]:24s} macro AUC = {m["auc"]:.4f}' for m in runs]
    axm.text(0.02, 0.95, "Macro (mean OvR) AUC\n" + "\n".join(lines),
             family="monospace", fontsize=9, va="top")
    fig.suptitle(a.title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(a.png)), exist_ok=True)
    fig.savefig(a.png)
    print(f"[roc] wrote {a.png}")

    hdr = "| run | variant | act bits | " + " | ".join(f"AUC({c})" for c in CLASS_NICE) \
          + " | macro | n_eval |"
    sep = "|" + "---|" * (len(CLASS_NICE) + 5)
    rows = [hdr, sep]
    rows += ["| {label} | {variant} | {act_bits} | ".format(**m)
             + " | ".join(f'{m["auc_per_class"][c]:.4f}' for c in CLASS_NICE)
             + f' | **{m["auc"]:.4f}** | {m["n"]} |'
             for m in runs]
    table = "\n".join(rows) + "\n"
    if a.table:
        with open(a.table, "w") as fh:
            fh.write(table)
        print(f"[roc] wrote {a.table}")
    print(table)

    if a.wandb:
        try:
            import wandb
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "bnjettag-bitnet"),
                             name="roc-overlay", job_type="roc-plot", reinit=True)
            run.log({"roc_overlay": wandb.Image(a.png)})
            cols = ["run", "variant", "act_bits"] + \
                   [f"auc_{c}" for c in CLASS_NICE] + ["auc_macro", "n_eval"]
            run.log({"roc_auc_table": wandb.Table(
                columns=cols,
                data=[[m["label"], m["variant"], m["act_bits"]]
                      + [m["auc_per_class"][c] for c in CLASS_NICE]
                      + [m["auc"], m["n"]]
                      for m in runs])})
            run.finish()
            print("[roc] wandb: logged overlay + AUC table")
        except Exception as e:
            sys.stderr.write(f"[roc] wandb logging skipped: {e!r}\n")


def main():
    p = argparse.ArgumentParser(description="ROC curves for BitNet jet-tagger runs "
                                            "(HLS4ML LHC Jet 150p, 5-class)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="evaluate one model -> .npz")
    e.add_argument("--out-dir", required=True, help="run dir holding *_bitnetJetTagModel.h5")
    e.add_argument("--data", required=True,
                   help="held-out eval dir of jetImage_*.h5 (NRP: /data/hls4ml_lhc_jet/val/val)")
    e.add_argument("--npz", required=True)
    e.add_argument("--label", required=True)
    e.add_argument("--seed", type=int, default=1234, help="subsample seed (with --max-n)")
    e.add_argument("--max-n", type=int, default=0, help="0 = evaluate the full val set")
    e.add_argument("--batch", type=int, default=8192)
    e.add_argument("--wandb", action="store_true",
                   help="also log AUCs + ROC curves to W&B (needs WANDB_API_KEY)")
    e.set_defaults(fn=cmd_eval)

    q = sub.add_parser("plot", help="aggregate .npz -> per-class ROC png + AUC table")
    q.add_argument("--glob", required=True)
    q.add_argument("--png", required=True)
    q.add_argument("--table", default="")
    q.add_argument("--title", default="BitNet jet tagger - ROC (HLS4ML LHC Jet 150p, held-out val)")
    q.add_argument("--wandb", action="store_true",
                   help="also log the overlay + AUC table to W&B (needs WANDB_API_KEY)")
    q.set_defaults(fn=cmd_plot)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
