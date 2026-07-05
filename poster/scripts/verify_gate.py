#!/usr/bin/env python3
"""Poster verification gate — independent recomputation from RAW arrays only.

Reads (read-only):
  qkeras-bitnet-run-2026-06-22/results/hgq2/runs/<hash>/scores.npz   (HGQ2 rebuild scores)
  qkeras-bitnet-run-2026-06-22/roc-results/r5/*.npz                  (trained QKeras reference scores)
  qkeras-bitnet-run-2026-06-22/results/hgq2/runs/<hash>/ebops.json   (HGQ2-native EBOPs)
Writes (poster/ only):
  poster/data/verification_results.json
  poster/data/roc_curves.npz   (per-model per-class ROC curves for the overlay figure)

No number from any summary table/markdown is used as input — only raw y/score arrays
and the raw per-layer EBOPs payloads. Claimed values are listed here solely as the
comparison targets the recomputation is checked against.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = "/Users/kaiyamaguchi/Downloads/bnjettag-training-results"
RUN = f"{ROOT}/qkeras-bitnet-run-2026-06-22"
OUT = f"{ROOT}/poster/data"

CLASSES = ["g", "q", "W", "Z", "t"]  # column order hypothesis; proven below via meta per-class match

REBUILDS = {  # hash -> (name, claimed macro AUC in tradeoff_table.md)
    "b224a8ea": ("W1A8", 0.8493),
    "a428e6e2": ("W1A6", 0.8222),
    "53b202bc": ("W1A4", 0.7115),
}
REFS = {  # r5 file stem -> claimed macro AUC in roc-results/r5/roc_auc.md
    "A8-lr05-s1": 0.8551, "A8-lr05-s2": 0.8452,
    "A6-lr05-s1": 0.8394, "A6-lr05-s2": 0.8220,
    "A4-lr05-s1": 0.7346, "A4-lr05-s2": 0.7312,
    "W8A8-lr05": 0.8642, "FP32-lr05": 0.8765,
}
# tradeoff_table.md claims: ref column + corr(scores) per rebuild (vs matching s1 ref)
TRADEOFF_CLAIMS = {
    "W1A8": {"ref": "A8-lr05-s1", "ref_auc": 0.8551, "delta": -0.0058, "corr": 0.9589},
    "W1A6": {"ref": "A6-lr05-s1", "ref_auc": 0.8394, "delta": -0.0173, "corr": 0.8944},
    "W1A4": {"ref": "A4-lr05-s1", "ref_auc": 0.7346, "delta": -0.0231, "corr": 0.6789},
}
EBOPS_CLAIMS = {"b224a8ea": 650_360_941, "a428e6e2": 501_183_824, "53b202bc": 352_214_162}

results = {"classes_column_order": None, "refs": {}, "rebuilds": {}, "ebops": {}, "integrity": {}}
curves = {}


def macro_and_per_class(y, s):
    per = [float(roc_auc_score(y[:, i], s[:, i])) for i in range(y.shape[1])]
    return float(np.mean(per)), per


# ---- 1. r5 reference runs: recompute macro + per-class, prove column order via meta ----
order_checks = []
ref_data = {}
for stem, claimed in REFS.items():
    d = np.load(f"{RUN}/roc-results/r5/{stem}.npz")
    y, s = d["y"], d["score"]
    meta = json.loads(str(d["meta"]))
    macro, per = macro_and_per_class(y, s)
    # column-order proof: recomputed per-column AUC must match meta's per-class dict (g,q,W,Z,t)
    meta_per = meta["auc_per_class"]
    col_match = all(abs(per[i] - meta_per[c]) < 5e-10 for i, c in enumerate(CLASSES))
    order_checks.append(col_match)
    ref_data[stem] = (y, s)
    results["refs"][stem] = {
        "recomputed_macro": macro,
        "claimed_macro_roc_auc_md": claimed,
        "match_claimed_4dp": bool(abs(macro - claimed) < 5e-5),
        "recomputed_per_class": dict(zip(CLASSES, per)),
        "meta_macro": meta["auc"],
        "match_meta_exact": bool(abs(macro - meta["auc"]) < 1e-9),
        "per_class_matches_meta_order_gqWZt": col_match,
        "n": int(y.shape[0]),
    }
results["classes_column_order"] = "g,q,W,Z,t CONFIRMED" if all(order_checks) else "ORDER CHECK FAILED"

# ---- 2. HGQ2 rebuilds: recompute macro, delta vs ref, corr(scores) ----
for h, (name, claimed) in REBUILDS.items():
    d = np.load(f"{RUN}/results/hgq2/runs/{h}/scores.npz")
    y, s = d["y"], d["score"]
    meta = json.loads(str(d["meta"]).replace("'", '"'))
    macro, per = macro_and_per_class(y, s)
    tc = TRADEOFF_CLAIMS[name]
    y_ref, s_ref = ref_data[tc["ref"]]
    y_identical = bool(np.array_equal(y, y_ref))
    corr = float(np.corrcoef(s.ravel(), s_ref.ravel())[0, 1])
    ref_macro, _ = macro_and_per_class(y_ref, s_ref)
    delta = macro - ref_macro
    results["rebuilds"][name] = {
        "hash": h,
        "recomputed_macro": macro,
        "claimed_macro_tradeoff_table": claimed,
        "match_claimed_4dp": bool(abs(macro - claimed) < 5e-5),
        "recomputed_per_class": dict(zip(CLASSES, per)),
        "npz_meta_auc_macro": float(meta["auc_macro"]),
        "match_meta_exact": bool(abs(macro - float(meta["auc_macro"])) < 1e-9),
        "ref_run": tc["ref"],
        "ref_recomputed_macro": ref_macro,
        "delta_recomputed": delta,
        "delta_claimed": tc["delta"],
        "delta_match_4dp": bool(abs(delta - tc["delta"]) < 5e-5),
        "corr_recomputed": corr,
        "corr_claimed": tc["corr"],
        "corr_match_4dp": bool(abs(corr - tc["corr"]) < 5e-5),
        "y_identical_to_ref": y_identical,
        "n": int(y.shape[0]),
    }
    ref_data[name] = (y, s)

# ---- 3. EBOPs: per-layer sums vs stored totals ----
for h, claimed_total in EBOPS_CLAIMS.items():
    e = json.load(open(f"{RUN}/results/hgq2/runs/{h}/ebops.json"))
    total = e["payload"]["total"]
    layer_sum = sum(e["payload"]["per_layer"].values())
    results["ebops"][e["config_name"]] = {
        "stored_total": total,
        "sum_per_layer": layer_sum,
        "sum_equals_total": layer_sum == total,
        "claimed_in_tradeoff_table": claimed_total,
        "matches_claim": total == claimed_total,
        "n_layers": len(e["payload"]["per_layer"]),
    }

# ---- 4. ROC curves for the overlay figure (only reconciled models) ----
for stem in ["FP32-lr05", "W8A8-lr05", "A8-lr05-s1", "A6-lr05-s1", "A4-lr05-s1",
             "W1A8", "W1A6", "W1A4"]:
    y, s = ref_data[stem]
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y[:, i], s[:, i])
        curves[f"{stem}__{c}__fpr"] = fpr.astype(np.float32)
        curves[f"{stem}__{c}__tpr"] = tpr.astype(np.float32)

np.savez_compressed(f"{OUT}/roc_curves.npz", **curves)
with open(f"{OUT}/verification_results.json", "w") as f:
    json.dump(results, f, indent=1)

# ---- console summary ----
print("column order:", results["classes_column_order"])
print(f"{'model':<12}{'recomputed':>12}{'claimed':>10}{'ok':>4}")
for stem, r in results["refs"].items():
    print(f"{stem:<12}{r['recomputed_macro']:>12.6f}{r['claimed_macro_roc_auc_md']:>10.4f}{str(r['match_claimed_4dp']):>6}")
for name, r in results["rebuilds"].items():
    print(f"{name:<12}{r['recomputed_macro']:>12.6f}{r['claimed_macro_tradeoff_table']:>10.4f}{str(r['match_claimed_4dp']):>6}"
          f"  d={r['delta_recomputed']:+.6f} (claim {r['delta_claimed']:+.4f}, ok={r['delta_match_4dp']})"
          f"  corr={r['corr_recomputed']:.6f} (claim {r['corr_claimed']:.4f}, ok={r['corr_match_4dp']})"
          f"  y_ident={r['y_identical_to_ref']}")
for nm, r in results["ebops"].items():
    print(f"{nm}: total={r['stored_total']:,} sum={r['sum_per_layer']:,} equal={r['sum_equals_total']} claim_ok={r['matches_claim']}")
