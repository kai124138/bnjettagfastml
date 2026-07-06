#!/usr/bin/env python3
"""Verify the per-function DSP split from the RAW Vitis csynth.xml module tables
(fetched into the store 2026-07-05). Independent of parse_csynth_modules.py and of
LEDGER.md. Per-module Area lives at ModuleInformation/Module/AreaEstimates/Resources.

Note on hierarchy: Vitis lists both the wrapper and inner instance of a function
(e.g. subln_1d_* wraps subln_*) with the same resources — module rows must not be
naively summed; the top-level total is authoritative and each function tree is
attributed by its outermost module row.
"""
import json
import xml.etree.ElementTree as ET

R = "/Users/kaiyamaguchi/Downloads/bnjettag-training-results/qkeras-bitnet-run-2026-06-22/results/hgq2/runs/b224a8ea"
OUT = "/Users/kaiyamaguchi/Downloads/bnjettag-training-results/poster/data/dsp_split_check.json"

PROBES = ["probe_bitlinear_head_fc2_rf32", "probe_bitlinear_rf256",
          "probe_subln_rf1", "probe_attn_core_rf1"]

out = {}
for p in PROBES:
    root = ET.parse(f"{R}/{p}/csynth.xml").getroot()
    top = {k.tag: int(k.text) for k in root.find("AreaEstimates/Resources")
           if k.text and k.text.lstrip("~").isdigit()}
    mods = []
    for m in root.findall(".//ModuleInformation/Module"):
        res = m.find("AreaEstimates/Resources")
        row = {"name": m.findtext("Name")}
        for k in res:
            if k.tag in ("BRAM_18K", "DSP", "FF", "LUT"):
                row[k.tag] = int(k.text)
        mods.append(row)
    out[p] = {"top": top, "modules": mods}
    print(f"===== {p}  (top DSP={top['DSP']}, LUT={top['LUT']})")
    for r in mods:
        print(f"  {r['name'][:72]:<74} DSP={r['DSP']:>6,} LUT={r['LUT']:>10,}")

json.dump(out, open(OUT, "w"), indent=1)

# ---- assertions for the poster claims ----
hf = {r["name"].split("_ap_")[0]: r for r in out["probe_bitlinear_head_fc2_rf32"]["modules"]}
rf = {r["name"].split("_ap_")[0]: r for r in out["probe_bitlinear_rf256"]["modules"]}
ac = {r["name"].split("_ap_")[0]: r for r in out["probe_attn_core_rf1"]["modules"]}
checks = {
    "head_fc2: SubLN tree DSP=112": hf["subln_1d"]["DSP"] == 112,
    "head_fc2: binary dense DSP=0": hf["dense_latency"]["DSP"] == 0,
    "head_fc2: CSD-2 affine DSP=0": hf["normalize"]["DSP"] == 0,
    "head_fc2: top total 112": out["probe_bitlinear_head_fc2_rf32"]["top"]["DSP"] == 112,
    "rf256: SubLN folded DSP=14": rf["subln_1d"]["DSP"] == 14,
    "rf256: Resource dense DSP=256": rf["dense_resource_rf_leq_nin"]["DSP"] == 256,
    "rf256: 14+256=270=top": 14 + 256 == out["probe_bitlinear_rf256"]["top"]["DSP"],
    "subln_rf1: SubLN-only 1792": out["probe_subln_rf1"]["top"]["DSP"] == 1792,
    "attn_core: top 52,000": out["probe_attn_core_rf1"]["top"]["DSP"] == 52000,
}
for k, v in checks.items():
    print(("PASS  " if v else "FAIL  ") + k)
print("\nattn_core module DSP rows (for the act-x-act cost statement):")
for r in out["probe_attn_core_rf1"]["modules"]:
    print(f"  DSP={r['DSP']:>6,}  {r['name'][:80]}")
