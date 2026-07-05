#!/usr/bin/env python3
"""Re-derive the analytic HGQ-v1 EBOPs table (results/ebops.md) from architecture
dimensions alone. Convention (ebops.md §0): EBOPs = sum over every scalar multiply
of b_i * b_j; NO accumulator term. Weight x act MACs cost b_w*b_a; attention
act x act MACs (QK^T, A.V) cost b_a*b_a.

Era-2 large model: D=256, L=8 blocks, FFN=1024, T=10 constituents, 16 feats, 5 classes.
"""
import json

D, L, FFN, T, F, C = 256, 8, 1024, 10, 16, 5

# weight x act MACs
input_proj = T * F * D                      # 10*16*256
per_block = 4 * (T * D * D) + T * D * FFN + T * FFN * D   # Wq,Wk,Wv,Wo + fc1 + fc2
head_fc1 = D * D                            # after GAP: 1 token
head_fc2 = D * C
wa_macs = input_proj + L * per_block + head_fc1 + head_fc2

# attention act x act MACs: QK^T (T*T*D) + A.V (T*T*D) per block
aa_macs = L * 2 * (T * T * D)

claims = {  # results/ebops.md:32-36
    "FP32": (32, 32, 64_954_302_464),
    "W8A8": (8, 8, 4_059_643_904),
    "W1A8": (1, 8, 530_393_088),
    "W1A6": (1, 6, 392_879_616),
    "W1A4": (1, 4, 258_642_944),
}
print(f"wa_macs = {wa_macs:,} (ebops.md claims 63,022,336) match={wa_macs == 63_022_336}")
print(f"aa_macs = {aa_macs:,} (ebops.md claims 409,600)    match={aa_macs == 409_600}")
out = {"wa_macs": wa_macs, "aa_macs": aa_macs, "rows": {}}
for name, (bw, ba, claim) in claims.items():
    ebops = wa_macs * bw * ba + aa_macs * ba * ba
    ok = ebops == claim
    out["rows"][name] = {"rederived": ebops, "claimed_ebops_md": claim, "match": ok}
    print(f"{name}: rederived {ebops:,} vs claimed {claim:,} -> {'MATCH' if ok else 'MISMATCH'}")
r = out["rows"]
ratio = r["W8A8"]["rederived"] / r["W1A8"]["rederived"]
print(f"W8A8/W1A8 = {ratio:.3f} (claimed 7.65x)")
out["ratio_w8a8_over_w1a8"] = ratio
# RESULTS.md §2f alternative table (era-1 param count) — flag only
out["results_md_2f_w1a8"] = {"value": 530_343_936, "matches_ebops_md": 530_343_936 == r["W1A8"]["rederived"]}
json.dump(out, open("/Users/kaiyamaguchi/Downloads/bnjettag-training-results/poster/data/ebops_analytic_check.json", "w"), indent=1)
