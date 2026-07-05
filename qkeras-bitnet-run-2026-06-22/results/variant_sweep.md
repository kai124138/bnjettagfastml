# BitLinear Transformer — Architecture Variant Sweep

**Status:** round-1 complete (2026-06-28). Numbers below are **single-run best `val_auc`**
extracted from the per-job PVC logs `/data/outputs/qk-variants/<key>/train_<key>.log`
(authoritative source; cross-checkable in W&B project `bnjettag-bitnet`).

> **⚠ Caveat — read before citing.** Every variant **peaked at epoch 2–6 then collapsed** under
> the lr15 recipe (peak LR 1.5e-4 was tuned for the 6.37M "large" model and is too hot at these
> smaller sizes — e.g. medium 0.7538@ep3 → 0.667@ep13; clspool 0.7475@ep4 → 0.65). So these are
> *best-transient* AUCs (effectively **lower bounds**), **single-run and unseeded**. The ~0.01–0.02
> ablation gaps are within plausible run-to-run noise. **Round-2** (below) adds an LR sweep +
> seed-repeats to fix the collapse and put error bars on the knob gaps.

## Motivation
The earlier sweep covered **precision** (A8/A6/A4) × **weight type** (binary / ternary / vanilla
FP32 / W8A8) × **attention** (softmax / softmax-free) — but all at a **fixed** model size
(D=256, L=8, H=8, FFN=1024) and fixed norm/structure. This sweep explores the two axes that were
never varied: **model size** and **norm / architecture structure**. Everything else is held at
the project's best-known binary recipe so each result is attributable to the one thing changed.

## Fixed across all variants
- Weights: **binary** BitNet `{−1,+1}` (`AbsMeanQuantizer`, centralized Sign(W−α)·β).
- Activations: **A8** (8-bit absmax).
- Attention: softmax (except the `sffree` ablation).
- Recipe (= lr15, empirically best for this model): LR=1.5e-4, warmup=1 ep, poly-decay=40 ep,
  Adam β2=0.98, weight-decay=0.01, grad-clip=1.0, batch=256, EarlyStop on `val_auc` (max,
  patience 10), max 200 epochs.
- Data: NRP PVC `kai-data` (merged_trainPart/Jet + QCD bkg), pT-reweighted, same as all runs.

## Variants (10 new runs)
### SIZE axis (default arch; constant ~4× FFN ratio)
| run | D | L | H | FFN | params | best val_auc |
|---|---|---|---|---|---|---|
| `var-size-tiny`   | 32  | 2 | 4 | 128 | 26,529  | 0.7112 (peak ep6) |
| `var-size-small`  | 64  | 3 | 8 | 256 | 153,793 | 0.7335 (peak ep6) |
| `var-size-medium` | 128 | 4 | 8 | 512 | 808,065 | **0.7538** (peak ep3) |
| `large` (= lr15)  | 256 | 8 | 8 | 1024| 6,373,633 | 0.7530 (existing run) |

**Size axis is monotonic and efficient:** tiny → small → medium climbs 0.711 → 0.734 → 0.754,
and **medium (808K params) already matches/edges the 6.37M large anchor at ~8× fewer parameters**
(0.7538 vs 0.7530). Because medium also peaked at ep3 before collapsing, 0.7538 is a lower bound —
the efficiency story is likely understated. (All four share the silent fixed-PE behaviour, so the
size axis is internally clean.)

### ABLATION axis (anchor = `var-size-small`, exactly one knob changed)
Anchor = `var-size-small` (best val_auc **0.7335**). Each row flips exactly one knob; Δ is vs that anchor.

| run | knob | params | best val_auc | Δ vs small | notes |
|---|---|---|---|---|---|
| `var-abl-clspool`    | `BN_POOL=cls`                    | 153,857 | **0.7475** | **+0.0140** | CLS token + CLS pooling |
| `var-abl-gelu`       | `BN_FFN_ACT=gelu`                | 153,793 | **0.7471** | **+0.0136** | GELU FFN instead of ReLU |
| `var-abl-sffree`     | `BN_SOFTMAX_FREE=1`              | 153,793 | **0.7462** | **+0.0127** | softmax-free linear attention |
| `var-abl-noposenc`   | `BN_POS_ENC=none`                | 153,793 | 0.7380 | +0.0045 | jet as a pure set |
| `var-abl-rmsnorm`    | `BN_NORM_TYPE=rmsnorm`           | 153,793 | 0.7358 | +0.0023 | true RMSNorm, no mean-subtraction |
| `var-abl-sharednorm` | `BN_NORM_PLACEMENT=shared_prenorm`| 153,793 | 0.7291 | −0.0044 | one pre-norm/sublayer (classic pre-LN) |
| `var-abl-trueposenc` | `BN_POS_ENC=learned_real`        | 154,433 | 0.7275 | −0.0060 | genuinely trainable (N,D) table |

(`var-abl-trueposenc` param count confirmed = small + N·D = 153,793 + 10·64 = 154,433.
`var-abl-clspool` = small + D = 153,793 + 64 = 153,857. Both match the positive-control prediction.)

## Round-1 findings (single-run, lr15, treat gaps as suggestive)
1. **CLS-pool, GELU-FFN, and softmax-free each lead the small-anchor ablations** (+0.013 to +0.014),
   and they're roughly tied — three independent knobs all worth ~the same. Promising to **stack**
   (→ round-2 combo).
2. **Positional encoding HURTS.** The genuinely-learned PE (`trueposenc`, −0.0060) is the *worst*
   ablation and is below dropping PE entirely (`noposenc`, +0.0045). Consistent with jet
   constituents being a permutation-invariant **set** — order is not signal here. (And it confirms
   the upstream "learned" PE being a silent fixed-random no-op was, if anything, lucky.)
3. **Norm structure barely moves it:** true RMSNorm ≈ neutral (+0.002); classic shared pre-LN is
   slightly *worse* (−0.004) than the per-BitLinear SubLN the model defaults to.
4. **All gaps are within single-run noise** until round-2 error bars land — do not over-read the
   clspool-vs-gelu-vs-sffree ordering.

## Round-2 (complete 2026-06-28) — the recipe, not the architecture, was the bottleneck
The round-1 collapse (peak ep2–6) said the lr15 recipe doesn't transfer down in scale, so round-2
ran a **per-size LR sweep** + a combo of the round-1 "winners" + **seed-repeats** for error bars.
10 jobs, same binary+A8 backbone; all complete, 0 failures; best `val_auc` from PVC logs
`/data/outputs/qk-variants-r2/<key>/train_<key>.log`.

### (a) LR sweep on `small` — recipe WAS the bottleneck
| run | LR | best val_auc | peak epoch | epochs ran |
|---|---|---|---|---|
| `small` (round-1)      | 1.5e-4            | 0.7335 | 6  | 16 |
| `r2-small-lr10`        | 1.0e-4            | 0.7344 | 6  | 16 |
| `r2-small-lr10-warm3`  | 1.0e-4 (warmup 3) | 0.7422 | 4  | 14 |
| `r2-small-lr075`       | 7.5e-5            | 0.7480 | 5  | 15 |
| `r2-small-lr05`        | **5.0e-5**        | **0.7540** | 17 | 27 |

Monotonic: **lower LR → higher AUC and a later, stabler peak.** At 5e-5 `small` trains 27 epochs
(peak ep17) instead of collapsing at ep6 — the round-1 "peak-then-collapse" was an LR artifact.
And **`small`@5e-5 = 0.7540 ≈ medium 0.7538 ≈ large 0.7530**: the 153,793-param model matches the
6.37M model once the LR is right. (lr05 peaked late ⇒ an even lower LR may have headroom — round-3.)

### (b) Seed repeats — the round-1 ablation gaps were NOISE
Three samples each at lr15 (round-1 unseeded + `BN_SEED=1,2`):
| config | unseeded / s1 / s2 | mean | spread |
|---|---|---|---|
| `small`   | 0.7335 / 0.7365 / 0.7359 | 0.7353 | 0.0030 |
| `clspool` | 0.7475 / 0.7247 / 0.7343 | 0.7355 | **0.0228** |

`clspool` mean (0.7355) ≈ `small` mean (0.7353): **the round-1 clspool "+0.014" was a single lucky
draw**, not a real effect — clspool just has far higher run-to-run variance. Paired same-seed diffs
(identical data split) clspool−small = {−0.0118, −0.0016, +0.0140}, mean ≈ **+0.0002**.
⇒ Treat any single-run gap below ~0.02 at lr15 as noise.

### (c) Combos — stacking the "winners" gives nothing
| run | knobs | best val_auc | vs plain |
|---|---|---|---|
| `r2-combo-small`  | cls+gelu+noPE on small  | 0.7323 | ≈ small 0.7335 (within noise) |
| `r2-combo-medium` | cls+gelu+noPE on medium | 0.7504 | ≤ medium 0.7538 (no gain) |

Consistent with (b): the knobs weren't real wins, so they don't stack.

### Round-2 verdict
**At this scale the BitLinear transformer's AUC is governed by the OPTIMIZER (peak LR), not the
architectural knobs.** A *plain* `small` (default LayerNorm/SubLN + GAP + ReLU + softmax) at
**LR = 5e-5** reaches **0.7540**, matching the 20×-larger model; the round-1 architecture "wins"
(CLS / GELU / softmax-free) evaporate under seed-averaging and fail to stack. **Recommended small
config: default-arch `small` @ LR 5e-5.** Round-3 confirms 5e-5 with seeds, pushes LR lower, and
re-runs `medium` below lr15.

## Round-3 (complete 2026-06-28) — LR ceiling, seed-confirm, and the size gap re-opens
All 6 `kai-bn3-*` complete, 0 failures; best `val_auc` from PVC logs `/data/outputs/qk-variants-r3/`.

### (a) small LR ceiling — plateaus at ~0.750 for LR ∈ [3.5e-5, 5e-5]
| LR | best val_auc | peak ep | epochs |
|---|---|---|---|
| 2.5e-5               | 0.7500 | 33 | 43 |
| 3.5e-5               | 0.7537 | 25 | 35 |
| 5.0e-5 (3-seed mean) | **0.7499 ± 0.0036** | 14–18 | 24–28 |
| └ 5e-5 samples       | 0.7540 / 0.7483 / 0.7474 | | |

`small` tops out around **0.750** between 3.5e-5 and 5e-5; going to 2.5e-5 only trains longer for no
gain (peak slides to ep33). The round-2 single 0.7540 was the high end of the 5e-5 spread — the
3-seed mean is **0.7499 ± 0.0036**. Still +0.015 over lr15 (0.7353): the LR fix is real and large.

### (b) medium WAS also LR-limited — and re-opens the size gap
| run | LR | best val_auc | vs medium@lr15 (0.7538) |
|---|---|---|---|
| `r3-medium-lr075` | 7.5e-5 | 0.7576 | +0.0038 |
| `r3-medium-lr05`  | 5.0e-5 | **0.7592** | **+0.0054** |

`medium` (808K) at a proper LR hits **0.7592** — *at round-3* the best result in the sweep **(superseded by
round-4: a tuned `large`@5e-5 = 0.7672 is the actual sweep-best — see Round-4 below)** — beating the
**6.37M** `large` **lr15** anchor (0.7530) by ~0.006 at **~8× fewer params**, and clearly above `small`
(~0.750). So once the LR is right the **bigger model wins again**: round-2's "small catches up" was
really "small was the most LR-starved."

### Efficiency frontier (best achieved per size)
| size | params | best val_auc | LR | note |
|---|---|---|---|---|
| tiny   | 26,529    | 0.7112 | 1.5e-4 | not LR-swept (likely also limited) |
| small  | 153,793   | ~0.750 (0.7499±0.0036 @5e-5; 0.7537 @3.5e-5) | 3.5–5e-5 | |
| medium | 808,065   | 0.7567±0.005 (3-seed; 0.7592 best) | 5e-5 | efficiency pick, ~8× fewer params than large |
| large  | 6,373,633 | **0.7672** | 5e-5 | **best AUC** (round-4); +0.0142 over its lr15 0.7530 |

**Updated by round-4:** `large` IS also LR-limited (0.7530 → 0.7672 @5e-5), so once every size gets a
proper LR, AUC is **monotonic in size**: small ~0.750 < medium 0.7567 < large 0.7672. The earlier
"medium beats large" was tuned-medium vs UN-tuned-large; a tuned large wins raw AUC, while medium is
the **efficiency** sweet spot (within ~0.010 of large at ~8× fewer params). NB the lr15 "large" is
**6.37M params** (round-4 preflight + the HLS doc), not the "~3M" earlier notes claimed.

### Recommendation (updated after round-4)
- **Best AUC:** `large` (D256/L8/H8/FFN1024, 6.37M, binary `{−1,+1}` + A8) @ **LR 5e-5** → **0.7672**
  (single run; seed-confirm pending) — +0.0142 over the originally-reported W1A8 0.7530, for free.
- **Best efficiency (deployment pick):** `medium` (808K) @ 5e-5 → **0.7567±0.005** — within ~0.010 of
  large at **~8× fewer params** and ~8× less FPGA logic.
- **Smallest viable:** `small` (D64/L3/H8/FFN256, 154K) @ LR ~4–5e-5 → ~0.750.
- **Architecture knobs:** keep the DEFAULTS (LayerNorm/SubLN · GAP · ReLU · softmax · no real PE) —
  none of the round-1 ablations beat default under seed-averaging.
- **Recipe is the lever, not the knobs:** lr15 → 5e-5 helps at **every** size (small +0.015,
  medium +0.005, large +0.014). The optimizer LR, not the architecture, was the hidden ceiling.

## Round-4 (complete 2026-06-28) — seed-confirm medium + large IS also LR-limited
All 4 `kai-bn4-*` complete, 0 failures; best `val_auc` from PVC logs `/data/outputs/qk-variants-r4/`
(re-extracted via kai-util3).

### (a) medium @ 5e-5 — seed-confirm: the 0.7592 was the high end of the spread
| run | seed | best val_auc | peak ep | epochs |
|---|---|---|---|---|
| `r3-medium-lr05` (round-3) | 0 (default) | 0.7592 | 12 | 22 |
| `r4-medium-lr05-s1`        | 1 | 0.7503 | 14 | 24 |
| `r4-medium-lr05-s2`        | 2 | 0.7606 | 14 | 24 |
| **3-seed mean ± spread**   |   | **0.7567 ± 0.005** | | |

Honest medium central estimate is **0.7567**, not 0.7592 (that was the high seed). Still clearly above
small's 0.7499 (+0.0068), but the medium seed spread (~0.010) is wider than small's (~0.004).

### (b) large (D256/L8/H8/FFN1024, **6.37M**) — also LR-limited, and now the best AUC
| run | LR | best val_auc | peak ep | epochs | vs large@lr15 (0.7530) |
|---|---|---|---|---|---|
| `r4-large-lr075` | 7.5e-5 | 0.7602 | 6  | 16 | +0.0072 (peaked early, still hot) |
| `r4-large-lr05`  | 5.0e-5 | **0.7672** | 37 | 47 | **+0.0142** (trained long, stable) |

**Verdict:** `large` IS also LR-limited — even the 6.37M model the lr15 recipe was nominally tuned for
does **better** at 5e-5 (0.7672 vs 0.7530). So the peak LR was the hidden lever at every size, and once
fixed, **bigger keeps winning**: small ~0.750 < medium 0.7567 < **large 0.7672**. "Medium beats large"
held only against the UN-tuned lr15 large; a tuned large takes the raw-AUC crown, while medium remains
the efficiency pick (~0.010 behind at ~8× fewer params). Net: the binary W1A8 tagger goes
**0.7530 → 0.7672 validation AUC for free**, just by lowering the peak LR. _Caveat: large@5e-5 is a
single run — seed-confirm it next; all numbers are validation AUC (during-training split), not ROC-test._

## Hardware cost — EBOPs on the now-FIXED model
**Decision (2026-06-29):** the architecture is **fixed at `large`** (D256/L8/H8/FFN1024, **6,373,633 params**);
the size/structure sweep above is retained as the *why-this-size* record, **not** an open architecture search.
The reporting axis going forward is **EBOPs (Effective Bit-Operations, HGQ arXiv:2405.00645)** — a static,
synthesis-free bit-op cost computed by `code/training/ebops.py`. On the fixed large model, going W8A8 →
**W1A8 binary cuts EBOPs 7.65×** (4.06 G → **530 M**; 122.5× below FP32), and the activation dial A8→A6→A4
trims it 530 M → 393 M → 259 M. The 7.65× (not a clean 8×) is the act×act attention floor (0.65% of MACs,
unaffected by binarization). At equal (W8A8) EBOPs budget a binary model could carry ~7.65× more matmul-MACs
for free — the cost-side converse of growing the model. **Full per-quant EBOPs table, ΔEBOPs, and the
EBOPs≈#LUT bridge: `results/RESULTS.md` §2f** (verified by results-analyst 2026-06-29).

## Methodology notes / caveats
- **Upstream "learned" PE is a silent no-op (verified 2026-06-27).** The default
  `BN_POS_ENC=learned` builds `Embedding(N,D)(tf.range(N))`; applied to a constant (not a
  KerasTensor), Keras folds it to a **fixed random** offset that trains **0 params** (small and
  noposenc have identical param counts; clspool = small+64 is the positive control). So:
  - The SIZE axis is clean — tiny/small/medium/large all share this same fixed-PE behaviour.
  - `noposenc` is really "fixed-random offset vs no offset".
  - `trueposenc` (`learned_real`, via `add_weight`) is the genuinely-learned PE — the honest test
    of whether positional information helps at all for jet constituents.
- All variants validated by a CPU `--sanity` build (construct + forward + `train_on_batch`) on
  NRP before GPU launch — 0 exceptions.

## Reproduce
```
cd code/jobs/training/variants
python3 gen_variant_jobs.py     # (re)generate the 10 yamls + preflight.sh
./launch_all.sh                 # apply all 10 GPU Jobs
./monitor.sh                    # snapshot progress
python3 fetch_results.py        # W&B comparison table (needs wandb + WANDB_API_KEY)
```
