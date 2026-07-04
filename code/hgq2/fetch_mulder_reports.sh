#!/usr/bin/env bash
# Pull any finished csynth reports home from mulder:~/bnjet_hgq2/, then refresh
# the tradeoff table, ROC figure, and dashboard from the store.
# Safe to run repeatedly; only copies reports that exist. (bash-3.2 compatible.)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
STORE="$HERE/../../results/hgq2"
PY=/Users/kaiyamaguchi/Downloads/bnjettag-training-results/.venv-hgq2/bin/python

PAIRS="
probe_subln_rf1:b224a8ea
probe_bitlinear_rf256:b224a8ea
probe_bitlinear_v2_rf256:b224a8ea
probe_bitlinear_v3lat_rf256:b224a8ea
probe_bitlinear_head_fc2_rf32:b224a8ea
probe_attn_core_rf1:b224a8ea
probe_bitlinear_a6lat_rf256:a428e6e2
probe_bitlinear_a4lat_rf256:53b202bc
"

for pair in $PAIRS; do
  probe="${pair%%:*}"
  h="${pair##*:}"
  # regenerate the json ONLY from a real xml; delete any stale one otherwise
  # (a report that rode along inside a tarball must never be mistaken for a result)
  ssh -o BatchMode=yes mulder "cd ~/bnjet_hgq2/$probe 2>/dev/null && \
    x=myproject_prj/solution1/syn/report/csynth.xml && \
    if [ -f \$x ]; then python3 ~/bnjet_hgq2/parse_csynth.py \$x > csynth_report.json; \
    else rm -f csynth_report.json; fi" 2>/dev/null
  mkdir -p "$STORE/runs/$h/$probe"
  if scp -q "mulder:~/bnjet_hgq2/$probe/csynth_report.json" \
        "$STORE/runs/$h/$probe/csynth_report.json" 2>/dev/null; then
    echo "[fetched] $probe -> runs/$h/$probe/"
  else
    echo "[pending] $probe (csynth not finished)"
  fi
done

cd "$HERE"
KERAS_BACKEND=tensorflow "$PY" aggregate.py
KERAS_BACKEND=tensorflow "$PY" generate_dashboard.py
echo "Refreshed: $STORE/tradeoff_table.md, roc_hgq2_overlay.png, dashboard.html"
echo "(Re-publish the dashboard artifact from Claude Code if you want the web copy updated.)"
