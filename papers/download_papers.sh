#!/usr/bin/env bash
# Download the 26 BNJetTag library PDFs from arXiv, next to their briefs.
#
# WHY THIS IS A SCRIPT: the research sandbox that built this library cannot reach
# arxiv.org, so the PDFs are fetched here, on your machine. Run it once:
#     cd papers && bash download_papers.sh
# Re-running skips files already present. Each PDF lands beside its .md brief.

set -u
cd "$(dirname "$0")"

# entries: "<topic-folder>/<basename>  <arxiv-id>"
read -r -d '' LIST <<'EOF'
bitnet-1bit-ternary/1605.04711_ternary_weight_networks 1605.04711
bitnet-1bit-ternary/1612.01064_trained_ternary_quantization 1612.01064
bitnet-1bit-ternary/2310.11453_bitnet_scaling_1bit_transformers 2310.11453
bitnet-1bit-ternary/2402.17764_era_of_1bit_llms_1p58 2402.17764
bitnet-1bit-ternary/2411.04965_bitnet_a4p8_4bit_activations 2411.04965
bitnet-1bit-ternary/2504.12285_bitnet_b1p58_2b4t_technical_report 2504.12285
jet-tagging-transformers/1810.05165_energy_flow_networks 1810.05165
jet-tagging-transformers/1902.08570_particlenet 1902.08570
jet-tagging-transformers/1902.09914_ml_landscape_top_taggers 1902.09914
jet-tagging-transformers/1908.05318_jedi_net 1908.05318
jet-tagging-transformers/2201.08187_lorentznet 2201.08187
jet-tagging-transformers/2202.03772_particle_transformer 2202.03772
hls4ml-fpga-triggers/1804.06913_fast_inference_dnn_fpga_particle_physics 1804.06913
hls4ml-fpga-triggers/2006.10159_autoqkeras_heterogeneous_quantization 2006.10159
hls4ml-fpga-triggers/2008.03601_garnet_gnn_fpga_particle_reco 2008.03601
hls4ml-fpga-triggers/2101.05108_fast_cnn_fpga_hls4ml 2101.05108
hls4ml-fpga-triggers/2102.11289_ps_and_qs_quantization_aware_pruning 2102.11289
hls4ml-fpga-triggers/2402.01876_ultrafast_jet_classification_hl_lhc 2402.01876
qat-binary-nn-foundations/1308.3432_straight_through_estimator 1308.3432
qat-binary-nn-foundations/1511.00363_binaryconnect 1511.00363
qat-binary-nn-foundations/1602.02830_binarized_neural_networks 1602.02830
qat-binary-nn-foundations/1603.05279_xnor_net 1603.05279
qat-binary-nn-foundations/1805.06085_pact 1805.06085
qat-binary-nn-foundations/1902.08153_lsq 1902.08153
qat-binary-nn-foundations/2004.03333_binary_neural_networks_survey 2004.03333
qat-binary-nn-foundations/2103.13630_quantization_survey 2103.13630
EOF

ok=0; skip=0; fail=0
while read -r base id; do
  [ -z "${base:-}" ] && continue
  out="${base}.pdf"
  if [ -s "$out" ]; then echo "skip  $out"; skip=$((skip+1)); continue; fi
  echo "get   $out  (arXiv:$id)"
  if curl -fsSL --retry 3 --retry-delay 2 -o "$out" "https://arxiv.org/pdf/${id}.pdf"; then
    ok=$((ok+1)); sleep 1   # be polite to arxiv
  else
    echo "  !! failed arXiv:$id"; rm -f "$out"; fail=$((fail+1))
  fi
done <<< "$LIST"

echo "----"
echo "downloaded:$ok  skipped:$skip  failed:$fail"
