# BNJetTag — Paper Library

A curated literature library for the **BNJetTag** project (1-bit BitNet binary
`{−1,+1}`-weight transformer jet tagger for the CMS Level-1 trigger, via hls4ml).

Every paper has a **rich-brief markdown** (abstract summary, key methods, main results,
and an explicit *Relevance to BNJetTag* section) so the team and AI agents can recall it fast.
**26 papers**, grouped into four areas. All arXiv IDs were verified via web search; no
benchmark numbers were invented (unconfirmed figures are marked "see paper").

## How to use this library
- Open any `.md` below for the brief. Each one links to the arXiv abstract + PDF.
- **To download the actual PDFs**, run `./download_papers.sh` locally (this folder). The
  research sandbox can't reach `arxiv.org`, so PDFs are fetched on your machine; they land
  in the matching topic subfolder next to each brief.
- Reference *code* (not papers) lives one level up in `../reference-code/`.

---

## 1 · BitNet & 1-bit / ternary quantization
*The core of the thesis: binary/ternary weights → multiply-free matmul → ~0 DSP on FPGA.*

| Paper | arXiv | Brief |
| --- | --- | --- |
| BitNet: Scaling 1-bit Transformers for Large Language Models | [2310.11453](https://arxiv.org/abs/2310.11453) | [brief](bitnet-1bit-ternary/2310.11453_bitnet_scaling_1bit_transformers.md) |
| The Era of 1-bit LLMs: All LLMs are in 1.58 Bits (BitNet b1.58, ternary) | [2402.17764](https://arxiv.org/abs/2402.17764) | [brief](bitnet-1bit-ternary/2402.17764_era_of_1bit_llms_1p58.md) |
| BitNet a4.8: 4-bit Activations for 1-bit LLMs | [2411.04965](https://arxiv.org/abs/2411.04965) | [brief](bitnet-1bit-ternary/2411.04965_bitnet_a4p8_4bit_activations.md) |
| BitNet b1.58 2B4T Technical Report | [2504.12285](https://arxiv.org/abs/2504.12285) | [brief](bitnet-1bit-ternary/2504.12285_bitnet_b1p58_2b4t_technical_report.md) |
| Ternary Weight Networks | [1605.04711](https://arxiv.org/abs/1605.04711) | [brief](bitnet-1bit-ternary/1605.04711_ternary_weight_networks.md) |
| Trained Ternary Quantization | [1612.01064](https://arxiv.org/abs/1612.01064) | [brief](bitnet-1bit-ternary/1612.01064_trained_ternary_quantization.md) |

## 2 · Jet tagging & transformers
*The model family we binarize, plus the benchmarks and baselines we measure AUC against.*

| Paper | arXiv | Brief |
| --- | --- | --- |
| Particle Transformer for Jet Tagging (ParT, JetClass) | [2202.03772](https://arxiv.org/abs/2202.03772) | [brief](jet-tagging-transformers/2202.03772_particle_transformer.md) |
| ParticleNet: Jet Tagging via Particle Clouds | [1902.08570](https://arxiv.org/abs/1902.08570) | [brief](jet-tagging-transformers/1902.08570_particlenet.md) |
| Energy Flow Networks: Deep Sets for Particle Jets | [1810.05165](https://arxiv.org/abs/1810.05165) | [brief](jet-tagging-transformers/1810.05165_energy_flow_networks.md) |
| JEDI-net: jet identification with interaction networks | [1908.05318](https://arxiv.org/abs/1908.05318) | [brief](jet-tagging-transformers/1908.05318_jedi_net.md) |
| An Efficient Lorentz Equivariant GNN for Jet Tagging (LorentzNet) | [2201.08187](https://arxiv.org/abs/2201.08187) | [brief](jet-tagging-transformers/2201.08187_lorentznet.md) |
| The Machine Learning Landscape of Top Taggers (benchmark) | [1902.09914](https://arxiv.org/abs/1902.09914) | [brief](jet-tagging-transformers/1902.09914_ml_landscape_top_taggers.md) |

## 3 · hls4ml & FPGA triggers
*The deployment flow, FPGA resource/latency budgets, and QKeras QAT.*

| Paper | arXiv | Brief |
| --- | --- | --- |
| Fast inference of DNNs in FPGAs for particle physics (hls4ml) | [1804.06913](https://arxiv.org/abs/1804.06913) | [brief](hls4ml-fpga-triggers/1804.06913_fast_inference_dnn_fpga_particle_physics.md) |
| Automatic heterogeneous quantization for low-latency inference (QKeras/AutoQKeras) | [2006.10159](https://arxiv.org/abs/2006.10159) | [brief](hls4ml-fpga-triggers/2006.10159_autoqkeras_heterogeneous_quantization.md) |
| Fast convolutional neural networks on FPGAs with hls4ml | [2101.05108](https://arxiv.org/abs/2101.05108) | [brief](hls4ml-fpga-triggers/2101.05108_fast_cnn_fpga_hls4ml.md) |
| Distance-Weighted GNNs on FPGAs (GarNet) | [2008.03601](https://arxiv.org/abs/2008.03601) | [brief](hls4ml-fpga-triggers/2008.03601_garnet_gnn_fpga_particle_reco.md) |
| Ps and Qs: Quantization-aware pruning for low-latency inference | [2102.11289](https://arxiv.org/abs/2102.11289) | [brief](hls4ml-fpga-triggers/2102.11289_ps_and_qs_quantization_aware_pruning.md) |
| Ultrafast jet classification on FPGAs for the HL-LHC | [2402.01876](https://arxiv.org/abs/2402.01876) | [brief](hls4ml-fpga-triggers/2402.01876_ultrafast_jet_classification_hl_lhc.md) |

## 4 · QAT & binary-NN foundations
*The lineage BitNet descends from: binarization, the straight-through estimator, activation quantization.*

| Paper | arXiv | Brief |
| --- | --- | --- |
| BinaryConnect: training with binary weights | [1511.00363](https://arxiv.org/abs/1511.00363) | [brief](qat-binary-nn-foundations/1511.00363_binaryconnect.md) |
| Binarized Neural Networks (weights + activations = ±1) | [1602.02830](https://arxiv.org/abs/1602.02830) | [brief](qat-binary-nn-foundations/1602.02830_binarized_neural_networks.md) |
| XNOR-Net: binary convolutional networks | [1603.05279](https://arxiv.org/abs/1603.05279) | [brief](qat-binary-nn-foundations/1603.05279_xnor_net.md) |
| Estimating/Propagating Gradients (the straight-through estimator) | [1308.3432](https://arxiv.org/abs/1308.3432) | [brief](qat-binary-nn-foundations/1308.3432_straight_through_estimator.md) |
| PACT: parameterized clipping activation | [1805.06085](https://arxiv.org/abs/1805.06085) | [brief](qat-binary-nn-foundations/1805.06085_pact.md) |
| LSQ: Learned Step Size Quantization | [1902.08153](https://arxiv.org/abs/1902.08153) | [brief](qat-binary-nn-foundations/1902.08153_lsq.md) |
| Binary Neural Networks: A Survey | [2004.03333](https://arxiv.org/abs/2004.03333) | [brief](qat-binary-nn-foundations/2004.03333_binary_neural_networks_survey.md) |
| A Survey of Quantization Methods for Efficient NN Inference | [2103.13630](https://arxiv.org/abs/2103.13630) | [brief](qat-binary-nn-foundations/2103.13630_quantization_survey.md) |

---

*Generated 2026-06-28. Briefs verified against arXiv via web search. To add a paper, drop a
new `.md` in the matching subfolder (same template) and add a row above.*
