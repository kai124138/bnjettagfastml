#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<31,15> inp_t;
typedef ap_fixed<4,3,AP_RND_CONV,AP_SAT,0> subln_t;
typedef ap_fixed<12,11> bit_block_0_attn_Wo_accum_t;
typedef ap_fixed<12,11,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_Wo_t;
typedef ap_fixed<2,2> bit_block_0_attn_Wo_weight_t;
typedef ap_ufixed<2,32> bit_block_0_attn_Wo_bias_t;
typedef ap_uint<1> layer4_index;
typedef ap_fixed<23,7> result_t;
typedef ap_ufixed<3,-4> bit_block_0_attn_Wo_affine_scale_t;
typedef ap_fixed<12,-4> bit_block_0_attn_Wo_affine_bias_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
