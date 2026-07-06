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
typedef ap_fixed<8,3,AP_RND_CONV,AP_SAT,0> subln_t;
typedef ap_fixed<19,7> bit_block_0_attn_Wo_accum_t;
typedef ap_fixed<19,7> result_t;
typedef ap_fixed<4,-3> bit_block_0_attn_Wo_weight_t;
typedef ap_fixed<7,1> bit_block_0_attn_Wo_bias_t;
typedef ap_uint<1> layer4_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
