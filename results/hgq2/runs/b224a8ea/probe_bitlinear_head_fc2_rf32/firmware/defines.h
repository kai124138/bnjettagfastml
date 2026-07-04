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
typedef ap_fixed<8,4,AP_RND_CONV,AP_SAT,0> subln_t;
typedef ap_fixed<16,12> head_fc2_accum_t;
typedef ap_fixed<15,11,AP_RND_CONV,AP_SAT,0> head_fc2_t;
typedef ap_fixed<2,2> head_fc2_weight_t;
typedef ap_ufixed<2,32> head_fc2_bias_t;
typedef ap_uint<1> layer4_index;
typedef ap_fixed<24,8> result_t;
typedef ap_ufixed<4,-3> head_fc2_affine_scale_t;
typedef ap_fixed<9,-7> head_fc2_affine_bias_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
