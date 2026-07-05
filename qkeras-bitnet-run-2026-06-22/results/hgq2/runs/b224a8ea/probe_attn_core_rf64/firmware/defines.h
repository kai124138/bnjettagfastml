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
typedef ap_fixed<16,10,AP_RND_CONV,AP_SAT,0> q_in_t;
typedef ap_fixed<16,10,AP_RND_CONV,AP_SAT,0> k_in_t;
typedef ap_fixed<37,25> attn_scores_accum_t;
typedef ap_fixed<37,25> attn_scores_t;
typedef ap_ufixed<12,1,AP_RND_CONV,AP_SAT,0> attn_softmax_exp_table_t;
typedef ap_ufixed<12,1,AP_RND_CONV,AP_SAT,0> attn_softmax_inv_table_t;
typedef ap_ufixed<12,4,AP_RND_CONV,AP_SAT,0> attn_softmax_inv_inp_t;
typedef ap_ufixed<10,14,AP_RND_CONV,AP_SAT,0> attn_softmax_inp_norm_t;
typedef ap_ufixed<27,8> attn_softmax_accum_t;
typedef ap_ufixed<23,1,AP_RND_CONV,AP_SAT,0> attn_softmax_t;
typedef ap_fixed<18,8> attn_softmax_table_t;
typedef ap_fixed<16,10,AP_RND_CONV,AP_SAT,0> v_in_t;
typedef ap_fixed<43,15> attn_ctx_accum_t;
typedef ap_fixed<43,15> result_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
