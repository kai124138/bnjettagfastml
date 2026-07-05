#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_einsum.h"

// hls-fpga-machine-learning insert weights


// hls-fpga-machine-learning insert layer-config
// attn_scores
struct config5_tpose_inp0 {
    static const unsigned dims = 3;
    static const unsigned N = 2560;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config5_tpose_inp0_from_shape[3] = {10, 8, 32};
unsigned config5_tpose_inp0_to_shape[3] = {8, 10, 32};
unsigned config5_tpose_inp0_perm[3] = {1, 0, 2};
unsigned config5_tpose_inp0_perm_strides[3] = {32, 256, 1};

const unsigned* const config5_tpose_inp0::from_shape = config5_tpose_inp0_from_shape;
const unsigned* const config5_tpose_inp0::to_shape = config5_tpose_inp0_to_shape;
const unsigned* const config5_tpose_inp0::perm = config5_tpose_inp0_perm;
const unsigned* const config5_tpose_inp0::perm_strides = config5_tpose_inp0_perm_strides;


struct config5_tpose_inp1 {
    static const unsigned dims = 3;
    static const unsigned N = 2560;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config5_tpose_inp1_from_shape[3] = {10, 8, 32};
unsigned config5_tpose_inp1_to_shape[3] = {8, 10, 32};
unsigned config5_tpose_inp1_perm[3] = {1, 0, 2};
unsigned config5_tpose_inp1_perm_strides[3] = {32, 256, 1};

const unsigned* const config5_tpose_inp1::from_shape = config5_tpose_inp1_from_shape;
const unsigned* const config5_tpose_inp1::to_shape = config5_tpose_inp1_to_shape;
const unsigned* const config5_tpose_inp1::perm = config5_tpose_inp1_perm;
const unsigned* const config5_tpose_inp1::perm_strides = config5_tpose_inp1_perm_strides;


struct config5_tpose_out {
    static const unsigned dims = 3;
    static const unsigned N = 800;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config5_tpose_out_from_shape[3] = {8, 10, 10};
unsigned config5_tpose_out_to_shape[3] = {8, 10, 10};
unsigned config5_tpose_out_perm[3] = {0, 1, 2};
unsigned config5_tpose_out_perm_strides[3] = {100, 10, 1};

const unsigned* const config5_tpose_out::from_shape = config5_tpose_out_from_shape;
const unsigned* const config5_tpose_out::to_shape = config5_tpose_out_to_shape;
const unsigned* const config5_tpose_out::perm = config5_tpose_out_perm;
const unsigned* const config5_tpose_out::perm_strides = config5_tpose_out_perm_strides;



struct config5 {
    typedef config5_tpose_inp0 tpose_inp0_config;
    typedef config5_tpose_inp1 tpose_inp1_config;
    typedef config5_tpose_out tpose_out_conf;

    typedef attn_scores_accum_t accum_t;

    // Layer Sizes
    static const unsigned n_free0 = 10;
    static const unsigned n_free1 = 10;
    static const unsigned n_contract = 32;
    static const unsigned n_inplace = 8;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 64;
    static const unsigned multiplier_limit = 400;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// attn_softmax
struct softmax_config6 : nnet::activ_config {
    static const unsigned n_in = 800;
    static const unsigned n_slice = 10;
    static const unsigned n_outer = 80;
    static const unsigned n_inner = 1;
    static const unsigned parallelization_factor = 80;
    static const unsigned exp_table_size = 1024;
    static const unsigned inv_table_size = 4096;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    static constexpr float exp_scale = 0.0005728639791670458;
    typedef attn_softmax_exp_table_t exp_table_t;
    typedef attn_softmax_inv_table_t inv_table_t;
    typedef attn_softmax_accum_t accum_t;
    typedef attn_softmax_inv_inp_t inv_inp_t;
    typedef attn_softmax_inp_norm_t inp_norm_t;
};

// attn_ctx
struct config10_tpose_inp0 {
    static const unsigned dims = 3;
    static const unsigned N = 800;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config10_tpose_inp0_from_shape[3] = {8, 10, 10};
unsigned config10_tpose_inp0_to_shape[3] = {8, 10, 10};
unsigned config10_tpose_inp0_perm[3] = {0, 1, 2};
unsigned config10_tpose_inp0_perm_strides[3] = {100, 10, 1};

const unsigned* const config10_tpose_inp0::from_shape = config10_tpose_inp0_from_shape;
const unsigned* const config10_tpose_inp0::to_shape = config10_tpose_inp0_to_shape;
const unsigned* const config10_tpose_inp0::perm = config10_tpose_inp0_perm;
const unsigned* const config10_tpose_inp0::perm_strides = config10_tpose_inp0_perm_strides;


struct config10_tpose_inp1 {
    static const unsigned dims = 3;
    static const unsigned N = 2560;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config10_tpose_inp1_from_shape[3] = {10, 8, 32};
unsigned config10_tpose_inp1_to_shape[3] = {8, 32, 10};
unsigned config10_tpose_inp1_perm[3] = {1, 2, 0};
unsigned config10_tpose_inp1_perm_strides[3] = {32, 1, 256};

const unsigned* const config10_tpose_inp1::from_shape = config10_tpose_inp1_from_shape;
const unsigned* const config10_tpose_inp1::to_shape = config10_tpose_inp1_to_shape;
const unsigned* const config10_tpose_inp1::perm = config10_tpose_inp1_perm;
const unsigned* const config10_tpose_inp1::perm_strides = config10_tpose_inp1_perm_strides;


struct config10_tpose_out {
    static const unsigned dims = 3;
    static const unsigned N = 2560;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config10_tpose_out_from_shape[3] = {8, 10, 32};
unsigned config10_tpose_out_to_shape[3] = {10, 8, 32};
unsigned config10_tpose_out_perm[3] = {1, 0, 2};
unsigned config10_tpose_out_perm_strides[3] = {32, 320, 1};

const unsigned* const config10_tpose_out::from_shape = config10_tpose_out_from_shape;
const unsigned* const config10_tpose_out::to_shape = config10_tpose_out_to_shape;
const unsigned* const config10_tpose_out::perm = config10_tpose_out_perm;
const unsigned* const config10_tpose_out::perm_strides = config10_tpose_out_perm_strides;



struct config10 {
    typedef config10_tpose_inp0 tpose_inp0_config;
    typedef config10_tpose_inp1 tpose_inp1_config;
    typedef config10_tpose_out tpose_out_conf;

    typedef attn_ctx_accum_t accum_t;

    // Layer Sizes
    static const unsigned n_free0 = 10;
    static const unsigned n_free1 = 32;
    static const unsigned n_contract = 10;
    static const unsigned n_inplace = 8;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 64;
    static const unsigned multiplier_limit = 400;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



#endif
