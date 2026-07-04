#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_subln.h"

// hls-fpga-machine-learning insert weights
#include "weights/w4.h"
#include "weights/b4.h"


// hls-fpga-machine-learning insert layer-config
// subln
struct config2 : nnet::subln_config {
    static const unsigned n_in = 256;
    static const unsigned seq_len = 1;
    static const unsigned table_size = 4096;
    static const unsigned table_size_log2 = 12;
    static constexpr double epsilon = 1.000000000000e-06;
    typedef ap_fixed<48,24> accum_t;
    typedef ap_fixed<40,16> diff_t;
    typedef ap_ufixed<65,39> var_t;
    typedef ap_ufixed<18,1> table_t;
    typedef ap_fixed<42,18> prod_t;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 256;
};

// bit_block_0_attn_Wo
struct config4 : nnet::dense_config {
    static const unsigned n_in = 256;
    static const unsigned n_out = 256;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 65536;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef bit_block_0_attn_Wo_accum_t accum_t;
    typedef bit_block_0_attn_Wo_bias_t bias_t;
    typedef bit_block_0_attn_Wo_weight_t weight_t;
    typedef layer4_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



#endif
