#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_subln.h"

// hls-fpga-machine-learning insert weights
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/s6.h"
#include "weights/b6.h"


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
    static const unsigned reuse_factor = 32;
};

// head_fc2
struct config4 : nnet::dense_config {
    static const unsigned n_in = 256;
    static const unsigned n_out = 5;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 1280;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef head_fc2_accum_t accum_t;
    typedef head_fc2_bias_t bias_t;
    typedef head_fc2_weight_t weight_t;
    typedef layer4_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// head_fc2_affine
struct config6 : nnet::batchnorm_config {
    static const unsigned n_in = 5;
    static const unsigned n_filt = 5;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 32;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef head_fc2_affine_bias_t bias_t;
    typedef head_fc2_affine_scale_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



#endif
