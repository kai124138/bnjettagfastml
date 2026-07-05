#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_subln.h"

// hls-fpga-machine-learning insert weights


// hls-fpga-machine-learning insert layer-config
// subln_256
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
    static const unsigned reuse_factor = 1;
};



#endif
