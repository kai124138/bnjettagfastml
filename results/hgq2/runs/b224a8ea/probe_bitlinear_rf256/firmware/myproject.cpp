#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    inp_t inp[256],
    result_t layer4_out[256]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=inp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=inp,layer4_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<bit_block_0_attn_Wo_weight_t, 65536>(w4, "w4.txt");
        nnet::load_weights_from_txt<bit_block_0_attn_Wo_bias_t, 256>(b4, "b4.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    subln_t layer2_out[256];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    nnet::subln<inp_t, subln_t, config2>(inp, layer2_out); // subln

    nnet::dense<subln_t, result_t, config4>(layer2_out, layer4_out, w4, b4); // bit_block_0_attn_Wo

}

