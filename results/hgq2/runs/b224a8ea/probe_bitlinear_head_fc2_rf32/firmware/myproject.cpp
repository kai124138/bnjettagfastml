#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    inp_t inp[256],
    result_t layer6_out[5]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=inp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=inp,layer6_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<head_fc2_weight_t, 1280>(w4, "w4.txt");
        nnet::load_weights_from_txt<head_fc2_bias_t, 5>(b4, "b4.txt");
        nnet::load_weights_from_txt<head_fc2_affine_scale_t, 5>(s6, "s6.txt");
        nnet::load_weights_from_txt<head_fc2_affine_bias_t, 5>(b6, "b6.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    subln_t layer2_out[256];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    head_fc2_t layer4_out[5];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    nnet::subln<inp_t, subln_t, config2>(inp, layer2_out); // subln

    nnet::dense<subln_t, head_fc2_t, config4>(layer2_out, layer4_out, w4, b4); // head_fc2

    nnet::normalize<head_fc2_t, result_t, config6>(layer4_out, layer6_out, s6, b6); // head_fc2_affine

}

