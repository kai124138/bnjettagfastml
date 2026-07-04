#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    q_in_t q_in[10*8*32], k_in_t k_in[10*8*32], v_in_t v_in[10*8*32],
    result_t layer10_out[10*8*32]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=q_in complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=k_in complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=v_in complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=q_in,k_in,v_in,layer10_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    attn_scores_t layer5_out[8*10*10];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    attn_softmax_t layer6_out[8*10*10];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    nnet::einsum<q_in_t, k_in_t, attn_scores_t, config5>(q_in, k_in, layer5_out); // attn_scores

    nnet::softmax_multidim<attn_scores_t, attn_softmax_t, softmax_config6>(layer5_out, layer6_out); // attn_softmax

    nnet::einsum<attn_softmax_t, v_in_t, result_t, config10>(layer6_out, v_in, layer10_out); // attn_ctx

}

