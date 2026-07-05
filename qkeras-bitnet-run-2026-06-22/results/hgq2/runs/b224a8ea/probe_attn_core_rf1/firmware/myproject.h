#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void myproject(
    q_in_t q_in[10*8*32], k_in_t k_in[10*8*32], v_in_t v_in[10*8*32],
    result_t layer10_out[10*8*32]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
