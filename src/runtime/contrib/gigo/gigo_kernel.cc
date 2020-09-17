/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file gigo_runtime.cc
 * \brief Execution handling of Ethos-N command streams.
 */

#include <iostream>
#include <string>
#include "gigo_kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {

void gigo_conv2d(float* data, float* weights, float* out, char* kernel_layout,
                 int data_N, int data_H,int data_W, int data_C,
                 int wght_O, int wght_H, int wght_W, int wght_I,
                 int group, int padding_T, int padding_L, int padding_B, int padding_R,
                 int stride_H, int stride_W)
{
    std::cout << "Send conv2d to the external device ---- list of params ----" << std::endl;
    std::cout << ">>>> data_N = " << data_N << std::endl;
    std::cout << ">>>> data_C = " << data_C << std::endl;
    std::cout << ">>>> data_H = " << data_H << std::endl;
    std::cout << ">>>> data_W = " << data_W << std::endl;
    std::cout << ">>>> wght_O = " << wght_O << std::endl;
    std::cout << ">>>> wght_H = " << wght_H << std::endl;
    std::cout << ">>>> wght_W = " << wght_W << std::endl;
    std::cout << ">>>> wght_I = " << wght_I << std::endl;
    std::cout << ">>>> group = " << group << std::endl;
    std::cout << ">>>> padding_T = " << padding_T << std::endl;
    std::cout << ">>>> padding_L = " << padding_L << std::endl;
    std::cout << ">>>> padding_B = " << padding_B << std::endl;
    std::cout << ">>>> padding_R = " << padding_R << std::endl;
    std::cout << ">>>> stride_H = " << stride_H << std::endl;
    std::cout << ">>>> stride_W = " << stride_W << std::endl;
    std::cout << ">>>> kernel_layout = " << kernel_layout << std::endl;

    if (std::string(kernel_layout) == "HWIO") {

    } else if (std::string(kernel_layout) == "HWOI") {

    } else {
        std::cout << "ERROR, un supported kernel layout" << std::endl;
    }

    int out_w = (data_W + padding_L + padding_R - wght_W)/2 + 1;
    int out_h = (data_H + padding_T + padding_B - wght_H)/2 + 1;
    for (int filter_index = 0; filter_index < wght_O; filter_index++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                // Assume out batch number is always 1
                // The out layout is the same as data layout, i.e. NHWC
                int out_idx = i * (out_w * wght_O) + j * wght_O + filter_index;

                float v = 0;
                for (int c = 0; c < data_C; c++) {
                    int data_idx = 0; // FIXME: to decide
                    int kernel_idx = 0; // FIXME: to decide
                    v += *(data+data_idx) * *(weights+kernel_idx);
                }
                *(out+out_idx) = v;
            }
        }
    }

    // try to access the (i, j, k, l) data, and we know the layout
    int i = std::min(1, wght_H-1);
    int j = std::min(1, wght_W-1);
    int k = std::min(1, wght_I-1);
    int l = std::min(1, wght_O-1);
    if (std::string(kernel_layout) == "HWIO") {
        auto* v = weights + i * (wght_W * wght_I * wght_O) +
                            j * (wght_I * wght_O) +
                            k * wght_O +
                            l;
        std::cout << "Weight value at (" <<
            i << "," << j << "," << k << "," << l << "): " <<
            *v << std::endl;
    }
    if (std::string(kernel_layout) == "HWOI") {
        auto* v = weights + i * (wght_W * wght_O * wght_I) +
                            j * (wght_O * wght_I) +
                            l * wght_I +
                            k;
        std::cout << "Weight value at (" <<
            i << "," << j << "," << l << "," << k << "): " <<
            *v << std::endl;
    }
}

} // namespace gigo
} // namespace runtime
} // namespace tvm
