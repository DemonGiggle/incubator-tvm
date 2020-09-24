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
#include <string.h>
#include "gigo_kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {

namespace {
void createNewDataFromPadding(float* data, float* new_data,
                                int data_H, int data_W, int data_C,
                                int new_h, int new_w,
                                int padding_T, int padding_L, int padding_B, int padding_R) {

    // clear the whole data (of course, includes the padding ones)
    memset(new_data, 0.0f, sizeof(float) * new_w * new_h * data_C);

    // assign old data to new data
    for (int i = 0; i < data_W; ++i) {
        for (int j = 0; j < data_H; ++j) {
            for (int c = 0; c < data_C; ++c) {
                int n_idx = (j+padding_T)*new_w*data_C + (i+padding_L)*data_C;
                int o_idx = j*data_W*data_C + i*data_C;

                *(new_data+n_idx) = *(data+o_idx);
            }
        }
    }
}

float* NHWCtoNCHW(float* data, int height, int width, int channels) {
    int size = height * width * channels;
    float* new_data = (float*)malloc(sizeof(float) * size);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int c = 0; c < channels; ++c) {
                int n_idx = c * height * width + j * width + i;
                int o_idx = j * width * channels + i * channels + c;
                *(new_data + n_idx) = *(data + o_idx);
            }
        }
    }

    memcpy(data, new_data, sizeof(float) * size);
    free(new_data);
    return data;
}

float* NCHWtoNHWC(float* data, int height, int width, int channels) {
    int size = height * width * channels;
    float* new_data = (float*)malloc(sizeof(float) * size);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int c = 0; c < channels; ++c) {
                int n_idx = j * width * channels + i * channels + c;
                int o_idx = c * height * width + j * width + i;
                *(new_data + n_idx) = *(data + o_idx);
            }
        }
    }

    memcpy(data, new_data, sizeof(float) * size);
    free(new_data);
    return data;
}

float* HWIOtoOIHW(float* weights, int filters, int height, int width, int channels) {
    int size = filters * height * width * channels;
    float* new_weights = (float*)malloc(sizeof(float) * size);

    for (int f = 0; f < filters; ++f) {
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                for (int c = 0; c < channels; ++c) {
                    int n_idx = f * channels * height * width + c * height * width + j * width + i;
                    int o_idx = j * width * channels * filters + i * channels * filters + c * filters + f;
                    *(new_weights + n_idx) = *(weights + o_idx);
                }
            }
        }
    }

    memcpy(weights, new_weights, sizeof(float) * size);
    free(new_weights);
    return weights;
}

float* HWOItoOIHW(float* weights, int filters, int height, int width, int channels) {
    int size = filters * height * width * channels;
    float* new_weights = (float*)malloc(sizeof(float) * size);

    for (int f = 0; f < filters; ++f) {
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                for (int c = 0; c < channels; ++c) {
                    int n_idx = f * channels * height * width + c * height * width + j * width + i;
                    int o_idx = j * width * channels * filters + i * channels * filters + f * channels + c;
                    *(new_weights + n_idx) = *(weights + o_idx);
                }
            }
        }
    }

    memcpy(weights, new_weights, sizeof(float) * size);
    free(new_weights);
    return weights;
}

void NormalConv2d(float* data, float* kernel, float* slice, int out_h, int out_w,
        int channels, int data_h, int data_w, int kern_h, int kern_w, int stride_h, int stride_w) {

    // The slice is in HW layout
    for (int dj = 0; dj + kern_h - 1 < data_h; dj += stride_h) {
        for (int di = 0; di + kern_w -1 < data_w; di += stride_w) {

            float v = 0;
            for (int c = 0; c < channels; ++c) {
                for (int kj = 0; kj < kern_h; ++kj) {
                    for (int ki =0; ki < kern_w; ++ki) {
                        int kern_idx = c * kern_h * kern_w + kj * kern_w + ki;
                        int data_idx = c * data_h * data_w + (dj+kj) * data_w + (di+ki);
                        v += *(data + data_idx) * *(kernel + kern_idx);
                    }
                }
            }
            *slice = v;
            slice++;
        }
    }
}

void Concat2D(float* out, float* source, int height, int width) {
    for (int i = 0; i < width*height; ++i) {
        *(out+i) = *(out+i) + *(source+i);
    }
}
}

void gigo_conv2d(float* data, float* weights, float* out, char* kernel_layout,
                 int data_N, int data_H,int data_W, int data_C,
                 int wght_O, int wght_H, int wght_W, int wght_I,
                 int group, int padding_T, int padding_L, int padding_B, int padding_R,
                 int stride_H, int stride_W)
{
    // The width and height of `out' buffer
    int out_w = (data_W + padding_L + padding_R - wght_W)/stride_W + 1;
    int out_h = (data_H + padding_T + padding_B - wght_H)/stride_H + 1;

    std::cout << "== out width: " << out_w << std::endl;
    std::cout << "== out hight: " << out_h << std::endl;

    if (data_N != 1) {
        std::cout << "ERROR, for impl simplicity, assume N = 1" << std::endl;
    }

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

    int new_d_w = data_W + padding_L + padding_R;
    int new_d_h = data_H + padding_T + padding_B;
    float* new_data = (float*)malloc(sizeof(float) * new_d_w * new_d_h * data_C);
    createNewDataFromPadding(data, new_data, data_H, data_W, data_C,
            new_d_h, new_d_w, padding_T, padding_L, padding_B, padding_R);

    /**
     * For easily perform convolution, converting the layout first
     * - data layout from NHWC to NCHW
     * - kernel layout to OIHW
     */
    new_data = NHWCtoNCHW(new_data, new_d_h, new_d_w, data_C);

    if (std::string(kernel_layout) == "HWIO") {
        weights = HWIOtoOIHW(weights, wght_O, wght_H, wght_W, wght_I);
    } else if (std::string(kernel_layout) == "HWOI") {
        weights = HWOItoOIHW(weights, wght_O, wght_H, wght_W, wght_I);
    } else {
        std::cout << "ERROR, un supported kernel layout" << std::endl;
    }

    /**
     * Start to perform convolution
     */
    float* slice = (float*)malloc(sizeof(float) * out_h * out_w);
    int grp_per_ch = data_C / group;
    for (int filter_idx = 0; filter_idx < wght_O; ++filter_idx) {
        // shift the start addr of weights according to the current filter index
        float* kernel = weights + filter_idx * wght_I * wght_H * wght_W;
        // shift the start addr of out according to the current filter index
        float* f_out = out + filter_idx * out_h * out_w;

        // handle for each group
        for (int g = 0; g < group; ++g) {
            // shift the start addr of data based on group
            float* g_data = new_data + g * new_d_h * new_d_w * grp_per_ch;

            // calculate conv from sub-data and sub-kernel
            NormalConv2d(g_data, kernel, slice, out_h, out_w,
                    grp_per_ch, new_d_h, new_d_w, wght_H, wght_W, stride_H, stride_W);

            // concate the result
            Concat2D(f_out, slice, out_h, out_w);
        }
    }

    /**
     * Convert data layout from NCHW back to NHWC
     */
    NCHWtoNHWC(out, out_h, out_w, wght_O);

    // free the temp allocate data
    free(slice);
    free(new_data);
}

void gigo_quant_conv2d(float* data, float* weights, float* out, char* kernel_layout,
                       int data_N, int data_H,int data_W, int data_C,
                       int wght_O, int wght_H, int wght_W, int wght_I,
                       int group, int padding_T, int padding_L, int padding_B, int padding_R,
                       int stride_H, int stride_W) {
    /**
     * Just forward to conv2d for impl's sake
     */
    gigo_conv2d(data, weights, out, kernel_layout,
            data_N, data_H, data_W, data_C,
            wght_O, wght_H, wght_W, wght_I,
            group, padding_T, padding_L, padding_B, padding_R,
            stride_H, stride_W);
}


} // namespace gigo
} // namespace runtime
} // namespace tvm
