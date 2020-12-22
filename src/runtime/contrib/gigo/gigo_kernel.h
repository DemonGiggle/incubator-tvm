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
 * \file src/runtime/contrib/gigo/gigo_kernel.h
 * \brief Use external gigo library kernels.
 */

#ifndef TVM_RUNTIME_CONTRIB_GIGO_GIGO_KERNEL_H_
#define TVM_RUNTIME_CONTRIB_GIGO_GIGO_KERNEL_H_

#include <tvm/runtime/c_runtime_api.h>

namespace tvm {
namespace runtime {
namespace contrib {

extern "C" TVM_DLL void gigo_conv2d(float* data, float* weights, float* out, char* kernel_layout,
                                    int data_N, int data_H,int data_W, int data_C,
                                    int wght_O, int wght_H, int wght_W, int wght_I,
                                    int group, int padding_T, int padding_L, int padding_B, int padding_R,
                                    int stride_H, int stride_W);

extern "C" TVM_DLL void gigo_quant_conv2d(float* data, float* weights, float* out, char* kernel_layout,
                                    int data_N, int data_H,int data_W, int data_C,
                                    int wght_O, int wght_H, int wght_W, int wght_I,
                                    int group, int padding_T, int padding_L, int padding_B, int padding_R,
                                    int stride_H, int stride_W);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_GIGO_GIGO_KERNEL_H_
