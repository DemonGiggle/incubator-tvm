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
#include "gigo_kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {

void gigo_conv2d(float* data, float* weights, float* out, int p_N_, int p_C_,
                 int p_H_, int p_W_, int p_O_, int p_G_, int p_Ph_, int p_Pw_,
                 int p_Kh_, int p_Kw_, int p_Sh_, int p_Sw_)
{
    std::cout << "Send conv2d to the external device" << std::endl;
}

} // namespace gigo
} // namespace runtime
} // namespace tvm
