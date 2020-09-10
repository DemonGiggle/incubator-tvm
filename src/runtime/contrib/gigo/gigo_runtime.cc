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

#include "gigo_runtime.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <string>

namespace tvm {
namespace runtime {
namespace gigo {

GigoRuntimeModule::GigoRuntimeModule(const std::string& symbol_name)
    : symbol_name_(symbol_name) {
}

PackedFunc GigoRuntimeModule::GetFunction(const std::string& name,
                                     const ObjectPtr<Object>& sptr_to_self) {
    std::cout << "GigoRuntimeModule::GetFunction w/ name = " << name <<std::endl;
    return PackedFunc(nullptr);
}

void GigoRuntimeModule::SaveToBinary(dmlc::Stream* stream) {
    std::cout << "GigoRuntimeModule::SaveToBinary" <<std::endl;

    // Save the symbol
    stream->Write(symbol_name_);
}

Module GigoRuntimeModule::LoadFromBinary(void* strm) {
    std::string symbol;

    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    CHECK(stream->Read(&symbol)) << "Loading symbol name failed";

    auto n = make_object<GigoRuntimeModule>(symbol);
    return Module(n);
}

runtime::Module GigoRuntimeCreate(String symbol_name) {
    std::cout << "Greeting! You just call GigoRuntimeCreate w/ name = " << symbol_name << std::endl;
    auto n = make_object<GigoRuntimeModule>(symbol_name);
    return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.GigoRuntimeCreate").set_body_typed(GigoRuntimeCreate);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_gigo")
    .set_body([](TVMArgs args, TVMRetValue* rv) { *rv = GigoRuntimeModule::LoadFromBinary(args[0]); });
}  // namespace gigo
}  // namespace runtime
}  // namespace tvm
