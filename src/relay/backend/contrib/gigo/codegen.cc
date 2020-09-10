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
 * \file src/relay/backend/contrib/gigo/codegen.cc
 * \brief Implementation of GIGO codegen APIs.
 */
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container.h>
#include <tvm/relay/function.h>

#include <string>

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief Get the external symbol of the Relay function name.
 *
 * \param func The provided function.
 *
 * \return An external symbol.
 */
std::string getExtSymbol(const Function& func) {
    const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    CHECK(name_node.defined()) << "Fail to retrieve external symbol";
    return std::string(name_node.value());
}

/*!
 * \brief The external codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module
 */
runtime::Module GIGOCompiler(const ObjectRef& ref) {
    CHECK(ref->IsInstance<FunctionNode>());
    auto func = Downcast<Function>(ref);
    auto func_name = getExtSymbol(func);

    const auto* pf = runtime::Registry::Get("runtime.module.GigoRuntimeCreate");
    CHECK(pf != nullptr) << "Cannot find GigoRuntimeModule to create";

    return (*pf)(func_name);
}

TVM_REGISTER_GLOBAL("relay.ext.gigo").set_body_typed(GIGOCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
