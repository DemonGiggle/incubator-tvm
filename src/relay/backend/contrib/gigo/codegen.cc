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
#include <tvm/relay/attrs/nn.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container.h>
#include <tvm/relay/function.h>

#include <string>

#include "../codegen_c/codegen_c.h"
#include "../../utils.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace tvm::relay::backend;

namespace gigo_ops {

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

std::vector<std::string> Conv2d(const CallNode* call) {
  std::vector<std::string> args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  CHECK(conv2d_attr);
  CHECK(conv2d_attr->data_layout == "NHWC")
      << "data layout only supports NHWC, but we have " << conv2d_attr->data_layout;

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  args.push_back(conv2d_attr->kernel_layout);

  // Args: Data dimensions: N, H, W, C
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: Weight HWIO (height, weight, in channels, channels)
  int w_h, w_w, w_o, w_i;

  w_h = wshape[0];
  w_w = wshape[1];

  if (conv2d_attr->kernel_layout == "HWOI") {
      w_o = wshape[2];
      w_i = wshape[3];
  } else if (conv2d_attr->kernel_layout == "HWIO") {
      w_o = wshape[3];
      w_i = wshape[2];
  } else {
      LOG(ERROR) << "Unsupported kernel layout: " << conv2d_attr->kernel_layout;
  }

  // paddings
  int pd_l, pd_r, pd_b, pd_t;

  pd_l = pd_r = pd_b = pd_t = 0;
  if (auto v = conv2d_attr->padding[0].as<IntImmNode>()->value != 0) {
      pd_l = pd_r = pd_b = pd_t = v;
  }
  if (auto v = conv2d_attr->padding[1].as<IntImmNode>()->value != 0) {
      pd_l = pd_r = v;
  }
  if (auto v = conv2d_attr->padding[3].as<IntImmNode>()->value != 0) {
      pd_b = conv2d_attr->padding[2].as<IntImmNode>()->value;
      pd_r = v;
  }

  args.push_back(std::to_string(w_o));
  args.push_back(std::to_string(w_h));
  args.push_back(std::to_string(w_w));
  args.push_back(std::to_string(w_i));
  args.push_back(std::to_string(conv2d_attr->groups));
  args.push_back(std::to_string(pd_t));
  args.push_back(std::to_string(pd_l));
  args.push_back(std::to_string(pd_b));
  args.push_back(std::to_string(pd_r));
  args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

  return args;
}

std::vector<std::string> QuantConv2d(const CallNode* call) {
    /**
     * Just use nn version for quantized version, for prototype of concept
     */
    return Conv2d(call);
}

} // namespace gigo_ops

/*!
 * \brief Get the external symbol of the Relay function name.
 *
 * \param func The provided function.
 *
 * \return An external symbol.
 */
std::string GetExtSymbol(const Function& func) {
    const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    CHECK(name_node.defined()) << "Fail to retrieve external symbol";
    return std::string(name_node.value());
}

class CodegenBuilder : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
    public:
        explicit CodegenBuilder(const std::string& id) : ext_func_id_(id) {}

        std::vector<Output> VisitExprDefault_(const Object* op) final {
            LOG(FATAL) << "GIGO codegen doesn't support: " << op->GetTypeKey();
            return {};
        }

        std::vector<Output> VisitExpr_(const CallNode* call) {
            GenerateBodyOutput ret;
            if (const auto* func = call->op.as<FunctionNode>()) {
                // FIXME: to impl
                LOG(FATAL) << "FunctionNode: not yet implemented";
            } else {
                ret = GenerateOpCall(call);
            }

            buf_decl_.insert(buf_decl_.end(), ret.buffers.begin(), ret.buffers.end());
            ext_func_body_.push_back(ret.decl);
            return ret.outputs;
        }

        std::vector<Output> VisitExpr_(const VarNode* node) final {
            ext_func_args_.push_back(GetRef<Var>(node));
            Output output;
            output.name = node->name_hint();
            return {output};
        }

        std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
            Output output;
            // Get const: static_cast<float*>(dnnl_0_consts[0]->data)
            output.name = CreateDataReference(ext_func_id_, const_idx_);
            output.dtype = "float";

            // Generate the global variable for needed ndarrays
            if (const_array_name_.empty()) {
                const_array_name_ = CreateNDArrayPool(ext_func_id_);
                std::string checker = CreateInitChecker(ext_func_id_);
                ext_func_body_.insert(ext_func_body_.begin(), checker);
            }

            // Give the ndarray a unique name to ease the initialization of it at
            // runtime.
            std::string const_var_name = CreateConstVar(ext_func_id_, const_idx_);
            const_vars_.push_back(const_var_name);
            const_idx_++;

            return {output};
        }

        std::string JIT(const std::vector<Output>& out) {
            return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
        }

	private:
        struct GenerateBodyOutput {
            std::string decl;
            std::vector<std::string> buffers;
            std::vector<Output> outputs;
        };

        GenerateBodyOutput GenerateOpCall(const CallNode* call) {
            const auto* op_node = call->op.as<OpNode>();
            CHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();

            using ArgFunType = std::function<std::vector<std::string>(const CallNode*)>;
            static const std::map<std::string, std::pair<std::string, ArgFunType>> op_map = {
                {"nn.conv2d", {"gigo_conv2d", gigo_ops::Conv2d}},
                {"qnn.conv2d", {"gigo_quant_conv2d", gigo_ops::QuantConv2d}},
            // TODO: add more supported ops
            //    {"nn.dense", {"gigo_dense", gigo_ops::Dense}},
            //    {"nn.relu", {"gigo_relu", gigo_ops::Relu}},
            //    {"nn.batch_norm", {"gigo_bn", gigo_ops::BatchNorm}},
            //    {"add", {"gigo_add", gigo_ops::Add}},
            };

            const auto op_name = GetRef<Op>(op_node)->name;
            const auto iter = op_map.find(op_name);
            if (iter != op_map.end()) {
                return GenerateBody(call, iter->second.first, iter->second.second(call));
            }

            LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
            return {};
        }

        std::vector<std::string> GetArgumentNames(const CallNode* call) {
            std::vector<std::string> arg_names;
            for (size_t i = 0; i < call->args.size(); ++i) {
                auto res = VisitExpr(call->args[i]);
                for (const auto& out : res) {
                    arg_names.push_back(out.name);
                }
            }
            return arg_names;
        }

        GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                const std::vector<std::string>& attribute_args) {
            return GenerateBody(root_call, func_name, GetArgumentNames(root_call), attribute_args);
        }

        GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                const std::vector<std::string>& func_args,
                const std::vector<std::string>& attribute_args) {
            // Make function call with input buffers when visiting arguments
            CHECK_GT(func_args.size(), 0);
            std::ostringstream decl_stream;
            decl_stream << "(" << func_args[0];
            for (size_t i = 1; i < func_args.size(); ++i) {
                decl_stream << ", " << func_args[i];
            }

            // Analyze the output buffers
            std::vector<Type> out_types;
            if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
                auto type_node = root_call->checked_type().as<TupleTypeNode>();
                for (auto field : type_node->fields) {
                    CHECK(field->IsInstance<TensorTypeNode>());
                    out_types.push_back(field);
                }
            } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
                CHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
                out_types.push_back(root_call->checked_type());
            } else {
                LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
            }

            GenerateBodyOutput ret;
            for (const auto& out_type : out_types) {
                this->PrintIndents();
                const std::string out = "buf_" + std::to_string(buf_idx_++);
                const auto out_size = gigo_ops::GetShape1DSize(out_type);
                decl_stream << ", " << out;

                Output output;
                output.name = out;
                output.size = out_size;
                output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
                output.need_copy = true;
                ret.buffers.push_back("float* " + out + " = (float*)std::malloc(4 * " +
                        std::to_string(out_size) + ");");
                ret.outputs.push_back(output);
            }

            // Attach attribute arguments
            // special handling for the first argument
            decl_stream << ", \"" << attribute_args[0] << "\"";
            for (size_t i = 1; i < attribute_args.size(); ++i) {
                decl_stream << ", " << attribute_args[i];
            }
            decl_stream << ");";
            ret.decl = func_name + decl_stream.str();
            return ret;
        }

		/*! \brief The id of the external GIGO ext_func. */
		std::string ext_func_id_;
		/*!
		 * \brief The index to track the output buffer. Each kernel will redirect the
		 * output to a buffer that may be consumed by other kernels.
		 */
		int buf_idx_{0};
		/*! \brief The index of global constants. */
		int const_idx_{0};
		/*! \brief The arguments used by a wrapped function that calls DNNL kernels. */
		Array<Var> ext_func_args_;
		/*! \brief Statement of the function that will be compiled using DNNL kernels. */
		std::vector<std::string> ext_func_body_;
		/*! \brief The array declared to store the constant values. */
		std::string const_array_name_;
		/*! \brief The declaration of intermeidate buffers. */
		std::vector<std::string> buf_decl_;
		/*! \brief The variable name to constant mapping. */
		Array<String> const_vars_;

        friend class GigoModuleCodegen;
};

/*!
 * \brief The Gigo codegen helper to generate wrapper function to Gigo
 * libraries. The code is a CSourceModule that can be compiled separately and
 * linked together with a DSOModule.
 */
class GigoModuleCodegen : public CSourceModuleCodegenBase {
    public:
        std::pair<std::string, Array<String>> GenFunc(const Function& func) {
            CHECK(func.defined()) << "Input error: expect a Relay function.";

            auto symbol_name = GetExtSymbol(func);

            CodegenBuilder builder(symbol_name);
            auto out = builder.VisitExpr(func->body);
            code_stream_ << builder.JIT(out);

            return {symbol_name, builder.const_vars_};
        }

        runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
            code_stream_ << "#include <cstdint>\n";
            code_stream_ << "#include <cstdlib>\n";
            code_stream_ << "#include <cstring>\n";
            code_stream_ << "#include <vector>\n";
            code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
            code_stream_ << "#include <tvm/runtime/container.h>\n";
            code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
            code_stream_ << "#include <dlpack/dlpack.h>\n";
            // gigo_kernel file is saved under src/runtime/contrib/gigo
            // To make export_library use it, users need to pass
            // -I${PATH_TO_TVM}/src/runtime/contrib
            code_stream_ << "#include <gigo/gigo_kernel.h>\n";
            code_stream_ << "using namespace tvm::runtime;\n";
            code_stream_ << "using namespace tvm::runtime::contrib;\n";
            code_stream_ << "\n";

            CHECK(ref->IsInstance<FunctionNode>());

            auto res = GenFunc(Downcast<Function>(ref));

            std::string code        = code_stream_.str();
            String symbol_name      = std::get<0>(res);
            Array<String> variables = std::get<1>(res);

            // Create a CSource module
            const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
            CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";

            return (*pf)(code, "c", symbol_name, variables);
        }

    private:
        /*!
         * \brief The code stream that prints the code that will be compiled using
         * external codegen tools.
         */
        std::ostringstream code_stream_;
};

/*!
 * \brief The external codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module
 */
runtime::Module GIGOCompiler(const ObjectRef& ref) {
    GigoModuleCodegen gen;
    return gen.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.gigo").set_body_typed(GIGOCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
