// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class GemmOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& op = node.OpType();
  const auto& input_defs(node.InputDefs());
  // We have already embedded the weights (matrix B and C(if any)) into the coreml layer
  // No need to copy them later to reduce memory consumption
  model_builder.AddInitializerToSkip(input_defs[1]->Name());
  if (op == "Gemm" && input_defs.size() > 2) {
    model_builder.AddInitializerToSkip(input_defs[2]->Name());
  }
}

// This is an internal function, requires input tensor to be 2d float tensor
// TODO, add support of other data types
static Status GetTensorFloatDataTransposed(const ONNX_NAMESPACE::TensorProto& tensor,
                                           std::vector<float>& transposed_data) {
  Initializer unpacked_tensor(tensor);
  auto src_data = unpacked_tensor.DataAsSpan<float>();
  const auto& tensor_shape = tensor.dims();
  auto x_t = SafeInt<size_t>(tensor_shape[0]);
  auto y_t = SafeInt<size_t>(tensor_shape[1]);
  transposed_data.resize(x_t * y_t);
  for (size_t x = 0; x < x_t; x++) {
    for (size_t y = 0; y < y_t; y++) {
      transposed_data[y * x_t + x] = src_data[x * y_t + y];
    }
  }

  return Status::OK();
}

Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto& b_tensor = *model_builder.GetConstantInitializer(input_defs[1]->Name());
  const auto& b_shape = b_tensor.dims();

  NodeAttrHelper helper(node);

  const bool is_gemm = op_type == "Gemm";

  const auto transB = is_gemm ? helper.Get("transB", 0) : 0;

  // B is {K, N} in ONNX spec by default, or {N, K} in Gemm if transB is true
  const auto K = transB ? b_shape[1] : b_shape[0];
  const auto N = transB ? b_shape[0] : b_shape[1];

  if (model_builder.CreateMLProgram()) {
    if (op_type == "MatMul") {
    } else {
    }
  } else {
    auto* coreml_inner_product = layer->mutable_innerproduct();

    *layer->mutable_input()->Add() = input_defs[0]->Name();

    coreml_inner_product->set_inputchannels(K);
    coreml_inner_product->set_outputchannels(N);

    // CoreML takes weight input as {N, K} which is the reverse of ONNX.
    // However if ONNX Gemm transB is true the input weight is {N, K} so can be added directly.
    if (transB) {
      ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*coreml_inner_product->mutable_weights(), b_tensor));
    } else {
      std::vector<float> b_transposed;
      ORT_RETURN_IF_ERROR(GetTensorFloatDataTransposed(b_tensor, b_transposed));
      CreateCoreMLWeight(*coreml_inner_product->mutable_weights(), b_transposed);
    }

    if (is_gemm && input_defs.size() > 2) {
      // Add bias if present
      const auto& bias_tensor = *model_builder.GetConstantInitializer(input_defs[2]->Name());
      const auto& bias_shape = bias_tensor.dims();

      coreml_inner_product->set_hasbias(true);

      // if scalar, or single value (which is broadcastable to M, N), expand to 1D tensor of size N
      if (bias_shape.empty() || bias_shape[0] != N && bias_shape.size() == 1) {
        Initializer unpacked_tensor(bias_tensor);
        auto src_data = unpacked_tensor.DataAsSpan<float>();
        assert(src_data.size() == 1);

        std::vector<float> bias_data(N, src_data[0]);
        CreateCoreMLWeight(*coreml_inner_product->mutable_bias(), bias_data);
      } else {
        ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*coreml_inner_product->mutable_bias(), bias_tensor));
      }
    }

    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }

  return Status::OK();
}

bool GemmOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs(node.InputDefs());
  size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C

  if (!input_params.graph_viewer.GetConstantInitializer(input_defs[b_idx]->Name())) {
    LOGS(logger, VERBOSE) << "B of Gemm/Matmul must be an initializer tensor";
    return false;
  }

  if (op_type == "Gemm") {
    NodeAttrHelper helper(node);
    const auto transA = helper.Get("transA", 0);
    const auto transB = helper.Get("transB", 0);
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);
    if (!(transA == 0 && alpha == 1.f && beta == 1.f)) {
      // TODO: We can support transA, alpha and beta by using multiple layers/operations if needed.
      LOGS(logger, VERBOSE) << "Only transA == 0, alpha == 1.0 "
                            << "and beta == 1.0 is supported."
                            << " transA " << transA
                            << " alpha " << alpha
                            << " beta " << beta;
      return false;
    }

    // C of Gemm
    // For now we only support {n} or {1,n} tensor.
    // We assume the ONNX model is valid.
    if (input_defs.size() == 3) {
      if (!input_params.graph_viewer.GetConstantInitializer(input_defs[c_idx]->Name())) {
        LOGS(logger, VERBOSE) << "C of Gemm must be a constant initializer";
        return false;
      }

      std::vector<int64_t> b_shape;
      if (!GetShape(*input_defs[b_idx], b_shape, logger)) {
        return false;
      }

      std::vector<int64_t> c_shape;
      if (!GetShape(*input_defs[c_idx], c_shape, logger)) {
        return false;
      }

      // B is {K, N} in ONNX spec by default, or {N, K} in Gemm if transB is true
      const auto K = transB ? b_shape[1] : b_shape[0];
      const auto N = transB ? b_shape[0] : b_shape[1];

      size_t c_rank = c_shape.size();

      // allowed: scalar, or 1D where the value is 1 or N, 2D with shape {1, N}
      bool c_valid = false;
      switch (c_rank) {
        case 0:
          c_valid = true;
          break;
        case 1:
          if (c_shape[0] == 1 || c_shape[0] == N) {
            c_valid = true;
          }
          break;
        case 2:
          if (c_shape[0] == 1 && c_shape[1] == N) {
            c_valid == true;
          }
          break;
      }

      if (!c_valid) {
        LOGS(logger, VERBOSE) << "Shape of C Gemm input must be {}, {1}, {N}, or {1, N}. N:" << N << " C shape: "
                              << Shape2String(c_shape);

        return false;
      }
    }
  }

  return true;
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "Gemm",
          "MatMul",
      };

  op_registrations.builders.push_back(std::make_unique<GemmOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace coreml
}  // namespace onnxruntime
