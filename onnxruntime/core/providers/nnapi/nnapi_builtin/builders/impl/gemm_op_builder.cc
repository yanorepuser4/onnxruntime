// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

namespace {
// Get the bias size (C) of Gemm op
// ANEURALNETWORKS_FULLY_CONNECTED only supports 1d bias
// Will test if C of Gemm can be squeezed and return the 1d vector size after squeeze
bool GetGemmBiasSize(const Shape& c_shape, int32_t android_feature_level, uint32_t& size,
                     const logging::Logger& logger) {
  // TODO add support of scalar C for Gemm
  size_t c_dim = c_shape.size();
  if (c_dim == 0) {
    LOGS(logger, VERBOSE) << "C of Gemm cannot be a scalar";
    return false;
  }

  if (c_dim != 1 && android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_2) {
    LOGS(logger, VERBOSE) << "C of Gemm can only be 1d tensor for API level " << android_feature_level
                          << " shape of C, " << Shape2String(c_shape);
    return false;
  }

  if (c_dim != 1) {
    // If C is a (2+)d tensor, it must have the format {1, 1, ..., 1, n}
    // where every except the last dimension should be 1
    for (size_t i = 0; i < c_dim - 1; ++i) {
      if (c_shape[i] != 1) {
        LOGS(logger, VERBOSE) << "C of Gemm must be a vector or a tensor with only last dimension != 1"
                              << " c_shape: " << Shape2String(c_shape);
        return false;
      }
    }
  }

  size = c_shape[c_dim - 1];
  return true;
}

bool IsSupportedBatchMatMul(const NodeUnit& node_unit, int32_t nnapi_feature_level, const logging::Logger& logger) {
  // Currently, batch MatMul is composed of various operations including ANEURALNETWORKS_SPLIT which requires
  // ANEURALNETWORKS_FEATURE_LEVEL_3.
  const auto min_nnapi_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_3;
  if (nnapi_feature_level < min_nnapi_feature_level) {
    LOGS(logger, VERBOSE) << "Minimum NNAPI feature level required: " << min_nnapi_feature_level
                          << ", actual: " << nnapi_feature_level;
    return false;
  }

  // Only support non-QDQ MatMul for now.
  // TODO could be expanded to support QLinearMatMul and QDQ MatMul
  if (node_unit.UnitType() != NodeUnit::Type::SingleNode ||
      node_unit.OpType() != "MatMul") {
    LOGS(logger, VERBOSE) << "Unsupported op type: "
                          << (node_unit.UnitType() == NodeUnit::Type::QDQGroup ? "QDQ " : "") << node_unit.OpType();
    return false;
  }

  const auto& inputs = node_unit.Inputs();

  // Verify shapes
  // A and B should have at least three dimensions* and have the same leading dimensions except for the last two.
  // [*] Having two dimensions is valid for a MatMul but for simplicity we don't support it in the current batch
  // MatMul implementation. That case is handled by the regular Gemm/MatMul op building logic.
  Shape a_shape;
  if (!GetShape(inputs[0].node_arg, a_shape, logger)) {
    return false;
  }

  Shape b_shape;
  if (!GetShape(inputs[1].node_arg, b_shape, logger)) {
    return false;
  }

  if (a_shape.size() < 3 ||
      a_shape.size() != b_shape.size() ||
      !std::equal(a_shape.begin(), a_shape.end() - 2,
                  b_shape.begin(), b_shape.end() - 2)) {
    LOGS(logger, VERBOSE)
        << "A and B must have at least three dimensions and have the same leading dimensions except for the last two. "
        << "A shape: " << Shape2String(a_shape) << ", B shape: " << Shape2String(b_shape);
    return false;
  }

  // Verify type
  int32_t a_type;
  if (!GetType(inputs[0].node_arg, a_type, logger)) {
    return false;
  }

  // Only support float for now.
  // TODO could be expanded to support other types
  if (a_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS(logger, VERBOSE) << "Unsupported element data type: " << a_type;
    return false;
  }

  return true;
}

Status BuildBatchMatMul(ModelBuilder& model_builder, const NodeUnit& node_unit) {
  // we will implement batch MatMul by composing NNAPI operations
  // this could be replaced with ANEURALNETWORKS_BATCH_MATMUL when that is more widely supported

  // assuming A and B have at least three dimensions and the same leading dimensions other than the last two
  const auto& logger = model_builder.GetLogger();
  const auto& inputs = node_unit.Inputs();

  // IsSupportedBatchMatMul verified we have shape info
  Shape a_shape;
  Shape b_shape;
  ORT_IGNORE_RETURN_VALUE(GetShape(inputs[0].node_arg, a_shape, logger));
  ORT_IGNORE_RETURN_VALUE(GetShape(inputs[1].node_arg, b_shape, logger));

  const std::string& a = inputs[0].node_arg.Name();
  const std::string& b = inputs[1].node_arg.Name();

  const std::string& output = node_unit.Outputs()[0].node_arg.Name();

  std::vector<std::string> gemm_a_inputs{a};
  std::vector<std::string> gemm_b_inputs{b};

  const auto m = a_shape[a_shape.size() - 2],
             k = a_shape[a_shape.size() - 1],
             n = b_shape[b_shape.size() - 1];

  const bool reshape_leading_dimensions = a_shape.size() > 3;
  const auto batch_size = ShapeSize(a_shape, 0, a_shape.size() - 2);

  auto add_reshape = [&model_builder](const std::string& input, const Shape& new_shape,
                                      const std::string& output) -> Status {
    const std::string new_shape_name = model_builder.GetUniqueName(input + "/new_shape");
    std::vector<int32_t> new_shape_i32{};
    new_shape_i32.reserve(new_shape.size());
    std::transform(new_shape.begin(), new_shape.end(), std::back_inserter(new_shape_i32),
                   [](uint32_t d) { return narrow<int32_t>(d); });
    ORT_RETURN_IF_ERROR(AddNnapiReshape(model_builder, input, new_shape_name, new_shape_i32, output));
    return Status::OK();
  };

  auto add_reshape_generate_output = [&model_builder, &add_reshape](const std::string& input, const Shape& new_shape,
                                                                    std::string& output) -> Status {
    std::string reshaped = model_builder.GetUniqueName(input + "/reshaped");
    ORT_RETURN_IF_ERROR(add_reshape(input, new_shape, reshaped));
    output = std::move(reshaped);
    return Status::OK();
  };

  // collapse leading dimensions to a single one
  if (reshape_leading_dimensions) {
    const Shape a_new_shape_value = {batch_size, m, k},
                b_new_shape_value = {batch_size, k, n};
    std::string a_reshaped, b_reshaped;

    ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_a_inputs.front(), a_new_shape_value, a_reshaped));
    gemm_a_inputs.front() = a_reshaped;

    ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_b_inputs.front(), b_new_shape_value, b_reshaped));
    gemm_b_inputs.front() = b_reshaped;
  }

  // transpose B
  {
    const std::string b_new_perm = model_builder.GetUniqueName(b + "/new_perm"),
                      b_transposed = model_builder.GetUniqueName(b + "/transposed");
    ORT_RETURN_IF_ERROR(AddNnapiTranspose(model_builder, gemm_b_inputs.front(), b_new_perm,
                                          AsSpan<int32_t>({0, 2, 1}), b_transposed));
    gemm_b_inputs.front() = b_transposed;
  }

  // split batch
  {
    auto add_split = [&model_builder, batch_size](const std::string& input,
                                                  std::vector<std::string>& outputs_result) -> Status {
      std::vector<std::string> outputs;
      outputs.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        outputs.push_back(model_builder.GetUniqueName(MakeString(input, "/split_", i)));
      }
      ORT_RETURN_IF_ERROR(AddNnapiSplit(model_builder, input, 0, outputs));
      outputs_result = std::move(outputs);
      return Status::OK();
    };

    std::vector<std::string> a_split_outputs;
    ORT_RETURN_IF_ERROR(add_split(gemm_a_inputs.front(), a_split_outputs));
    gemm_a_inputs = std::move(a_split_outputs);

    std::vector<std::string> b_split_outputs;
    ORT_RETURN_IF_ERROR(add_split(gemm_b_inputs.front(), b_split_outputs));
    gemm_b_inputs = std::move(b_split_outputs);
  }

  // GEMM per matrix pair
  std::vector<std::string> gemm_outputs;
  gemm_outputs.reserve(batch_size);
  {
    const std::string bias = model_builder.GetUniqueName(node_unit.Name() + "/zero_bias");
    {
      if (model_builder.GetOperandTypes().at(b).type != Type::TENSOR_FLOAT32) {
        ORT_NOT_IMPLEMENTED("Only float input is supported now.");
      }
      const Shape bias_shape{n};
      const std::vector<float> buffer(n, 0.0f);
      const OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_shape);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    }

    auto add_fc = [&model_builder, &bias](const std::string& a, const std::string& b_transposed,
                                          const std::string& output) -> Status {
      const auto& operand_indices = model_builder.GetOperandIndices();
      const auto& operand_types = model_builder.GetOperandTypes();
      auto& shaper = model_builder.GetShaper();
      InlinedVector<uint32_t> input_indices;
      input_indices.push_back(operand_indices.at(a));             // A
      input_indices.push_back(operand_indices.at(b_transposed));  // B'
      input_indices.push_back(operand_indices.at(bias));          // C
      int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
      ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, fuse_code));

      const auto a_dimen = shaper[a];
      const auto b_transposed_dimen = shaper[b_transposed];  // num_units, input_size
      Shape output_dimen{a_dimen[0], b_transposed_dimen[0]};
      shaper.AddShape(output, output_dimen);
      const OperandType output_operand_type(operand_types.at(a).type, output_dimen);
      ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indices,
                                                     {output}, {output_operand_type}));
      return Status::OK();
    };

    for (uint32_t i = 0; i < batch_size; ++i) {
      const auto &gemm_a_input = gemm_a_inputs[i],
                 &gemm_b_input = gemm_b_inputs[i];

      // make inputs 2D ([1, x, y] -> [x, y])
      std::string a_2d, b_transposed_2d;
      ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_a_input, Shape{m, k}, a_2d));
      ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_b_input, Shape{n, k}, b_transposed_2d));

      const std::string gemm_output = model_builder.GetUniqueName(MakeString(node_unit.Name(), "/gemm_", i));
      ORT_RETURN_IF_ERROR(add_fc(a_2d, b_transposed_2d, gemm_output));

      // reshape output for concatenation ([x, y] -> [1, x, y])
      std::string gemm_output_3d;
      ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_output, Shape{1, m, n}, gemm_output_3d));

      gemm_outputs.push_back(gemm_output_3d);
    }
  }

  // concat batch
  const std::string joined_gemm_output =
      reshape_leading_dimensions ? model_builder.GetUniqueName(node_unit.Name() + "/joined_gemm_output") : output;
  {
    auto add_concat = [&model_builder](const std::vector<std::string>& inputs,
                                       const std::string& output) -> Status {
      const auto& operand_indices = model_builder.GetOperandIndices();
      const auto& operand_types = model_builder.GetOperandTypes();
      auto& shaper = model_builder.GetShaper();
      InlinedVector<uint32_t> input_indices;
      input_indices.reserve(inputs.size() + 1);
      std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_indices),
                     [&operand_indices](const std::string& input) { return operand_indices.at(input); });
      const int32_t axis = 0;
      ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, axis));

      // Calculate_concat_output_shape
      {
        std::vector<Shape> dimens;
        for (const auto& input_name : inputs) {
          const Shape dimen = shaper[input_name];
          dimens.push_back(dimen);
        }

        // If one of the inputs has dynamic shape (at axis), we will keep the dimen[axis] as 0 (dynamic)
        auto output_dimen = dimens[0];
        if (output_dimen[axis] != 0) {
          for (size_t i = 1; i < dimens.size(); i++) {
            if (dimens[i][axis] == 0) {
              output_dimen[axis] = 0;
              break;
            }
            output_dimen[axis] += dimens[i][axis];
          }
        }
        shaper.AddShape(output, output_dimen);
      }

      OperandType output_operand_type = operand_types.at(inputs[0]);
      output_operand_type.SetDimensions(shaper[output]);
      ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_CONCATENATION,
                                                     input_indices, {output}, {output_operand_type}));
      return Status::OK();
    };

    ORT_RETURN_IF_ERROR(add_concat(gemm_outputs, joined_gemm_output));
  }

  // reshape to original dimensions
  if (reshape_leading_dimensions) {
    Shape new_shape = a_shape;
    new_shape[new_shape.size() - 2] = m;
    new_shape[new_shape.size() - 1] = n;
    ORT_RETURN_IF_ERROR(add_reshape(joined_gemm_output, new_shape, output));
  }

  return Status::OK();
}

}  // namespace

class GemmOpBuilder : public BaseOpBuilder {
 private:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params, const logging::Logger& logger) const override;
  bool HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                    const OpSupportCheckParams& params, const logging::Logger& logger) const override;
  int GetMinSupportedOpSet(const NodeUnit& node_unit) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& /*node_unit*/, const logging::Logger& /*logger*/) const override {
    return true;
  }

  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& logger = model_builder.GetLogger();

  if (IsSupportedBatchMatMul(node_unit, model_builder.GetEffectiveFeatureLevel(), logger)) {
    // no initializers to skip for batch matmul
    return;
  }

  const auto& inputs = node_unit.Inputs();
  if (IsQuantizedOp(node_unit)) {
    if (node_unit.OpType() == "QLinearMatMul" || node_unit.OpType() == "MatMul") {                 // QLinear/QDQMatMul
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // a_scale, a_zp
      AddInputToSkip(model_builder, inputs[1]);                                                    // b, b_scale, b_zp
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
    } else if (node_unit.OpType() == "Gemm") {                                                     // QDQGemm
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // a_scale, a_zp
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[1].quant_param);               // b_scale, b_zp

      NodeAttrHelper helper(node_unit);
      const auto transB = helper.Get("transB", 0);
      // For transB == 0, we need to transpose it and add transposed initializer later into nnapi model,
      // not directly using it here, so add to skip list.
      if (transB == 0)
        model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());

      if (inputs.size() > 2) {
        AddInputToSkip(model_builder, inputs[2]);  // c, c_scale, c_zp (bias)
      }
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
    }
  } else {
    const auto& op = node_unit.OpType();
    if (op == "MatMul") {
      model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());
    } else if (op == "Gemm") {
      NodeAttrHelper helper(node_unit);
      const auto transB = helper.Get("transB", 0);
      if (transB == 0)
        model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());
    }
  }
}

Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& logger = model_builder.GetLogger();
  if (IsSupportedBatchMatMul(node_unit, model_builder.GetEffectiveFeatureLevel(), logger)) {
    return BuildBatchMatMul(model_builder, node_unit);
  }

  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());

  const auto& op = node_unit.OpType();
  const auto& inputs = node_unit.Inputs();
  NodeAttrHelper helper(node_unit);

  const auto quant_type = GetQuantizedOpType(node_unit);
  const bool is_quant_matmul = (quant_type == QuantizedOpType::QDQMatMul ||
                                quant_type == QuantizedOpType::QLinearMatMul);
  const bool is_quant_gemm = quant_type == QuantizedOpType::QDQGemm;

  const auto& input0_name = inputs[0].node_arg.Name();
  // if we skipped the preceeding Reshape we need to use it's input name instead
  // e.g. x -> Reshape -> y -> Gemm, gets adjusted to x -> Gemm if we skip the Reshape
  std::optional<std::string> skipped_reshape_input_name = model_builder.GetSkippedReshapeInput(input0_name);
  const auto& input0 = skipped_reshape_input_name ? *skipped_reshape_input_name : input0_name;
  const auto& input1 = inputs[1].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  const auto transB = helper.Get("transB", 0);

  float a_scale = 0.0f,
        b_scale = 0.0f,
        y_scale = 0.0f;
  int32_t a_zero_point = 0,
          b_zero_point = 0,
          y_zero_point = 0;

  bool is_per_tensor_u8s8 = false;
  if (is_quant_matmul || is_quant_gemm) {
    optional<std::vector<float>> w_scales;
    ORT_RETURN_IF_ERROR(
        GetConvMatMulOpQuantizationScaleAndZeroPoint(model_builder, node_unit,
                                                     a_scale, b_scale, y_scale,
                                                     a_zero_point, b_zero_point, y_zero_point,
                                                     w_scales, is_per_tensor_u8s8));
  }

  uint32_t input_1_idx;
  if (transB == 0) {
    Type onnx_mat_b_type;
    if (!is_quant_matmul && !is_quant_gemm)
      onnx_mat_b_type = Type::TENSOR_FLOAT32;
    else
      onnx_mat_b_type = Type::TENSOR_QUANT8_ASYMM;

    const auto& mat_b_tensor = *initializers.at(input1);
    Shape onnx_mat_b_shape;
    for (auto dim : mat_b_tensor.dims())
      onnx_mat_b_shape.push_back(SafeInt<uint32_t>(dim));

    const OperandType onnx_mat_b_operand_type(onnx_mat_b_type, onnx_mat_b_shape, b_scale, b_zero_point);
    ORT_RETURN_IF_ERROR(AddInitializerTransposed(model_builder, onnx_mat_b_operand_type, input1, is_per_tensor_u8s8));
  }

  input_1_idx = operand_indices.at(input1);
  // Verify if the scale and zero point matchs from onnx input and nnapi input
  if (is_quant_matmul || is_quant_gemm) {
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input0, a_scale, a_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, b_scale, b_zero_point));
  }

  uint32_t bias_idx;
  bool has_bias = inputs.size() > 2;
  if (has_bias) {
    const auto& bias = inputs[2].node_arg.Name();
    if (!is_quant_gemm) {
      // We need to squeeze the input tensor to 1d if necessary
      if (shaper[bias].size() > 1) {
        std::string bias_squeezed = model_builder.GetUniqueName(node_unit.Name() + op + "_bias_squeezed");
        // We will use squeeze all here
        ORT_RETURN_IF_ERROR(AddSqueezeOp(model_builder, node_unit.Name(),
                                         bias, bias_squeezed, {} /* axes */));
        bias_idx = operand_indices.at(bias_squeezed);
        LOGS(logger, VERBOSE) << "GemmOpBuilder - Operand [" << bias << "] squeezed from "
                              << Shape2String(shaper[bias])
                              << " to "
                              << Shape2String(shaper[bias_squeezed]);
      } else {
        bias_idx = operand_indices.at(bias);
      }
    } else {  // is_quant_gemm
      const auto& bias_tensor = *model_builder.GetInitializerTensors().at(bias);
      // QGemm has a contraint on input C to be int32 type
      ORT_RETURN_IF_NOT(bias_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32,
                        "bias of QDQGemm should be int32, actual type: ", bias_tensor.data_type());
      Shape bias_dimen;
      for (auto dim : bias_tensor.dims())
        bias_dimen.push_back(SafeInt<uint32_t>(dim));
      Initializer unpacked_tensor(bias_tensor);
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale);
      ORT_RETURN_IF_ERROR(
          model_builder.AddOperandFromPersistMemoryBuffer(
              bias,
              unpacked_tensor.data<int32_t>(), bias_operand_type));

      bias_idx = operand_indices.at(bias);
    }

  } else {
    // No C supplied, we need a vector of 0
    std::string bias = model_builder.GetUniqueName(node_unit.Name() + op + "_bias");
    const auto& bias_type = operand_types.at(input1).type;
    const Shape& bias_dimen = {shaper[input1][0]};
    if (bias_type == Type::TENSOR_FLOAT32) {
      std::vector<float> buffer(bias_dimen[0], 0.f);
      OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_dimen);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else if (bias_type == Type::TENSOR_QUANT8_ASYMM) {
      std::vector<int32_t> buffer(bias_dimen[0], 0);
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale, 0);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown weight type ", TypeToStr(bias_type));
    }

    bias_idx = operand_indices.at(bias);
  }

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input0));  // A
  input_indices.push_back(input_1_idx);                 // B
  input_indices.push_back(bias_idx);                    // C
  int32_t fuse_code = model_builder.FindActivation(node_unit);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  const OperandType output_operand_type(operand_types.at(input0).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

bool GemmOpBuilder::HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                                 const OpSupportCheckParams& params,
                                                 const logging::Logger& logger) const {
  if (!IsQuantizedOp(node_unit)) {
    return InputIsFloat(node_unit, 0, logger);
  }

  // QLinearMatMul/QDQGemm/QDQMatMul
  if (!HasValidBinaryOpQuantizedInputTypes(node_unit, logger))
    return false;

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0, 1}, params, ArgType::kInput, logger))
    return false;

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kOutput, logger))
    return false;

  return true;
}

int GemmOpBuilder::GetMinSupportedOpSet(const NodeUnit& node_unit) const {
  const auto& op(node_unit.OpType());

  // Gemm opset 6- has broadcast attributes we do not support now
  if (op == "Gemm")
    return 7;

  return 1;
}

bool GemmOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedGemm(GetQuantizedOpType(node_unit));
}

bool GemmOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                      const OpSupportCheckParams& params, const logging::Logger& logger) const {
  // check batch matmul first, then fall back to checking single gemm/matmul
  {
    const bool is_supported_batch_matmul = IsSupportedBatchMatMul(node_unit, params.android_feature_level, logger);
    LOGS(logger, VERBOSE) << "Supported batch matmul: [" << is_supported_batch_matmul << "]";
    if (is_supported_batch_matmul) {
      return true;
    }
  }

  const auto& op_type = node_unit.OpType();
  const auto& inputs = node_unit.Inputs();
  const bool is_qlinear_matmul = op_type == "QLinearMatMul";
  const auto quant_type = GetQuantizedOpType(node_unit);
  const bool is_quant_gemm = quant_type == QuantizedOpType::QDQGemm;

  Shape a_shape;
  {
    if (!GetShape(inputs[0].node_arg, a_shape, logger))
      return false;

    if (a_shape.size() != 2) {
      LOGS(logger, VERBOSE) << "A must be 2D";
      return false;
    }
  }

  Shape b_shape;
  {
    if (!GetShape(inputs[1].node_arg, b_shape, logger))
      return false;

    if (b_shape.size() != 2) {
      LOGS(logger, VERBOSE) << "B must be 2D";
      return false;
    }
  }

  if (op_type == "Gemm") {
    // Only support
    // 1. A*B'+C
    // 2. A*B+C and B is an initializer
    NodeAttrHelper helper(node_unit);
    const auto transA = helper.Get("transA", 0);
    const auto transB = helper.Get("transB", 0);
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);

    if (!(transA == 0 && alpha == 1.f && beta == 1.f)) {
      LOGS(logger, VERBOSE) << "Only transA == 0, alpha == 1.0 "
                            << "and beta == 1.0 is supported."
                            << " transA " << transA
                            << " transB " << transB
                            << " alpha " << alpha
                            << " beta " << beta;
      return false;
    }

    if (transB == 0 && !graph_viewer.GetConstantInitializer(inputs[1].node_arg.Name())) {
      LOGS(logger, VERBOSE) << "B of Gemm must be a constant initializer if transB != 1";
      return false;
    }

    if (inputs.size() == 3) {
      Shape c_shape;
      if (!GetShape(inputs[2].node_arg, c_shape, logger))
        return false;

      uint32_t c_size;
      if (!GetGemmBiasSize(c_shape, params.android_feature_level, c_size, logger))
        return false;

      if (c_size != (transB == 0 ? b_shape[1] : b_shape[0])) {
        LOGS(logger, VERBOSE) << "C of Gemm must be a vector of b_shape["
                              << (transB == 0 ? "1" : "0") << "]"
                              << " b_shape: " << Shape2String(b_shape)
                              << " c_shape: " << Shape2String(c_shape);

        return false;
      }
    }
  } else if (op_type == "MatMul" || is_qlinear_matmul) {
    // Only support A*B B is an initializer
    if (!graph_viewer.GetConstantInitializer(inputs[1].node_arg.Name())) {
      LOGS(logger, VERBOSE) << "B of MatMul must be a constant initializer";
      return false;
    }
  } else {
    LOGS(logger, VERBOSE) << "GemmOpSupportChecker, unknown op: " << op_type;
  }

  if (is_quant_gemm) {
    if (inputs.size() > 2 && !graph_viewer.GetConstantInitializer(inputs[2].node_arg.Name())) {
      LOGS(logger, VERBOSE) << "Bias of QDQ Gemm must be a constant initializer";
      return false;
    }
  }

  return true;
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<GemmOpBuilder>(
      op_type, op_registrations,
      {
          "Gemm",
          "MatMul",
          "QLinearMatMul",
      });
}

}  // namespace nnapi
}  // namespace onnxruntime
