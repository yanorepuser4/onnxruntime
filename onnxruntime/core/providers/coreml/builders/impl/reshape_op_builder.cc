// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class ReshapeOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  // Reshape opset 4- uses attributes for new shape which we do not support for now
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 5; }

  bool SupportsMLProgram() const override { return true; }
};

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  if (model_builder.CreateMLProgram()) {
    // we add the initializers as 'const' operations via ModelBuilder::RegisterInitializers
    return;
  }

  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  // TODO: This doesn't seem like it handles a 0 with allowzero of false where we need to copy the
  // value from the input shape. As we reject nodes with allowzero=true and zeros, we know that any zero here
  // means copy the value from the original shape.

  // One -1 is allowed and means to infer the size from the original shape
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  // this should always succeed given we checked it in IsOpSupportedImpl
  ORT_RETURN_IF_NOT(GetStaticShape(*input_defs[0], input_shape, logger), "Cannot get shape of data");

  const auto& data_name = input_defs[0]->Name();
  const auto& new_shape_name = input_defs[1]->Name();
  Initializer unpacked_tensor(*model_builder.GetConstantInitializer(new_shape_name));
  TensorShapeVector new_shape = ToShapeVector(unpacked_tensor.DataAsSpan<int64_t>());

  // ReshapeHelper applies the ONNX rules to create the concrete output shape
  ReshapeHelper helper(TensorShape(input_shape), new_shape);

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    std::unique_ptr<Operation> reshape_op = model_builder.CreateOperation(node, "reshape");

    AddOperationInput(*reshape_op, "x", data_name);

    // convert from TensorShapeVector to std::vector<int64_t> to work with more common helper
    // TODO: is it worth the binary size cost of adding a specialization to AddValueAsConstantOperationInput
    // to handle TensorShapeVector directly? Would only be used when a shape is an input to an op.
    AddOperationInput(*reshape_op, "shape",
                      model_builder.AddConstant(reshape_op->type(), "shape",
                                                std::vector<int64_t>(new_shape.begin(), new_shape.end())));

    AddOperationOutput(*reshape_op, *node.OutputDefs()[0]);

    model_builder.AddOperation(std::move(reshape_op));
  } else
#endif  // defined(COREML_ENABLE_MLPROGRAM)
  {
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

    *layer->mutable_reshapestatic()->mutable_targetshape() = {new_shape.cbegin(), new_shape.cend()};
    *layer->mutable_input()->Add() = data_name;
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool ReshapeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& new_shape_name = input_defs[1]->Name();
  const auto* new_shape = input_params.graph_viewer.GetConstantInitializer(new_shape_name);
  if (!new_shape) {
    // technically a dynamic shape _could_ be used with ML Program, however ONNX has different rules around how
    // -1 and 0 values are used/combined, so we can't check if those can be translated to CoreML if the shape is
    // unknown.
    LOGS(logger, VERBOSE) << "New shape of reshape must be a constant initializer";
    return false;
  }

  Initializer unpacked_tensor(*new_shape);
  auto new_shape_dims = unpacked_tensor.DataAsSpan<int64_t>();
  if (new_shape_dims.empty()) {
    LOGS(logger, VERBOSE) << "New shape of reshape cannot be empty";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetStaticShape(*input_defs[0], input_shape, logger))
    return false;

  if (input_shape.empty()) {
    LOGS(logger, VERBOSE) << "Reshape does not support empty input shape";
    return false;
  }

  // CoreML reshape doesn't support new shape with more than 5 dimensions. This applies to both NN and MLProgram.
  if (new_shape_dims.size() > 5) {
    LOGS(logger, VERBOSE) << "Reshape does not support new shape with rank greater than 5. Input shape: "
                          << Shape2String(input_shape) << ", new shape: " << Shape2String(new_shape_dims);
    return false;
  }

  // CoreML reshape does not support 0 as dimension (i.e. a tensor that has no data)
  NodeAttrHelper helper(node);
  const bool allow_zero = helper.Get("allowzero", 0) == 1;
  if (allow_zero) {
    if (std::find(new_shape_dims.begin(), new_shape_dims.end(), int64_t{0}) != new_shape_dims.end()) {
      LOGS(logger, VERBOSE) << "Reshape does not support new shape with 0 as dimension when allowzero is enabled. "
                               "Input shape: "
                            << Shape2String(input_shape) << ", new shape: " << Shape2String(new_shape_dims);
      return false;
    }
  }

  return true;
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
