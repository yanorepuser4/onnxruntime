// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/shared/utils/utils.h"

// #ifdef __APPLE__OR__TEST__
#include "core/providers/coreml/builders/model_builder.h"
// #endif

using namespace CoreML::Specification;

namespace onnxruntime {
namespace coreml {

namespace {
// TODO, move this to shared_library
bool HasExternalInitializer(const InitializedTensorSet& initializers, const Node& node,
                            const logging::Logger& logger) {
  for (const auto* node_arg : node.InputDefs()) {
    const auto& input_name(node_arg->Name());
    const auto initializer_it = initializers.find(input_name);
    if (initializer_it == initializers.end()) {
      continue;
    }

    const auto& tensor = *initializer_it->second;
    if (tensor.has_data_location() &&
        tensor.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS(logger, VERBOSE) << "Initializer [" << input_name
                            << "] with external data location are not currently supported";
      return true;
    }
  }

  return false;
}

}  // namespace

// Add operator related
// #if defined(__APPLE__OR__TEST__) || defined(__linux__)
Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                        const logging::Logger& logger) const {
  Status status = AddToModelBuilderImpl(model_builder, node, logger);

  if (status.IsOK()) {
    LOGS(logger, VERBOSE) << "Operator name: [" << node.Name() << "] type: [" << node.OpType() << "] was added";
  }

  return status;
}

// #endif

// Operator support related

bool BaseOpBuilder::IsOpSupported(const Node& node, const OpBuilderInputParams& input_params,
                                  const logging::Logger& logger) const {
  if (input_params.create_mlprogram && !SupportsMLProgram()) {
    LOGS(logger, VERBOSE) << "Operator [" << node.OpType() << "] does not support MLProgram";
    return false;
  }

  if (!HasSupportedOpSet(node, logger)) {
    return false;
  }

  if (!HasSupportedInputs(node, input_params, logger)) {
    return false;
  }

  // We do not support external initializers for now
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
  if (HasExternalInitializer(initializers, node, logger)) {
    return false;
  }

  return IsOpSupportedImpl(node, input_params, logger);
}

bool BaseOpBuilder::HasSupportedInputs(const Node& node, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  for (const auto* input : node.InputDefs()) {
    if (!IsInputSupported(node, *input, input_params, logger)) {
      return false;
    }
  }

  return HasSupportedInputsImpl(node, logger);
}

/* static */
bool BaseOpBuilder::IsInput0Supported(const Node& node, const logging::Logger& logger) {
  const auto& input = *node.InputDefs()[0];

  int32_t input_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;

  // TODO: With ML Program can we expand the allowed input types?
  // If so, we need to check if we're generating an ML Program and need OpBuilderInputParams.create_mlprogram
  // to be passed through.
  if (!GetType(input, input_type, logger) ||
      (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
    LOGS(logger, VERBOSE) << "[" << node.OpType() << "] Input type: [" << input_type << "] is not currently supported";
    return false;
  }

  return true;
}

///* static */
// void BaseOpBuilder::AddOperationArgument(google::protobuf::Map<std::string, COREML_SPEC::MILSpec::Argument>& params,
//                                          const std::string& param_name, const std::string& value_name) {
//   COREML_SPEC::MILSpec::Argument arg;
//   arg.mutable_arguments()->Add()->set_name(value_name);
//   params[param_name] = std::move(arg);
// }
//
///* static */
// template <typename T>
// COREML_SPEC::MILSpec::TensorValue BaseOpBuilder::CreateTensorValue(const std::string& name,
//                                                                    const gsl::span<const T>& data,
//                                                                    const gsl::span<int32_t>& shape,
//                                                                    COREML_SPEC::MILSpec::DataType data_type) {
//   MILSpec::Value value;
//   MILSpec::ValueType& value_type = *value.mutable_type();
//   MILSpec::TensorType& tensor_type = *value_type.mutable_tensortype();
//   tensor_type.set_datatype(data_type);
//   tensor_type.set_rank(shape.size());
//   tensor_type.mutable_dimensions()->Add(shape.begin(), shape.end());
//
//   MILSpec::TensorValue& tensor_value = *value.mutable_immediatevalue()->mutable_tensor();
//   TensorValueDataWriter.Copy(tensor_value, data);
//
//   return value;
// }

bool BaseOpBuilder::HasSupportedInputsImpl(const Node& node, const logging::Logger& logger) const {
  // We only check the type of input 0 by default
  // specific op builder can override this
  return IsInput0Supported(node, logger);
}

bool BaseOpBuilder::HasSupportedOpSet(const Node& node, const logging::Logger& logger) const {
  auto since_version = node.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node) || since_version > GetMaxSupportedOpSet(node)) {
    LOGS(logger, VERBOSE) << node.OpType() << "is only supported for opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

}  // namespace coreml
}  // namespace onnxruntime
