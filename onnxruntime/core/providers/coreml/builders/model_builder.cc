// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/providers/coreml/model/host_utils.h"
#include "core/providers/coreml/shape_utils.h"

using namespace COREML_SPEC;

namespace onnxruntime {
namespace coreml {

ModelBuilder::ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger,
                           int32_t coreml_version, uint32_t coreml_flags,
                           const std::string& model_output_path)
    : graph_viewer_(graph_viewer),
      logger_(logger),
      coreml_version_(coreml_version),
      coreml_flags_(coreml_flags),
      model_output_path_(model_output_path),
      create_ml_program_((coreml_flags_ & COREML_FLAG_CREATE_MLPROGRAM) != 0) {
}

std::unique_ptr<NeuralNetworkLayer> ModelBuilder::CreateNNLayer(const Node& node, std::string_view suffix) {
  auto layer_name = node.Name();
  if (layer_name.empty()) {
    // CoreML requires layer has a name, while the node name is optional in ONNX
    // In this case, create a unique name for the layer
    layer_name = GetUniqueName(MakeString("Node_", node.Index(), "_", node.OpType(), suffix));
  } else if (!suffix.empty()) {
    layer_name += suffix;
  }

  std::unique_ptr<NeuralNetworkLayer> layer = std::make_unique<NeuralNetworkLayer>();
  layer->set_name(layer_name);
  return layer;
}

void ModelBuilder::PreprocessInitializers() {
  // TODO: We should be using GetConstantInitializer not GetAllInitializedTensors in all places
  const auto& initializers = graph_viewer_.GetAllInitializedTensors();
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();

  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto& node = *graph_viewer_.GetNode(node_indices[i]);

    // find all initializers consumed. AddInitializersToSkip will potentially decrement the usage count.
    for (const auto* input : node.InputDefs()) {
      if (input->Exists() && Contains(initializers, input->Name())) {
        initializer_usage_[input->Name()]++;
      }
    }

    if (const auto* op_builder = GetOpBuilder(node)) {
      op_builder->AddInitializersToSkip(*this, node);
    }
  }
}

Status ModelBuilder::RegisterInitializers() {
  for (const auto& pair : GetInitializerTensors()) {
    const auto& tensor = *pair.second;
    const auto& name = tensor.name();

    // skip initializer if there is no remaining usage
    auto usage_count = initializer_usage_[name];
    if (usage_count == 0)
      continue;

    //
    // TODO: For MLProgram we need to create a Value with ValueType, and potentially write to weights.bin
    //

    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = std::make_unique<COREML_SPEC::NeuralNetworkLayer>();
    layer->set_name(GetUniqueName("initializer_" + name));

    // TODO,look at using LoadConstantLayer instead of LoadConstantNDLayer
    auto* constant_tensor = layer->mutable_loadconstantnd();
    const auto& shape = tensor.dims();
    if (shape.empty()) {
      // This is a scalar initializer, CoreML constant layer requires a shape, make this a {1} tensor
      constant_tensor->mutable_shape()->Add(1);
    } else {
      std::transform(shape.cbegin(), shape.cend(),
                     google::protobuf::RepeatedFieldBackInserter(constant_tensor->mutable_shape()),
                     [](int64_t dim) -> uint64_t { return SafeInt<uint64_t>(dim); });
    }

    ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*constant_tensor->mutable_data(), tensor));
    *layer->mutable_output()->Add() = name;
    AddLayer(std::move(layer));
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputOutput(const NodeArg& node_arg, bool is_input) {
  const auto& name = node_arg.Name();
  const std::string input_output_type = is_input ? "input" : "output";

  if (is_input) {
    // input should not be an initializer
    if (Contains(GetInitializerTensors(), name))
      return Status::OK();

    // This input will not be used
    if (Contains(skipped_inputs_, name))
      return Status::OK();
  }

  auto* model_description = coreml_model_->mutable_description();
  auto& input_output = is_input
                           ? *model_description->mutable_input()->Add()
                           : *model_description->mutable_output()->Add();

  input_output.set_name(name);
  auto* multi_array = input_output.mutable_type()->mutable_multiarraytype();

  std::vector<int64_t> shape;
  ORT_RETURN_IF_NOT(GetShape(node_arg, shape, logger_),
                    "Unable to get shape for ", input_output_type, ": ", name);

  if (shape.empty()) {
    // If we have an empty shape, this is a scalar input,
    // Since all the input output of CoreML EP is MultiArray, we will make the scalar input output as a {1} MultiArray
    shape.push_back(1);

    // we need to change the shapes of these scalar outputs back to {} when CoreML EP returns these values to ORT
    if (!is_input) {
      AddScalarOutput(name);
    }
  }

  if (IsStaticShape(shape)) {
    *multi_array->mutable_shape() = {shape.cbegin(), shape.cend()};
  } else {
    if (is_input) {
      auto& multi_array_shape_range = *multi_array->mutable_shaperange();
      auto& multi_array_shape = *multi_array->mutable_shape();

      for (const auto dim : shape) {
        auto& multi_array_dim_size_range = *multi_array_shape_range.mutable_sizeranges()->Add();
        if (dim == -1) {
          multi_array_dim_size_range.set_lowerbound(0);
          multi_array_dim_size_range.set_upperbound(-1);  // unbounded

          multi_array_shape.Add(1);  // pick 1 as an arbitrary default dynamic dimension value
        } else {
          multi_array_dim_size_range.set_lowerbound(dim);
          multi_array_dim_size_range.set_upperbound(dim);

          multi_array_shape.Add(dim);
        }
      }
    } else {
      // Leave dynamic output shapes unspecified.
      // If we specify an output shape that doesn't match the actual output shape at runtime, CoreML returns a 5D shape
      // padded with ones.
    }
  }

  int32_t data_type;
  {  // type
    const auto* type_proto = node_arg.TypeAsProto();
    if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The ", input_output_type, " of graph doesn't have elem_type: ", name);
    }

    data_type = type_proto->tensor_type().elem_type();
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        multi_array->set_datatype(COREML_SPEC::ArrayFeatureType::FLOAT32);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        multi_array->set_datatype(COREML_SPEC::ArrayFeatureType::INT32);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        // If we have an int64 input/output type, since COREML_SPEC:ArrayFeatureType does not support INT64
        // we assign it to be INT32 here
        multi_array->set_datatype(COREML_SPEC::ArrayFeatureType::INT32);
        if (!is_input) {
          // Record the output names and we need to change them back to Int64 when CoreML EP returns these values to ORT
          AddInt64Output(name);
        }
        break;
      default: {
        // TODO: support other type
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "The ", input_output_type, " of graph doesn't have valid type, name: ", name,
                               " type: ", type_proto->tensor_type().elem_type());
      }
    }
  }

  input_output_info_.emplace(name, OnnxTensorInfo{data_type, shape});

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputs() {
  for (const auto* node_arg : graph_viewer_.GetInputs()) {
    ORT_RETURN_IF_ERROR(RegisterModelInputOutput(*node_arg, true /* is_input */));
  }

  return Status::OK();
}

Status ModelBuilder::ProcessNodes() {
  // const auto builder_params = MakeOpBuilderParams(graph_viewer_, coreml_version_, coreml_flags_);

  for (const auto node_idx : graph_viewer_.GetNodesInTopologicalOrder()) {
    const auto& node = *graph_viewer_.GetNode(node_idx);
    if (const auto* op_builder = GetOpBuilder(node)) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(*this, node, logger_));
    } else {
      // This shouldn't happen as this is called from CoreMLExecutionProvider::Compile and should only be processing
      // nodes that we said were supported and were returned from CoreMLExecutionProvider::GetCapability.
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Node [", node.Name(), "], type [", node.OpType(), "] is not supported");
    }
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelOutputs() {
  for (const auto* node_arg : graph_viewer_.GetOutputs()) {
    ORT_RETURN_IF_ERROR(RegisterModelInputOutput(*node_arg, false /* is_input */));
  }

  return Status::OK();
}

Status ModelBuilder::CreateModel() {
  coreml_model_ = std::make_unique<CoreML::Specification::Model>();

  // initialize CoreML model
  // We support CorelML Specification Version 4 (Core ML 3)

  coreml_model_->set_specificationversion(4);
  auto* neural_network = coreml_model_->mutable_neuralnetwork();
  neural_network->set_arrayinputshapemapping(CoreML::Specification::NeuralNetworkMultiArrayShapeMapping::EXACT_ARRAY_MAPPING);

  PreprocessInitializers();

  ORT_RETURN_IF_ERROR(RegisterInitializers());
  ORT_RETURN_IF_ERROR(RegisterModelInputs());
  ORT_RETURN_IF_ERROR(ProcessNodes());
  ORT_RETURN_IF_ERROR(RegisterModelOutputs());

  return Status::OK();
}

Status ModelBuilder::SaveModel() {
  std::ofstream stream(model_output_path_, std::ofstream::out | std::ofstream::binary);
  ORT_RETURN_IF_NOT(coreml_model_->SerializeToOstream(&stream), "Save the CoreML model failed");

#if !defined(NDEBUG)
  // Debug infra to allow also saving to an alternate path using an env var.
  std::string debug_path = onnxruntime::Env::Default().GetEnvironmentVar("ORT_COREML_EP_CONVERTED_MODEL_PATH");
  if (!debug_path.empty()) {
    std::filesystem::copy(model_output_path_, debug_path, std::filesystem::copy_options::overwrite_existing);
  }
#endif

  return Status::OK();
}

Status ModelBuilder::Build(std::unique_ptr<Model>& model) {
  ORT_RETURN_IF_ERROR(CreateModel());
  ORT_RETURN_IF_ERROR(SaveModel());

  model = std::make_unique<Model>(model_output_path_,
                                  std::move(input_output_info_),
                                  std::move(scalar_outputs_),
                                  std::move(int64_outputs_),
                                  logger_, coreml_flags_);

  return model->LoadModel();  // load using CoreML API, including compilation
}

void ModelBuilder::AddScalarOutput(const std::string& output_name) {
  scalar_outputs_.insert(output_name);
}

void ModelBuilder::AddInt64Output(const std::string& output_name) {
  int64_outputs_.insert(output_name);
}

void ModelBuilder::AddLayer(std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer) {
  auto* neural_network = coreml_model_->mutable_neuralnetwork();
  neural_network->mutable_layers()->AddAllocated(layer.release());
}

void ModelBuilder::AddInitializerToSkip(const std::string& tensor_name) {
  // decrement usage count if this is a known initializer.
  // For simplicity the OpBuilder::AddInitializersToSkip implementations may call this for arbitrary input names
  // without first checking if the value is an initializer.
  auto entry = initializer_usage_.find(tensor_name);
  if (entry != initializer_usage_.end()) {
    entry->second -= 1;
  }
}

void ModelBuilder::AddInputToSkip(const std::string& input_name) {
  skipped_inputs_.insert(input_name);
}

std::string ModelBuilder::GetUniqueName(const std::string& base_name) {
  std::string unique_name;
  do {
    std::ostringstream os;
    os << base_name << "_token_" << name_token_++;
    unique_name = os.str();
  } while (Contains(unique_names_, unique_name));

  return unique_name;
}

bool ModelBuilder::UseWeightFile(const onnx::TensorProto& weight) {
  /*
  https://github.com/apple/coremltools/blob/dbb0094fd0cb936469e35320bf37e866ef7a1da4/coremltools/converters/mil/backend/mil/load.py#L51-L57
  def should_use_weight_file(val):
    return (
        val is not None
        and isinstance(val, (np.ndarray, np.generic))
        and val.size >= 10
        and val.dtype in ['float16', 'float32', 'uint8', 'int8']
    )*/

  bool use_weight_file = false;

  switch (weight.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      auto num_elements = TensorShape(utils::GetTensorShapeFromTensorProto(weight)).Size();
      use_weight_file = num_elements >= 10;
      break;
    }
    default:
      break;
  }

  return use_weight_file;
}

void ModelBuilder::AddWeightToFile(const onnx::TensorProto& /*weight*/) {
}
}  // namespace coreml
}  // namespace onnxruntime
