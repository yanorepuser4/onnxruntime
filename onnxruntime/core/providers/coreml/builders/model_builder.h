// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_viewer.h"
#include "core/providers/coreml/builders/coreml_spec.h"
#include "core/providers/coreml/model/model.h"

namespace onnxruntime {
namespace coreml {

class IOpBuilder;
class Model;

class ModelBuilder {
 public:
  ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger,
               int32_t coreml_version, uint32_t coreml_flags,
               const std::string& model_output_path);
  ~ModelBuilder() = default;

  // Create the CoreML model, serialize to disk, load and compile using the CoreML API and return in `model`
  Status Build(std::unique_ptr<Model>& model);

  // Accessors for members
  const GraphViewer& GetGraphViewer() const { return graph_viewer_; }
  const InitializedTensorSet& GetInitializerTensors() const { return graph_viewer_.GetAllInitializedTensors(); }
  int32_t CoreMLVersion() const { return coreml_version_; }
  bool CreateMLProgram() const { return create_ml_program_; }

  // Create a NeuralNetwork layer using the node name and optional suffix for the name.
  // If Node has no name a unique name will be generated from the node index and operator.
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> CreateNNLayer(const Node& node, std::string_view suffix = "");

  // Add layer to the Core ML NeuralNetwork model
  void AddLayer(std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer);

  // Add operator to Core ML MLProgram model
  void AddOperation(std::unique_ptr<COREML_SPEC::MILSpec::Operation> operation);

  // The initializer is processed separately (e.g. layout is transformed) by the operator builder,
  // so we don't do a copy of the original initializer into the model.
  void AddInitializerToSkip(const std::string& tensor_name);

  // There are some input which will not be used, add it to a list which will not
  // be added to CoreML model, since CoreML does not like input unused
  void AddInputToSkip(const std::string& input_name);

  std::string GetUniqueName(const std::string& base_name);

 private:
  // Convert the ONNX model in graph_viewer_ to a CoreML::Specification::Model and serialize to disk
  Status CreateModel();
  Status SaveModel();

  // when generating an mlpackage, should a weight be written to the external file or added directly
  bool UseWeightFile(const onnx::TensorProto& weight);
  void AddWeightToFile(const onnx::TensorProto& weight);

  // If a CoreML operation will use initializers directly, we will add the initializers to the skip list
  void PreprocessInitializers();

  // Copy and process all the initializers to CoreML model
  Status RegisterInitializers();

  Status ProcessNodes();
  Status RegisterModelInputs();
  Status RegisterModelOutputs();
  Status RegisterModelInputOutput(const NodeArg& node_arg, bool is_input);

  // Record the onnx scalar output names
  void AddScalarOutput(const std::string& output_name);

  // Record the onnx int64 type output names
  void AddInt64Output(const std::string& output_name);

  const GraphViewer& graph_viewer_;
  const logging::Logger& logger_;
  const int32_t coreml_version_;
  const uint32_t coreml_flags_;
  const std::string model_output_path_;
  const bool create_ml_program_;  // ML Program (CoreML5, iOS 15+, macOS 12+) or NeuralNetwork (old)

  std::unique_ptr<CoreML::Specification::Model> coreml_model_;
  std::unordered_set<std::string> scalar_outputs_;
  std::unordered_set<std::string> int64_outputs_;
  std::unordered_map<std::string, OnnxTensorInfo> input_output_info_;

  std::unordered_map<std::string, int> initializer_usage_;
  std::unordered_set<std::string> skipped_inputs_;

  uint32_t name_token_{0};
  std::unordered_set<std::string> unique_names_;
};

}  // namespace coreml
}  // namespace onnxruntime
