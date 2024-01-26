// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/coreml/builders/op_builder.h"
#include "core/common/span_utils.h"

#if defined(__APPLE__OR__TEST__)
#include "core/providers/coreml/builders/coreml_spec.h"
#endif

namespace onnxruntime {
namespace coreml {

class ModelBuilder;

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;

  bool SupportsMLProgram() const override { return false; }

  // Add operator related
 public:
  bool IsOpSupported(const Node& node, const OpBuilderInputParams& input_params,
                     const logging::Logger& logger) const override final;

#if defined(__APPLE__OR__TEST__)
  Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                           const logging::Logger& logger) const override final;

  void AddInitializersToSkip(ModelBuilder& /*model_builder*/, const Node& /*node*/) const override {}

#endif

 protected:
  // check if the first input is supported. used for
  static bool IsInput0Supported(const Node& node, const OpBuilderInputParams& input_params,
                                const logging::Logger& logger);

  // Operator support related
 private:
  virtual bool IsOpSupportedImpl(const Node& /*node*/, const OpBuilderInputParams& /*input_params*/,
                                 const logging::Logger& /*logger*/) const {
    return true;
  }

  virtual bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const;

  virtual int GetMinSupportedOpSet(const Node& /*node*/) const { return 1; }
  virtual int GetMaxSupportedOpSet(const Node& /*node*/) const { return 20; }

  bool HasSupportedOpSet(const Node& node, const logging::Logger& logger) const;
  bool HasSupportedInputs(const Node& node, const OpBuilderInputParams& input_params,
                          const logging::Logger& logger) const;

#if defined(__APPLE__OR__TEST__)
  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                       const logging::Logger& logger) const = 0;
#endif
};

}  // namespace coreml
}  // namespace onnxruntime
