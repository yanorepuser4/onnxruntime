// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This contains the utility functions which will be used to build a coreml model

#pragma once

#ifdef __APPLE__OR__TEST__

#include <optional>

#include "core/common/gsl.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/providers/common.h"

#include "core/providers/coreml/builders/coreml_spec.h"

// namespace CoreML {
// namespace Specification {
// class WeightParams;
// }
// }  // namespace CoreML

namespace onnxruntime {
class NodeArg;

namespace coreml {

// Try to see if we can map explicit padding to auto padding for Conv/Pool
// Since usually use auto padding is more efficient
Status HandleAutoPad(const std::vector<int64_t> input_shape,
                     const int64_t weight_size_y,
                     const int64_t weight_size_x,
                     const std::vector<int64_t>& onnx_pads,
                     const std::vector<int64_t>& onnx_strides,
                     const std::vector<int64_t>& onnx_dilations,
                     AutoPadType auto_pad_type,
                     AutoPadType& auto_pad_type_out);

//
// NeuralNetwork utils
//

// Copy an onnx initializer data to a coreml weight
Status CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, const ONNX_NAMESPACE::TensorProto& tensor);

// Copy the float array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const float> data);

// Copy the int32_t array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const int32_t> data);

// Copy the int64_t array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const int64_t> data);

//
// MLProgram utils
//

// helper for static_assert at end of is_same tests
template <typename>
constexpr bool false_for_T = false;

template <typename T>
COREML_SPEC::MILSpec::DataType DataTypeToMILSpec() {
  if constexpr (std::is_same_v<T, float>) {
    return COREML_SPEC::MILSpec::DataType::FLOAT32;
  } else if constexpr (std::is_same_v<T, double>) {
    return COREML_SPEC::MILSpec::DataType::FLOAT64;
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return COREML_SPEC::MILSpec::DataType::BFLOAT16;
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    return COREML_SPEC::MILSpec::DataType::FLOAT16;

  } else if constexpr (std::is_same_v<T, int8_t>) {
    return COREML_SPEC::MILSpec::DataType::INT8;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return COREML_SPEC::MILSpec::DataType::INT16;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return COREML_SPEC::MILSpec::DataType::INT32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return COREML_SPEC::MILSpec::DataType::INT64;

  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return COREML_SPEC::MILSpec::DataType::UINT8;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return COREML_SPEC::MILSpec::DataType::UINT16;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return COREML_SPEC::MILSpec::DataType::UINT32;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return COREML_SPEC::MILSpec::DataType::UINT64;

  } else if constexpr (std::is_same_v<T, bool>) {
    return COREML_SPEC::MILSpec::DataType::BOOL;
  } else if constexpr (std::is_same_v<T, std::string>) {
    return COREML_SPEC::MILSpec::DataType::STRING;
  } else {
    static_assert(false_for_T<T>, "Unsupported type.");
  }
}

COREML_SPEC::MILSpec::DataType OnnxDataTypeToMILSpec(int onnx_type);

// convert int64_t ONNX shape to int32_t CoreML shape
std::vector<int32_t> GetCoreMLShape(const gsl::span<const int64_t> dims);

/// <summary>
/// Create a CoreML MILSpec::TensorValue for an ONNX tensor.
/// </summary>
/// <typeparam name="T1">ONNX C++ data type</typeparam>
/// <typeparam name="T2">CoreML C++ data type</typeparam>
/// <param name="data_type">CoreML protobuf data type</param>
/// <param name="data">ONNX data</param>
/// <param name="shape">ONNX data shape</param>
/// <returns>New TensorValue</returns>
template <typename T1, typename T2 = T1>
COREML_SPEC::MILSpec::Value CreateTensorValue(const gsl::span<const T1> data,
                                              std::optional<const gsl::span<const int32_t>> shape = std::nullopt);

template <typename T>
COREML_SPEC::MILSpec::Value CreateScalarTensorValue(const T& data);

// Add an input argument to a MILSpec::Operation
// The input name is defined by the spec for the operation.
// The value_name is the value that is providing the input.
void AddOperationInput(COREML_SPEC::MILSpec::Operation& op,
                       std::string_view input_name, std::string_view value_name);

void AddOperationOutput(COREML_SPEC::MILSpec::Operation& op, const NodeArg& output);
}  // namespace coreml
}  // namespace onnxruntime

#endif
