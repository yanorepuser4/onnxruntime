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

using namespace CoreML::Specification;

namespace onnxruntime {
namespace coreml {

namespace {

// The TensorProto.data_type field is an int, but must be a valid TensorProto_DataType value.
// Use int for the arg so the caller can pass TensorProto.data_type() value and do the cast to enum internally
MILSpec::DataType OnnxDataTypeToMILSpec(int onnx_type) {
  switch (static_cast<ONNX_NAMESPACE::TensorProto_DataType>(onnx_type)) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return MILSpec::DataType::FLOAT32;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return MILSpec::DataType::FLOAT64;
    // case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: Not sure if this is supported
    //   return MILSpec::DataType::BFLOAT16;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return MILSpec::DataType::FLOAT16;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return MILSpec::DataType::INT8;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return MILSpec::DataType::INT16;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return MILSpec::DataType::INT32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return MILSpec::DataType::INT64;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return MILSpec::DataType::UINT8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return MILSpec::DataType::UINT16;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return MILSpec::DataType::UINT32;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return MILSpec::DataType::UINT64;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return MILSpec::DataType::BOOL;
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      return MILSpec::DataType::STRING;
    default:
      ORT_THROW("Unsupported data type: ", onnx_type);
  }
}

// Should the Tensor be written to file or kept as an immediate value
bool UseWeightFile(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  // https://github.com/apple/coremltools/blob/dbb0094fd0cb936469e35320bf37e866ef7a1da4/coremltools/converters/mil/backend/mil/load.py#L51-L57

  bool use_weight_file = false;

  switch (tensor_proto.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      auto num_elements = TensorShape(utils::GetTensorShapeFromTensorProto(tensor_proto)).Size();
      use_weight_file = num_elements >= 10;
      break;
    }
    default:
      break;
  }

  return use_weight_file;
}

// write the weight to weight.bin and return the offset
uint64_t AddWeightToFile(const onnx::TensorProto& tensor_proto) {
  // TEMP hack to test. needs to use BlobWriter from coremltools
  static uint64_t offset = 0;
  offset += TensorShape(utils::GetTensorShapeFromTensorProto(tensor_proto)).Size();
  return offset;
}

// copy from the ONNX TensorProto to a CoreML field.
// T1 is the source type. T2 is the target type. If the types differ, T1 must be smaller than T2.
// e.g. uint32_t data can be written to RepeatedField<uint64_t>
template <typename T1, typename T2 = T1>
void CopyRawDataToRepeatedField(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                google::protobuf::RepeatedField<T2>& repeated_field) {
  const auto& raw_data = tensor_proto.raw_data();
  const T1* data = reinterpret_cast<const T1*>(raw_data.data());
  const T1* data_end = data + (raw_data.size() / sizeof(T1));
  if constexpr (sizeof(T1) == sizeof(T2)) {
    repeated_field.Add(data, data_end);
  } else {
    static_assert(sizeof(T1) < sizeof(T2));
    // we need to iterate over the data and copy to the repeated field, converting to T2 as we go.
    repeated_field.Resize(data_end - data, T2(0));
    for (int i = 0; data != data_end; ++data, ++i) {
      repeated_field[i] = static_cast<T2>(*data);
    }
  }
}

// copy T data from the TensorProto.int32_t field to TensorValue.bytes
template <typename T>
void CopyInt32DataToBytes(const ONNX_NAMESPACE::TensorProto& tensor_proto, MILSpec::TensorValue tensor_value) {
  const int num_entries = tensor_proto.int32_data_size();
  std::string& bytes = *tensor_value.mutable_bytes()->mutable_values();
  bytes.resize(num_entries * sizeof(T));
  T* out = reinterpret_cast<T*>(bytes.data());

  const int32_t* in = tensor_proto.int32_data().data();
  for (int i = 0; i < num_entries; ++i) {
    out[i] = static_cast<T>(in[i]);
  }
}

// copy T data from the TensorProto.uint64_data field to TensorValue.bytes
template <typename T>
void CopyUInt64DataToBytes(const ONNX_NAMESPACE::TensorProto& tensor_proto, MILSpec::TensorValue tensor_value) {
  const int num_entries = tensor_proto.uint64_data_size();
  std::string& bytes = *tensor_value.mutable_bytes()->mutable_values();
  bytes.resize(num_entries * sizeof(T));
  T* out = reinterpret_cast<T*>(bytes.data());

  const uint64_t* in = tensor_proto.uint64_data().data();
  for (int i = 0; i < num_entries; ++i) {
    out[i] = static_cast<T>(in[i]);
  }
}

void AddTensorProtoDataToMILSpecTensorValue(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                            MILSpec::TensorValue& tensor_value) {
  bool has_raw_data = tensor_proto.has_raw_data();
  auto data_type = tensor_proto.data_type();

  // handling based on
  // ONNX TensorProto field usage
  // https://github.com/onnx/onnx/blob/b86cc54efce19530fb953e4b21f57e6b3888534c/onnx/onnx.proto#L544-L572
  // CoreMLTools conversion implementation that maps data types to fields
  // https://github.com/apple/coremltools/blob/dbb0094fd0cb936469e35320bf37e866ef7a1da4/coremltools/converters/mil/backend/mil/helper.py#L98
  // along with some special cased types that are stored in bytes
  // https://github.com/apple/coremltools/blob/dbb0094fd0cb936469e35320bf37e866ef7a1da4/coremltools/converters/mil/backend/mil/helper.py#L23
  //   IMMEDIATE_VALUE_TYPES_IN_BYTES = (types.fp16, types.int8, types.uint8, types.uint32)

  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      // from: float_data/raw, to: floats
      if (has_raw_data) {
        CopyRawDataToRepeatedField<float>(tensor_proto, *tensor_value.mutable_floats()->mutable_values());
      } else {
        tensor_value.mutable_floats()->mutable_values()->CopyFrom(tensor_proto.float_data());
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      // from: double_data/raw, to: doubles
      if (has_raw_data) {
        CopyRawDataToRepeatedField<double>(tensor_proto, *tensor_value.mutable_doubles()->mutable_values());
      } else {
        tensor_value.mutable_doubles()->mutable_values()->CopyFrom(tensor_proto.double_data());
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      // from: int32_data/raw, to: ints
      if (has_raw_data) {
        CopyRawDataToRepeatedField<int32_t>(tensor_proto, *tensor_value.mutable_ints()->mutable_values());
      } else {
        tensor_value.mutable_ints()->mutable_values()->CopyFrom(tensor_proto.int32_data());
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      // from: int64_data/raw, to: longints
      if (has_raw_data) {
        CopyRawDataToRepeatedField<int64_t>(tensor_proto, *tensor_value.mutable_longints()->mutable_values());

      } else {
        tensor_value.mutable_longints()->mutable_values()->CopyFrom(tensor_proto.int64_data());
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
      // from: int32_data/raw, to: bytes
      if (has_raw_data) {
        *tensor_value.mutable_bytes()->mutable_values() = tensor_proto.raw_data();
      } else {
        // iterate the int32_data, taking the 16-bits from each entry, and copying to the bytes.
        // we use uint16_t as only the size of the data type matters
        CopyInt32DataToBytes<uint16_t>(tensor_proto, tensor_value);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      // from: int32_data/raw, to: bytes
      if (has_raw_data) {
        *tensor_value.mutable_bytes()->mutable_values() = tensor_proto.raw_data();
      } else {
        // copy from int32_data to bytes. uint8_t for both as only the size of the data type matters when copying
        CopyInt32DataToBytes<uint8_t>(tensor_proto, tensor_value);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {
      // from: uint64_data/raw, to: bytes
      if (has_raw_data) {
        *tensor_value.mutable_bytes()->mutable_values() = tensor_proto.raw_data();
      } else {
        // copy uint32_t values from TensorProto.uint64_data
        CopyUInt64DataToBytes<uint32_t>(tensor_proto, tensor_value);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
      // from: uint64_data/raw, to: longints
      if (has_raw_data) {
        CopyRawDataToRepeatedField<uint64_t>(tensor_proto, *tensor_value.mutable_longints()->mutable_values());
      } else {
        // TODO: Is this safe? Need to check the CopyFrom implementation. As it's a straight copy of bytes this
        // hopefully can do it as one block instead of iterating and potentially doing a static_cast of each
        // individual value.
        tensor_value.mutable_longints()->mutable_values()->CopyFrom(
            reinterpret_cast<const google::protobuf::RepeatedField<int64_t>&>(tensor_proto.uint64_data()));
      }

      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL: {
      // from: int32_data/raw, to: bools
      if (has_raw_data) {
        CopyRawDataToRepeatedField<bool>(tensor_proto, *tensor_value.mutable_bools()->mutable_values());
      } else {
        const auto& int32s = tensor_proto.int32_data();
        auto& bools = *tensor_value.mutable_bools()->mutable_values();
        const int num_entries = int32s.size();
        bools.Reserve(num_entries);
        const int32_t* in = int32s.data();
        for (int i = 0; i < num_entries; ++i) {
          *bools.AddAlreadyReserved() = *in++;
        }
      }

      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_STRING: {
      // from: string_data (which is protobuf type bytes), to: strings (protobuf type string)
      // due to the protobuf type mismatch we need to iterate and copy
      auto& in = tensor_proto.string_data();
      auto& out = *tensor_value.mutable_strings()->mutable_values();
      out.Reserve(in.size());
      for (const auto& iter : in) {
        *out.Add() = iter;
      }

      break;
    }
    /* Not clear there's an actual use-case for 16-bit int data currently, so leaving commented out
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      // from: int32_data/raw, to: ints
      // WARNING: This may change to write to mutable_bytes
      // https://github.com/apple/coremltools/blob/dbb0094fd0cb936469e35320bf37e866ef7a1da4/coremltools/converters/mil/backend/mil/helper.py#L113-L115
      if (has_raw_data) {
          CopyRawDataToRepeatedField<uint16_t, int32_t>(tensor_proto, *tensor_value.mutable_ints()->mutable_values());
      } else {
          tensor_value.mutable_ints()->mutable_values()->CopyFrom(tensor_proto.int32_data());
      }
      break;
    } */
    default:
      ORT_THROW("AddTensorProtoDataToMILSpecTensorValue: Unsupported data type: ", data_type);
  }
}

// MILSpec::Value OnnxValueInfoToMILSpec(const ONNX_NAMESPACE::ValueInfoProto& tensor_proto) {
// }

// TODO: Could turn this into the more generic CreateScalarValue if it supports more types
MILSpec::Value CreateNameValue(const std::string& name) {
  MILSpec::Value value;
  MILSpec::ValueType& value_type = *value.mutable_type();
  MILSpec::TensorType& tensor_type = *value_type.mutable_tensortype();
  tensor_type.set_datatype(MILSpec::DataType::STRING);
  tensor_type.set_rank(0);

  MILSpec::TensorValue& tensor_value = *value.mutable_immediatevalue()->mutable_tensor();
  std::string& data = *tensor_value.mutable_strings()->add_values();
  data = name;

  return value;
}

MILSpec::Value OnnxTensorProtoToMILSpec(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  MILSpec::Value value;

  // populate ValueType with tensor data type, dims and rank
  MILSpec::ValueType& value_type = *value.mutable_type();
  MILSpec::TensorType* tensor_type = value_type.mutable_tensortype();
  tensor_type->set_datatype(OnnxDataTypeToMILSpec(tensor_proto.data_type()));
  // create_valuetype_scalar in coremltools/converters/mil/backend/mil/helper.py creates a ValueType with empty
  // shape and rank of zero, so it looks like it's fine for an ML Program to have a rank 0 scalar.
  // i.e. we don't need to convert rank 0 to rank 1 like we did with NeuralNetwork

  tensor_type->set_rank(tensor_proto.dims().size());
  for (const auto& dim : tensor_proto.dims()) {
    tensor_type->add_dimensions()->mutable_constant()->set_size(dim);
  }

  // add data to either weights.bin or as an immediate value
  if (UseWeightFile(tensor_proto)) {
    uint64_t offset = AddWeightToFile(tensor_proto);

    auto* file_value = value.mutable_blobfilevalue();
    // Filename copied from
    // https://github.com/apple/coremltools/blob/dbb0094fd0cb936469e35320bf37e866ef7a1da4/coremltools/converters/mil/backend/mil/helper.py#L329
    file_value->set_filename("@model_path/weights/weight.bin");
    file_value->set_offset(offset);

  } else {
    MILSpec::TensorValue* tensor_value = value.mutable_immediatevalue()->mutable_tensor();
    AddTensorProtoDataToMILSpecTensorValue(tensor_proto, *tensor_value);
  }

  return value;
}

}  // namespace

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
    if (usage_count == 0) {
      continue;
    }

    if (create_ml_program_) {
      auto value = OnnxTensorProtoToMILSpec(tensor);
      MILSpec::Operation const_op;
      const_op.set_type("const");

      /*
      return pm.Operation(
        type="const",
        attributes={"name": create_scalar_value(op.name), "val": value},
        outputs=[
            pm.NamedValueType(
                name=output_var.name, type=types_to_proto(output_var.sym_type)
            )
        ],

        Seems to create a TensorValue for the name of the weight

        def create_scalar_value(py_scalar):
            """
            Return TensorValue (since there's no ScalarValue)
            """
            # Create the "scalar" (rank 0) tensor
            builtin_type = type_to_builtin_type(type(py_scalar))
            value_type = create_valuetype_scalar(types_to_proto_primitive(builtin_type))
            val = pm.Value(type=value_type)
            t_val = val.immediateValue.tensor

            # Set the tensor value
            t_field = _tensor_field_by_type(t_val, builtin_type)
            if builtin_type in IMMEDIATE_VALUE_TYPES_IN_BYTES:
                # Serialize to bytes because MIL read them from the "bytes" field in TensorValue.
                val.immediateValue.tensor.bytes.values = np_val_to_py_type(py_scalar)
            else:
                if builtin_type == types.str:
                    py_scalar = py_scalar.encode("utf-8")
                t_field.append(np_val_to_py_type(py_scalar))

            return val

      */
      auto* attr_map = const_op.mutable_attributes();
      (*attr_map)["name"] = CreateNameValue(name);
      (*attr_map)["val"] = value;
    } else {
      std::unique_ptr<NeuralNetworkLayer> layer = std::make_unique<NeuralNetworkLayer>();
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
        multi_array->set_datatype(ArrayFeatureType::FLOAT32);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        multi_array->set_datatype(ArrayFeatureType::INT32);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        // If we have an int64 input/output type, since COREML_SPEC:ArrayFeatureType does not support INT64
        // we assign it to be INT32 here
        multi_array->set_datatype(ArrayFeatureType::INT32);
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
  if (create_ml_program_) {
    // We support CorelML Specification Version 4 (Core ML 3)
    coreml_model_->set_specificationversion(4);
    auto* neural_network = coreml_model_->mutable_neuralnetwork();
    neural_network->set_arrayinputshapemapping(CoreML::Specification::NeuralNetworkMultiArrayShapeMapping::EXACT_ARRAY_MAPPING);
  } else {
    // target the CoreML version supported by this device.
    // TODO: Validate this returns the Core ML version for the device we are running on and not the device we
    // did the build on.
    // from Core ML 2 onwards the spec version is one greater due to Core ML 1.2 being spec version 2.
    int32_t coreml_version = CoreMLVersion();
    std::string coreml_opset = "CoreML" + std::to_string(coreml_version);
    coreml_model_->set_specificationversion(coreml_version + 1);
    MILSpec::Program* mlprogram = coreml_model_->mutable_mlprogram();
    MILSpec::Function& main = (*mlprogram->mutable_functions())["main"];  // ??? Does this create the Function instance
    *main.mutable_opset() = coreml_opset;
    mlprogram_main_ = &(*main.mutable_block_specializations())[coreml_opset];
  }

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

void ModelBuilder::AddLayer(std::unique_ptr<NeuralNetworkLayer> layer) {
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

}  // namespace coreml
}  // namespace onnxruntime
