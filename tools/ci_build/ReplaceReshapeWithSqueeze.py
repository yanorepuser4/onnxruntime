import onnx
import struct
import sys
from onnx import TensorProto
from pathlib import Path

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <onnx model path>")
    # TODO: We could also look for 4D ReduceMean/ReduceMax with keepdims=1 and axes of {2, 3} as that is equivalent
    #       to GlobalAveragePool/GlobalMaxPool
    print("""
    Search for GlobalAveragePool or GlobalMaxPool with 4D input -> Reshape to {x, -1} -> Gemm and replace the Reshape 
    with Squeeze. If the model is updated, it will be saved with '.update.onnx' as the suffix in the same location 
    as the original model
    
    Prior to running this script:
      - ensure the model is updated to ONNX opset 13 or later
      - optimize the ONNX model to 'basic' level to ensure the Reshape `shape` input is an initializer if possible
      
      python -m onnxruntime.tools.update_onnx_opset --opset 13 model.onnx model.opset13.onnx
      python -m onnxruntime.tools.optimize_onnx_model --opt_level basic model.opset13.onnx model.opset13. basic.onnx

    NOTE: Subgraphs of Scan/Loop/If nodes are not searched.
    """)
    sys.exit(-1)

model_path = Path(sys.argv[1])
model_path.resolve(strict=True)

m = onnx.load(str(model_path))
orig_model = onnx.shape_inference.infer_shapes(m)  # run shape_inference as we need shape info
onnx_opset = [i.version for i in orig_model.opset_import if i.domain == ''][0]
if onnx_opset < 13:
    # we currently  support Squeeze axes as inputs, which requires opset 13 or later
    raise ValueError("Please update the model to opset 13 or later.")

value_info_map = {}
for vi in orig_model.graph.value_info:
    value_info_map[vi.name] = vi

initializer_map = {}
for tp in orig_model.graph.initializer:
    initializer_map[tp.name] = tp

pool_outputs = set()
reshape_output_to_node = {}
reshape_nodes_to_replace = []  # NodeProto isn't hashable so need to keep in list

# first see if any nodes match
for node in orig_model.graph.node:
    if node.op_type == "GlobalAveragePool" or node.op_type == "GlobalMaxPool":
        if node.input[0] in value_info_map:
            input_info = value_info_map[node.input[0]]
            if len(input_info.type.tensor_type.shape.dim) == 4:
                pool_outputs.add(node.output[0])

    # GlobalAveragePool or GlobalMaxPool -> Reshape
    if node.op_type == "Reshape" and node.input[0] in pool_outputs:
        replace = False
        # look for Reshape to [x, -1]. `shape` input must be initializer
        shape_input = node.input[1]
        if shape_input in initializer_map:
            reshape_to = initializer_map[shape_input]
            if reshape_to.dims[0] == 2:
                if reshape_to.HasField("raw_data"):
                    dims = struct.unpack('qq', reshape_to.raw_data)
                else:
                    dims = reshape_to.int64_data

                if dims[1] == -1:
                    reshape_output_to_node[node.output[0]] = node

    # GlobalAveragePool or GlobalMaxPool -> Reshape to {x, -1} -> Gemm
    if node.op_type == "Gemm" and node.input[0] in reshape_output_to_node:
        # add the Reshape node providing the Gemm input.
        reshape_node = reshape_output_to_node[node.input[0]]
        assert reshape_node not in reshape_nodes_to_replace
        reshape_nodes_to_replace.append(reshape_node)

if reshape_nodes_to_replace:
    replacement_idx = 0

    # edit the model without shape info as we don't want to save that
    # we clear out the nodes first and add from orig_model, replacing any Reshape as we go
    del m.graph.node[:]

    for node in orig_model.graph.node:
        if node in reshape_nodes_to_replace:
            squeeze_name = f"reshape_to_squeeze_{replacement_idx}"
            axes_name = f"reshape_to_squeeze_axes_{replacement_idx}"

            # Squeeze the trailing 2 dims from the GlobalAveragePool/GlobalMaxPool which are both 1.
            # Output from those operators is always 4D.
            squeeze_axes = onnx.helper.make_tensor(axes_name, TensorProto.INT64, dims=[2], vals=[2, 3])
            m.graph.initializer.append(squeeze_axes)

            squeeze_node = onnx.helper.make_node("Squeeze", [node.input[0], axes_name], [node.output[0]],
                                                 name=squeeze_name)
            m.graph.node.append(squeeze_node)

            replacement_idx += 1

        else:
            m.graph.node.append(node)

    new_model_path = model_path.with_suffix(suffix=".updated.onnx")
    onnx.checker.check_model(m)
    onnx.save(m, new_model_path)
else:
    print("No nodes were replaced.")
