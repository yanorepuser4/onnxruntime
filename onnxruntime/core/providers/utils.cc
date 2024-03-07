// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/utils.h"

namespace onnxruntime {
namespace utils {

#if !defined(DISABLE_OPTIONAL_TYPE)
common::Status OutputOptionalWithoutDataHelper(const ONNX_NAMESPACE::TypeProto& input_type_proto,
                                               OpKernelContext* context, int output_index) {
  if (utils::HasOptionalTensorType(input_type_proto)) {
    context->OutputOptionalWithoutData<Tensor>(output_index);
  } else if (utils::HasOptionalTensorSequenceType(input_type_proto)) {
    context->OutputOptionalWithoutData<TensorSeq>(output_index);
  } else {
    // Will never hit this as we don't support any other type than Tensor and TensorSeq
    // for optional type
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported type");
  }

  return Status::OK();
}
#endif

std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
GetAllNodeUnits(const GraphViewer& graph_viewer) {
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

  const auto add_node_unit_to_map = [&](const std::vector<NodeIndex>& node_indices, const NodeUnit* node_unit) {
    for (const auto& node_idx : node_indices) {
      const auto* node = graph_viewer.GetNode(node_idx);
      node_unit_map.insert({node, node_unit});
    }
  };

  // Get QDQ NodeUnits first
  QDQ::SelectorManager selector_mgr;
  const auto qdq_selections = selector_mgr.GetQDQSelections(graph_viewer);

  for (const auto& qdq_selection : qdq_selections) {
    auto qdq_unit = std::make_unique<NodeUnit>(graph_viewer, qdq_selection);

    // Fill the node to node_unit map for all nodes in the QDQ Group
    add_node_unit_to_map(qdq_selection.dq_nodes, qdq_unit.get());
    add_node_unit_to_map(qdq_selection.q_nodes, qdq_unit.get());
    add_node_unit_to_map({qdq_selection.target_node}, qdq_unit.get());

    node_unit_holder.push_back(std::move(qdq_unit));
  }

  // Get the left over SingleNode NodeUnits
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    const auto* node(graph_viewer.GetNode(node_idx));

    // This is already part of a QDQ NodeUnit
    if (node_unit_map.find(node) != node_unit_map.cend())
      continue;

    auto node_unit = std::make_unique<NodeUnit>(*node);
    node_unit_map[node] = node_unit.get();
    node_unit_holder.push_back(std::move(node_unit));
  }

  return std::make_pair(std::move(node_unit_holder), std::move(node_unit_map));
}

}  // namespace utils
}  // namespace onnxruntime
