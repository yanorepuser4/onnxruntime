// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/framework_common.h"
#include "core/framework/node_unit.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace utils {

#if !defined(DISABLE_OPTIONAL_TYPE)
common::Status OutputOptionalWithoutDataHelper(const ONNX_NAMESPACE::TypeProto& input_type_proto,
                                               OpKernelContext* context, int output_index);
#endif

// Get all the nodes in the given graph_viewer as NodeUnits (SingleNode or QDQGroup)
// And return a map to quick query the NodeUnit which contains the given Node,
// Note, the value of the map is owned by the vector of std::unique_ptr<NodeUnit>
std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
GetAllNodeUnits(const GraphViewer& graph_viewer);

}  // namespace utils
}  // namespace onnxruntime
