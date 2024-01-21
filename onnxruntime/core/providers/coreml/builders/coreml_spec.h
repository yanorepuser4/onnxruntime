// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// TODO come up with a more intuitive way of limiting this to Apple platform builds
// E.g., putting CoreML EP files that should be enabled iff `defined(__APPLE__OR__TEST__)` in a separate directory.
// #if !defined(__APPLE__OR__TEST__)
// #error "This file should only be included when building on Apple platforms."
// #endif

// Model.pb.h is generated in the build output directory from the CoreML protobuf files in
// onnxruntime/core/providers/coreml/coremltools/mlmodel/format
#include "coreml_proto/Model.pb.h"

namespace COREML_SPEC = CoreML::Specification;
namespace COREML_SPEC_MIL = CoreML::Specification::MILSpec;
