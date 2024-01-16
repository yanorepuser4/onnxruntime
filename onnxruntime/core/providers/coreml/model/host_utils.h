// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file hosts the c++ bridge functions for some host utility functions which
// are available using Objective c only

#pragma once

#include <string>

#define API_AVAILABLE_OS_VERSIONS API_AVAILABLE(macos(10.15), ios(13))
#define MLPROGRAM_AVAILABLE_OS_VERSIONS API_AVAILABLE(macos(12), ios(15))

// Base required OS to run CoreML Specification Version 4 (Core ML 3)
#define HAS_VALID_BASE_OS_VERSION_COREML_3 @available(macOS 10.15, iOS 13, *)

// Base required OS to run CoreML Specification Version 6 (Core ML 4) which added MLProgram
// https://github.com/apple/coremltools/blob/4392c6be9506edf56aa00a95074635af4b729dee/mlmodel/format/Model.proto#L253-L255
#define HAS_VALID_BASE_OS_VERSION_COREML_4 @available(macOS 12, iOS 15, *)

namespace onnxruntime {
namespace coreml {
namespace util {

// Return if we are running on the required OS to enable CoreML EP
// This corresponds to [CoreML Specification Version 4 (Core ML 3)]
bool HasRequiredBaseOS();

bool HasMLProgram();

// Get a temporary macOS/iOS temp file path
std::string GetTemporaryFilePath();

}  // namespace util
}  // namespace coreml
}  // namespace onnxruntime
