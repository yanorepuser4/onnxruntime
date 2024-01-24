// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/model/host_utils.h"

namespace onnxruntime {
namespace coreml {
namespace util {

// Manually adjust these when testing as needed.
// e.g. set HasMLProgram to return false to test fallback to creating NeuralNetwork model instead of MLProgram
bool HasRequiredBaseOS() {
  return true;
}

int CoreMLVersion() {
  return 7;  // CoreML 7 is the latest we support
}

std::string GetTemporaryFilePath() {
  // TODO: could do something more advanced if needed. for local testing only so starting with very basic.
  return "_COREML_EP_TEMP_FILE_PATH_";
}

}  // namespace util
}  // namespace coreml
}  // namespace onnxruntime
