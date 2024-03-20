// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include "endian.h"
#include "core/common/common.h"

namespace onnxruntime {
struct Int4Pair {
  int8_t val_0 : 4;
  int8_t val_1 : 4;

  Int4Pair() : val_0{0}, val_1{0} {}
  Int4Pair(uint8_t bits) {
    val_0 = static_cast<int8_t>(bits & 0xFF);
    val_1 = static_cast<int8_t>((bits >> 4) & 0xFF);
  }

  inline int8_t operator[](size_t index) const {
    assert(index <= 1);
    return index == 0 ? val_0 : val_1;
  }

  inline uint8_t ToBits() const {
    return (static_cast<uint8_t>(val_1) << 4) | (static_cast<uint8_t>(val_0) & 0xFF);
  }
};

static_assert(sizeof(Int4Pair) == sizeof(int8_t));

struct UInt4Pair {
  uint8_t val_0 : 4;
  uint8_t val_1 : 4;

  UInt4Pair() : val_0{0}, val_1{0} {}
  UInt4Pair(uint8_t bits) {
    val_0 = bits & 0xFF;
    val_1 = (bits >> 4) & 0xFF;
  }

  inline uint8_t operator[](size_t index) const {
    assert(index <= 1);
    return index == 0 ? val_0 : val_1;
  }

  inline uint8_t ToBits() const {
    return (static_cast<uint8_t>(val_1) << 4) | (static_cast<uint8_t>(val_0) & 0xFF);
  }
};

static_assert(sizeof(UInt4Pair) == sizeof(uint8_t));
}  // namespace onnxruntime
