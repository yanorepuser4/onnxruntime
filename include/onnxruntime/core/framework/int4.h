// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include "endian.h"
#include "core/common/common.h"
#include "core/common/gsl.h"

namespace onnxruntime {
struct UnpackedInt4 {
  int8_t val : 4;
  UnpackedInt4() : val{0} {}
  UnpackedInt4(uint8_t bits) {
    val = static_cast<int8_t>(bits & 0xF);
  }

  UnpackedInt4& operator=(uint8_t bits) {
    val = static_cast<int8_t>(bits & 0xF);
    return *this;
  }
};

struct UnpackedUInt4 {
  uint8_t val : 4;
  UnpackedUInt4() : val{0} {}
  UnpackedUInt4(uint8_t bits) {
    val = bits & 0xF;
  }

  UnpackedUInt4& operator=(uint8_t bits) {
    val = bits & 0xF;
    return *this;
  }
};

struct Int4Pair {
  int8_t val_0 : 4;
  int8_t val_1 : 4;

  Int4Pair() : val_0{0}, val_1{0} {}
  Int4Pair(uint8_t bits) {
    val_0 = static_cast<int8_t>(bits & 0xF);
    val_1 = static_cast<int8_t>((bits >> 4) & 0xF);
  }
  Int4Pair(int8_t lo, int8_t hi) : val_0{lo}, val_1{hi} {}

  inline int8_t operator[](size_t index) const {
    assert(index <= 1);
    return index == 0 ? val_0 : val_1;
  }

  inline uint8_t ToBits() const {
    return (static_cast<uint8_t>(val_1) << 4) | (static_cast<uint8_t>(val_0) & 0xF);
  }

  static bool Unpack(gsl::span<UnpackedInt4> dst, gsl::span<const Int4Pair> src) {
    if (((dst.size() + 1) / 2) != src.size()) {
      return false;
    }

    for (size_t i = 0; i < dst.size(); i++) {
      size_t r = i >> 1;  // i / 2;
      size_t c = i & 0x1;  // i % 2;
      dst[i].val = src[r][c];
    }

    return true;
  }
};

static_assert(sizeof(Int4Pair) == sizeof(int8_t));

struct UInt4Pair {
  uint8_t val_0 : 4;
  uint8_t val_1 : 4;

  UInt4Pair() : val_0{0}, val_1{0} {}
  UInt4Pair(uint8_t bits) {
    val_0 = bits & 0xF;
    val_1 = (bits >> 4) & 0xF;
  }
  UInt4Pair(uint8_t lo, uint8_t hi) : val_0{lo}, val_1{hi} {}

  inline uint8_t operator[](size_t index) const {
    assert(index <= 1);
    return index == 0 ? val_0 : val_1;
  }

  inline uint8_t ToBits() const {
    return (static_cast<uint8_t>(val_1) << 4) | (static_cast<uint8_t>(val_0) & 0xF);
  }

  static bool Unpack(gsl::span<UnpackedUInt4> dst, gsl::span<const UInt4Pair> src) {
    if (((dst.size() + 1) / 2) != src.size()) {
      return false;
    }

    for (size_t i = 0; i < dst.size(); i++) {
      size_t r = i >> 1;  // i / 2;
      size_t c = i & 0x1;  // i % 2;
      dst[i].val = src[r][c];
    }

    return true;
  }
};

static_assert(sizeof(UInt4Pair) == sizeof(uint8_t));
}  // namespace onnxruntime
