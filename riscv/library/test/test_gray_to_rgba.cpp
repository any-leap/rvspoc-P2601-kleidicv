// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kleidicv/kleidicv.h"

extern "C" const char *kleidicv_riscv_active_backend();

namespace {

int failures = 0;
#define EXPECT(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); ++failures; } } while(0)

void test_backend() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_gray_to_rgba] backend=%s\n", b);
  if (std::strcmp(b, std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv") != 0) std::exit(2);
}

void test_bulk() {
  constexpr size_t kW = 257, kH = 3;
  std::vector<uint8_t> src(kW * kH), out(4 * kW * kH);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i * 7 + 3);
  (void)kleidicv_gray_to_rgba_u8(src.data(), kW, out.data(), 4 * kW, kW, kH);
  for (size_t i = 0; i < src.size(); ++i) {
    if (out[4*i] != src[i] || out[4*i+1] != src[i] || out[4*i+2] != src[i] || out[4*i+3] != 0xFF) {
      std::fprintf(stderr, "FAIL i=%zu\n", i); ++failures; break;
    }
  }
}

void test_null() {
  uint8_t b[1] = {0};
  EXPECT(kleidicv_gray_to_rgba_u8(nullptr, 1, b, 4, 1, 1) == KLEIDICV_ERROR_NULL_POINTER, "null");
}

}  // namespace

int main() { test_backend(); test_bulk(); test_null(); if (failures == 0) { std::printf("[test_gray_to_rgba] all checks passed\n"); return 0; } return 1; }
