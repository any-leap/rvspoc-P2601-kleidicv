// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
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
#define EXPECT(cond, msg)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);  \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

void test_backend_active() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_in_range] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

void test_bulk() {
  constexpr size_t kW = 257, kH = 3;
  std::vector<uint8_t> src(kW * kH), out(kW * kH);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i * 11 + 1);
  uint8_t lo = 50, hi = 200;
  (void)kleidicv_in_range_u8(src.data(), kW, out.data(), kW, kW, kH, lo, hi);
  for (size_t i = 0; i < src.size(); ++i) {
    uint8_t exp = (src[i] >= lo && src[i] <= hi) ? 0xFF : 0;
    if (out[i] != exp) {
      std::fprintf(stderr, "FAIL i=%zu src=%u got=%u exp=%u\n", i, src[i],
                   out[i], exp);
      ++failures;
      break;
    }
  }
}

void test_inclusive_bounds() {
  uint8_t src[4] = {49, 50, 200, 201};
  uint8_t out[4] = {0};
  (void)kleidicv_in_range_u8(src, 4, out, 4, 4, 1, 50, 200);
  EXPECT(out[0] == 0, "below");
  EXPECT(out[1] == 0xFF, "lower inclusive");
  EXPECT(out[2] == 0xFF, "upper inclusive");
  EXPECT(out[3] == 0, "above");
}

void test_null() {
  uint8_t buf[1] = {0};
  EXPECT(kleidicv_in_range_u8(nullptr, 1, buf, 1, 1, 1, 0, 1) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null");
}

void test_f32_bulk() {
  constexpr size_t kW = 65, kH = 2;
  std::vector<float> src(kW * kH);
  std::vector<uint8_t> out(kW * kH);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<float>(i) * 0.1f - 3.0f;
  float lo = -1.5f, hi = 2.5f;
  (void)kleidicv_in_range_f32(src.data(), kW * sizeof(float), out.data(), kW,
                              kW, kH, lo, hi);
  for (size_t i = 0; i < src.size(); ++i) {
    uint8_t exp = (src[i] >= lo && src[i] <= hi) ? 0xFF : 0;
    if (out[i] != exp) {
      std::fprintf(stderr, "FAIL f32 i=%zu src=%g got=%u exp=%u\n", i, src[i],
                   out[i], exp);
      ++failures;
      break;
    }
  }
}

}  // namespace

int main() {
  test_backend_active();
  test_bulk();
  test_inclusive_bounds();
  test_f32_bulk();
  test_null();
  if (failures == 0) {
    std::printf("[test_in_range] all checks passed\n");
    return 0;
  }
  std::printf("[test_in_range] %d FAILURES\n", failures);
  return 1;
}
