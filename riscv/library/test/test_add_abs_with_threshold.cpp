// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
  std::printf("[test_add_abs_with_threshold] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

int16_t ref(int16_t a, int16_t b, int16_t threshold) {
  int32_t aa = std::abs(static_cast<int32_t>(a));
  int32_t bb = std::abs(static_cast<int32_t>(b));
  int32_t s = aa + bb;
  constexpr int32_t kMax = std::numeric_limits<int16_t>::max();
  if (s > kMax) s = kMax;
  return s > threshold ? static_cast<int16_t>(s) : int16_t{0};
}

void test_bulk() {
  constexpr size_t kW = 257, kH = 3;
  std::vector<int16_t> a(kW * kH), b(kW * kH), out(kW * kH);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<int16_t>(static_cast<int>(i) * 19 - 5000);
    b[i] = static_cast<int16_t>(static_cast<int>(i * i) - static_cast<int>(i) * 11);
  }
  int16_t threshold = 4000;
  (void)kleidicv_saturating_add_abs_with_threshold_s16(
      a.data(), kW * 2, b.data(), kW * 2, out.data(), kW * 2, kW, kH,
      threshold);
  for (size_t i = 0; i < a.size(); ++i) {
    int16_t exp = ref(a[i], b[i], threshold);
    if (out[i] != exp) {
      std::fprintf(stderr, "FAIL i=%zu a=%d b=%d got=%d exp=%d\n", i, a[i],
                   b[i], out[i], exp);
      ++failures;
      break;
    }
  }
}

void test_int16_min_saturation() {
  // |INT16_MIN| would be 32768 — saturate to INT16_MAX.
  int16_t a[1] = {std::numeric_limits<int16_t>::min()};
  int16_t b[1] = {0};
  int16_t out[1] = {0};
  (void)kleidicv_saturating_add_abs_with_threshold_s16(a, 2, b, 2, out, 2, 1,
                                                       1, 0);
  EXPECT(out[0] == std::numeric_limits<int16_t>::max(), "INT16_MIN abs sat");
}

void test_threshold_strict() {
  // sum exactly == threshold should produce 0 (strictly greater).
  int16_t a[1] = {50}, b[1] = {50}, out[1] = {0};
  (void)kleidicv_saturating_add_abs_with_threshold_s16(a, 2, b, 2, out, 2, 1,
                                                       1, 100);
  EXPECT(out[0] == 0, "threshold equal => 0");
  out[0] = 0;
  (void)kleidicv_saturating_add_abs_with_threshold_s16(a, 2, b, 2, out, 2, 1,
                                                       1, 99);
  EXPECT(out[0] == 100, "above threshold => sum");
}

void test_null() {
  int16_t buf[1] = {0};
  EXPECT(kleidicv_saturating_add_abs_with_threshold_s16(nullptr, 2, buf, 2,
                                                        buf, 2, 1, 1, 0) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null");
}

}  // namespace

int main() {
  test_backend_active();
  test_bulk();
  test_int16_min_saturation();
  test_threshold_strict();
  test_null();
  if (failures == 0) {
    std::printf("[test_add_abs_with_threshold] all checks passed\n");
    return 0;
  }
  std::printf("[test_add_abs_with_threshold] %d FAILURES\n", failures);
  return 1;
}
