// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
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

double ref_sum(const std::vector<float> &v, size_t row_stride_floats,
               size_t width, size_t height) {
  double s = 0.0;
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      s += static_cast<double>(v[y * row_stride_floats + x]);
    }
  }
  return s;
}

void test_small() {
  std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float result = -1.0f;
  kleidicv_error_t err = kleidicv_sum_f32(v.data(), 6 * sizeof(float), 6, 1,
                                          &result);
  EXPECT(err == KLEIDICV_OK, "small returned error");
  EXPECT(result == 21.0f, "small sum value");
}

void test_two_d_with_padding() {
  // 13 valid + 3 padding floats per row, 5 rows. Padding bytes contain values
  // that would corrupt the sum if read.
  constexpr size_t kW = 13, kH = 5, kStrideFloats = 16;
  std::vector<float> buf(kStrideFloats * kH);
  // Fill padding with absurd values that would dominate the sum if read.
  for (auto &x : buf) x = 1.0e20f;
  for (size_t y = 0; y < kH; ++y) {
    for (size_t x = 0; x < kW; ++x) {
      buf[y * kStrideFloats + x] =
          static_cast<float>(static_cast<double>(x + y) * 0.5);
    }
  }
  float result = 0.0f;
  kleidicv_error_t err = kleidicv_sum_f32(
      buf.data(), kStrideFloats * sizeof(float), kW, kH, &result);
  EXPECT(err == KLEIDICV_OK, "2d returned error");
  double expected = ref_sum(buf, kStrideFloats, kW, kH);
  EXPECT(std::fabs(result - static_cast<float>(expected)) < 1e-3f,
         "2d sum mismatch");
}

void test_large_random_vs_reference() {
  // 1027 wide × 17 high — prime-ish width forces strip-mine tail iterations
  // at every plausible vlen, large enough that f32-only accumulation would
  // visibly drift from the f64 reference.
  constexpr size_t kW = 1027, kH = 17;
  std::vector<float> buf(kW * kH);
  uint32_t lcg = 0xDEADBEEFu;
  for (auto &x : buf) {
    lcg = lcg * 1103515245u + 12345u;
    // Centre around zero so partial sums fluctuate (worst case for f32 acc).
    x = (static_cast<float>(static_cast<int32_t>(lcg)) / 2147483648.0f) * 1.0e3f;
  }
  float result = 0.0f;
  kleidicv_error_t err = kleidicv_sum_f32(buf.data(), kW * sizeof(float), kW,
                                          kH, &result);
  EXPECT(err == KLEIDICV_OK, "large returned error");
  double expected = ref_sum(buf, kW, kW, kH);
  // Allow a bit of slack: even with f64 accumulator, vector reduction order
  // differs from a sequential one and the final cast is to f32. ~1e-3
  // relative is fine here.
  double rel = std::fabs(result - expected) / std::max(1.0, std::fabs(expected));
  if (rel > 1e-3) {
    std::fprintf(stderr,
                 "FAIL large rel-err=%g  result=%.9g expected=%.9g\n", rel,
                 static_cast<double>(result), expected);
    ++failures;
  }
}

void test_zero_size() {
  float result = 42.0f;
  // height=0 → must return 0 sum.
  kleidicv_error_t err = kleidicv_sum_f32(reinterpret_cast<float *>(0x1), 0,
                                          5, 0, &result);
  EXPECT(err == KLEIDICV_OK, "zero size returned error");
  EXPECT(result == 0.0f, "zero size should yield 0.0");
}

void test_null_pointer() {
  float result = 0.0f;
  kleidicv_error_t err = kleidicv_sum_f32(nullptr, 1, 1, 1, &result);
  EXPECT(err == KLEIDICV_ERROR_NULL_POINTER, "null src not flagged");
  err = kleidicv_sum_f32(reinterpret_cast<float *>(0x1), 1, 1, 1, nullptr);
  EXPECT(err == KLEIDICV_ERROR_NULL_POINTER, "null sum not flagged");
}

void test_backend_active() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_sum] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

}  // namespace

int main() {
  test_backend_active();
  test_small();
  test_two_d_with_padding();
  test_large_random_vs_reference();
  test_zero_size();
  test_null_pointer();
  if (failures == 0) {
    std::printf("[test_sum] all checks passed\n");
    return 0;
  }
  std::printf("[test_sum] %d FAILURES\n", failures);
  return 1;
}
