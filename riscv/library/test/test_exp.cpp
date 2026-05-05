// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
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
  std::printf("[test_exp] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

void test_bulk() {
  // 257 inputs spanning a moderate range. Tolerance: poly maxerr per
  // upstream is ~0.4 + 0.5 ulp, but in our SPOC compare to std::exp with a
  // generous relative tolerance.
  constexpr size_t kW = 257, kH = 1;
  std::vector<float> src(kW * kH), out(kW * kH);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = -10.0f + static_cast<float>(i) * (20.0f / (kW * kH - 1));
  (void)kleidicv_exp_f32(src.data(), kW * sizeof(float), out.data(),
                         kW * sizeof(float), kW, kH);
  for (size_t i = 0; i < src.size(); ++i) {
    float exp_ref = std::exp(src[i]);
    float got = out[i];
    float rel = std::fabs(got - exp_ref) / std::fmax(std::fabs(exp_ref), 1e-30f);
    if (rel > 5e-6f) {
      std::fprintf(stderr,
                   "FAIL i=%zu src=%g got=%g ref=%g rel=%g\n", i, src[i],
                   got, exp_ref, rel);
      ++failures;
      break;
    }
  }
}

void test_zero() {
  float src[1] = {0.0f}, out[1] = {0.0f};
  (void)kleidicv_exp_f32(src, 4, out, 4, 1, 1);
  EXPECT(std::fabs(out[0] - 1.0f) < 1e-6f, "exp(0) ≈ 1");
}

void test_null() {
  float buf[1] = {0};
  EXPECT(kleidicv_exp_f32(nullptr, 4, buf, 4, 1, 1) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null");
}

}  // namespace

int main() {
  test_backend_active();
  test_bulk();
  test_zero();
  test_null();
  if (failures == 0) {
    std::printf("[test_exp] all checks passed\n");
    return 0;
  }
  std::printf("[test_exp] %d FAILURES\n", failures);
  return 1;
}
