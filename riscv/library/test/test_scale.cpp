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

void test_backend_active() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_scale] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

uint8_t ref_u8(uint8_t v, double scale, double shift) {
  long iv = std::lrintf(static_cast<float>(v) * static_cast<float>(scale) +
                        static_cast<float>(shift));
  if (iv < 0) return 0;
  if (iv > 255) return 255;
  return static_cast<uint8_t>(iv);
}

void test_u8_bulk() {
  constexpr size_t kW = 257, kH = 3;
  std::vector<uint8_t> src(kW * kH), out(kW * kH);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i & 0xff);
  double scale = 0.5, shift = 10.0;
  (void)kleidicv_scale_u8(src.data(), kW, out.data(), kW, kW, kH, scale, shift);
  for (size_t i = 0; i < src.size(); ++i) {
    uint8_t exp = ref_u8(src[i], scale, shift);
    if (out[i] != exp) {
      std::fprintf(stderr, "FAIL u8 i=%zu src=%u got=%u exp=%u\n", i, src[i],
                   out[i], exp);
      ++failures;
      break;
    }
  }
}

void test_u8_saturate() {
  // scale=2, shift=0: 200*2 = 400 → saturate to 255
  uint8_t src[1] = {200}, out[1] = {0};
  (void)kleidicv_scale_u8(src, 1, out, 1, 1, 1, 2.0, 0.0);
  EXPECT(out[0] == 255, "u8 sat");
  // scale=1, shift=-1000: → 0
  (void)kleidicv_scale_u8(src, 1, out, 1, 1, 1, 1.0, -1000.0);
  EXPECT(out[0] == 0, "u8 floor");
}

void test_f32_bulk() {
  constexpr size_t kW = 65, kH = 2;
  std::vector<float> src(kW * kH), out(kW * kH);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i) * 0.25f - 7.0f;
  double scale = 1.5, shift = 2.0;
  (void)kleidicv_scale_f32(src.data(), kW * sizeof(float), out.data(),
                           kW * sizeof(float), kW, kH, scale, shift);
  for (size_t i = 0; i < src.size(); ++i) {
    float exp = src[i] * static_cast<float>(scale) + static_cast<float>(shift);
    if (out[i] != exp) {
      std::fprintf(stderr, "FAIL f32 i=%zu src=%g got=%g exp=%g\n", i, src[i],
                   out[i], exp);
      ++failures;
      break;
    }
  }
}

void test_null() {
  uint8_t buf[1] = {0};
  EXPECT(kleidicv_scale_u8(nullptr, 1, buf, 1, 1, 1, 1.0, 0.0) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null u8");
  float fb[1] = {0};
  EXPECT(kleidicv_scale_f32(nullptr, 4, fb, 4, 1, 1, 1.0, 0.0) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null f32");
}

}  // namespace

int main() {
  test_backend_active();
  test_u8_bulk();
  test_u8_saturate();
  test_f32_bulk();
  test_null();
  if (failures == 0) {
    std::printf("[test_scale] all checks passed\n");
    return 0;
  }
  std::printf("[test_scale] %d FAILURES\n", failures);
  return 1;
}
