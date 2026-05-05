// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
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
#define EXPECT(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); ++failures; } } while(0)

void test_backend() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_float_conv] backend=%s\n", b);
  if (std::strcmp(b, std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv") != 0) std::exit(2);
}

uint8_t ref_f32_to_u8(float v) {
  long iv = std::lrintf(v);
  if (iv < 0) return 0;
  if (iv > 255) return 255;
  return static_cast<uint8_t>(iv);
}

int8_t ref_f32_to_s8(float v) {
  long iv = std::lrintf(v);
  if (iv < -128) return -128;
  if (iv > 127) return 127;
  return static_cast<int8_t>(iv);
}

void test_f32_to_u8() {
  constexpr size_t kW = 65;
  std::vector<float> src(kW);
  std::vector<uint8_t> out(kW);
  for (size_t i = 0; i < kW; ++i) src[i] = -50.0f + static_cast<float>(i) * 5.0f;
  (void)kleidicv_f32_to_u8(src.data(), kW * 4, out.data(), kW, kW, 1);
  for (size_t i = 0; i < kW; ++i) {
    uint8_t exp = ref_f32_to_u8(src[i]);
    if (out[i] != exp) {
      std::fprintf(stderr, "FAIL u8 i=%zu src=%g got=%u exp=%u\n", i, src[i], out[i], exp);
      ++failures; break;
    }
  }
}

void test_f32_to_s8() {
  constexpr size_t kW = 65;
  std::vector<float> src(kW);
  std::vector<int8_t> out(kW);
  for (size_t i = 0; i < kW; ++i) src[i] = -200.0f + static_cast<float>(i) * 7.0f;
  (void)kleidicv_f32_to_s8(src.data(), kW * 4, out.data(), kW, kW, 1);
  for (size_t i = 0; i < kW; ++i) {
    int8_t exp = ref_f32_to_s8(src[i]);
    if (out[i] != exp) {
      std::fprintf(stderr, "FAIL s8 i=%zu src=%g got=%d exp=%d\n", i, src[i], out[i], exp);
      ++failures; break;
    }
  }
}

void test_u8_to_f32() {
  constexpr size_t kW = 65;
  std::vector<uint8_t> src(kW);
  std::vector<float> out(kW);
  for (size_t i = 0; i < kW; ++i) src[i] = static_cast<uint8_t>(i * 3 + 7);
  (void)kleidicv_u8_to_f32(src.data(), kW, out.data(), kW * 4, kW, 1);
  for (size_t i = 0; i < kW; ++i)
    EXPECT(out[i] == static_cast<float>(src[i]), "u8->f32");
}

void test_s8_to_f32() {
  constexpr size_t kW = 65;
  std::vector<int8_t> src(kW);
  std::vector<float> out(kW);
  for (size_t i = 0; i < kW; ++i) src[i] = static_cast<int8_t>(i * 5 - 100);
  (void)kleidicv_s8_to_f32(src.data(), kW, out.data(), kW * 4, kW, 1);
  for (size_t i = 0; i < kW; ++i)
    EXPECT(out[i] == static_cast<float>(src[i]), "s8->f32");
}

}  // namespace

int main() {
  test_backend();
  test_f32_to_u8();
  test_f32_to_s8();
  test_u8_to_f32();
  test_s8_to_f32();
  if (failures == 0) { std::printf("[test_float_conv] all checks passed\n"); return 0; }
  return 1;
}
