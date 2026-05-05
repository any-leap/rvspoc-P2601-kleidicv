// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
// SPDX-License-Identifier: Apache-2.0
//
// Tests for separable_filter_2d (5x5) + gaussian_blur (3x3) + remap_s16.

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
  std::printf("[test_separable] backend=%s\n", b);
  if (std::strcmp(b, std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv") != 0) std::exit(2);
}

// Apply 5x5 box filter (kernel [1,1,1,1,1]) with our impl, compare to manual.
void test_separable_5x5_box() {
  constexpr size_t W = 17, H = 5;
  std::vector<uint8_t> src(W * H), dst(W * H);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i);
  uint8_t kx[5] = {1, 1, 1, 1, 1};
  uint8_t ky[5] = {1, 1, 1, 1, 1};
  EXPECT(kleidicv_separable_filter_2d_u8(src.data(), W, dst.data(), W, W, H, 1,
                                         kx, 5, ky, 5,
                                         KLEIDICV_BORDER_TYPE_REPLICATE) ==
             KLEIDICV_OK,
         "sep err");
  // For a 5x5 box (kernel sum = 25), output saturates to 255 quickly. Just
  // sanity-check that interior pixel matches the formula.
  // Pick (y, x) = (2, 8) — fully interior.
  size_t y = 2, x = 8;
  uint32_t acc = 0;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx)
      acc += src[(y + dy) * W + (x + dx)];
  uint8_t exp = static_cast<uint8_t>(acc > 255U ? 255U : acc);
  EXPECT(dst[y * W + x] == exp, "sep 5x5 box interior");
}

void test_gaussian_3x3() {
  constexpr size_t W = 5, H = 5;
  uint8_t src[25];
  for (size_t i = 0; i < 25; ++i) src[i] = static_cast<uint8_t>(i * 10);
  uint8_t dst[25] = {0};
  EXPECT(kleidicv_gaussian_blur_u8(src, W, dst, W, W, H, 1, 3, 3, 0.0f, 0.0f,
                                   KLEIDICV_BORDER_TYPE_REPLICATE) ==
             KLEIDICV_OK,
         "gauss err");
  // Check centre pixel (2,2): kernel applied with replicate borders.
  // sum = 1*src[1,1] + 2*src[1,2] + 1*src[1,3]
  //     + 2*src[2,1] + 4*src[2,2] + 2*src[2,3]
  //     + 1*src[3,1] + 2*src[3,2] + 1*src[3,3], / 16
  int s = src[6] + 2 * src[7] + src[8] + 2 * src[11] + 4 * src[12] +
          2 * src[13] + src[16] + 2 * src[17] + src[18];
  uint8_t exp = static_cast<uint8_t>((s + 8) >> 4);
  EXPECT(dst[12] == exp, "gauss centre");
}

void test_remap_s16_basic() {
  // 4x4 src; pick out a 2x2 region (offset by 1).
  uint8_t src[16];
  for (size_t i = 0; i < 16; ++i) src[i] = static_cast<uint8_t>(i);
  int16_t mapxy[8] = {1, 1, 2, 1, 1, 2, 2, 2};  // (1,1) (2,1) (1,2) (2,2)
  uint8_t dst[4] = {0};
  EXPECT(kleidicv_remap_s16_u8(src, 4, 4, 4, dst, 2, 2, 2, 1, mapxy, 4 * 2,
                               KLEIDICV_BORDER_TYPE_REPLICATE,
                               nullptr) == KLEIDICV_OK,
         "remap err");
  EXPECT(dst[0] == src[1 * 4 + 1], "remap [0]");
  EXPECT(dst[1] == src[1 * 4 + 2], "remap [1]");
  EXPECT(dst[2] == src[2 * 4 + 1], "remap [2]");
  EXPECT(dst[3] == src[2 * 4 + 2], "remap [3]");
}

void test_remap_s16_constant_border() {
  uint8_t src[4] = {1, 2, 3, 4};
  // Map all dst to (-1, -1), (10, 10) — both out of range.
  int16_t mapxy[4] = {-1, -1, 10, 10};
  uint8_t dst[2] = {0};
  uint8_t fill = 99;
  EXPECT(kleidicv_remap_s16_u8(src, 2, 2, 2, dst, 2, 2, 1, 1, mapxy, 4,
                               KLEIDICV_BORDER_TYPE_CONSTANT,
                               &fill) == KLEIDICV_OK,
         "remap const err");
  EXPECT(dst[0] == 99 && dst[1] == 99, "remap constant fill");
}

}  // namespace

int main() {
  test_backend();
  test_separable_5x5_box();
  test_gaussian_3x3();
  test_remap_s16_basic();
  test_remap_s16_constant_border();
  if (failures == 0) { std::printf("[test_separable] all checks passed\n"); return 0; }
  return 1;
}
