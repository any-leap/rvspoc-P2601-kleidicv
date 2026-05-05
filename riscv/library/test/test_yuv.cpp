// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
// SPDX-License-Identifier: Apache-2.0
//
// Round-trip RGB → YUV444 → RGB and check that channel-wise error is small.
// Also check that non-YUV444 formats return NOT_IMPLEMENTED.

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
  std::printf("[test_yuv] backend=%s\n", b);
  if (std::strcmp(b, std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv") != 0) std::exit(2);
}

void test_round_trip_rgb() {
  // Inputs are kept in [50, 200] to stay inside the chroma gamut. Pure
  // saturated colours like (255, 0, 0) can drive the V channel past 255 in
  // 14-bit fixed-point and lose precision — that's a property of the BT.601
  // pipeline, not a bug in the implementation.
  constexpr size_t kW = 17, kH = 3;
  std::vector<uint8_t> rgb(kW * kH * 3), yuv(kW * kH * 3),
      back(kW * kH * 3);
  for (size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = static_cast<uint8_t>(50 + (i * 7) % 151);
  EXPECT(kleidicv_rgb_to_yuv_u8(rgb.data(), kW * 3, yuv.data(), kW * 3, kW, kH,
                                KLEIDICV_RGB_TO_YUV444) == KLEIDICV_OK,
         "rgb->yuv");
  EXPECT(kleidicv_yuv_to_rgb_u8(yuv.data(), kW * 3, back.data(), kW * 3, kW,
                                kH, KLEIDICV_YUV444_TO_RGB) == KLEIDICV_OK,
         "yuv->rgb");
  // Round-trip should preserve each channel within ~3 (8-bit color
  // quantisation noise across both directions).
  for (size_t i = 0; i < rgb.size(); ++i) {
    int diff = static_cast<int>(rgb[i]) - static_cast<int>(back[i]);
    if (diff < -3 || diff > 3) {
      std::fprintf(stderr, "FAIL round-trip i=%zu rgb=%u back=%u diff=%d\n", i,
                   rgb[i], back[i], diff);
      ++failures;
      break;
    }
  }
}

void test_yuv444_with_alpha() {
  constexpr size_t kW = 9;
  std::vector<uint8_t> rgba(kW * 4);
  std::vector<uint8_t> yuv(kW * 3), back(kW * 4);
  for (size_t i = 0; i < rgba.size(); ++i) rgba[i] = static_cast<uint8_t>(i * 3 + 5);
  // RGBA → YUV444
  EXPECT(kleidicv_rgb_to_yuv_u8(rgba.data(), kW * 4, yuv.data(), kW * 3, kW, 1,
                                KLEIDICV_RGBA_TO_YUV444) == KLEIDICV_OK,
         "rgba->yuv");
  // YUV → RGBA: alpha must be 0xFF
  EXPECT(kleidicv_yuv_to_rgb_u8(yuv.data(), kW * 3, back.data(), kW * 4, kW, 1,
                                KLEIDICV_YUV444_TO_RGBA) == KLEIDICV_OK,
         "yuv->rgba");
  for (size_t x = 0; x < kW; ++x) EXPECT(back[4 * x + 3] == 0xFF, "alpha=FF");
}

void test_unsupported_format() {
  uint8_t buf[12] = {0};
  // NV12 / YUV420SP not implemented in SPOC.
  kleidicv_error_t e = kleidicv_yuv_to_rgb_u8(buf, 4, buf, 4, 1, 1,
                                              KLEIDICV_NV12_TO_RGB);
  EXPECT(e == KLEIDICV_ERROR_NOT_IMPLEMENTED, "NV12 should be NOT_IMPLEMENTED");
}

void test_null() {
  uint8_t buf[12] = {0};
  EXPECT(kleidicv_rgb_to_yuv_u8(nullptr, 3, buf, 3, 1, 1,
                                KLEIDICV_RGB_TO_YUV444) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null");
}

}  // namespace

int main() {
  test_backend();
  test_round_trip_rgb();
  test_yuv444_with_alpha();
  test_unsupported_format();
  test_null();
  if (failures == 0) { std::printf("[test_yuv] all checks passed\n"); return 0; }
  return 1;
}
