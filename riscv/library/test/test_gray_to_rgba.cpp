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

void test_in_place_multi_row() {
  // src == dst with the canonical 4× expanded layout. Every gray byte must
  // come back as (g, g, g, 0xFF) at the corresponding RGBA slot, even though
  // expanding row 0 would clobber row 1's gray bytes if the impl wasn't
  // processing rows bottom-up.
  constexpr size_t W = 19, H = 4;
  std::vector<uint8_t> buf(W * 4 * H, 0);
  for (size_t y = 0; y < H; ++y)
    for (size_t x = 0; x < W; ++x)
      buf[y * W + x] = static_cast<uint8_t>((y * 41 + x * 11 + 3) & 0xff);
  std::vector<uint8_t> exp(W * 4 * H);
  for (size_t y = 0; y < H; ++y)
    for (size_t x = 0; x < W; ++x) {
      uint8_t g = static_cast<uint8_t>((y * 41 + x * 11 + 3) & 0xff);
      exp[y * W * 4 + 4 * x + 0] = g;
      exp[y * W * 4 + 4 * x + 1] = g;
      exp[y * W * 4 + 4 * x + 2] = g;
      exp[y * W * 4 + 4 * x + 3] = 0xFF;
    }
  EXPECT(kleidicv_gray_to_rgba_u8(buf.data(), W, buf.data(), W * 4, W, H) ==
             KLEIDICV_OK,
         "gray_to_rgba in-place ok");
  if (std::memcmp(buf.data(), exp.data(), W * 4 * H) != 0) {
    std::fprintf(stderr, "FAIL gray_to_rgba multi-row in-place\n");
    ++failures;
  }
}

}  // namespace

int main() {
  test_backend();
  test_bulk();
  test_null();
  test_in_place_multi_row();
  if (failures == 0) {
    std::printf("[test_gray_to_rgba] all checks passed\n");
    return 0;
  }
  return 1;
}
