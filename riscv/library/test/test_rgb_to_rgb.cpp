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
  std::printf("[test_rgb_to_rgb] backend=%s\n", b);
  if (std::strcmp(b, std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv") != 0) std::exit(2);
}

void test_rgb_to_bgr() {
  constexpr size_t kW = 65;
  std::vector<uint8_t> in(kW * 3), out(kW * 3);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i);
  (void)kleidicv_rgb_to_bgr_u8(in.data(), kW * 3, out.data(), kW * 3, kW, 1);
  for (size_t x = 0; x < kW; ++x) {
    EXPECT(out[3*x+0] == in[3*x+2], "B");
    EXPECT(out[3*x+1] == in[3*x+1], "G");
    EXPECT(out[3*x+2] == in[3*x+0], "R");
  }
}

void test_rgb_to_rgba() {
  constexpr size_t kW = 65;
  std::vector<uint8_t> in(kW * 3), out(kW * 4);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i);
  (void)kleidicv_rgb_to_rgba_u8(in.data(), kW * 3, out.data(), kW * 4, kW, 1);
  for (size_t x = 0; x < kW; ++x) {
    EXPECT(out[4*x+0] == in[3*x+0], "R");
    EXPECT(out[4*x+1] == in[3*x+1], "G");
    EXPECT(out[4*x+2] == in[3*x+2], "B");
    EXPECT(out[4*x+3] == 0xFF, "A=FF");
  }
}

void test_rgba_to_bgr() {
  constexpr size_t kW = 65;
  std::vector<uint8_t> in(kW * 4), out(kW * 3);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i);
  (void)kleidicv_rgba_to_bgr_u8(in.data(), kW * 4, out.data(), kW * 3, kW, 1);
  for (size_t x = 0; x < kW; ++x) {
    EXPECT(out[3*x+0] == in[4*x+2], "B");
    EXPECT(out[3*x+1] == in[4*x+1], "G");
    EXPECT(out[3*x+2] == in[4*x+0], "R");
  }
}

void test_rgba_to_bgra() {
  constexpr size_t kW = 65;
  std::vector<uint8_t> in(kW * 4), out(kW * 4);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i);
  (void)kleidicv_rgba_to_bgra_u8(in.data(), kW * 4, out.data(), kW * 4, kW, 1);
  for (size_t x = 0; x < kW; ++x) {
    EXPECT(out[4*x+0] == in[4*x+2], "B");
    EXPECT(out[4*x+1] == in[4*x+1], "G");
    EXPECT(out[4*x+2] == in[4*x+0], "R");
    EXPECT(out[4*x+3] == in[4*x+3], "A");
  }
}

// Round 6 review #58: the four variants below were wired but never tested.
// Identity ops (rgb_to_rgb, rgba_to_rgba) just verify pass-through; the
// other two cover the remaining channel-reorder shapes.
void test_rgb_to_rgb() {
  constexpr size_t W = 33;
  std::vector<uint8_t> in(W * 3), out(W * 3, 0);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 11);
  EXPECT(kleidicv_rgb_to_rgb_u8(in.data(), W * 3, out.data(), W * 3, W, 1) ==
             KLEIDICV_OK,
         "rgb_to_rgb err");
  EXPECT(std::memcmp(in.data(), out.data(), W * 3) == 0, "rgb_to_rgb identity");
}
void test_rgba_to_rgba() {
  constexpr size_t W = 33;
  std::vector<uint8_t> in(W * 4), out(W * 4, 0);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 13);
  EXPECT(kleidicv_rgba_to_rgba_u8(in.data(), W * 4, out.data(), W * 4, W, 1) ==
             KLEIDICV_OK,
         "rgba_to_rgba err");
  EXPECT(std::memcmp(in.data(), out.data(), W * 4) == 0,
         "rgba_to_rgba identity");
}
void test_rgb_to_bgra() {
  constexpr size_t W = 21;
  std::vector<uint8_t> in(W * 3), out(W * 4);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 17);
  EXPECT(kleidicv_rgb_to_bgra_u8(in.data(), W * 3, out.data(), W * 4, W, 1) ==
             KLEIDICV_OK,
         "rgb_to_bgra err");
  for (size_t x = 0; x < W; ++x) {
    EXPECT(out[4 * x + 0] == in[3 * x + 2], "B");
    EXPECT(out[4 * x + 1] == in[3 * x + 1], "G");
    EXPECT(out[4 * x + 2] == in[3 * x + 0], "R");
    EXPECT(out[4 * x + 3] == 0xFF, "A");
  }
}
void test_rgba_to_rgb() {
  constexpr size_t W = 21;
  std::vector<uint8_t> in(W * 4), out(W * 3);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 23);
  EXPECT(kleidicv_rgba_to_rgb_u8(in.data(), W * 4, out.data(), W * 3, W, 1) ==
             KLEIDICV_OK,
         "rgba_to_rgb err");
  for (size_t x = 0; x < W; ++x) {
    EXPECT(out[3 * x + 0] == in[4 * x + 0], "R");
    EXPECT(out[3 * x + 1] == in[4 * x + 1], "G");
    EXPECT(out[3 * x + 2] == in[4 * x + 2], "B");
  }
}

}  // namespace

int main() {
  test_backend();
  test_rgb_to_bgr();
  test_rgb_to_rgba();
  test_rgba_to_bgr();
  test_rgba_to_bgra();
  test_rgb_to_rgb();
  test_rgba_to_rgba();
  test_rgb_to_bgra();
  test_rgba_to_rgb();
  if (failures == 0) { std::printf("[test_rgb_to_rgb] all checks passed\n"); return 0; }
  return 1;
}
