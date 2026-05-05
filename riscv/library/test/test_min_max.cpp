// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
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
#define EXPECT(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); ++failures; } } while(0)

void test_backend() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_min_max] backend=%s\n", b);
  if (std::strcmp(b, std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv") != 0) std::exit(2);
}

void test_u8() {
  constexpr size_t kW = 257, kH = 3;
  std::vector<uint8_t> src(kW * kH);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i * 7);
  uint8_t mn, mx;
  EXPECT(kleidicv_min_max_u8(src.data(), kW, kW, kH, &mn, &mx) == KLEIDICV_OK,
         "err");
  uint8_t emn = 255, emx = 0;
  for (auto v : src) { if (v < emn) emn = v; if (v > emx) emx = v; }
  EXPECT(mn == emn, "u8 min");
  EXPECT(mx == emx, "u8 max");
}

void test_s16() {
  constexpr size_t kW = 257, kH = 3;
  std::vector<int16_t> src(kW * kH);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<int16_t>(static_cast<int>(i) * 13 - 5000);
  int16_t mn, mx;
  EXPECT(kleidicv_min_max_s16(src.data(), kW * 2, kW, kH, &mn, &mx) ==
             KLEIDICV_OK,
         "err");
  int16_t emn = std::numeric_limits<int16_t>::max();
  int16_t emx = std::numeric_limits<int16_t>::min();
  for (auto v : src) { if (v < emn) emn = v; if (v > emx) emx = v; }
  EXPECT(mn == emn, "s16 min");
  EXPECT(mx == emx, "s16 max");
}

void test_optional_outs() {
  uint8_t src[4] = {10, 5, 20, 15};
  uint8_t mn = 0, mx = 0;
  EXPECT(kleidicv_min_max_u8(src, 4, 4, 1, &mn, nullptr) == KLEIDICV_OK,
         "min only");
  EXPECT(mn == 5, "min only val");
  EXPECT(kleidicv_min_max_u8(src, 4, 4, 1, nullptr, &mx) == KLEIDICV_OK,
         "max only");
  EXPECT(mx == 20, "max only val");
}

void test_null() {
  uint8_t mn, mx;
  EXPECT(kleidicv_min_max_u8(nullptr, 1, 1, 1, &mn, &mx) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null");
}

}  // namespace

int main() {
  test_backend();
  test_u8();
  test_s16();
  test_optional_outs();
  test_null();
  if (failures == 0) { std::printf("[test_min_max] all checks passed\n"); return 0; }
  return 1;
}
