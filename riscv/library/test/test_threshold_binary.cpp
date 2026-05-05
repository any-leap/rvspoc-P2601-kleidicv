// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
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

#define EXPECT(cond, msg)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);  \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

void test_backend_active() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_threshold_binary] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

void test_bulk() {
  constexpr size_t kW = 257, kH = 3;
  std::vector<uint8_t> src(kW * kH), out(kW * kH);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i * 7 + 3);
  uint8_t threshold = 100, value = 200;
  kleidicv_error_t err = kleidicv_threshold_binary_u8(
      src.data(), kW, out.data(), kW, kW, kH, threshold, value);
  EXPECT(err == KLEIDICV_OK, "err");
  for (size_t i = 0; i < out.size(); ++i) {
    uint8_t exp = src[i] > threshold ? value : 0;
    if (out[i] != exp) {
      std::fprintf(stderr, "FAIL i=%zu src=%u got=%u exp=%u\n", i, src[i],
                   out[i], exp);
      ++failures;
      break;
    }
  }
}

void test_corner() {
  // Strict greater: equal-to-threshold becomes 0.
  uint8_t src[3] = {99, 100, 101};
  uint8_t out[3] = {0};
  (void)kleidicv_threshold_binary_u8(src, 3, out, 3, 3, 1, 100, 250);
  EXPECT(out[0] == 0 && out[1] == 0 && out[2] == 250, "strict greater");
}

void test_null() {
  uint8_t buf[1] = {0};
  EXPECT(kleidicv_threshold_binary_u8(nullptr, 1, buf, 1, 1, 1, 0, 1) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null");
}

}  // namespace

int main() {
  test_backend_active();
  test_bulk();
  test_corner();
  test_null();
  if (failures == 0) {
    std::printf("[test_threshold_binary] all checks passed\n");
    return 0;
  }
  std::printf("[test_threshold_binary] %d FAILURES\n", failures);
  return 1;
}
