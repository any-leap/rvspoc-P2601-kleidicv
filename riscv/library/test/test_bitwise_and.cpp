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
  std::printf("[test_bitwise_and] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

void test_bulk() {
  constexpr size_t kW = 257, kH = 3;
  std::vector<uint8_t> a(kW * kH), b(kW * kH), out(kW * kH);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<uint8_t>(i * 13 + 7);
    b[i] = static_cast<uint8_t>(i * i - i * 5);
  }
  kleidicv_error_t err =
      kleidicv_bitwise_and(a.data(), kW, b.data(), kW, out.data(), kW, kW, kH);
  EXPECT(err == KLEIDICV_OK, "bulk error");
  for (size_t i = 0; i < out.size(); ++i) {
    uint8_t exp = static_cast<uint8_t>(a[i] & b[i]);
    if (out[i] != exp) {
      std::fprintf(stderr, "FAIL i=%zu got=%u exp=%u\n", i, out[i], exp);
      ++failures;
      break;
    }
  }
}

void test_null_pointer() {
  uint8_t buf[4] = {0};
  EXPECT(kleidicv_bitwise_and(nullptr, 1, buf, 1, buf, 1, 1, 1) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null check");
}

void test_stride() {
  uint8_t a[6] = {0xF0, 0x0F, 0xAA, 0xFF, 0x00, 0xBB};
  uint8_t b[6] = {0xCC, 0xCC, 0x55, 0x33, 0xFF, 0xDD};
  uint8_t out[6] = {0};
  (void)kleidicv_bitwise_and(a, 3, b, 3, out, 3, 2, 2);
  EXPECT(out[0] == (0xF0 & 0xCC), "row0 col0");
  EXPECT(out[1] == (0x0F & 0xCC), "row0 col1");
  EXPECT(out[3] == (0xFF & 0x33), "row1 col0");
  EXPECT(out[4] == (0x00 & 0xFF), "row1 col1");
  EXPECT(out[2] == 0 && out[5] == 0, "padding clobbered");
}

}  // namespace

int main() {
  test_backend_active();
  test_bulk();
  test_null_pointer();
  test_stride();
  if (failures == 0) {
    std::printf("[test_bitwise_and] all checks passed\n");
    return 0;
  }
  std::printf("[test_bitwise_and] %d FAILURES\n", failures);
  return 1;
}
