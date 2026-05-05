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
  std::printf("[test_compare] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

void test_bulk() {
  constexpr size_t kW = 257, kH = 3;
  std::vector<uint8_t> a(kW * kH), b(kW * kH), eq(kW * kH), gt(kW * kH);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<uint8_t>(i * 5 + 7);
    b[i] = static_cast<uint8_t>(i * 5 + (i % 17 == 0 ? 7 : 8));
  }
  (void)kleidicv_compare_equal_u8(a.data(), kW, b.data(), kW, eq.data(), kW,
                                  kW, kH);
  (void)kleidicv_compare_greater_u8(a.data(), kW, b.data(), kW, gt.data(), kW,
                                    kW, kH);
  for (size_t i = 0; i < a.size(); ++i) {
    uint8_t exp_eq = a[i] == b[i] ? 0xFF : 0;
    uint8_t exp_gt = a[i] > b[i] ? 0xFF : 0;
    if (eq[i] != exp_eq || gt[i] != exp_gt) {
      std::fprintf(stderr, "FAIL i=%zu a=%u b=%u eq=%u/%u gt=%u/%u\n", i,
                   a[i], b[i], eq[i], exp_eq, gt[i], exp_gt);
      ++failures;
      break;
    }
  }
}

void test_null() {
  uint8_t buf[1] = {0};
  EXPECT(kleidicv_compare_equal_u8(nullptr, 1, buf, 1, buf, 1, 1, 1) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "eq null");
  EXPECT(kleidicv_compare_greater_u8(nullptr, 1, buf, 1, buf, 1, 1, 1) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "gt null");
}

}  // namespace

int main() {
  test_backend_active();
  test_bulk();
  test_null();
  if (failures == 0) {
    std::printf("[test_compare] all checks passed\n");
    return 0;
  }
  std::printf("[test_compare] %d FAILURES\n", failures);
  return 1;
}
