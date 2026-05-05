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
  std::printf("[test_split_merge] backend=%s\n", b);
  if (std::strcmp(b, std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv") != 0) std::exit(2);
}

template <size_t CH, size_t ES>
void test_split_merge() {
  constexpr size_t kW = 65, kH = 2;
  using T = uint8_t;  // bytes; multiply width by ES for storage
  std::vector<uint8_t> interleaved(kW * kH * CH * ES);
  std::vector<std::vector<uint8_t>> planes(CH);
  for (auto &p : planes) p.resize(kW * kH * ES);
  for (size_t i = 0; i < interleaved.size(); ++i)
    interleaved[i] = static_cast<uint8_t>(i * 13 + CH * 17 + ES);

  // split
  void *dsts[CH];
  size_t dst_strides[CH];
  for (size_t c = 0; c < CH; ++c) {
    dsts[c] = planes[c].data();
    dst_strides[c] = kW * ES;
  }
  EXPECT(kleidicv_split(interleaved.data(), kW * CH * ES, dsts, dst_strides,
                        kW, kH, CH, ES) == KLEIDICV_OK,
         "split err");
  // verify split
  for (size_t y = 0; y < kH; ++y) {
    for (size_t x = 0; x < kW; ++x) {
      for (size_t c = 0; c < CH; ++c) {
        for (size_t b = 0; b < ES; ++b) {
          uint8_t got = planes[c][(y * kW + x) * ES + b];
          uint8_t exp = interleaved[(y * kW + x) * CH * ES + c * ES + b];
          if (got != exp) {
            std::fprintf(stderr,
                         "FAIL split CH=%zu ES=%zu y=%zu x=%zu c=%zu b=%zu\n",
                         CH, ES, y, x, c, b);
            ++failures;
            return;
          }
        }
      }
    }
  }

  // merge back
  std::vector<uint8_t> rebuilt(interleaved.size(), 0);
  const void *srcs[CH];
  size_t src_strides[CH];
  for (size_t c = 0; c < CH; ++c) {
    srcs[c] = planes[c].data();
    src_strides[c] = kW * ES;
  }
  EXPECT(kleidicv_merge(srcs, src_strides, rebuilt.data(), kW * CH * ES, kW, kH,
                        CH, ES) == KLEIDICV_OK,
         "merge err");
  if (std::memcmp(rebuilt.data(), interleaved.data(), interleaved.size()) != 0) {
    std::fprintf(stderr, "FAIL merge round-trip CH=%zu ES=%zu\n", CH, ES);
    ++failures;
  }
}

void test_null() {
  uint8_t buf[4] = {0};
  void *d[2] = {buf, buf};
  size_t s[2] = {1, 1};
  EXPECT(kleidicv_split(nullptr, 1, d, s, 1, 1, 2, 1) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null");
}

}  // namespace

int main() {
  test_backend();
  test_split_merge<2, 1>();
  test_split_merge<3, 1>();
  test_split_merge<4, 1>();
  test_split_merge<2, 2>();
  test_split_merge<3, 2>();
  test_split_merge<4, 2>();
  test_split_merge<2, 4>();
  test_split_merge<3, 4>();
  test_split_merge<4, 4>();
  test_split_merge<2, 8>();
  test_split_merge<3, 8>();
  test_split_merge<4, 8>();
  test_null();
  if (failures == 0) { std::printf("[test_split_merge] all checks passed\n"); return 0; }
  return 1;
}
