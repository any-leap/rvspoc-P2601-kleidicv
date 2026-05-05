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

void run_case(size_t width, size_t height, size_t src_pad, size_t dst_pad) {
  size_t src_stride = width + src_pad;
  size_t dst_stride = 3 * width + dst_pad;
  std::vector<uint8_t> src(src_stride * height, 0xCC);
  std::vector<uint8_t> dst(dst_stride * height, 0xAA);
  // Fill the *valid* region of src with a deterministic pattern.
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      src[y * src_stride + x] = static_cast<uint8_t>((x * 7 + y * 13) & 0xff);
    }
  }
  std::vector<uint8_t> dst_initial = dst;

  kleidicv_error_t err = kleidicv_gray_to_rgb_u8(
      src.data(), src_stride, dst.data(), dst_stride, width, height);
  EXPECT(err == KLEIDICV_OK, "returned error");

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint8_t g = src[y * src_stride + x];
      uint8_t r0 = dst[y * dst_stride + 3 * x + 0];
      uint8_t r1 = dst[y * dst_stride + 3 * x + 1];
      uint8_t r2 = dst[y * dst_stride + 3 * x + 2];
      if (r0 != g || r1 != g || r2 != g) {
        std::fprintf(stderr,
                     "FAIL pixel (x=%zu y=%zu) g=%u got=(%u,%u,%u)\n", x, y,
                     g, r0, r1, r2);
        ++failures;
        return;
      }
    }
    // Padding bytes after the last RGB triplet must be untouched.
    for (size_t p = 3 * width; p < dst_stride; ++p) {
      if (dst[y * dst_stride + p] != dst_initial[y * dst_stride + p]) {
        std::fprintf(stderr, "FAIL dst padding clobbered at row %zu byte %zu\n",
                     y, p);
        ++failures;
        return;
      }
    }
  }
}

void test_null_pointer() {
  uint8_t buf[16] = {0};
  kleidicv_error_t err =
      kleidicv_gray_to_rgb_u8(nullptr, 1, buf, 3, 1, 1);
  EXPECT(err == KLEIDICV_ERROR_NULL_POINTER, "null check missing");
}

void test_backend_active() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_gray_to_rgb] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) {
    std::fprintf(stderr, "FAIL backend mismatch: got %s expected %s\n", b,
                 expected);
    std::exit(2);
  }
}

// In-place: src and dst point at the same buffer, gray pixels are at the
// leftmost width bytes, and after the call the same buffer holds 3·width
// RGB bytes (each gray replicated three times). Right-to-left expansion in
// the scalar impl is what makes this work.
void test_in_place() {
  constexpr size_t W = 17;
  std::vector<uint8_t> buf(W * 3, 0);
  for (size_t i = 0; i < W; ++i) buf[i] = static_cast<uint8_t>(i * 13 + 5);
  std::vector<uint8_t> expected(W * 3);
  for (size_t i = 0; i < W; ++i) {
    uint8_t g = static_cast<uint8_t>(i * 13 + 5);
    expected[3 * i + 0] = g;
    expected[3 * i + 1] = g;
    expected[3 * i + 2] = g;
  }
  if (kleidicv_gray_to_rgb_u8(buf.data(), W, buf.data(), W * 3, W, 1) !=
      KLEIDICV_OK) {
    std::fprintf(stderr, "FAIL gray_to_rgb_u8 in-place returned error\n");
    std::exit(1);
  }
  if (std::memcmp(buf.data(), expected.data(), W * 3) != 0) {
    std::fprintf(stderr, "FAIL gray_to_rgb_u8 in-place corrupted\n");
    std::exit(1);
  }
}

}  // namespace

int main() {
  test_in_place();
  test_backend_active();
  // Single small row — fits in one vl on every plausible vlen.
  run_case(7, 1, 0, 0);
  // Multi-row with non-tight strides on both sides. 257 forces an unaligned
  // tail iteration of the strip-mined RVV loop.
  run_case(257, 5, 4, 7);
  // Wide row, no padding.
  run_case(1024, 2, 0, 0);
  test_null_pointer();
  if (failures == 0) {
    std::printf("[test_gray_to_rgb] all checks passed\n");
    return 0;
  }
  std::printf("[test_gray_to_rgb] %d FAILURES\n", failures);
  return 1;
}
