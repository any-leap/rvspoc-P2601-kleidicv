// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
// SPDX-License-Identifier: Apache-2.0
//
// Combined smoke test for sobel, scharr, blur_and_downsample, median_blur,
// plus the NOT_IMPLEMENTED stubs for separable_filter_2d / gaussian_blur.

#include <cmath>
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
  std::printf("[test_filters] backend=%s\n", b);
  if (std::strcmp(b, std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv") != 0) std::exit(2);
}

// ---- sobel ----

uint8_t fetch(const std::vector<uint8_t> &src, size_t w, size_t h, ptrdiff_t y,
              ptrdiff_t x) {
  if (y < 0) y = 0; if ((size_t)y >= h) y = h - 1;
  if (x < 0) x = 0; if ((size_t)x >= w) x = w - 1;
  return src[y * w + x];
}

void test_sobel() {
  constexpr size_t W = 17, H = 5;
  std::vector<uint8_t> src(W * H);
  std::vector<int16_t> dx(W * H), dy(W * H);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i * 7 + 11);
  EXPECT(kleidicv_sobel_3x3_horizontal_s16_u8(src.data(), W, dx.data(), W * 2,
                                              W, H, 1) == KLEIDICV_OK,
         "h err");
  EXPECT(kleidicv_sobel_3x3_vertical_s16_u8(src.data(), W, dy.data(), W * 2, W,
                                            H, 1) == KLEIDICV_OK,
         "v err");
  // Compare a few interior pixels to scalar reference.
  auto ref_h = [&](size_t y, size_t x) {
    int s = 0;
    for (int yy = -1; yy <= 1; ++yy) {
      int wy = yy == 0 ? 2 : 1;
      s += wy * (-fetch(src, W, H, (ptrdiff_t)y + yy, (ptrdiff_t)x - 1) +
                 fetch(src, W, H, (ptrdiff_t)y + yy, (ptrdiff_t)x + 1));
    }
    return static_cast<int16_t>(s);
  };
  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      EXPECT(dx[y * W + x] == ref_h(y, x), "sobel-h match");
    }
  }
  // multi-channel: deinterleave→sobel-per-plane→reinterleave. Verify
  // bit-equality against running channels=1 sobel on each interleaved plane.
  constexpr size_t MW = 8, MH = 4;
  std::vector<uint8_t> mc_src(MW * MH * 3);
  for (size_t i = 0; i < mc_src.size(); ++i)
    mc_src[i] = static_cast<uint8_t>(i * 19 + 5);
  std::vector<int16_t> mc_got(MW * MH * 3, 0), mc_ref(MW * MH * 3, 0);
  EXPECT(kleidicv_sobel_3x3_horizontal_s16_u8(mc_src.data(), MW * 3,
                                                mc_got.data(),
                                                MW * 3 * sizeof(int16_t), MW,
                                                MH, 3) == KLEIDICV_OK,
         "sobel mc channels=3 ok");
  // Reference: extract each channel into a planar W*H buffer, run channels=1
  // sobel on it, then re-interleave.
  for (size_t c = 0; c < 3; ++c) {
    std::vector<uint8_t> plane(MW * MH);
    for (size_t y = 0; y < MH; ++y)
      for (size_t x = 0; x < MW; ++x)
        plane[y * MW + x] = mc_src[(y * MW + x) * 3 + c];
    std::vector<int16_t> plane_dst(MW * MH);
    kleidicv_sobel_3x3_horizontal_s16_u8(plane.data(), MW, plane_dst.data(),
                                            MW * sizeof(int16_t), MW, MH, 1);
    for (size_t y = 0; y < MH; ++y)
      for (size_t x = 0; x < MW; ++x)
        mc_ref[(y * MW + x) * 3 + c] = plane_dst[y * MW + x];
  }
  if (std::memcmp(mc_got.data(), mc_ref.data(),
                   mc_got.size() * sizeof(int16_t)) != 0) {
    std::fprintf(stderr, "FAIL sobel mc mismatch\n");
    ++failures;
  }
}

// ---- scharr ----

void test_scharr() {
  constexpr size_t W = 9, H = 4;
  std::vector<uint8_t> src(W * H);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i * 13 + 3);
  size_t out_w = W - 2, out_h = H - 2;
  std::vector<int16_t> dst(out_w * out_h * 2);
  EXPECT(kleidicv_scharr_interleaved_s16_u8(src.data(), W, W, H, 1, dst.data(),
                                            out_w * 4) == KLEIDICV_OK,
         "scharr err");
  // Verify pixel (0,0) of output (== source pixel (1,1))
  auto S = [&](size_t y, size_t x) -> int { return src[y * W + x]; };
  int dx_exp = 3 * (S(0, 2) - S(0, 0) + S(2, 2) - S(2, 0)) +
               10 * (S(1, 2) - S(1, 0));
  int dy_exp = 3 * (S(2, 0) + S(2, 2) - S(0, 0) - S(0, 2)) +
               10 * (S(2, 1) - S(0, 1));
  EXPECT(dst[0] == dx_exp, "scharr dx@0");
  EXPECT(dst[1] == dy_exp, "scharr dy@0");
}

// ---- blur_and_downsample ----

void test_blur_downsample() {
  constexpr size_t W = 10, H = 10;
  std::vector<uint8_t> src(W * H);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i & 0xff);
  size_t dW = (W + 1) / 2, dH = (H + 1) / 2;
  std::vector<uint8_t> dst(dW * dH);
  EXPECT(kleidicv_blur_and_downsample_u8(src.data(), W, W, H, dst.data(), dW,
                                         1, KLEIDICV_BORDER_TYPE_REPLICATE) ==
             KLEIDICV_OK,
         "blur-down err");
  // Verify dst[0][0] manually: source center (0,0), 5x5 with replicate
  // border. Kernel [1,4,6,4,1] both axes, normalize /256.
  const int K[5] = {1, 4, 6, 4, 1};
  int acc = 0;
  for (int yy = -2; yy <= 2; ++yy) {
    for (int xx = -2; xx <= 2; ++xx) {
      ptrdiff_t sy = std::max<ptrdiff_t>(0, yy);
      ptrdiff_t sx = std::max<ptrdiff_t>(0, xx);
      acc += K[yy + 2] * K[xx + 2] * src[sy * W + sx];
    }
  }
  uint8_t exp = static_cast<uint8_t>((acc + 128) >> 8);
  EXPECT(dst[0] == exp, "blur-down dst[0][0]");
  // unsupported border returns NOT_IMPL
  EXPECT(kleidicv_blur_and_downsample_u8(src.data(), W, W, H, dst.data(), dW,
                                         1, KLEIDICV_BORDER_TYPE_REFLECT) ==
             KLEIDICV_ERROR_NOT_IMPLEMENTED,
         "reflect should be NOT_IMPL");
}

// ---- median ----

void test_median() {
  // 5x5 image with one obvious salt-and-pepper noise pixel.
  uint8_t src[25] = {
      10, 10, 10, 10, 10,
      10, 50, 10, 10, 10,
      10, 10, 255, 10, 10,
      10, 10, 10, 10, 10,
      10, 10, 10, 10, 10,
  };
  uint8_t dst[25] = {0};
  EXPECT(kleidicv_median_blur_u8(src, 5, dst, 5, 5, 5, 1, 3, 3,
                                 KLEIDICV_BORDER_TYPE_REPLICATE) ==
             KLEIDICV_OK,
         "median err");
  // Center pixel (2,2): 9 neighbours = {50,10,10,10,255,10,10,10,10} sorted
  // → median is 10.
  EXPECT(dst[12] == 10, "median removes salt");
  // 5x5 kernel returns NOT_IMPL
  EXPECT(kleidicv_median_blur_u8(src, 5, dst, 5, 5, 5, 1, 5, 5,
                                 KLEIDICV_BORDER_TYPE_REPLICATE) ==
             KLEIDICV_ERROR_NOT_IMPLEMENTED,
         "5x5 should be NOT_IMPL");
}

// ---- separable_filter_2d / gaussian_blur (stubs) ----

void test_stubs_return_not_implemented() {
  uint8_t b[1] = {0};
  uint8_t k[1] = {1};
  EXPECT(kleidicv_separable_filter_2d_u8(b, 1, b, 1, 1, 1, 1, k, 1, k, 1,
                                         KLEIDICV_BORDER_TYPE_REPLICATE) ==
             KLEIDICV_ERROR_NOT_IMPLEMENTED,
         "separable u8 stub");
  EXPECT(kleidicv_gaussian_blur_u8(b, 1, b, 1, 1, 1, 1, 3, 3, 1.0f, 1.0f,
                                   KLEIDICV_BORDER_TYPE_REPLICATE) ==
             KLEIDICV_ERROR_NOT_IMPLEMENTED,
         "gaussian stub");
}

// Verifies that a multi-channel filter result equals running the same
// channels=1 kernel separately on each plane and re-interleaving.
template <typename SrcT, typename DstT, typename McCall, typename ScCall>
void check_mc_equals_per_plane(McCall mc, ScCall sc, size_t W, size_t H,
                                 size_t channels, size_t out_w, size_t out_h,
                                 size_t out_per_pixel, const char *tag) {
  std::vector<SrcT> src(W * H * channels);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<SrcT>(i * 41 + 7);
  std::vector<DstT> got(out_w * out_h * channels * out_per_pixel, 0);
  std::vector<DstT> ref(out_w * out_h * channels * out_per_pixel, 0);
  EXPECT(mc(src.data(), got.data(), W, H, channels) == KLEIDICV_OK, tag);
  for (size_t c = 0; c < channels; ++c) {
    std::vector<SrcT> plane(W * H);
    for (size_t y = 0; y < H; ++y)
      for (size_t x = 0; x < W; ++x)
        plane[y * W + x] = src[(y * W + x) * channels + c];
    std::vector<DstT> plane_out(out_w * out_h * out_per_pixel);
    sc(plane.data(), plane_out.data(), W, H);
    for (size_t y = 0; y < out_h; ++y)
      for (size_t x = 0; x < out_w; ++x)
        for (size_t k = 0; k < out_per_pixel; ++k)
          ref[(y * out_w + x) * channels * out_per_pixel +
              c * out_per_pixel + k] =
              plane_out[(y * out_w + x) * out_per_pixel + k];
  }
  if (std::memcmp(got.data(), ref.data(), got.size() * sizeof(DstT)) != 0) {
    std::fprintf(stderr, "FAIL mc mismatch (%s)\n", tag);
    ++failures;
  }
}

void test_multichannel_filters() {
  // Each lambda matches the (mc, sc) callback shape expected above.
  // separable_filter_2d_u8 5x5 box.
  uint8_t kx[5] = {1, 2, 3, 2, 1};
  uint8_t ky[5] = {1, 1, 1, 1, 1};
  check_mc_equals_per_plane<uint8_t, uint8_t>(
      [&](const uint8_t *s, uint8_t *d, size_t W, size_t H, size_t C) {
        return kleidicv_separable_filter_2d_u8(s, W * C, d, W * C, W, H, C,
                                                  kx, 5, ky, 5,
                                                  KLEIDICV_BORDER_TYPE_REPLICATE);
      },
      [&](const uint8_t *s, uint8_t *d, size_t W, size_t H) {
        kleidicv_separable_filter_2d_u8(s, W, d, W, W, H, 1, kx, 5, ky, 5,
                                          KLEIDICV_BORDER_TYPE_REPLICATE);
      },
      16, 12, 3, 16, 12, 1, "separable_filter_2d_u8 c=3");

  // gaussian_blur_u8 3x3 binomial.
  check_mc_equals_per_plane<uint8_t, uint8_t>(
      [](const uint8_t *s, uint8_t *d, size_t W, size_t H, size_t C) {
        return kleidicv_gaussian_blur_u8(s, W * C, d, W * C, W, H, C, 3, 3,
                                          0.0F, 0.0F,
                                          KLEIDICV_BORDER_TYPE_REPLICATE);
      },
      [](const uint8_t *s, uint8_t *d, size_t W, size_t H) {
        kleidicv_gaussian_blur_u8(s, W, d, W, W, H, 1, 3, 3, 0.0F, 0.0F,
                                    KLEIDICV_BORDER_TYPE_REPLICATE);
      },
      14, 10, 4, 14, 10, 1, "gaussian_blur_u8 c=4");

  // blur_and_downsample_u8.
  check_mc_equals_per_plane<uint8_t, uint8_t>(
      [](const uint8_t *s, uint8_t *d, size_t W, size_t H, size_t C) {
        return kleidicv_blur_and_downsample_u8(
            s, W * C, W, H, d, ((W + 1) / 2) * C, C,
            KLEIDICV_BORDER_TYPE_REPLICATE);
      },
      [](const uint8_t *s, uint8_t *d, size_t W, size_t H) {
        kleidicv_blur_and_downsample_u8(s, W, W, H, d, (W + 1) / 2, 1,
                                          KLEIDICV_BORDER_TYPE_REPLICATE);
      },
      20, 16, 3, 10, 8, 1, "blur_and_downsample_u8 c=3");

  // morph (dilate) u8.
  check_mc_equals_per_plane<uint8_t, uint8_t>(
      [](const uint8_t *s, uint8_t *d, size_t W, size_t H, size_t C) {
        return kleidicv_dilate_u8(s, W * C, d, W * C, W, H, C, 3, 3, 1, 1,
                                    KLEIDICV_BORDER_TYPE_REPLICATE, nullptr,
                                    1);
      },
      [](const uint8_t *s, uint8_t *d, size_t W, size_t H) {
        kleidicv_dilate_u8(s, W, d, W, W, H, 1, 3, 3, 1, 1,
                              KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1);
      },
      12, 9, 2, 12, 9, 1, "dilate_u8 c=2");

  // median_blur_u8 3x3.
  check_mc_equals_per_plane<uint8_t, uint8_t>(
      [](const uint8_t *s, uint8_t *d, size_t W, size_t H, size_t C) {
        return kleidicv_median_blur_u8(s, W * C, d, W * C, W, H, C, 3, 3,
                                          KLEIDICV_BORDER_TYPE_REPLICATE);
      },
      [](const uint8_t *s, uint8_t *d, size_t W, size_t H) {
        kleidicv_median_blur_u8(s, W, d, W, W, H, 1, 3, 3,
                                  KLEIDICV_BORDER_TYPE_REPLICATE);
      },
      11, 8, 3, 11, 8, 1, "median_blur_u8 c=3");
}

}  // namespace

int main() {
  test_backend();
  test_sobel();
  test_scharr();
  test_blur_downsample();
  test_median();
  test_multichannel_filters();
  test_stubs_return_not_implemented();
  if (failures == 0) { std::printf("[test_filters] all checks passed\n"); return 0; }
  return 1;
}
