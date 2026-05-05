// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for the LK optical-flow stack on RISC-V:
// - kleidicv_build_optical_flow_pyr_lk_pyramid / *_release / *_get_*
// - kleidicv_optical_flow_pyr_lk_u8 (image-to-image, builds pyramids inline)
// - kleidicv_optical_flow_pyr_lk_u8_from_pyramid (uses prebuilt pyramids)
// - kleidicv_standalone_lucas_kanade_alg_u8 (single-level primitive)
//
// Builds two images (the second is the first shifted by a known dx/dy) with
// a strong textured pattern, picks corner-like points, and asserts the
// tracker recovers the displacement to within sub-pixel tolerance.

#include <cassert>
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
#define EXPECT(cond, msg)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);  \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

void test_backend_active() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_optical_flow] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) {
    std::fprintf(stderr, "FAIL backend mismatch: got %s, expected %s\n", b,
                 expected);
    std::exit(2);
  }
}

// Synthesise a textured image: combine multiple sinusoids so the structure
// tensor has good rank everywhere (no ambiguous gradients).
void make_image(uint8_t *img, int width, int height, int stride, float dx,
                float dy) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float fx = static_cast<float>(x) - dx;
      float fy = static_cast<float>(y) - dy;
      float v = 128.0F +
                60.0F * std::sin(0.4F * fx + 0.7F * fy) +
                40.0F * std::cos(0.9F * fx - 0.3F * fy);
      int iv = static_cast<int>(v + 0.5F);
      if (iv < 0) iv = 0;
      if (iv > 255) iv = 255;
      img[y * stride + x] = static_cast<uint8_t>(iv);
    }
  }
}

void test_pyr_lk_image_to_image() {
  constexpr int W = 96, H = 96;
  std::vector<uint8_t> prev(W * H), next(W * H);
  const float TRUE_DX = 1.5F, TRUE_DY = -0.7F;
  make_image(prev.data(), W, H, W, 0, 0);
  make_image(next.data(), W, H, W, TRUE_DX, TRUE_DY);

  // Points kept well clear of the level-2 (96/4=24) border so the coarsest
  // pyramid level still has a full window around each.
  std::vector<float> prev_pts = {40, 40, 50, 50, 56, 44, 44, 56};
  std::vector<float> next_pts = prev_pts;  // initial guess = prev
  size_t n = prev_pts.size() / 2;
  std::vector<uint8_t> status(n, 0);
  std::vector<float> err(n, 0);

  kleidicv_optflow_lk_context_t ctx{};
  ctx.window_width = 15;
  ctx.window_height = 15;
  ctx.max_level = 2;
  ctx.termination_count = 30;
  ctx.termination_epsilon = 0.01F;
  ctx.min_eig_threshold = 1e-4F;
  ctx.flags = 0;

  kleidicv_error_t e = kleidicv_optical_flow_pyr_lk_u8(
      prev.data(), W, next.data(), W, W, H, 1, prev_pts.data(),
      next_pts.data(), n, status.data(), err.data(), ctx);
  if (e != KLEIDICV_OK) std::fprintf(stderr, "  pyr_lk_u8 err=%d\n", e);
  EXPECT(e == KLEIDICV_OK, "pyr_lk_u8 returned ok");

  int tracked = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!status[i]) continue;
    ++tracked;
    float estimated_dx = next_pts[i * 2] - prev_pts[i * 2];
    float estimated_dy = next_pts[i * 2 + 1] - prev_pts[i * 2 + 1];
    float ex = estimated_dx - TRUE_DX;
    float ey = estimated_dy - TRUE_DY;
    if (std::fabs(ex) > 0.5F || std::fabs(ey) > 0.5F) {
      std::fprintf(stderr,
                   "  point %zu: est=(%.3f,%.3f) truth=(%.3f,%.3f) err=(%.3f,%.3f)\n",
                   i, estimated_dx, estimated_dy, TRUE_DX, TRUE_DY, ex, ey);
      EXPECT(false, "tracked displacement within 0.5 px of truth");
    }
  }
  EXPECT(tracked >= 3, "at least 3 of 4 points tracked");
}

void test_pyr_lk_pyramid_apis() {
  constexpr int W = 80, H = 80;
  std::vector<uint8_t> img(W * H);
  make_image(img.data(), W, H, W, 0, 0);

  kleidicv_optical_flow_pyr_lk_pyramid_t *pyr = nullptr;
  kleidicv_error_t e = kleidicv_build_optical_flow_pyr_lk_pyramid(
      &pyr, img.data(), W, W, H, 1, 3, 11, 11);
  if (e != KLEIDICV_OK) std::fprintf(stderr, "  build_pyramid err=%d\n", e);
  EXPECT(e == KLEIDICV_OK, "build pyramid ok");
  EXPECT(pyr != nullptr, "pyramid handle non-null");

  size_t levels = 0;
  e = kleidicv_optical_flow_pyr_lk_pyramid_get_level_count(pyr, &levels);
  EXPECT(e == KLEIDICV_OK && levels >= 1 && levels <= 3,
         "get_level_count plausible");

  for (size_t L = 0; L < levels; ++L) {
    const uint8_t *idata = nullptr;
    size_t istride = 0, iw = 0, ih = 0;
    e = kleidicv_optical_flow_pyr_lk_pyramid_get_image_level(pyr, L, &idata,
                                                              &istride, &iw, &ih);
    EXPECT(e == KLEIDICV_OK && idata != nullptr && iw > 0 && ih > 0,
           "get_image_level ok");
    const int16_t *sdata = nullptr;
    size_t sstride = 0, sw = 0, sh = 0;
    e = kleidicv_optical_flow_pyr_lk_pyramid_get_scharr_level(pyr, L, &sdata,
                                                               &sstride, &sw,
                                                               &sh);
    EXPECT(e == KLEIDICV_OK && sdata != nullptr && sw == iw && sh == ih,
           "get_scharr_level ok and matches image dims");
  }

  e = kleidicv_optical_flow_pyr_lk_pyramid_release(pyr);
  EXPECT(e == KLEIDICV_OK, "pyramid release ok");
}

void test_pyr_lk_from_pyramid() {
  constexpr int W = 96, H = 96;
  std::vector<uint8_t> prev(W * H), next(W * H);
  const float TRUE_DX = 0.9F, TRUE_DY = 1.2F;
  make_image(prev.data(), W, H, W, 0, 0);
  make_image(next.data(), W, H, W, TRUE_DX, TRUE_DY);

  kleidicv_optical_flow_pyr_lk_pyramid_t *pp = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t *np = nullptr;
  EXPECT(kleidicv_build_optical_flow_pyr_lk_pyramid(&pp, prev.data(), W, W, H,
                                                     1, 3, 15, 15) ==
             KLEIDICV_OK,
         "prev pyramid built");
  EXPECT(kleidicv_build_optical_flow_pyr_lk_pyramid(&np, next.data(), W, W, H,
                                                     1, 3, 15, 15) ==
             KLEIDICV_OK,
         "next pyramid built");

  std::vector<float> prev_pts = {40, 40, 50, 30};
  std::vector<float> next_pts = prev_pts;
  size_t n = prev_pts.size() / 2;
  std::vector<uint8_t> status(n, 0);
  std::vector<float> err(n, 0);

  kleidicv_optflow_lk_context_t ctx{};
  ctx.window_width = 15;
  ctx.window_height = 15;
  ctx.max_level = 2;
  ctx.termination_count = 30;
  ctx.termination_epsilon = 0.01F;
  ctx.min_eig_threshold = 1e-4F;
  ctx.flags = 0;

  kleidicv_error_t e = kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
      pp, np, prev_pts.data(), next_pts.data(), n, status.data(), err.data(),
      ctx);
  EXPECT(e == KLEIDICV_OK, "from_pyramid ok");
  for (size_t i = 0; i < n; ++i) {
    if (!status[i]) continue;
    float dx = next_pts[i * 2] - prev_pts[i * 2];
    float dy = next_pts[i * 2 + 1] - prev_pts[i * 2 + 1];
    EXPECT(std::fabs(dx - TRUE_DX) < 0.5F && std::fabs(dy - TRUE_DY) < 0.5F,
           "from_pyramid displacement within 0.5 px");
  }

  kleidicv_optical_flow_pyr_lk_pyramid_release(pp);
  kleidicv_optical_flow_pyr_lk_pyramid_release(np);
}

void test_standalone_lk() {
  // standalone_lucas_kanade_alg_u8: single level, no pyramid. We hand it
  // precomputed Scharr derivatives via the public scharr API and run one
  // iteration batch.
  constexpr int W = 96, H = 96;
  std::vector<uint8_t> prev(W * H), next(W * H);
  const float TRUE_DX = 0.6F, TRUE_DY = -0.4F;
  make_image(prev.data(), W, H, W, 0, 0);
  make_image(next.data(), W, H, W, TRUE_DX, TRUE_DY);

  std::vector<int16_t> scharr(W * H * 2);
  size_t scharr_stride_bytes = W * 2 * sizeof(int16_t);
  kleidicv_error_t e = kleidicv_scharr_interleaved_s16_u8(
      prev.data(), W, W, H, 1, scharr.data(), scharr_stride_bytes);
  EXPECT(e == KLEIDICV_OK, "scharr ok");

  std::vector<float> prev_pts = {48, 48, 32, 60};
  std::vector<float> next_pts = prev_pts;
  size_t n = prev_pts.size() / 2;
  std::vector<uint8_t> status(n, 1);
  std::vector<float> err(n, 0);

  e = kleidicv_standalone_lucas_kanade_alg_u8(
      prev.data(), W, scharr.data(), scharr_stride_bytes, next.data(), W, W,
      H, 1, prev_pts.data(), next_pts.data(), n, status.data(), err.data(),
      11, 11, 30, 0.0001, false, 1e-4F);
  EXPECT(e == KLEIDICV_OK, "standalone_lk ok");

  for (size_t i = 0; i < n; ++i) {
    if (!status[i]) continue;
    float dx = next_pts[i * 2] - prev_pts[i * 2];
    float dy = next_pts[i * 2 + 1] - prev_pts[i * 2 + 1];
    // Single-level only — sub-pixel ground truth converges within ~0.3 px.
    EXPECT(std::fabs(dx - TRUE_DX) < 0.4F && std::fabs(dy - TRUE_DY) < 0.4F,
           "standalone_lk displacement within 0.4 px");
  }
}

}  // namespace

int main() {
  test_backend_active();
  test_pyr_lk_image_to_image();
  test_pyr_lk_pyramid_apis();
  test_pyr_lk_from_pyramid();
  test_standalone_lk();
  if (failures != 0) {
    std::fprintf(stderr, "[test_optical_flow] %d failure(s)\n", failures);
    return 1;
  }
  std::printf("[test_optical_flow] OK\n");
  return 0;
}
