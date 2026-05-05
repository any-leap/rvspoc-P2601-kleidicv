// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Micro-benchmark for the RISC-V port. Times a handful of representative
// operators on synthetic inputs and prints ns/iteration plus an RVV-vs-scalar
// speedup ratio. Run twice, once with KLEIDICV_FORCE_SCALAR=1 and once
// without — the harness picks up which backend is live and labels output.
//
// Caveat: timings under qemu-user are emulator instruction counts, not real
// silicon. Use them only to verify the RVV path is on the hot loop and the
// speedup is non-negative; treat absolute numbers as relative-only.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kleidicv/kleidicv.h"

extern "C" const char *kleidicv_riscv_active_backend();

namespace {

template <typename Fn>
double bench_ns(Fn &&fn, int iters) {
  // Warm up.
  fn();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) fn();
  auto t1 = std::chrono::steady_clock::now();
  double ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  return ns / static_cast<double>(iters);
}

void bench_add_u8(int iters) {
  constexpr size_t W = 1024, H = 256;
  std::vector<uint8_t> a(W * H), b(W * H), c(W * H);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<uint8_t>(i & 0xff);
    b[i] = static_cast<uint8_t>((i * 7) & 0xff);
  }
  double ns = bench_ns(
      [&]() {
        kleidicv_saturating_add_u8(a.data(), W, b.data(), W, c.data(), W, W, H);
      },
      iters);
  std::printf("  saturating_add_u8 1024x256:  %9.1f ns/op\n", ns);
}

void bench_sobel(int iters) {
  constexpr size_t W = 512, H = 512;
  std::vector<uint8_t> src(W * H);
  std::vector<int16_t> dst(W * H);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i * 11);
  double ns = bench_ns(
      [&]() {
        kleidicv_sobel_3x3_horizontal_s16_u8(src.data(), W, dst.data(),
                                              W * sizeof(int16_t), W, H, 1);
      },
      iters);
  std::printf("  sobel_3x3_horizontal 512^2:  %9.1f ns/op\n", ns);
}

void bench_blur_downsample(int iters) {
  constexpr size_t W = 512, H = 512;
  std::vector<uint8_t> src(W * H);
  std::vector<uint8_t> dst(((W + 1) / 2) * ((H + 1) / 2));
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i * 13);
  double ns = bench_ns(
      [&]() {
        kleidicv_blur_and_downsample_u8(src.data(), W, W, H, dst.data(),
                                         (W + 1) / 2, 1,
                                         KLEIDICV_BORDER_TYPE_REPLICATE);
      },
      iters);
  std::printf("  blur_and_downsample 512^2:   %9.1f ns/op\n", ns);
}

void bench_lk_optical_flow(int iters) {
  constexpr size_t W = 256, H = 256;
  std::vector<uint8_t> prev(W * H), next(W * H);
  for (size_t y = 0; y < H; ++y)
    for (size_t x = 0; x < W; ++x) {
      prev[y * W + x] = static_cast<uint8_t>((x * 17 + y * 23) & 0xff);
      next[y * W + x] = static_cast<uint8_t>(
          (((x + 1) * 17) + ((y - 1) * 23)) & 0xff);
    }
  std::vector<float> prev_pts(64 * 2), next_pts(64 * 2);
  for (int p = 0; p < 64; ++p) {
    prev_pts[p * 2] = 64.0F + static_cast<float>(p % 8) * 16.0F;
    prev_pts[p * 2 + 1] = 64.0F + static_cast<float>(p / 8) * 16.0F;
  }
  std::vector<uint8_t> status(64);
  std::vector<float> err(64);
  kleidicv_optflow_lk_context_t ctx{};
  ctx.window_width = 15;
  ctx.window_height = 15;
  ctx.max_level = 2;
  ctx.termination_count = 30;
  ctx.termination_epsilon = 0.01F;
  ctx.min_eig_threshold = 1e-4F;
  ctx.flags = 0;
  double ns = bench_ns(
      [&]() {
        std::memcpy(next_pts.data(), prev_pts.data(),
                    prev_pts.size() * sizeof(float));
        kleidicv_optical_flow_pyr_lk_u8(prev.data(), W, next.data(), W, W, H, 1,
                                         prev_pts.data(), next_pts.data(), 64,
                                         status.data(), err.data(), ctx);
      },
      iters);
  std::printf("  optical_flow_pyr_lk_u8 64pts: %9.1f ns/op\n", ns);
}

void bench_warp(int iters) {
  constexpr size_t W = 512, H = 512;
  std::vector<uint8_t> src(W * H), dst(W * H);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i);
  // A mildly non-trivial homography (rotation + translation).
  const float M[9] = {0.95F, -0.05F, 5.0F, 0.05F, 0.95F, -3.0F,
                      0.0001F, 0.0002F, 1.0F};
  uint8_t fill = 0;
  double ns = bench_ns(
      [&]() {
        kleidicv_warp_perspective_u8(src.data(), W, W, H, dst.data(), W, W, H,
                                      M, 1, KLEIDICV_INTERPOLATION_LINEAR,
                                      KLEIDICV_BORDER_TYPE_CONSTANT, &fill);
      },
      iters);
  std::printf("  warp_perspective bilinear 512^2: %9.1f ns/op\n", ns);
}

}  // namespace

int main(int argc, char **argv) {
  int iters = 5;
  if (argc > 1) iters = std::atoi(argv[1]);
  if (iters < 1) iters = 1;

  const char *backend = kleidicv_riscv_active_backend();
  std::printf("[bench_kleidicv] backend=%s, iters=%d\n", backend, iters);

  bench_add_u8(iters);
  bench_sobel(iters);
  bench_blur_downsample(iters);
  bench_warp(iters);
  bench_lk_optical_flow(iters);

  std::printf(
      "[bench_kleidicv] done. Pair this with KLEIDICV_FORCE_SCALAR=1 to get\n"
      "  the scalar baseline; ratio = scalar_ns / rvv_ns is the qemu speedup.\n"
      "  Real-silicon numbers will differ — qemu emulates instruction counts.\n");
  return 0;
}
