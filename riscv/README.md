<!--
SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors

SPDX-License-Identifier: Apache-2.0
-->

# KleidiCV RISC-V Backend

This directory contains the RVSPOC **P2601** deliverable: a port of
KleidiCV 26.03 to RISC-V (RV64GCV) with RVV 1.0 vector intrinsics.

Upstream Arm sources are not modified; everything new is added here so
the upstream tree stays diff-friendly.

## Submission deliverables (mapped to the P2601 task page)

| Required item | Where it lives |
|---|---|
| Source code (Pull Request) | `riscv/library/src/*` (RVV + scalar paths), `riscv/library/test/*` (cross-checks), built by `riscv/library/CMakeLists.txt` |
| Verification documentation | `riscv/README.md` § *Validation methods*, `riscv/PORTING_GUIDE.md` (per-operator status), `docs/findings.md` / `docs/decisions.md` / `docs/gotchas.md` (engineering notes) |
| `kleidicv-benchmark` results vs OpenCV 4.13.0 | Build script: `riscv/scripts/build-benchmark.sh`. Comparison procedure: § *OpenCV 4.13.0 comparative benchmark* below. **Real-silicon numbers pending hardware access** |
| Real-board (SG2044/A210) measurements | **Pending** — A210 remote env not yet available. The cross-built artifacts (`libkleidicv.a` and `kleidicv-benchmark`) are riscv64 ELFs that run unmodified on any RV64GCV Linux board |
| CMake-based cross-compile build | `riscv/cmake/toolchain-rv64gcv.cmake` + `riscv/library/CMakeLists.txt`. One-command reproduction in `riscv/scripts/build-lib.sh` |
| Apache-2.0 licensing + SPDX headers | Every new source file carries `SPDX-License-Identifier: Apache-2.0`; verified with `find … -name '*.cpp' -o -name '*.h' \| xargs grep -L SPDX` (returns nothing) |
| AI-assisted code disclosure | `riscv/AI_DISCLOSURE.md` |

What's covered and what's not is broken down further in the
*Coverage and constraints* section below.

## Layout

```
riscv/
├── README.md              ← this file
├── docker/
│   └── Dockerfile         ← Ubuntu 24.04 + riscv64-linux-gnu toolchain + qemu-user
├── cmake/
│   └── toolchain-rv64gcv.cmake  ← CMake toolchain for RV64GCV (RVV 1.0)
├── smoke/
│   ├── CMakeLists.txt
│   └── hello_rvv.c        ← scalar + RVV intrinsics smoke test
└── scripts/
    ├── build-image.sh     ← build the dev docker image
    ├── build-smoke.sh     ← compile smoke test inside container
    └── run-smoke.sh       ← run smoke test under qemu-riscv64
```

## Quick start (host: macOS arm64 with Docker Desktop)

```bash
# 1. Build dev image (~5 min first time)
./riscv/scripts/build-image.sh

# 2. Compile and run RVV smoke test under qemu
./riscv/scripts/build-smoke.sh
./riscv/scripts/run-smoke.sh
```

Expected output:
```
[hello_rvv] vlen reported by qemu: 256 bits
[hello_rvv] scalar sum   = 4950
[hello_rvv] RVV sum (e32m1) = 4950
[hello_rvv] match: yes
```

## QEMU configuration

Smoke tests target `qemu-riscv64` (Linux user-mode emulation, simpler than
full-system QEMU for early development). Default `RISCV_QEMU_CPU` is
`rv64,v=true,vlen=256,zba=true,zbb=true,zbs=true`. Override via env var.

System-mode QEMU (`qemu-system-riscv64 -machine virt -cpu rv64,v=true`) will
be needed later for benchmarking; not required for unit tests.

## Status

- [x] Dev container + RVV smoke test running under qemu-riscv64
- [x] Parallel CMake root produces `libkleidicv.a` for riscv64 and runs
      ctest under qemu at VLEN ∈ {128, 256, 512}
- [x] Runtime dispatcher (`KLEIDICV_FORCE_SCALAR=1` to force scalar; otherwise
      RVV when HWCAP advertises 'V')
- [x] Bulk port of every public operator. Inventory in
      [PORTING_GUIDE.md](PORTING_GUIDE.md). Highlights:
  - Element-wise arithmetic: `add`/`sub`/`multiply`/`scale`/`exp`/`compare`/
    `threshold_binary`/`in_range`/`bitwise_and`/`add_abs_with_threshold`/
    `saturating_absdiff` with widening MAC + saturating narrow
  - Channel ops: `split` / `merge` / `gray_to_rgb*` / `rgb_to_rgb` family /
    `rgb_to_yuv` / `yuv_to_rgb` (YUV444) / `float_conv` via `vlsegN`/`vssegN`
  - Filters: `sobel_3x3`, `scharr_interleaved`, `separable_filter_2d` (k=5),
    `gaussian_blur` (3×3 binomial), `blur_and_downsample`,
    `morph_u8` dilate/erode (rectangular SE), `median_blur_u8` (3×3
    sorting net)
  - Geometric: `transpose` / `rotate` (90/180/270; pixel_size 1/2/4/8 RVV,
    3/6 scalar), `resize_linear_u8/f32`, `remap_s16_u8/u16`,
    `warp_perspective_u8` (nearest+bilinear)
  - Optical flow: full 7-API LK pyramid stack
    (`build_optical_flow_pyr_lk_pyramid`, `_release`, `_get_level_count`,
    `_get_image_level`, `_get_scharr_level`, `optical_flow_pyr_lk_u8`,
    `_from_pyramid`) + `standalone_lucas_kanade_alg_u8`. Pyramid build
    reuses upstream's template scaffold on top of our blur+scharr; the two
    LK SIMD primitives have a real RVV path
  - Reductions: `sum_f32`, `min_max_*` for all integer SEW
- [x] 46 ctests (rvv + scalar_forced for 23 test executables) green at
      VLEN=128/256/512 under qemu-user
- [ ] Real-board benchmarks: blocked on A210 hardware access. Build is set
      up so a `cmake -DCMAKE_TOOLCHAIN_FILE=…` from a real RV64GCV host
      will work directly, but no numbers in this PR yet
- [ ] OpenCV 4.13.0 conformity benchmark: harness lives at
      `conformity/opencv/` upstream — wiring it through this parallel build
      tree is a follow-up

## Build + test (any toolchain that ships gcc-14 RV64GCV + qemu-user)

```bash
./riscv/scripts/build-image.sh        # one-time dev image
./riscv/scripts/build-lib.sh           # default VLEN=256
VLEN=128 ./riscv/scripts/build-lib.sh  # alternate vector lengths
VLEN=512 ./riscv/scripts/build-lib.sh
```

Each test executable runs twice: once on the dispatched RVV path and once
with `KLEIDICV_FORCE_SCALAR=1`, so any RVV regression vs the scalar
reference shows up immediately. `riscv64-linux-gnu-objdump -d
build/CMakeFiles/kleidicv.dir/src/<op>_rvv.cpp.o` confirms each op emits
`vsetvli`/`vle*`/`v…v`/`vse*`/`vnclip*` etc. and was not silently degraded.

## Microbenchmark

A small bench harness lives at `riscv/library/test/bench_kleidicv.cpp`,
compiled but not added to ctest. Run it manually:

```bash
# Inside the dev container:
qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbs=true \
  riscv/library/build/test/bench_kleidicv 20
KLEIDICV_FORCE_SCALAR=1 qemu-riscv64 -cpu … \
  riscv/library/build/test/bench_kleidicv 20
```

Sample qemu-user output at VLEN=256 (qemu emulates each vector op as many
host instructions, so RVV is artificially **slower** here — the harness is
intended for real-silicon comparisons; treat qemu numbers only as
"is the path live" smoke):

```
[bench_kleidicv] backend=rvv,    iters=5
  saturating_add_u8 1024x256:        3.2 ms/op
  sobel_3x3_horizontal 512^2:        7.5 ms/op
  blur_and_downsample 512^2:         9.1 ms/op
  warp_perspective bilinear 512^2:  67.7 ms/op
  optical_flow_pyr_lk_u8 64pts:     33.7 ms/op

[bench_kleidicv] backend=scalar, iters=5
  saturating_add_u8 1024x256:        0.6 ms/op
  sobel_3x3_horizontal 512^2:        3.0 ms/op
  blur_and_downsample 512^2:         4.2 ms/op
  warp_perspective bilinear 512^2:  79.7 ms/op
  optical_flow_pyr_lk_u8 64pts:      9.1 ms/op
```

On real RV64GCV silicon (e.g. SG2044/A210) the same binary should show RVV
> scalar by a wide margin for memory-streaming and saturating-arithmetic
ops; numbers will be filled in once hardware access is available.

## OpenCV 4.13.0 comparative benchmark

The upstream KleidiCV repo includes a Google-Benchmark suite at
`benchmark/benchmark.cpp` that exercises every public C API. We wire it
into the riscv64 build via the `KLEIDICV_UPSTREAM_BENCHMARK` option:

```bash
./riscv/scripts/build-image.sh           # one-time
./riscv/scripts/build-benchmark.sh        # builds kleidicv-benchmark for riscv64
```

The resulting binary lives at
`riscv/library/build-bench/upstream_benchmark/kleidicv-benchmark`. It is
a riscv64 ELF that runs unmodified on any RV64GCV Linux host or under
qemu-user.

To produce the comparative numbers required by the P2601 spec:

1. **Our impl numbers (RVV path):** on the target hardware,
   ```bash
   ./kleidicv-benchmark --image_width=1280 --image_height=720 \
     --benchmark_format=json --benchmark_out=ours_rvv.json
   ```
   then with `KLEIDICV_FORCE_SCALAR=1` to capture the scalar baseline.

2. **OpenCV 4.13.0 RISC-V impl numbers:** build OpenCV 4.13.0 from
   source on the same host with RVV enabled:
   ```bash
   cmake -S opencv-4.13.0 -B build-opencv \
     -DCPU_BASELINE=RVV -DCV_RVV=ON \
     -DBUILD_PERF_TESTS=ON -DBUILD_EXAMPLES=OFF
   cmake --build build-opencv -j
   ```
   Then run the OpenCV `opencv_perf_imgproc` / `opencv_perf_core`
   binaries and align test cases per operator (e.g. `Resize` ↔
   `kleidicv_resize_linear_u8`). OpenCV's perf tests output JSON via
   `--perf_write_xml=`.

3. **Comparison:** match operator+size pairs between the two JSON outputs
   and compute the speedup ratio. A small Python helper for this is
   intentionally not committed yet — the report format depends on which
   board the organisers provide.

Real-silicon numbers will be appended to this README in a follow-up
commit once SG2044 / A210 hardware access lands. The build is set up so
the same binary runs on real hardware with no changes — just copy the
ELF over and run.

The bench harness was smoke-tested under qemu-user (output sample below
is *emulator instruction counts*, not silicon time):

```
qemu-riscv64 -cpu rv64,v=true,vlen=256 \
  riscv/library/build-bench/upstream_benchmark/kleidicv-benchmark \
  --benchmark_filter='min_max_u8|sobel' --benchmark_min_time=0.05s
…
min_max_u8                  5121289 ns      4432791 ns           16
sobel_filter_vertical      27928917 ns     27927499 ns            2
sobel_filter_horizontal    27825042 ns     27827645 ns            2
```

## Validation methods

Every operator has been validated through one or more of:

- **ctest cross-check (default + KLEIDICV_FORCE_SCALAR=1):** every test
  executable runs twice, once on the dispatched RVV path and once forced
  to the scalar reference. 46 ctests pass at VLEN ∈ {128, 256, 512}.
- **Bit-exact reference comparison on randomised inputs:** for ops
  where the scalar and RVV paths use the same numeric domain, the test
  generates pseudo-random input and asserts `memcmp(scalar_ref, rvv) == 0`
  on output (see `test_transform.cpp::test_rvv_strip_mining`,
  `::test_remap_s16_compare`, `::test_warp_perspective_compare`).
- **Bounded-tolerance comparison for f32-vs-f64 paths:** resize_linear
  RVV uses `f32` while the scalar reference uses `f64`; the test
  asserts `|got - ref| ≤ 1` on u8 and `~1e-5` on f32 across a
  64×48→97×53 random image (see `::test_resize_linear_compare`).
- **End-to-end algorithmic check:** Lucas-Kanade tracker is verified on
  synthetic textured images with known sub-pixel shift; recovered
  displacement is asserted to within 0.5 px of ground truth at
  multi-level pyramids (`test_optical_flow.cpp`).
- **objdump spot-check:** for each new RVV translation unit,
  `riscv64-linux-gnu-objdump -d` is inspected to confirm the inner loop
  emits the expected vector instructions
  (`vsetvli`/`vle*`/`vlse*`/`vluxei*`/`vwmul*`/`vnclip*`/`vsseg*` etc.)
  and was not silently degraded to scalar.
- **VLEN independence:** `VLEN=128 / 256 / 512 ./riscv/scripts/build-lib.sh`
  rebuilds and reruns the entire suite at three vector lengths to catch
  any inadvertently VLEN-coupled assumptions.
- **Multi-iteration smoke under qemu:** every test executable runs in
  qemu-user via `qemu-riscv64 -cpu rv64,v=true,vlen=$VLEN`, so the
  full toolchain (cross-compile → user-mode emulation → vsetvli
  enforcement) is validated end-to-end.

## Coverage and constraints

- **Operators:** every public KleidiCV 26.03 C API symbol either has
  an RVV implementation, or is a thin call into an upstream C++
  template fed by RVV-implemented primitives, or returns a clean
  `KLEIDICV_ERROR_NOT_IMPLEMENTED` for out-of-scope variants
  (s16point5/float remap, yuv-semiplanar conversions, scale-to-f16,
  count_nonzeros, min_max_loc, min_max_f32). The not-implemented set
  is explicitly out of P2601 scope.
- **RVV coverage:** ≥ 90% of operators have a real RVV path
  (the fully-RVV list is in `PORTING_GUIDE.md`); the few remaining
  scalar-fallback paths are guarded by `pixel_size ∈ {3, 6}` or
  unusual border modes that the SPOC scope explicitly excludes.
- **Multi-channel:** every operator that takes a `channels` parameter
  now accepts channels ∈ {1, 2, 3, 4} (`KLEIDICV_MAXIMUM_CHANNEL_COUNT`).
  Element-wise / conversion ops were already channel-agnostic; filter
  ops (sobel, scharr, separable_filter_2d_u8/u16, gaussian_blur_u8,
  blur_and_downsample_u8, morph_u8 dilate/erode, median_blur_u8) get
  multi-channel via vectorised deinterleave/reinterleave (`vlsegN` /
  `vssegN`) wrapping the existing channels=1 kernels. Bit-exact
  against per-plane scalar reference.
- **Border modes:** REPLICATE everywhere; CONSTANT additionally on
  remap_s16 and warp_perspective. REVERSE/REFLECT_101 in
  `blur_and_downsample` is accepted but treated as REPLICATE — the LK
  pyramid pre-fills border pixels with reflect_101 data, so the
  numeric impact is zero on its primary caller.

## What's known not to be bit-exact vs scalar

| Operator | Reason | Cross-check tolerance |
|----------|--------|----------------------|
| `resize_linear_u8/f32` | RVV uses `f32` ops; scalar uses `f64` | u8: ±1 LSB / f32: ~1e-5 |
| `warp_perspective_u8` (bilinear) | RVV uses `f32`; scalar uses `f32` (matches) | bit-exact for matching modes |
| `blur_and_downsample_u8` | REVERSE/REFLECT_101 border treated as REPLICATE | edge 2 pixels per side |

All other operators are bit-exact across RVV/scalar.

## Repository layout extension

The riscv/library/ tree is a parallel CMake root because the upstream
top-level CMake builds AArch64 OBJECT libraries unconditionally and
cannot configure for riscv64 until the entire tree folds together. Once
this work merges and a few open items above land, the riscv/library/
sources are intended to become a target-namespace partition under the
existing top-level build (`kleidicv_rvv` OBJECT library following the
`kleidicv_neon` / `kleidicv_sve2` precedent).

## Phase 2 reproduction

```bash
./riscv/scripts/build-image.sh   # only first time
./riscv/scripts/build-lib.sh     # configure, build, ctest under qemu
```

Override the QEMU vector length: `VLEN=512 ./riscv/scripts/build-lib.sh`.
