<!--
SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
SPDX-License-Identifier: Apache-2.0
-->

# Operator Porting Guide

State as of branch `riscv/scaffolding`.

## Status snapshot

46 ctests (rvv + scalar_forced × 23 test executables) pass at VLEN=128/256/512.

**Fully implemented with RVV** (saturating arithmetic / lane intrinsics
verified via objdump):
- `saturating_absdiff` (u8/s8/u16/s16/s32)
- `gray_to_rgb_u8`, `gray_to_rgba_u8`
- `sum_f32`, `min_max_*` (u8/s8/u16/s16/s32)
- Bucket A: `add` / `sub` (u8/s8/u16/s16/u32/s32/u64/s64), `bitwise_and` (u8),
  `multiply` (u8/s8/u16/s16/s32; scale param ignored, matches upstream TODO),
  `threshold_binary_u8`, `compare_equal_u8`, `compare_greater_u8`,
  `in_range` (u8 + f32), `add_abs_with_threshold_s16`, `scale` (u8 + f32),
  `exp_f32`
- Bucket B: `split`, `merge` (channels∈{2,3,4} × element_size∈{1,2,4,8}),
  `rgb_to_rgb` family (8 variants), `float_conv` (f32↔u8/s8)
- Bucket D: `sobel_3x3_horizontal/vertical_s16_u8`, `scharr_interleaved_s16_u8`,
  `separable_filter_2d_u8/u16` (kernel_size=5), `gaussian_blur_u8` (3x3
  binomial), `morph_u8` (`dilate`/`erode`, rectangular structuring element)
- `blur_and_downsample_u8` (5×5 binomial + 2× downsample): horizontal pass
  unit-stride `vle8`/`vwaddu`/`vwmaccu` into a u16 row buffer, vertical
  pass + downsample uses `vlse16` (stride 4) and saturating `vnclipu`. The
  2-pixel left/right border still uses scalar replicate clip
- `transpose` and `rotate` (90/180/270) for pixel_size ∈ {1, 2, 4, 8}
  via SEW=8/16/32/64 strided store/load (negative stride for 180° and
  270°); pixel_size ∈ {3, 6} falls through to the scalar memcpy loop
- Bucket F: `optical_flow_pyr_lk_*` (full 7-API stack: build/release/get_*,
  `pyr_lk_u8`, `pyr_lk_u8_from_pyramid`) and
  `standalone_lucas_kanade_alg_u8`. Pyramid build reuses upstream's template
  scaffold (`kleidicv/analysis/build_optical_flow_pyr_lk_pyramid.h`,
  `calc_optical_flow_pyr_lk.h`) on top of our `blur_and_downsample_u8` +
  `scharr_interleaved_s16_u8`. The two LK SIMD primitives
  (`sample_patch_and_gradients`, `accumulate_mismatch_vector`) have an RVV
  path using `vlseg2e16` / `vwmul` / `vnclip.wi` / `vwredsum` (verified via
  objdump) and a scalar fallback. Bit-exact against the scalar oracle on the
  smoke test.

- `resize_linear_u8/f32` (bilinear, channels=1): per-row index/weight tables
  precomputed once, then `vluxei32` gathers four neighbours per stripe and
  fused-multiply-add blends; u8 narrows back via two-stage `vnclipu`. Scalar
  is bit-exact in double precision; RVV uses single precision and matches
  scalar within ±1 LSB on cross-checked random sizes
- `remap_s16_u8/u16` (integer pickup, REPLICATE/CONSTANT border): `vlseg2e16`
  loads the (sx, sy) pairs, `vmslt`/`vmsge` build the OOB mask, clamped
  indices feed `vluxei32` and `vmerge` substitutes the constant-fill pixels
- `warp_perspective_u8` (perspective + nearest/bilinear, REPLICATE/CONSTANT):
  per row, the three projective accumulators are built as fused multiply-add
  vectors, then divided by sw_p once. Nearest path is single gather +
  optional fill merge; bilinear is four gathers + per-stage `vfmacc` blend
  with saturating narrow back to u8

**Implemented scalar-only (RVV path = scalar)**:
- *(none — every operator has either an RVV path, an upstream template fed
  by RVV primitives, or a multi-byte memcpy fallback for pixel_size ∈ {3, 6}
  / unusual border modes)*

**Implemented partial subset**:
- `median_blur_u8` (3×3 only via 9-element sorting network — 5×5/7×7
  return `KLEIDICV_ERROR_NOT_IMPLEMENTED`. Generalising the sorting net
  to 25/49 elements is non-trivial and not in P2601 scope)
- `gaussian_blur_u8` — fast path for 3×3 zero-sigma (binomial integer
  kernel); generic f32 separable path for any other odd kernel size and
  any sigma (sigma=0 uses OpenCV's default `0.3·((ks-1)/2-1)+0.8`).
  Even kernel sizes return `KLEIDICV_ERROR_RANGE` per the upstream
  contract
- `blur_and_downsample_u8` border modes: REPLICATE (vectorised),
  REFLECT_101 / REVERSE (scalar, proper reflect_101 mapping), REFLECT
  (scalar, edge-doubled mirror). WRAP and CONSTANT return
  `KLEIDICV_ERROR_NOT_IMPLEMENTED` (not a meaningful border mode for a
  blur pyramid)
- `rgb_to_yuv_u8`, `yuv_to_rgb_u8` — **YUV444 only**, other base formats
  (NV12/NV21/YUYV/IYUV/etc) return `KLEIDICV_ERROR_NOT_IMPLEMENTED`

**Constraints common to filter/transform ops**: `KLEIDICV_BORDER_TYPE_REPLICATE`
only (or REPLICATE+CONSTANT for remap/warp). Other border types return
`NOT_IMPLEMENTED`.

Multi-channel filter ops (sobel, scharr, separable_filter_2d_u8/u16,
gaussian_blur_u8 3×3, blur_and_downsample_u8, morph_u8 dilate/erode,
median_blur_u8 3×3) now accept channels ∈ {1, 2, 3, 4}: channels=1 hits
the existing fast path; channels>1 deinterleaves with `vlsegN`, runs the
channels=1 kernel on each plane, and reinterleaves with `vssegN`. The
deinterleave/reinterleave passes are themselves vectorised, so the
channels>1 path stays on the vector unit even though the kernel itself
is reused as-is. Bit-exact against per-plane scalar reference on the
test vectors in `test_filters.cpp::test_multichannel_filters` and
`test_filters.cpp::test_sobel`.

**Stubbed `NOT_IMPLEMENTED`** (planned, not yet ported):
- `separable_filter_2d` for kernel_size ≠ 5
- `gaussian_blur_u8` for kernel_size ≠ 3 or non-zero sigma
- `remap_s16point5_u8/u16` (fractional bilinear), `remap_f32_u8/u16` (float
  coordinates)

## Original kept for reference

## Per-operator file layout (4 files each)

```
riscv/library/src/<op>_decls.h       — prototypes in kleidicv::scalar / kleidicv::rvv
riscv/library/src/<op>_scalar.cpp    — straight loops, also serves as oracle
riscv/library/src/<op>_rvv.cpp       — RVV intrinsics, strip-mined
riscv/library/src/<op>_api.cpp       — extern "C" function pointers wired via dispatcher
```

Tests: `riscv/library/test/test_<op>.cpp`, registered twice in
`test/CMakeLists.txt` (default RVV + `KLEIDICV_FORCE_SCALAR=1`).

When adding an op:

1. add the 4 source files (copy a similar existing op as a starting point)
2. append source files to `add_library(kleidicv …)` in
   `riscv/library/CMakeLists.txt`
3. append the test executable + two `add_test` lines to
   `riscv/library/test/CMakeLists.txt`
4. `./riscv/scripts/build-lib.sh` should make ctest grow by 2 entries

## Difficulty buckets for the remaining 32 operators

### Bucket A — element-wise, copy-paste from absdiff (~10 ops, ~1 hr each)

Same shape as `saturating_absdiff`: image-to-image, per-pixel computation,
no borders. Differ only in the per-element formula.

- `arithmetics/add` (saturating_add): vsadd_vv / vsaddu_vv
- `arithmetics/sub` (saturating_sub): vssub_vv / vssubu_vv
- `arithmetics/multiply`: widening multiply + narrow w/ saturation
- `arithmetics/bitwise_and`: vand_vv (trivial)
- `arithmetics/threshold`: compare-and-mask + select
- `arithmetics/in_range`: two compares + and
- `arithmetics/compare`: vmseq/vmslt/etc. → mask → byte
- `arithmetics/scale`: a*x + b, fused multiply-add
- `arithmetics/exp`: f32 polynomial — uses upstream's polynomial directly
- `arithmetics/add_abs_with_threshold`: composed of above

### Bucket B — channel conversion, segment ld/st (~7 ops, ~1.5 hr each)

Now feasible thanks to the gcc 14 upgrade.

- `conversions/split`: 1 interleaved input → N planar outputs (vlseg → store)
- `conversions/merge`: inverse of split (load → vsseg)
- `conversions/gray_to_rgba`: same as gray_to_rgb but 4 channels
- `conversions/rgb_to_rgb`: channel reorder (vlseg + reorder + vsseg)
- `conversions/rgb_to_yuv`: matrix multiply + interleave; per-pixel fixed-point coefficients
- `conversions/yuv_to_rgb`: inverse of above; needs saturating clip
- `conversions/float_conv`: f32↔u8/s16 with rounding/saturation

### Bucket C — reductions, similar to sum (~1 op, ~2 hr)

- `analysis/min_max`: dual-output (min and max) via vredmin/vredmax
- `analysis/sum_api.cpp` already has `kleidicv_sum_f32` only — but check
  whether u8/s16 sum variants are also expected (read public header).

### Bucket D — separable filters, the real work (~6 ops, ~1 day each)

These need either porting upstream's `separable.h` workspace framework or
writing a self-contained equivalent. Border replication, intermediate s16
buffer, the `(y_begin, y_end]` stripe contract.

- `filters/sobel`: 3x3 Gx and Gy, separable
- `filters/scharr`: 3x3 weighted, similar to sobel
- `filters/gaussian_blur`: arbitrary kernel size, separable
- `filters/blur_and_downsample`: gaussian + 2x downsample
- `filters/separable_filter_2d`: generic separable, user-supplied kernels
- `filters/median_blur`: not separable; sliding-window selection (242 LOC api)

**Recommended approach for the first one (sobel)**: build a self-contained
implementation that allocates its own intermediate row buffer. Don't try to
port the upstream framework yet — that's a separate refactor. Once 2-3
filters exist and we see what they share, factor out the common scaffold.

### Bucket E — geometric transforms (~4 ops, ~1 day each)

- `transform/transpose`: simplest; mostly memory shuffle (could exercise vrgather)
- `transform/rotate`: 90/180/270; coarse-grained
- `transform/remap`: per-pixel coordinate lookup (vluxei / scalar fallback)
- `transform/warp_perspective`: full perspective; precision-critical

### Bucket F — heavyweight (the rest)

- `morphology/morphology`: erode/dilate w/ arbitrary structuring element
- `resize/resize_linear` (286 LOC api): bilinear, multiple type combinations
- `resize/resize_to_quarter`: special-case 4x downsample
- `analysis/build_optical_flow_pyr_lk_pyramid` (126 LOC api): pyramid construction
- `analysis/calc_optical_flow_pyr_lk` (51 LOC api): LK iteration
- `analysis/standalone_lucas_kanade_alg`: standalone variant

These each warrant a planning session before coding.

## Suggested next session order

All buckets above are now ported and live in `riscv/library/src/`. Open
items left, in rough priority for a follow-up:

1. **Real-board benchmarks.** Re-run `bench_kleidicv` on SG2044/A210 (or
   another RV64GCV host) and pin numbers in the README. Qemu-user numbers
   are emulator instruction counts, not silicon time.
2. **OpenCV 4.13.0 conformity.** Wire the upstream `conformity/opencv/`
   suite (which already includes `test_standalone_lucas_kanade_alg.cpp`)
   through this parallel build tree so it cross-checks against OpenCV's
   reference implementation. Needs an OpenCV install in the dev image.
3. **Multi-channel LK pyramid.** Pyramid build itself is channels-aware
   (it forwards `channels_` to `blur_and_downsample` and `scharr`, both
   of which now accept channels ∈ {1..4}). End-to-end LK tracking on
   multi-channel images works mechanically but isn't covered by a smoke
   test in this PR.
4. **Strict REVERSE/REFLECT_101 in `blur_and_downsample`.** The LK pyramid
   pre-fills the border with reflect_101 so our REPLICATE-clip impl works,
   but a clean pass would teach the inner kernel to clip with
   reflect_101 and drop the API-level alias.
5. **Fold the `riscv/library/` parallel CMake root back into the upstream
   top-level build.** Mirror the `kleidicv_neon` / `kleidicv_sve2` OBJECT
   library pattern so a single `cmake -S kleidicv` build produces all
   targets when the host toolchain supports them.

## Things to keep in mind

- **Always test under both backends.** ctest registers each op's test twice
  (default + `KLEIDICV_FORCE_SCALAR=1`). Don't add an op without both.
- **Use `vlen=128/256/512` to verify VLEN-independence.** The Phase 1 helper
  `VLEN=N ./riscv/scripts/build-lib.sh` works for all ops.
- **Saturating-narrow uses 4-arg vnclip in gcc 14.** Always pass
  `__RISCV_VXRM_RNU` as the rmode unless you specifically need rounding.
- **objdump-spot-check the .o** after writing RVV code. If you don't see
  `vsetvli`/`v…v`/`vse…` instructions in your inner loop, the compiler
  silently degraded to scalar.
- **`kleidicv_riscv_active_backend()`** is the test hook. Don't assert
  outputs without first asserting which backend is live; otherwise a
  dispatcher init bug looks like a correctness bug.

## Open questions

- Bucket A templated vs file-per-op: settled. We kept file-per-op but
  factored the inner loops into `elementwise_{scalar,rvv}.h` after the
  third op, which cut net LOC ~60% as predicted.
- Folding `riscv/library/` back into upstream top-level CMake: still open,
  intentionally deferred until real-board numbers land so the consolidation
  doesn't churn under benchmark iteration.
- Real-board access: still pending A210 environment from the organisers.
  Build is set up so a `cmake -DCMAKE_TOOLCHAIN_FILE=…` from a real
  RV64GCV host produces the same artifacts; only thing missing is silicon
  time.
