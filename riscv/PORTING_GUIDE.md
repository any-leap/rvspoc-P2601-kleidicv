<!--
SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
SPDX-License-Identifier: Apache-2.0
-->

# Operator Porting Guide

State as of branch `riscv/scaffolding`.

## What's done (3/35 operators)

| Operator | Types | Pattern proven | Notes |
|---|---|---|---|
| `saturating_absdiff` | u8/s8/u16/s16/s32 | Element-wise binary; widen+abs+saturating-narrow for signed | Phase 3 |
| `gray_to_rgb_u8` | u8 | 1-in 3-out interleave via `vsseg3e8` | Phase 4 |
| `sum_f32` | f32 | Scalar reduction via widening `vfwredusum` (f32→f64) | Phase 5 |

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

1. **Bucket A bulk-port** (1 sitting): finish all 10 element-wise. Aim to
   refactor partway through into a header-only template (`elementwise_rvv.h`)
   that takes a per-op functor — will cut LOC by ~60%.
2. **Bucket B** (1-2 sittings): channel conversions. `split`/`merge` first
   (simple), then RGB↔YUV.
3. **Bucket C** (small): `min_max`.
4. **Bucket D first op**: `sobel_3x3`. Plan it explicitly: sketch the
   stripe contract, intermediate buffer lifetime, border policy on paper
   before any code.
5. **Reassess.**

After Bucket A+B+C, we'll be at ~21/35 operators. That's 60% — enough to
start drafting a partial PR to upstream and showing progress to organisers.

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

## Open questions for next session

- Should Bucket A be templated to share code, or keep file-per-op?
  Recommendation: do 3 by hand first (`add`, `sub`, `bitwise_and`), then
  decide based on observed duplication.
- When does the parallel `riscv/library/` tree fold back into upstream
  top-level CMake? Probably after Bucket A+B (i.e. when ~21 operators exist
  and the "scaffold proves itself" risk is gone). See DEC-001.
- Real-board access: organisers offered A210 remote env. We should request
  it before starting Bucket D so benchmark runs aren't qemu-only.
