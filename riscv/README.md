<!--
SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors

SPDX-License-Identifier: Apache-2.0
-->

# KleidiCV RISC-V Backend

This directory contains scaffolding and additions for the **RVSPOC P2601** task:
porting KleidiCV to RISC-V (RV64GCV) with RVV 1.0 vectorization.

It does **not** modify upstream Arm sources; everything new lives here so the
upstream tree stays diff-friendly.

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

- [x] Phase 1: dev container + RVV smoke test running under qemu-riscv64
- [x] Phase 2: parallel CMake root produces `libkleidicv.a` for riscv64 and runs
      ctest under qemu (1 operator scalar-only).
- [x] Phase 3: RVV impls + runtime dispatcher; saturating_absdiff covers u8/s8/u16/s16/s32.
- [x] Phase 4: gray_to_rgb_u8 with vsseg3 segment store (after upgrading dev image to gcc 14.2).
- [x] Phase 5: sum_f32 with widening reduce-sum (vfwredusum).
- [ ] Phase 6+: bulk-port remaining 32 operators. See [PORTING_GUIDE.md](PORTING_GUIDE.md)
      for difficulty buckets and recommended next-session order.
- [ ] Phase final: kleidicv-benchmark integration + comparison vs OpenCV RVV HAL,
      real-board (SG2044/A210) numbers, fold parallel riscv/library/ tree back
      into upstream top-level CMake.

## Phase 2 reproduction

```bash
./riscv/scripts/build-image.sh   # only first time
./riscv/scripts/build-lib.sh     # configure, build, ctest under qemu
```

Override the QEMU vector length: `VLEN=512 ./riscv/scripts/build-lib.sh`.
