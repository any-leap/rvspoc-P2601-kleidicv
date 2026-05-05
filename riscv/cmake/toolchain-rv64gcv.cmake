# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# CMake toolchain file for RV64GCV (RVV 1.0).
# Use:  cmake -DCMAKE_TOOLCHAIN_FILE=<this-file> ...

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

# rv64gcv = RV64I + M + A + F + D + C + V (vector). Zba/Zbb/Zbs are widely
# available on real hardware (SG2044, K1) and let the compiler emit better
# scalar code.
set(KLEIDICV_RVV_ARCH "rv64gcv_zba_zbb_zbs" CACHE STRING
    "RISC-V ISA string used for the RVV build")

set(CMAKE_C_FLAGS_INIT   "-march=${KLEIDICV_RVV_ARCH} -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "-march=${KLEIDICV_RVV_ARCH} -mabi=lp64d")

# Run cross-built test/bench binaries through qemu-user automatically when the
# host runs `ctest` or executes a target.
find_program(QEMU_RISCV64 qemu-riscv64)
if(QEMU_RISCV64)
  # VLEN is read from the environment so the build scripts can drive multi-VLEN
  # validation (VLEN=128 ./build-lib.sh, VLEN=512 ./build-lib.sh, …) without
  # requiring an explicit -DKLEIDICV_QEMU_VLEN= override at every cmake call.
  if(DEFINED ENV{VLEN})
    set(KLEIDICV_QEMU_VLEN_DEFAULT "$ENV{VLEN}")
  else()
    set(KLEIDICV_QEMU_VLEN_DEFAULT "256")
  endif()
  set(KLEIDICV_QEMU_VLEN "${KLEIDICV_QEMU_VLEN_DEFAULT}" CACHE STRING
      "qemu-user RVV VLEN (bits) — must be one of 128, 256, 512, 1024")

  # qemu-user needs `-L <sysroot>` to find the dynamic linker for cross-built
  # ELFs; without it ctest reports "Could not open ld-linux-riscv64-lp64d".
  set(CMAKE_CROSSCOMPILING_EMULATOR
      "${QEMU_RISCV64};-L;/usr/riscv64-linux-gnu;-cpu;rv64,v=true,vlen=${KLEIDICV_QEMU_VLEN},zba=true,zbb=true,zbs=true"
      CACHE STRING "qemu-user emulator for cross-built binaries" FORCE)
endif()

set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
