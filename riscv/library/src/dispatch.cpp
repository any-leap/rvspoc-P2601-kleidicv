// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"

#include <sys/auxv.h>

#include <cstdlib>
#include <cstring>

namespace kleidicv::riscv_dispatch {

namespace {

// Linux RISC-V hwcap bits are letter-indexed: bit 0 is 'A', bit 21 is 'V'.
// (See arch/riscv/include/asm/hwcap.h; userspace exposes it via getauxval.)
constexpr unsigned long kHwcapV = 1UL << ('V' - 'A');

bool env_force_scalar() {
  const char* v = std::getenv("KLEIDICV_FORCE_SCALAR");
  return v && (std::strcmp(v, "1") == 0 || std::strcmp(v, "ON") == 0 ||
               std::strcmp(v, "true") == 0);
}

bool kernel_advertises_v() {
  return (getauxval(AT_HWCAP) & kHwcapV) != 0;
}

Backend detect_once() {
  if (env_force_scalar()) return Backend::Scalar;
  if (kernel_advertises_v()) return Backend::Rvv;
  return Backend::Scalar;
}

}  // namespace

Backend active_backend() {
  static const Backend cached = detect_once();
  return cached;
}

}  // namespace kleidicv::riscv_dispatch
