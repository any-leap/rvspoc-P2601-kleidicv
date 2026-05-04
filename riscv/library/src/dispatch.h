// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Runtime backend selection for the RISC-V port. The decision is made once at
// load time: if the kernel's HWCAP advertises 'V' and the user has not set
// KLEIDICV_FORCE_SCALAR=1, the RVV variant is selected; otherwise scalar.
//
// This header is intentionally tiny — Phase 3 only has one operator and one
// pair of backends. When more backends arrive (e.g. an RVV1.0+Zvbb fast path),
// the macros below should grow rather than the call sites.

#ifndef KLEIDICV_RISCV_DISPATCH_H
#define KLEIDICV_RISCV_DISPATCH_H

namespace kleidicv::riscv_dispatch {

enum class Backend { Scalar, Rvv };

// Selected once at first call, cached forever after.
Backend active_backend();

// Forwarding helper: pick the function pointer that matches the active
// backend. `scalar_fn` and `rvv_fn` must share a type.
template <typename Fn>
inline Fn select(Fn scalar_fn, Fn rvv_fn) {
  return active_backend() == Backend::Rvv ? rvv_fn : scalar_fn;
}

}  // namespace kleidicv::riscv_dispatch

#endif
