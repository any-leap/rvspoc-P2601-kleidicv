// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Smoke test: sums an int32 array using a plain scalar loop and an RVV
// intrinsics loop, reports both, and verifies they match. Also prints the
// VLEN reported by the implementation so we can confirm qemu's `-cpu vlen=`
// took effect.

#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>

#define N 100

static int32_t sum_scalar(const int32_t *x, size_t n) {
    int32_t s = 0;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

static int32_t sum_rvv(const int32_t *x, size_t n) {
    // Strip-mined reduction with dynamic vl. LMUL=1 for simplicity; the
    // compiler picks the actual vlen at runtime via vsetvl.
    size_t vl;
    vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vint32m1_t v = __riscv_vle32_v_i32m1(x + i, vl);
        acc = __riscv_vadd_vv_i32m1_tu(acc, acc, v, vl);
    }
    // Tree reduction to scalar. vredsum is the canonical way.
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t r = __riscv_vredsum_vs_i32m1_i32m1(acc, zero,
                                                  __riscv_vsetvlmax_e32m1());
    return __riscv_vmv_x_s_i32m1_i32(r);
}

int main(void) {
    // VLEN in bits = vlmax_e8m1 * 8 (since e8m1 fits VLEN/8 elements).
    size_t vlmax_e8 = __riscv_vsetvlmax_e8m1();
    printf("[hello_rvv] vlen reported by qemu: %zu bits\n", vlmax_e8 * 8);

    int32_t a[N];
    for (int i = 0; i < N; ++i) a[i] = i;

    int32_t s_scalar = sum_scalar(a, N);
    int32_t s_rvv = sum_rvv(a, N);
    printf("[hello_rvv] scalar sum   = %d\n", s_scalar);
    printf("[hello_rvv] RVV sum (e32m1) = %d\n", s_rvv);
    printf("[hello_rvv] match: %s\n", s_scalar == s_rvv ? "yes" : "no");
    return s_scalar == s_rvv ? 0 : 1;
}
