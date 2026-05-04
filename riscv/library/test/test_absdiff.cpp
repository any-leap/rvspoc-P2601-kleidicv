// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for kleidicv_saturating_absdiff_* on RISC-V.
// Runs under qemu via CMAKE_CROSSCOMPILING_EMULATOR.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "kleidicv/kleidicv.h"

extern "C" const char *kleidicv_riscv_active_backend();

namespace {

void test_backend_active() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_absdiff] active backend = %s\n", b);
  // The default test invocation runs without KLEIDICV_FORCE_SCALAR under qemu
  // with V=true, so we expect rvv. The dedicated scalar test (run via env var)
  // sets the expectation differently and is checked there.
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) {
    std::fprintf(stderr,
                 "FAIL backend mismatch: got %s, expected %s "
                 "(KLEIDICV_FORCE_SCALAR=%s)\n",
                 b, expected,
                 std::getenv("KLEIDICV_FORCE_SCALAR")
                     ? std::getenv("KLEIDICV_FORCE_SCALAR")
                     : "<unset>");
    std::exit(2);
  }
}

template <typename T>
T ref_absdiff(T a, T b) {
  using U = std::make_unsigned_t<T>;
  U ua = static_cast<U>(a);
  U ub = static_cast<U>(b);
  U diff = (a > b) ? static_cast<U>(ua - ub) : static_cast<U>(ub - ua);
  if constexpr (std::is_unsigned_v<T>) return static_cast<T>(diff);
  constexpr U kMax = static_cast<U>(std::numeric_limits<T>::max());
  return static_cast<T>(diff > kMax ? kMax : diff);
}

int failures = 0;

#define EXPECT(cond, msg)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);  \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

void test_u8_basic() {
  constexpr size_t kW = 32, kH = 4;
  std::vector<uint8_t> a(kW * kH), b(kW * kH), out(kW * kH);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<uint8_t>(i & 0xff);
    b[i] = static_cast<uint8_t>((i * 7 + 3) & 0xff);
  }
  kleidicv_error_t err = kleidicv_saturating_absdiff_u8(
      a.data(), kW, b.data(), kW, out.data(), kW, kW, kH);
  EXPECT(err == KLEIDICV_OK, "u8 returned error");
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT(out[i] == ref_absdiff<uint8_t>(a[i], b[i]), "u8 value mismatch");
  }
}

void test_s8_saturation() {
  // |INT8_MIN - INT8_MAX| = 255 which overflows int8_t and must saturate to
  // INT8_MAX (127).
  int8_t a[1] = {std::numeric_limits<int8_t>::min()};
  int8_t b[1] = {std::numeric_limits<int8_t>::max()};
  int8_t out[1] = {0};
  kleidicv_error_t err = kleidicv_saturating_absdiff_s8(a, 1, b, 1, out, 1, 1, 1);
  EXPECT(err == KLEIDICV_OK, "s8 returned error");
  EXPECT(out[0] == std::numeric_limits<int8_t>::max(),
         "s8 saturation incorrect");
}

template <typename T>
void test_signed_bulk_vs_reference() {
  // Large enough to drive multiple RVV strip-mine iterations and to mix
  // saturating + non-saturating regions across many vl windows.
  constexpr size_t kW = 257, kH = 3;  // 257 is prime — guarantees an unaligned
                                      // tail on every plausible vlen.
  std::vector<T> a(kW * kH), b(kW * kH), out(kW * kH);
  using S = typename std::make_signed<T>::type;
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<T>(static_cast<S>(i) * 13 - 1000);
    b[i] = static_cast<T>(static_cast<S>(i * i) - static_cast<S>(i) * 7 - 500);
  }
  kleidicv_error_t err;
  if constexpr (std::is_same_v<T, int8_t>) {
    err = kleidicv_saturating_absdiff_s8(a.data(), kW * sizeof(T), b.data(),
                                         kW * sizeof(T), out.data(),
                                         kW * sizeof(T), kW, kH);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    err = kleidicv_saturating_absdiff_s16(a.data(), kW * sizeof(T), b.data(),
                                          kW * sizeof(T), out.data(),
                                          kW * sizeof(T), kW, kH);
  } else {
    err = kleidicv_saturating_absdiff_s32(a.data(), kW * sizeof(T), b.data(),
                                          kW * sizeof(T), out.data(),
                                          kW * sizeof(T), kW, kH);
  }
  EXPECT(err == KLEIDICV_OK, "signed bulk returned error");
  for (size_t i = 0; i < out.size(); ++i) {
    T expected = ref_absdiff<T>(a[i], b[i]);
    if (out[i] != expected) {
      std::fprintf(stderr,
                   "FAIL signed bulk i=%zu sz=%zu a=%lld b=%lld got=%lld "
                   "expected=%lld\n",
                   i, sizeof(T), static_cast<long long>(a[i]),
                   static_cast<long long>(b[i]),
                   static_cast<long long>(out[i]),
                   static_cast<long long>(expected));
      ++failures;
      break;
    }
  }
}

void test_s32_max_saturation() {
  // (INT32_MIN, INT32_MAX) must saturate to INT32_MAX.
  int32_t a[4] = {std::numeric_limits<int32_t>::min(),
                  std::numeric_limits<int32_t>::max(), 0, -1};
  int32_t b[4] = {std::numeric_limits<int32_t>::max(),
                  std::numeric_limits<int32_t>::min(), 0, 1};
  int32_t out[4] = {0, 0, 0, 0};
  kleidicv_error_t err = kleidicv_saturating_absdiff_s32(
      a, 4 * sizeof(int32_t), b, 4 * sizeof(int32_t), out,
      4 * sizeof(int32_t), 4, 1);
  EXPECT(err == KLEIDICV_OK, "s32 sat returned error");
  EXPECT(out[0] == std::numeric_limits<int32_t>::max(), "s32 sat min-vs-max");
  EXPECT(out[1] == std::numeric_limits<int32_t>::max(), "s32 sat max-vs-min");
  EXPECT(out[2] == 0, "s32 zero diff");
  EXPECT(out[3] == 2, "s32 small diff");
}

void test_null_pointer() {
  uint8_t buf[4] = {0};
  kleidicv_error_t err =
      kleidicv_saturating_absdiff_u8(nullptr, 1, buf, 1, buf, 1, 1, 1);
  EXPECT(err == KLEIDICV_ERROR_NULL_POINTER, "null check missing");
}

void test_stride() {
  // 2x2 with non-tight stride (3 bytes per row, only first 2 used).
  uint8_t a[6] = {10, 20, 0xAA,  30, 40, 0xBB};
  uint8_t b[6] = { 5,  8, 0xCC,  50, 36, 0xDD};
  uint8_t out[6] = {0};
  kleidicv_error_t err =
      kleidicv_saturating_absdiff_u8(a, 3, b, 3, out, 3, 2, 2);
  EXPECT(err == KLEIDICV_OK, "stride returned error");
  EXPECT(out[0] == 5 && out[1] == 12, "stride row 0");
  EXPECT(out[3] == 20 && out[4] == 4, "stride row 1");
  // Padding bytes must remain untouched.
  EXPECT(out[2] == 0 && out[5] == 0, "padding clobbered");
}

void test_sme_alias_matches() {
  // On RISC-V the _sme alias points at the same scalar impl; both must
  // produce identical output. This catches accidental divergence.
  std::vector<int32_t> a(64), b(64), o1(64), o2(64);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<int32_t>(i) - 30;
    b[i] = static_cast<int32_t>(i * i) - 100;
  }
  (void)kleidicv_saturating_absdiff_s32(
      a.data(), 64 * sizeof(int32_t), b.data(), 64 * sizeof(int32_t),
      o1.data(), 64 * sizeof(int32_t), 64, 1);
  (void)kleidicv_saturating_absdiff_s32_sme(
      a.data(), 64 * sizeof(int32_t), b.data(), 64 * sizeof(int32_t),
      o2.data(), 64 * sizeof(int32_t), 64, 1);
  EXPECT(std::memcmp(o1.data(), o2.data(), o1.size() * sizeof(int32_t)) == 0,
         "default vs _sme alias diverged");
}

}  // namespace

int main() {
  test_backend_active();
  test_u8_basic();
  test_s8_saturation();
  test_signed_bulk_vs_reference<int8_t>();
  test_signed_bulk_vs_reference<int16_t>();
  test_signed_bulk_vs_reference<int32_t>();
  test_s32_max_saturation();
  test_null_pointer();
  test_stride();
  test_sme_alias_matches();
  if (failures == 0) {
    std::printf("[test_absdiff] all checks passed\n");
    return 0;
  }
  std::printf("[test_absdiff] %d FAILURES\n", failures);
  return 1;
}
