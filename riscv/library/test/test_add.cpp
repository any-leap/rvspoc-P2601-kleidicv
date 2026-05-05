// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for kleidicv_saturating_add_* on RISC-V.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "kleidicv/kleidicv.h"

extern "C" const char *kleidicv_riscv_active_backend();

namespace {

int failures = 0;

#define EXPECT(cond, msg)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);  \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

void test_backend_active() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_add] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) {
    std::fprintf(stderr, "FAIL backend mismatch: got %s expected %s\n", b,
                 expected);
    std::exit(2);
  }
}

template <typename T>
T ref_sat_add(T a, T b) {
  if constexpr (sizeof(T) == 8) {
    if constexpr (std::is_unsigned_v<T>) {
      T s = static_cast<T>(a + b);
      return s < a ? std::numeric_limits<T>::max() : s;
    } else {
      using U = std::make_unsigned_t<T>;
      U sum = static_cast<U>(static_cast<U>(a) + static_cast<U>(b));
      T r = static_cast<T>(sum);
      bool sa = a < 0, sb = b < 0;
      if (sa == sb && (r < 0) != sa)
        return sa ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
      return r;
    }
  } else {
    using L = std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;
    L w = static_cast<L>(a) + static_cast<L>(b);
    constexpr L kMax = static_cast<L>(std::numeric_limits<T>::max());
    constexpr L kMin = static_cast<L>(std::numeric_limits<T>::min());
    if (w > kMax) return std::numeric_limits<T>::max();
    if (w < kMin) return std::numeric_limits<T>::min();
    return static_cast<T>(w);
  }
}

template <typename T>
kleidicv_error_t call(const T *a, size_t sa, const T *b, size_t sb, T *d,
                      size_t sd, size_t w, size_t h);

#define BIND(suffix, T, FN)                                                  \
  template <>                                                                \
  kleidicv_error_t call<T>(const T *a, size_t sa, const T *b, size_t sb,     \
                           T *d, size_t sd, size_t w, size_t h) {            \
    return FN(a, sa, b, sb, d, sd, w, h);                                    \
  }

BIND(u8, uint8_t, kleidicv_saturating_add_u8)
BIND(s8, int8_t, kleidicv_saturating_add_s8)
BIND(u16, uint16_t, kleidicv_saturating_add_u16)
BIND(s16, int16_t, kleidicv_saturating_add_s16)
BIND(u32, uint32_t, kleidicv_saturating_add_u32)
BIND(s32, int32_t, kleidicv_saturating_add_s32)
BIND(u64, uint64_t, kleidicv_saturating_add_u64)
BIND(s64, int64_t, kleidicv_saturating_add_s64)
#undef BIND

template <typename T>
void test_bulk_vs_reference(const char *tag) {
  constexpr size_t kW = 257, kH = 3;
  std::vector<T> a(kW * kH), b(kW * kH), out(kW * kH);
  using S = std::make_signed_t<T>;
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<T>(static_cast<S>(i) * 17 - 13);
    b[i] = static_cast<T>(static_cast<S>(i * i) - static_cast<S>(i) * 5 + 7);
  }
  kleidicv_error_t err =
      call<T>(a.data(), kW * sizeof(T), b.data(), kW * sizeof(T), out.data(),
              kW * sizeof(T), kW, kH);
  EXPECT(err == KLEIDICV_OK, "bulk returned error");
  for (size_t i = 0; i < out.size(); ++i) {
    T expected = ref_sat_add<T>(a[i], b[i]);
    if (out[i] != expected) {
      std::fprintf(stderr,
                   "FAIL %s i=%zu a=%lld b=%lld got=%lld exp=%lld\n", tag, i,
                   static_cast<long long>(a[i]), static_cast<long long>(b[i]),
                   static_cast<long long>(out[i]),
                   static_cast<long long>(expected));
      ++failures;
      break;
    }
  }
}

void test_saturation_corners() {
  // Unsigned overflow saturates to MAX.
  {
    uint8_t a[1] = {255}, b[1] = {1}, o[1] = {0};
    (void)kleidicv_saturating_add_u8(a, 1, b, 1, o, 1, 1, 1);
    EXPECT(o[0] == 255, "u8 sat to MAX");
  }
  // Signed positive overflow.
  {
    int8_t a[1] = {120}, b[1] = {30}, o[1] = {0};
    (void)kleidicv_saturating_add_s8(a, 1, b, 1, o, 1, 1, 1);
    EXPECT(o[0] == 127, "s8 sat to MAX");
  }
  // Signed negative overflow.
  {
    int8_t a[1] = {-120}, b[1] = {-30}, o[1] = {0};
    (void)kleidicv_saturating_add_s8(a, 1, b, 1, o, 1, 1, 1);
    EXPECT(o[0] == -128, "s8 sat to MIN");
  }
  // 64-bit cases.
  {
    uint64_t a[1] = {std::numeric_limits<uint64_t>::max()}, b[1] = {1}, o[1] = {0};
    (void)kleidicv_saturating_add_u64(a, 8, b, 8, o, 8, 1, 1);
    EXPECT(o[0] == std::numeric_limits<uint64_t>::max(), "u64 sat to MAX");
  }
  {
    int64_t a[1] = {std::numeric_limits<int64_t>::max()}, b[1] = {1}, o[1] = {0};
    (void)kleidicv_saturating_add_s64(a, 8, b, 8, o, 8, 1, 1);
    EXPECT(o[0] == std::numeric_limits<int64_t>::max(), "s64 sat to MAX");
  }
  {
    int64_t a[1] = {std::numeric_limits<int64_t>::min()}, b[1] = {-1}, o[1] = {0};
    (void)kleidicv_saturating_add_s64(a, 8, b, 8, o, 8, 1, 1);
    EXPECT(o[0] == std::numeric_limits<int64_t>::min(), "s64 sat to MIN");
  }
}

void test_null_pointer() {
  uint8_t buf[4] = {0};
  EXPECT(kleidicv_saturating_add_u8(nullptr, 1, buf, 1, buf, 1, 1, 1) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null check missing");
}

void test_stride_padding_untouched() {
  uint8_t a[6] = {1, 2, 0xAA, 3, 4, 0xBB};
  uint8_t b[6] = {10, 20, 0xCC, 30, 40, 0xDD};
  uint8_t out[6] = {0};
  (void)kleidicv_saturating_add_u8(a, 3, b, 3, out, 3, 2, 2);
  EXPECT(out[0] == 11 && out[1] == 22 && out[3] == 33 && out[4] == 44, "values");
  EXPECT(out[2] == 0 && out[5] == 0, "padding clobbered");
}

}  // namespace

int main() {
  test_backend_active();
  test_bulk_vs_reference<uint8_t>("u8");
  test_bulk_vs_reference<int8_t>("s8");
  test_bulk_vs_reference<uint16_t>("u16");
  test_bulk_vs_reference<int16_t>("s16");
  test_bulk_vs_reference<uint32_t>("u32");
  test_bulk_vs_reference<int32_t>("s32");
  test_bulk_vs_reference<uint64_t>("u64");
  test_bulk_vs_reference<int64_t>("s64");
  test_saturation_corners();
  test_null_pointer();
  test_stride_padding_untouched();
  if (failures == 0) {
    std::printf("[test_add] all checks passed\n");
    return 0;
  }
  std::printf("[test_add] %d FAILURES\n", failures);
  return 1;
}
