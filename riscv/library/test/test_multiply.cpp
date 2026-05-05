// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// scale param is intentionally ignored (matches upstream's TODO state); we
// pass scale=1.0 in all calls below.

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
  std::printf("[test_multiply] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

template <typename T>
T ref(T a, T b) {
  using W = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
  W w = static_cast<W>(a) * static_cast<W>(b);
  if constexpr (std::is_signed_v<T>) {
    constexpr W kMax = std::numeric_limits<T>::max();
    constexpr W kMin = std::numeric_limits<T>::min();
    if (w > kMax) return std::numeric_limits<T>::max();
    if (w < kMin) return std::numeric_limits<T>::min();
  } else {
    constexpr W kMax = std::numeric_limits<T>::max();
    if (w > kMax) return std::numeric_limits<T>::max();
  }
  return static_cast<T>(w);
}

#define BIND(T, FN)                                                          \
  inline kleidicv_error_t call(const T *a, size_t sa, const T *b, size_t sb, \
                               T *d, size_t sd, size_t w, size_t h) {        \
    return FN(a, sa, b, sb, d, sd, w, h, 1.0);                               \
  }

template <typename T> kleidicv_error_t call_t(const T *, size_t, const T *, size_t, T *, size_t, size_t, size_t);
template <> kleidicv_error_t call_t<uint8_t>(const uint8_t *a, size_t sa, const uint8_t *b, size_t sb, uint8_t *d, size_t sd, size_t w, size_t h) { return kleidicv_saturating_multiply_u8(a, sa, b, sb, d, sd, w, h, 1.0); }
template <> kleidicv_error_t call_t<int8_t>(const int8_t *a, size_t sa, const int8_t *b, size_t sb, int8_t *d, size_t sd, size_t w, size_t h) { return kleidicv_saturating_multiply_s8(a, sa, b, sb, d, sd, w, h, 1.0); }
template <> kleidicv_error_t call_t<uint16_t>(const uint16_t *a, size_t sa, const uint16_t *b, size_t sb, uint16_t *d, size_t sd, size_t w, size_t h) { return kleidicv_saturating_multiply_u16(a, sa, b, sb, d, sd, w, h, 1.0); }
template <> kleidicv_error_t call_t<int16_t>(const int16_t *a, size_t sa, const int16_t *b, size_t sb, int16_t *d, size_t sd, size_t w, size_t h) { return kleidicv_saturating_multiply_s16(a, sa, b, sb, d, sd, w, h, 1.0); }
template <> kleidicv_error_t call_t<int32_t>(const int32_t *a, size_t sa, const int32_t *b, size_t sb, int32_t *d, size_t sd, size_t w, size_t h) { return kleidicv_saturating_multiply_s32(a, sa, b, sb, d, sd, w, h, 1.0); }

template <typename T>
void test_bulk(const char *tag) {
  constexpr size_t kW = 257, kH = 3;
  std::vector<T> a(kW * kH), b(kW * kH), out(kW * kH);
  using S = std::make_signed_t<T>;
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<T>(static_cast<S>(i) * 13 - 7);
    b[i] = static_cast<T>(static_cast<S>(i * i) - static_cast<S>(i) * 5 + 3);
  }
  (void)call_t<T>(a.data(), kW * sizeof(T), b.data(), kW * sizeof(T),
                  out.data(), kW * sizeof(T), kW, kH);
  for (size_t i = 0; i < a.size(); ++i) {
    T expected = ref<T>(a[i], b[i]);
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

void test_corners() {
  // u8 sat: 200 * 200 = 40000 → 255
  uint8_t a8[1] = {200}, b8[1] = {200}, o8[1] = {0};
  (void)kleidicv_saturating_multiply_u8(a8, 1, b8, 1, o8, 1, 1, 1, 1.0);
  EXPECT(o8[0] == 255, "u8 sat");
  // s8: -100 * -100 = 10000 → 127
  int8_t as[1] = {-100}, bs[1] = {-100}, os[1] = {0};
  (void)kleidicv_saturating_multiply_s8(as, 1, bs, 1, os, 1, 1, 1, 1.0);
  EXPECT(os[0] == 127, "s8 pos sat");
  // s8: 100 * -100 = -10000 → -128
  int8_t a2[1] = {100}, b2[1] = {-100}, o2[1] = {0};
  (void)kleidicv_saturating_multiply_s8(a2, 1, b2, 1, o2, 1, 1, 1, 1.0);
  EXPECT(o2[0] == -128, "s8 neg sat");
}

void test_null() {
  uint8_t buf[1] = {0};
  EXPECT(kleidicv_saturating_multiply_u8(nullptr, 1, buf, 1, buf, 1, 1, 1, 1.0) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null");
}

}  // namespace

int main() {
  test_backend_active();
  test_bulk<uint8_t>("u8");
  test_bulk<int8_t>("s8");
  test_bulk<uint16_t>("u16");
  test_bulk<int16_t>("s16");
  test_bulk<int32_t>("s32");
  test_corners();
  test_null();
  if (failures == 0) {
    std::printf("[test_multiply] all checks passed\n");
    return 0;
  }
  std::printf("[test_multiply] %d FAILURES\n", failures);
  return 1;
}
