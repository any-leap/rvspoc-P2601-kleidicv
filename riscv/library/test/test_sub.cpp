// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for kleidicv_saturating_sub_*.

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
  std::printf("[test_sub] active backend = %s\n", b);
  const char *expected = std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv";
  if (std::strcmp(b, expected) != 0) std::exit(2);
}

template <typename T>
T ref_sat_sub(T a, T b) {
  if constexpr (sizeof(T) == 8) {
    if constexpr (std::is_unsigned_v<T>) {
      return a < b ? T{0} : static_cast<T>(a - b);
    } else {
      using U = std::make_unsigned_t<T>;
      U diff = static_cast<U>(static_cast<U>(a) - static_cast<U>(b));
      T r = static_cast<T>(diff);
      bool sa = a < 0, sb = b < 0;
      if (sa != sb && (r < 0) != sa)
        return sa ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
      return r;
    }
  } else {
    long long w = static_cast<long long>(a) - static_cast<long long>(b);
    constexpr long long kMax = static_cast<long long>(std::numeric_limits<T>::max());
    constexpr long long kMin = static_cast<long long>(std::numeric_limits<T>::min());
    if (w > kMax) return std::numeric_limits<T>::max();
    if (w < kMin) return std::numeric_limits<T>::min();
    return static_cast<T>(w);
  }
}

template <typename T>
kleidicv_error_t call(const T *a, size_t sa, const T *b, size_t sb, T *d,
                      size_t sd, size_t w, size_t h);

#define BIND(T, FN)                                                          \
  template <>                                                                \
  kleidicv_error_t call<T>(const T *a, size_t sa, const T *b, size_t sb,     \
                           T *d, size_t sd, size_t w, size_t h) {            \
    return FN(a, sa, b, sb, d, sd, w, h);                                    \
  }

BIND(uint8_t, kleidicv_saturating_sub_u8)
BIND(int8_t, kleidicv_saturating_sub_s8)
BIND(uint16_t, kleidicv_saturating_sub_u16)
BIND(int16_t, kleidicv_saturating_sub_s16)
BIND(uint32_t, kleidicv_saturating_sub_u32)
BIND(int32_t, kleidicv_saturating_sub_s32)
BIND(uint64_t, kleidicv_saturating_sub_u64)
BIND(int64_t, kleidicv_saturating_sub_s64)
#undef BIND

template <typename T>
void test_bulk(const char *tag) {
  constexpr size_t kW = 257, kH = 3;
  std::vector<T> a(kW * kH), b(kW * kH), out(kW * kH);
  using S = std::make_signed_t<T>;
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<T>(static_cast<S>(i) * 11 - 17);
    b[i] = static_cast<T>(static_cast<S>(i * i) - static_cast<S>(i) * 3 + 5);
  }
  kleidicv_error_t err =
      call<T>(a.data(), kW * sizeof(T), b.data(), kW * sizeof(T), out.data(),
              kW * sizeof(T), kW, kH);
  EXPECT(err == KLEIDICV_OK, "bulk error");
  for (size_t i = 0; i < out.size(); ++i) {
    T expected = ref_sat_sub<T>(a[i], b[i]);
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
  {
    uint8_t a[1] = {0}, b[1] = {1}, o[1] = {0};
    (void)kleidicv_saturating_sub_u8(a, 1, b, 1, o, 1, 1, 1);
    EXPECT(o[0] == 0, "u8 underflow -> 0");
  }
  {
    int8_t a[1] = {-120}, b[1] = {30}, o[1] = {0};
    (void)kleidicv_saturating_sub_s8(a, 1, b, 1, o, 1, 1, 1);
    EXPECT(o[0] == -128, "s8 underflow -> MIN");
  }
  {
    int8_t a[1] = {120}, b[1] = {-30}, o[1] = {0};
    (void)kleidicv_saturating_sub_s8(a, 1, b, 1, o, 1, 1, 1);
    EXPECT(o[0] == 127, "s8 overflow -> MAX");
  }
  {
    int64_t a[1] = {std::numeric_limits<int64_t>::min()}, b[1] = {1}, o[1] = {0};
    (void)kleidicv_saturating_sub_s64(a, 8, b, 8, o, 8, 1, 1);
    EXPECT(o[0] == std::numeric_limits<int64_t>::min(), "s64 sat to MIN");
  }
  {
    uint64_t a[1] = {0}, b[1] = {std::numeric_limits<uint64_t>::max()}, o[1] = {0};
    (void)kleidicv_saturating_sub_u64(a, 8, b, 8, o, 8, 1, 1);
    EXPECT(o[0] == 0, "u64 underflow -> 0");
  }
}

void test_null_pointer() {
  uint8_t buf[4] = {0};
  EXPECT(kleidicv_saturating_sub_u8(nullptr, 1, buf, 1, buf, 1, 1, 1) ==
             KLEIDICV_ERROR_NULL_POINTER,
         "null check missing");
}

}  // namespace

int main() {
  test_backend_active();
  test_bulk<uint8_t>("u8");
  test_bulk<int8_t>("s8");
  test_bulk<uint16_t>("u16");
  test_bulk<int16_t>("s16");
  test_bulk<uint32_t>("u32");
  test_bulk<int32_t>("s32");
  test_bulk<uint64_t>("u64");
  test_bulk<int64_t>("s64");
  test_corners();
  test_null_pointer();
  if (failures == 0) {
    std::printf("[test_sub] all checks passed\n");
    return 0;
  }
  std::printf("[test_sub] %d FAILURES\n", failures);
  return 1;
}
