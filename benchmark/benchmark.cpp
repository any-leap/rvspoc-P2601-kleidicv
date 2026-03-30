// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

#include "kleidicv/kleidicv.h"

// These variables can be set runtime, from command line
extern size_t image_width, image_height;

// Setting up buffers can be time-consuming and is not of interest when
// benchmarking. Therefore endeavour to reuse buffers where possible. This
// function creates a single shared buffer for each pair of template arguments.
// All bytes of the buffer are filled with the Value argument.
template <int PixelSize, int Value>
uint8_t* get_buffer() {
  static std::vector<uint8_t> result(image_width * image_height * PixelSize,
                                     Value);
  return result.data();
}

// Get a buffer suitable for using as the first input buffer.
template <typename T, int Channels = 1>
const T* get_source_buffer_a() {
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0xA3>());
}

// Get a buffer suitable for using as the second input buffer.
template <typename T, int Channels = 1>
const T* get_source_buffer_b() {
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0x9E>());
}

// Get a buffer suitable for using as the third input buffer.
template <typename T, int Channels = 1>
const T* get_source_buffer_c() {
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0xB8>());
}

// Get a buffer suitable for using as the forth input buffer.
template <typename T, int Channels = 1>
const T* get_source_buffer_d() {
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0x4A>());
}

// Get a buffer suitable for using as the destination buffer.
template <typename T, int Channels = 1>
T* get_destination_buffer_a() {
  // Value argument is only used here to differentiate from the source buffers.
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0xC1>());
}

// Get a buffer suitable for using as the second destination buffer.
template <typename T, int Channels = 1>
T* get_destination_buffer_b() {
  // Value argument is only used here to differentiate from the source buffers.
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0x83>());
}

// Get a buffer suitable for using as the third destination buffer.
template <typename T, int Channels = 1>
T* get_destination_buffer_c() {
  // Value argument is only used here to differentiate from the source buffers.
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0xF6>());
}

// Get a buffer suitable for using as the forth destination buffer.
template <typename T, int Channels = 1>
T* get_destination_buffer_d() {
  // Value argument is only used here to differentiate from the source buffers.
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0x7D>());
}

// A hook before and after the benchmarks can be useful in some environments.
#ifndef KLEIDICV_BENCHMARK_PRE_HOOK
#define KLEIDICV_BENCHMARK_PRE_HOOK()
#endif
#ifndef KLEIDICV_BENCHMARK_POST_HOOK
#define KLEIDICV_BENCHMARK_POST_HOOK()
#endif

// Warms up the functor then benchmarks it.
template <typename F>
void bench_functor(benchmark::State& state, F functor) {
  // warm up
  functor();

  KLEIDICV_BENCHMARK_PRE_HOOK();

  for (auto _ : state) {
    // This code gets benchmarked
    functor();
  }

  KLEIDICV_BENCHMARK_POST_HOOK();
}

template <typename T, typename Function>
static void bench_binary_op(Function f, benchmark::State& state) {
  bench_functor(state, [f]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T),
            get_source_buffer_b<T>(), image_width * sizeof(T),
            get_destination_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height);
  });
}

#define BENCH_BINARY_OP(name, type)                \
  static void name(benchmark::State& state) {      \
    bench_binary_op<type>(kleidicv_##name, state); \
  }                                                \
  BENCHMARK(name)

BENCH_BINARY_OP(saturating_add_s8, int8_t);
BENCH_BINARY_OP(saturating_add_u8, uint8_t);
BENCH_BINARY_OP(saturating_add_s16, int16_t);
BENCH_BINARY_OP(saturating_add_u16, uint16_t);
BENCH_BINARY_OP(saturating_add_s32, int32_t);
BENCH_BINARY_OP(saturating_add_u32, uint32_t);
BENCH_BINARY_OP(saturating_add_s64, int64_t);
BENCH_BINARY_OP(saturating_add_u64, uint64_t);

BENCH_BINARY_OP(saturating_sub_s8, int8_t);
BENCH_BINARY_OP(saturating_sub_u8, uint8_t);
BENCH_BINARY_OP(saturating_sub_s16, int16_t);
BENCH_BINARY_OP(saturating_sub_u16, uint16_t);
BENCH_BINARY_OP(saturating_sub_s32, int32_t);
BENCH_BINARY_OP(saturating_sub_u32, uint32_t);
BENCH_BINARY_OP(saturating_sub_s64, int64_t);
BENCH_BINARY_OP(saturating_sub_u64, uint64_t);

BENCH_BINARY_OP(saturating_absdiff_s8, int8_t);
BENCH_BINARY_OP(saturating_absdiff_u8, uint8_t);
BENCH_BINARY_OP(saturating_absdiff_s16, int16_t);
BENCH_BINARY_OP(saturating_absdiff_u16, uint16_t);
BENCH_BINARY_OP(saturating_absdiff_s32, int32_t);

template <typename T, typename Function>
static void bench_saturating_multiply(Function f, benchmark::State& state) {
  bench_functor(state, [f]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T),
            get_source_buffer_b<T>(), image_width * sizeof(T),
            get_destination_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, 0.0);
  });
}

#define BENCH_SATURATING_MULTIPLY(name, type)                \
  static void name(benchmark::State& state) {                \
    bench_saturating_multiply<type>(kleidicv_##name, state); \
  }                                                          \
  BENCHMARK(name)

BENCH_SATURATING_MULTIPLY(saturating_multiply_s8, int8_t);
BENCH_SATURATING_MULTIPLY(saturating_multiply_u8, uint8_t);
BENCH_SATURATING_MULTIPLY(saturating_multiply_s16, int16_t);
BENCH_SATURATING_MULTIPLY(saturating_multiply_u16, uint16_t);
BENCH_SATURATING_MULTIPLY(saturating_multiply_s32, int32_t);

static void saturating_add_abs_with_threshold_s16(benchmark::State& state) {
  bench_functor(state, []() {
    (void)kleidicv_saturating_add_abs_with_threshold_s16(
        get_source_buffer_a<int16_t>(), image_width * sizeof(int16_t),
        get_source_buffer_b<int16_t>(), image_width * sizeof(int16_t),
        get_destination_buffer_a<int16_t>(), image_width * sizeof(int16_t),
        image_width, image_height, 0x1000);
  });
}
BENCHMARK(saturating_add_abs_with_threshold_s16);

BENCH_BINARY_OP(bitwise_and, uint8_t);
BENCH_BINARY_OP(compare_equal_u8, uint8_t);
BENCH_BINARY_OP(compare_greater_u8, uint8_t);

static void threshold_binary_u8(benchmark::State& state) {
  bench_functor(state, []() {
    (void)kleidicv_threshold_binary_u8(
        get_source_buffer_a<uint8_t>(), image_width * sizeof(uint8_t),
        get_destination_buffer_a<uint8_t>(), image_width * sizeof(uint8_t),
        image_width, image_height, 0x90, 0xF0);
  });
}
BENCHMARK(threshold_binary_u8);

static void count_nonzeros_u8(benchmark::State& state) {
  bench_functor(state, []() {
    size_t res = 0;
    (void)kleidicv_count_nonzeros_u8(get_source_buffer_a<uint8_t>(),
                                     image_width * sizeof(uint8_t), image_width,
                                     image_height, &res);
  });
}
BENCHMARK(count_nonzeros_u8);

template <typename T, int kAngle>
static void bench_rotate(benchmark::State& state) {
  bench_functor(state, []() {
    (void)kleidicv_rotate(get_source_buffer_a<T>(), image_width * sizeof(T),
                          image_width, image_height,
                          get_destination_buffer_a<T>(),
                          image_height * sizeof(T), kAngle, sizeof(T));
  });
}

#define BENCH_ROTATE(name, type, angle)       \
  static void name(benchmark::State& state) { \
    bench_rotate<type, angle>(state);         \
  }                                           \
  BENCHMARK(name)

BENCH_ROTATE(rotate_cw_u8, uint8_t, 90);
BENCH_ROTATE(rotate_cw_u16, uint16_t, 90);
BENCH_ROTATE(rotate_cw_u32, uint32_t, 90);
BENCH_ROTATE(rotate_cw_u64, uint64_t, 90);
BENCH_ROTATE(rotate_cw_u24, uint8_t[3], 90);
BENCH_ROTATE(rotate_cw_u48, uint16_t[3], 90);

BENCH_ROTATE(rotate_ccw_u8, uint8_t, -90);
BENCH_ROTATE(rotate_ccw_u16, uint16_t, -90);
BENCH_ROTATE(rotate_ccw_u32, uint32_t, -90);
BENCH_ROTATE(rotate_ccw_u64, uint64_t, -90);
BENCH_ROTATE(rotate_ccw_u24, uint8_t[3], -90);
BENCH_ROTATE(rotate_ccw_u48, uint16_t[3], -90);

template <typename T>
static void bench_transpose(benchmark::State& state) {
  bench_functor(state, []() {
    (void)kleidicv_transpose(get_source_buffer_a<T>(), image_width * sizeof(T),
                             get_destination_buffer_a<T>(),
                             image_height * sizeof(T), image_width,
                             image_height, sizeof(T));
  });
}

#define BENCH_TRANSPOSE(name, type)                                           \
  static void name(benchmark::State& state) { bench_transpose<type>(state); } \
  BENCHMARK(name)

BENCH_TRANSPOSE(transpose_u8, uint8_t);
BENCH_TRANSPOSE(transpose_u16, uint16_t);
BENCH_TRANSPOSE(transpose_u32, uint32_t);
BENCH_TRANSPOSE(transpose_u64, uint64_t);
BENCH_TRANSPOSE(transpose_u24, uint8_t[3]);
BENCH_TRANSPOSE(transpose_u48, uint16_t[3]);

template <typename T, size_t channels>
static void bench_split(benchmark::State& state) {
  bench_functor(state, []() {
    void* dst_data[] = {
        get_destination_buffer_a<T>(), get_destination_buffer_b<T>(),
        get_destination_buffer_c<T>(), get_destination_buffer_d<T>()};
    size_t dst_strides[] = {image_width * sizeof(T), image_width * sizeof(T),
                            image_width * sizeof(T), image_width * sizeof(T)};
    (void)kleidicv_split(
        get_source_buffer_a<T, channels>(), image_width * sizeof(T) * channels,
        dst_data, dst_strides, image_width, image_height, channels, sizeof(T));
  });
}

#define BENCH_SPLIT(name, type, channels)     \
  static void name(benchmark::State& state) { \
    bench_split<type, channels>(state);       \
  }                                           \
  BENCHMARK(name)

BENCH_SPLIT(split_u8_2ch, uint8_t, 2);
BENCH_SPLIT(split_u16_2ch, uint16_t, 2);
BENCH_SPLIT(split_u32_2ch, uint32_t, 2);
BENCH_SPLIT(split_u64_2ch, uint64_t, 2);

BENCH_SPLIT(split_u8_3ch, uint8_t, 3);
BENCH_SPLIT(split_u16_3ch, uint16_t, 3);
BENCH_SPLIT(split_u32_3ch, uint32_t, 3);
BENCH_SPLIT(split_u64_3ch, uint64_t, 3);

BENCH_SPLIT(split_u8_4ch, uint8_t, 4);
BENCH_SPLIT(split_u16_4ch, uint16_t, 4);
BENCH_SPLIT(split_u32_4ch, uint32_t, 4);
BENCH_SPLIT(split_u64_4ch, uint64_t, 4);

template <typename T, size_t channels>
static void bench_merge(benchmark::State& state) {
  bench_functor(state, []() {
    const void* src_data[] = {
        get_source_buffer_a<T>(), get_source_buffer_b<T>(),
        get_source_buffer_c<T>(), get_source_buffer_d<T>()};
    size_t src_strides[] = {image_width * sizeof(T), image_width * sizeof(T),
                            image_width * sizeof(T), image_width * sizeof(T)};
    (void)kleidicv_merge(src_data, src_strides,
                         get_destination_buffer_a<T, channels>(),
                         image_width * sizeof(T) * channels, image_width,
                         image_height, channels, sizeof(T));
  });
}

#define BENCH_MERGE(name, type, channels)     \
  static void name(benchmark::State& state) { \
    bench_merge<type, channels>(state);       \
  }                                           \
  BENCHMARK(name)

BENCH_MERGE(merge_u8_2ch, uint8_t, 2);
BENCH_MERGE(merge_u16_2ch, uint16_t, 2);
BENCH_MERGE(merge_u32_2ch, uint32_t, 2);
BENCH_MERGE(merge_u64_2ch, uint64_t, 2);

BENCH_MERGE(merge_u8_3ch, uint8_t, 3);
BENCH_MERGE(merge_u16_3ch, uint16_t, 3);
BENCH_MERGE(merge_u32_3ch, uint32_t, 3);
BENCH_MERGE(merge_u64_3ch, uint64_t, 3);

BENCH_MERGE(merge_u8_4ch, uint8_t, 4);
BENCH_MERGE(merge_u16_4ch, uint16_t, 4);
BENCH_MERGE(merge_u32_4ch, uint32_t, 4);
BENCH_MERGE(merge_u64_4ch, uint64_t, 4);

template <typename I, typename O, size_t InChannels, size_t OutChannels,
          typename Function>
static void bench_unary_op(Function f, benchmark::State& state) {
  bench_functor(state, [f]() {
    (void)f(get_source_buffer_a<I, InChannels>(),
            image_width * InChannels * sizeof(I),
            get_destination_buffer_a<O, OutChannels>(),
            image_width * OutChannels * sizeof(O), image_width, image_height);
  });
}

#define BENCH_UNARY_OP(name, channels, type)                                \
  static void name(benchmark::State& state) {                               \
    bench_unary_op<type, type, channels, channels>(kleidicv_##name, state); \
  }                                                                         \
  BENCHMARK(name)

BENCH_UNARY_OP(exp_f32, 1, float);

#define BENCH_UNARY_OP_DIFFERENT_IO_TYPES(name, itype, otype)   \
  static void name(benchmark::State& state) {                   \
    bench_unary_op<itype, otype, 1, 1>(kleidicv_##name, state); \
  }                                                             \
  BENCHMARK(name)

BENCH_UNARY_OP_DIFFERENT_IO_TYPES(f32_to_s8, float, int8_t);
BENCH_UNARY_OP_DIFFERENT_IO_TYPES(f32_to_u8, float, uint8_t);
BENCH_UNARY_OP_DIFFERENT_IO_TYPES(s8_to_f32, int8_t, float);
BENCH_UNARY_OP_DIFFERENT_IO_TYPES(u8_to_f32, uint8_t, float);

#define BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(name, in_channels,         \
                                                out_channels, type)        \
  static void name(benchmark::State& state) {                              \
    bench_unary_op<type, type, in_channels, out_channels>(kleidicv_##name, \
                                                          state);          \
  }                                                                        \
  BENCHMARK(name)

BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(gray_to_rgb_u8, 1, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(gray_to_rgba_u8, 1, 4, uint8_t);

BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgb_to_bgr_u8, 3, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgb_to_rgb_u8, 3, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgba_to_bgra_u8, 4, 4, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgba_to_rgba_u8, 4, 4, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgb_to_bgra_u8, 3, 4, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgb_to_rgba_u8, 3, 4, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgba_to_bgr_u8, 4, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgba_to_rgb_u8, 4, 3, uint8_t);

static void min_max_loc_u8(benchmark::State& state) {
  bench_functor(state, []() {
    size_t min_offset, max_offset;
    (void)kleidicv_min_max_loc_u8(get_source_buffer_a<uint8_t>(),
                                  image_width * sizeof(uint8_t), image_width,
                                  image_height, &min_offset, &max_offset);
  });
}
BENCHMARK(min_max_loc_u8);

static void sum_f32(benchmark::State& state) {
  bench_functor(state, []() {
    float total;
    (void)kleidicv_sum_f32(get_source_buffer_a<float>(),
                           image_width * sizeof(float), image_width,
                           image_height, &total);
  });
}
BENCHMARK(sum_f32);

template <typename T, typename U, typename Function>
static void scale(Function f, float factor, float shift,
                  benchmark::State& state) {
  bench_functor(state, [f, factor, shift]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T),
            get_destination_buffer_a<U>(), image_width * sizeof(U), image_width,
            image_height, factor, shift);
  });
}

#define BENCH_SCALE(benchname, name, factor, shift, src_type, dst_type) \
  static void benchname(benchmark::State& state) {                      \
    scale<src_type, dst_type>(kleidicv_##name, factor, shift, state);   \
  }                                                                     \
  BENCHMARK(benchname)

BENCH_SCALE(scale_u8_1, scale_u8, 1.0, 4.567, uint8_t, uint8_t);
BENCH_SCALE(scale_u8_generic, scale_u8, 1.234, 4.567, uint8_t, uint8_t);
BENCH_SCALE(scale_f32_1, scale_f32, 1.0, 4.567, float, float);
BENCH_SCALE(scale_f32_generic, scale_f32, 1.234, 4.567, float, float);
BENCH_SCALE(scale_u8_f16_1, scale_u8_f16, 1.0, 10, uint8_t, float16_t);
BENCH_SCALE(scale_u8_f16_generic, scale_u8_f16, 1.234, 4.567, uint8_t,
            float16_t);

template <typename T, typename F>
static void min_max(F f, benchmark::State& state) {
  bench_functor(state, [f]() {
    T min_value = 0, max_value = 0;
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, &min_value, &max_value);
  });
}

#define BENCH_MIN_MAX(name, type)             \
  static void name(benchmark::State& state) { \
    min_max<type>(kleidicv_##name, state);    \
  }                                           \
  BENCHMARK(name)

BENCH_MIN_MAX(min_max_s8, int8_t);
BENCH_MIN_MAX(min_max_u8, uint8_t);
BENCH_MIN_MAX(min_max_s16, int16_t);
BENCH_MIN_MAX(min_max_u16, uint16_t);
BENCH_MIN_MAX(min_max_s32, int32_t);
BENCH_MIN_MAX(min_max_f32, float);

template <typename T, typename F>
static void resize(F f, size_t src_width, size_t src_height, size_t dst_width,
                   size_t dst_height, benchmark::State& state,
                   size_t channels) {
  bench_functor(
      state, [f, src_width, src_height, dst_width, dst_height, channels]() {
        (void)f(get_source_buffer_a<T>(), src_width * sizeof(T), src_width,
                src_height, get_destination_buffer_a<T>(),
                dst_width * sizeof(T), dst_width, dst_height, channels);
      });
}

template <typename T, typename F>
static void resize_upscale(F f, size_t scale_x, size_t scale_y,
                           benchmark::State& state, size_t channels = 1) {
  size_t src_width = image_width / scale_x;
  size_t src_height = image_height / scale_y;
  resize<T>(f, src_width, src_height, src_width * scale_x, src_height * scale_y,
            state, channels);
}

template <typename T, typename F>
static void resize_downscale(F f, double scale_x, double scale_y,
                             benchmark::State& state, size_t channels = 1) {
  // Casting to uint32 to avoid clang-tidy errors
  uint32_t image_width_u32 = static_cast<uint32_t>(image_width);
  uint32_t image_height_u32 = static_cast<uint32_t>(image_height);
  uint32_t dst_width = static_cast<uint32_t>(image_width_u32 / scale_x);
  uint32_t dst_height = static_cast<uint32_t>(image_height_u32 / scale_y);
  resize<T>(f, dst_width * scale_x, dst_height * scale_y, dst_width, dst_height,
            state, channels);
}

static void resize_quarter_channels(benchmark::internal::Benchmark* b) {
  b->ArgNames({"channels"});
  for (int channel : {1, 2, 3, 4}) {
    b->Args({channel});
  }
}

static void resize_quarter_u8(benchmark::State& state) {
  int channels = state.range(0);
  resize_downscale<uint8_t>(kleidicv_resize_linear_u8, 2, 2, state,
                            static_cast<size_t>(channels));
}
BENCHMARK(resize_quarter_u8)->Apply(resize_quarter_channels);

static void resize_downscale_channels(benchmark::internal::Benchmark* b) {
  b->ArgNames({"channels"});
  for (int channel : {1, 2, 3}) {
    b->Args({channel});
  }
}

static void resize_linear_u8_r2(benchmark::State& state) {
  int channels = state.range(0);
  resize_downscale<uint8_t>(kleidicv_resize_linear_u8, 1.7, 2, state,
                            static_cast<size_t>(channels));
}
BENCHMARK(resize_linear_u8_r2)->Apply(resize_downscale_channels);

static void resize_linear_u8_r3(benchmark::State& state) {
  int channels = state.range(0);
  resize_downscale<uint8_t>(kleidicv_resize_linear_u8, 2.7, 2, state,
                            static_cast<size_t>(channels));
}
BENCHMARK(resize_linear_u8_r3)->Apply(resize_downscale_channels);

static void resize_linear_2x2_u8_1ch(benchmark::State& state) {
  resize_upscale<uint8_t>(kleidicv_resize_linear_u8, 2, 2, state);
}
BENCHMARK(resize_linear_2x2_u8_1ch);

static void resize_linear_4x4_u8_1ch(benchmark::State& state) {
  resize_upscale<uint8_t>(kleidicv_resize_linear_u8, 4, 4, state);
}
BENCHMARK(resize_linear_4x4_u8_1ch);

static void resize_linear_2x2_f32_1ch(benchmark::State& state) {
  resize_upscale<float>(kleidicv_resize_linear_f32, 2, 2, state);
}
BENCHMARK(resize_linear_2x2_f32_1ch);

static void resize_linear_4x4_f32_1ch(benchmark::State& state) {
  resize_upscale<float>(kleidicv_resize_linear_f32, 4, 4, state);
}
BENCHMARK(resize_linear_4x4_f32_1ch);

static void resize_linear_8x8_f32_1ch(benchmark::State& state) {
  resize_upscale<float>(kleidicv_resize_linear_f32, 8, 8, state);
}
BENCHMARK(resize_linear_8x8_f32_1ch);

template <typename T, size_t KernelSize, int Channels, typename F>
static void separable_filter_2d(benchmark::State& state, F function) {
  std::vector<T> kernel(KernelSize, 2);

  bench_functor(state, [kernel, function]() {
    (void)function(get_source_buffer_a<T, Channels>(),
                   image_width * Channels * sizeof(T),
                   get_destination_buffer_a<T, Channels>(),
                   image_width * Channels * sizeof(T), image_width,
                   image_height, Channels, kernel.data(), KernelSize,
                   kernel.data(), KernelSize, KLEIDICV_BORDER_TYPE_REPLICATE);
  });
}

static void separable_filter_2d_u8_5x5_1ch(benchmark::State& state) {
  separable_filter_2d<uint8_t, 5, 1>(state, kleidicv_separable_filter_2d_u8);
}
BENCHMARK(separable_filter_2d_u8_5x5_1ch);

static void separable_filter_2d_u8_5x5_3ch(benchmark::State& state) {
  separable_filter_2d<uint8_t, 5, 3>(state, kleidicv_separable_filter_2d_u8);
}
BENCHMARK(separable_filter_2d_u8_5x5_3ch);

static void separable_filter_2d_u16_5x5_1ch(benchmark::State& state) {
  separable_filter_2d<uint16_t, 5, 1>(state, kleidicv_separable_filter_2d_u16);
}
BENCHMARK(separable_filter_2d_u16_5x5_1ch);

static void separable_filter_2d_u16_5x5_3ch(benchmark::State& state) {
  separable_filter_2d<uint16_t, 5, 3>(state, kleidicv_separable_filter_2d_u16);
}
BENCHMARK(separable_filter_2d_u16_5x5_3ch);

template <typename T, size_t KernelSize, int Channels, bool Binomial>
static void gaussian_blur(benchmark::State& state) {
  bench_functor(state, []() {
    (void)kleidicv_gaussian_blur_u8(
        get_source_buffer_a<T, Channels>(), image_width * Channels * sizeof(T),
        get_destination_buffer_a<T, Channels>(),
        image_width * Channels * sizeof(T), image_width, image_height, Channels,
        KernelSize, KernelSize, (Binomial ? 0.0 : 2.0), (Binomial ? 0.0 : 2.0),
        KLEIDICV_BORDER_TYPE_REFLECT);
  });
}

#define BENCH_GAUSSIAN_BLUR(kernel_size, channel_number)                                 \
  static void                                                                            \
  gaussian_blur_binomial_u8##_##kernel_size##x##kernel_size##_##channel_number##ch(      \
      benchmark::State& state) {                                                         \
    gaussian_blur<uint8_t, kernel_size, channel_number, true>(state);                    \
  }                                                                                      \
  BENCHMARK(                                                                             \
      gaussian_blur_binomial_u8##_##kernel_size##x##kernel_size##_##channel_number##ch); \
                                                                                         \
  static void                                                                            \
  gaussian_blur_custom_sigma_u8##_##kernel_size##x##kernel_size##_##channel_number##ch(  \
      benchmark::State& state) {                                                         \
    gaussian_blur<uint8_t, kernel_size, channel_number, false>(state);                   \
  }                                                                                      \
  BENCHMARK(                                                                             \
      gaussian_blur_custom_sigma_u8##_##kernel_size##x##kernel_size##_##channel_number##ch);

BENCH_GAUSSIAN_BLUR(3, 1);
BENCH_GAUSSIAN_BLUR(3, 3);
BENCH_GAUSSIAN_BLUR(3, 4);
BENCH_GAUSSIAN_BLUR(5, 1);
BENCH_GAUSSIAN_BLUR(5, 3);
BENCH_GAUSSIAN_BLUR(5, 4);
BENCH_GAUSSIAN_BLUR(7, 1);
BENCH_GAUSSIAN_BLUR(7, 3);
BENCH_GAUSSIAN_BLUR(7, 4);
BENCH_GAUSSIAN_BLUR(9, 1);
BENCH_GAUSSIAN_BLUR(9, 3);
BENCH_GAUSSIAN_BLUR(9, 4);
BENCH_GAUSSIAN_BLUR(15, 1);
BENCH_GAUSSIAN_BLUR(15, 3);
BENCH_GAUSSIAN_BLUR(15, 4);
BENCH_GAUSSIAN_BLUR(21, 1);
BENCH_GAUSSIAN_BLUR(21, 3);
BENCH_GAUSSIAN_BLUR(21, 4);

template <typename T, int Channels, typename Function>
static void median_blur(benchmark::State& state, Function func) {
  int kernel_size = state.range(0);

  bench_functor(state, [&]() {
    (void)func(
        get_source_buffer_a<T, Channels>(), image_width * Channels * sizeof(T),
        get_destination_buffer_a<T, Channels>(),
        image_width * Channels * sizeof(T), image_width, image_height, Channels,
        kernel_size, kernel_size, KLEIDICV_BORDER_TYPE_REPLICATE);
  });
}

BENCHMARK_TEMPLATE2_CAPTURE(median_blur, uint8_t, 1, , kleidicv_median_blur_u8)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7)
    ->Arg(9)
    ->Arg(11)
    ->Arg(13)
    ->Arg(15)
    ->Arg(17)
    ->Arg(27)
    ->Arg(35);
BENCHMARK_TEMPLATE2_CAPTURE(median_blur, uint8_t, 4, , kleidicv_median_blur_u8)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7)
    ->Arg(9)
    ->Arg(11)
    ->Arg(13)
    ->Arg(15)
    ->Arg(17)
    ->Arg(27)
    ->Arg(35);

BENCHMARK_TEMPLATE2_CAPTURE(median_blur, int8_t, 1, , kleidicv_median_blur_s8)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);
BENCHMARK_TEMPLATE2_CAPTURE(median_blur, int8_t, 4, , kleidicv_median_blur_s8)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);

BENCHMARK_TEMPLATE2_CAPTURE(median_blur, uint16_t, 1, ,
                            kleidicv_median_blur_u16)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);
BENCHMARK_TEMPLATE2_CAPTURE(median_blur, uint16_t, 4, ,
                            kleidicv_median_blur_u16)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);

BENCHMARK_TEMPLATE2_CAPTURE(median_blur, int16_t, 1, , kleidicv_median_blur_s16)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);
BENCHMARK_TEMPLATE2_CAPTURE(median_blur, int16_t, 4, , kleidicv_median_blur_s16)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);

BENCHMARK_TEMPLATE2_CAPTURE(median_blur, uint32_t, 1, ,
                            kleidicv_median_blur_u32)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);
BENCHMARK_TEMPLATE2_CAPTURE(median_blur, uint32_t, 4, ,
                            kleidicv_median_blur_u32)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);

BENCHMARK_TEMPLATE2_CAPTURE(median_blur, int32_t, 1, , kleidicv_median_blur_s32)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);
BENCHMARK_TEMPLATE2_CAPTURE(median_blur, int32_t, 4, , kleidicv_median_blur_s32)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);

BENCHMARK_TEMPLATE2_CAPTURE(median_blur, float, 1, , kleidicv_median_blur_f32)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);
BENCHMARK_TEMPLATE2_CAPTURE(median_blur, float, 4, , kleidicv_median_blur_f32)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7);

template <typename Function>
static void sobel_filter(Function f, benchmark::State& state) {
  bench_functor(state, [f]() {
    (void)f(get_source_buffer_a<uint8_t, 1>(), image_width * sizeof(uint8_t),
            get_destination_buffer_a<int16_t, 1>(),
            image_width * sizeof(int16_t), image_width, image_height, 1);
  });
}

static void sobel_filter_vertical(benchmark::State& state) {
  sobel_filter(kleidicv_sobel_3x3_vertical_s16_u8, state);
}
BENCHMARK(sobel_filter_vertical);

static void sobel_filter_horizontal(benchmark::State& state) {
  sobel_filter(kleidicv_sobel_3x3_horizontal_s16_u8, state);
}
BENCHMARK(sobel_filter_horizontal);

template <size_t Channels>
static void yuv422_to_rgbx(benchmark::State& state,
                           kleidicv_color_conversion_t color_format) {
  bench_functor(state, [&]() {
    (void)kleidicv_yuv_to_rgb_u8(
        get_source_buffer_a<uint8_t, 2>(),
        image_width * 2 *
            sizeof(uint8_t),  // YUV422: 2 bytes per pixel (interleaved)
        get_destination_buffer_a<uint8_t, Channels>(),
        image_width * Channels * sizeof(uint8_t), image_width, image_height,
        color_format);
  });
}

BENCHMARK_CAPTURE(yuv422_to_rgbx<3>, , KLEIDICV_YUYV_TO_BGR)
    ->Name("yuyv_to_bgr");
BENCHMARK_CAPTURE(yuv422_to_rgbx<3>, , KLEIDICV_UYVY_TO_BGR)
    ->Name("uyvy_to_bgr");
BENCHMARK_CAPTURE(yuv422_to_rgbx<3>, , KLEIDICV_YVYU_TO_BGR)
    ->Name("yvyu_to_bgr");
BENCHMARK_CAPTURE(yuv422_to_rgbx<3>, , KLEIDICV_YUYV_TO_RGB)
    ->Name("yuyv_to_rgb");
BENCHMARK_CAPTURE(yuv422_to_rgbx<3>, , KLEIDICV_UYVY_TO_RGB)
    ->Name("uyvy_to_rgb");
BENCHMARK_CAPTURE(yuv422_to_rgbx<3>, , KLEIDICV_YVYU_TO_RGB)
    ->Name("yvyu_to_rgb");
BENCHMARK_CAPTURE(yuv422_to_rgbx<4>, , KLEIDICV_YUYV_TO_BGRA)
    ->Name("yuyv_to_bgra");
BENCHMARK_CAPTURE(yuv422_to_rgbx<4>, , KLEIDICV_UYVY_TO_BGRA)
    ->Name("uyvy_to_bgra");
BENCHMARK_CAPTURE(yuv422_to_rgbx<4>, , KLEIDICV_YVYU_TO_BGRA)
    ->Name("yvyu_to_bgra");
BENCHMARK_CAPTURE(yuv422_to_rgbx<4>, , KLEIDICV_YUYV_TO_RGBA)
    ->Name("yuyv_to_rgba");
BENCHMARK_CAPTURE(yuv422_to_rgbx<4>, , KLEIDICV_UYVY_TO_RGBA)
    ->Name("uyvy_to_rgba");
BENCHMARK_CAPTURE(yuv422_to_rgbx<4>, , KLEIDICV_YVYU_TO_RGBA)
    ->Name("yvyu_to_rgba");

template <size_t Channels>
static void yuv420sp_to_rgbx(benchmark::State& state,
                             kleidicv_color_conversion_t color_format) {
  bench_functor(state, [&] {
    (void)kleidicv_yuv_semiplanar_to_rgb_u8(
        get_source_buffer_a<uint8_t, 1>(), image_width * sizeof(uint8_t),
        get_source_buffer_b<uint8_t, 1>(), (image_width / 2) * sizeof(uint8_t),
        get_destination_buffer_a<uint8_t, Channels>(),
        image_width * Channels * sizeof(uint8_t), image_width, image_height,
        color_format);
  });
}

BENCHMARK_CAPTURE(yuv420sp_to_rgbx<3>, , KLEIDICV_NV12_TO_BGR)
    ->Name("nv12_to_bgr");
BENCHMARK_CAPTURE(yuv420sp_to_rgbx<3>, , KLEIDICV_NV12_TO_RGB)
    ->Name("nv12_to_rgb");
BENCHMARK_CAPTURE(yuv420sp_to_rgbx<4>, , KLEIDICV_NV12_TO_BGRA)
    ->Name("nv12_to_bgra");
BENCHMARK_CAPTURE(yuv420sp_to_rgbx<4>, , KLEIDICV_NV12_TO_RGBA)
    ->Name("nv12_to_rgba");
BENCHMARK_CAPTURE(yuv420sp_to_rgbx<3>, , KLEIDICV_NV21_TO_BGR)
    ->Name("nv21_to_bgr");
BENCHMARK_CAPTURE(yuv420sp_to_rgbx<3>, , KLEIDICV_NV21_TO_RGB)
    ->Name("nv21_to_rgb");
BENCHMARK_CAPTURE(yuv420sp_to_rgbx<4>, , KLEIDICV_NV21_TO_BGRA)
    ->Name("nv21_to_bgra");
BENCHMARK_CAPTURE(yuv420sp_to_rgbx<4>, , KLEIDICV_NV21_TO_RGBA)
    ->Name("nv21_to_rgba");

template <size_t Channels>
static void yuv420p_to_rgbx(benchmark::State& state,
                            kleidicv_color_conversion_t color_format) {
  bench_functor(state, [&] {
    (void)kleidicv_yuv_to_rgb_u8(get_source_buffer_a<uint8_t, 2>(),
                                 image_width * sizeof(uint8_t),
                                 get_destination_buffer_a<uint8_t, Channels>(),
                                 image_width * Channels * sizeof(uint8_t),
                                 image_width, image_height, color_format);
  });
}

BENCHMARK_CAPTURE(yuv420p_to_rgbx<3>, , KLEIDICV_YV12_TO_BGR)
    ->Name("yv12_to_bgr");
BENCHMARK_CAPTURE(yuv420p_to_rgbx<3>, , KLEIDICV_YV12_TO_RGB)
    ->Name("yv12_to_rgb");
BENCHMARK_CAPTURE(yuv420p_to_rgbx<4>, , KLEIDICV_YV12_TO_BGRA)
    ->Name("yv12_to_bgra");
BENCHMARK_CAPTURE(yuv420p_to_rgbx<4>, , KLEIDICV_YV12_TO_RGBA)
    ->Name("yv12_to_rgba");
BENCHMARK_CAPTURE(yuv420p_to_rgbx<3>, , KLEIDICV_IYUV_TO_BGR)
    ->Name("iyuv_to_bgr");
BENCHMARK_CAPTURE(yuv420p_to_rgbx<3>, , KLEIDICV_IYUV_TO_RGB)
    ->Name("iyuv_to_rgb");
BENCHMARK_CAPTURE(yuv420p_to_rgbx<4>, , KLEIDICV_IYUV_TO_BGRA)
    ->Name("iyuv_to_bgra");
BENCHMARK_CAPTURE(yuv420p_to_rgbx<4>, , KLEIDICV_IYUV_TO_RGBA)
    ->Name("iyuv_to_rgba");

template <size_t Channels>
static void yuv444_to_rgbx(benchmark::State& state,
                           kleidicv_color_conversion_t color_format) {
  bench_functor(state, [&] {
    (void)kleidicv_yuv_to_rgb_u8(get_source_buffer_a<uint8_t, 3>(),
                                 image_width * 3 * sizeof(uint8_t),
                                 get_destination_buffer_a<uint8_t, Channels>(),
                                 image_width * Channels * sizeof(uint8_t),
                                 image_width, image_height, color_format);
  });
}

BENCHMARK_CAPTURE(yuv444_to_rgbx<3>, , KLEIDICV_YUV444_TO_BGR)
    ->Name("yuv444_to_bgr ");
BENCHMARK_CAPTURE(yuv444_to_rgbx<3>, , KLEIDICV_YUV444_TO_RGB)
    ->Name("yuv444_to_rgb ");
BENCHMARK_CAPTURE(yuv444_to_rgbx<4>, , KLEIDICV_YUV444_TO_BGRA)
    ->Name("yuv444_to_bgra");
BENCHMARK_CAPTURE(yuv444_to_rgbx<4>, , KLEIDICV_YUV444_TO_RGBA)
    ->Name("yuv444_to_rgba");

template <size_t Channels>
static void rgbx_to_yuv420sp(benchmark::State& state,
                             kleidicv_color_conversion_t color_format) {
  bench_functor(state, [&] {
    (void)kleidicv_rgb_to_yuv_semiplanar_u8(
        get_source_buffer_a<uint8_t, Channels>(),
        Channels * image_width * sizeof(uint8_t),
        get_destination_buffer_a<uint8_t, 1>(), image_width * sizeof(uint8_t),
        get_destination_buffer_b<uint8_t, 2>(),
        (image_width / 2) * sizeof(uint8_t), image_width, image_height,
        color_format);
  });
}

BENCHMARK_CAPTURE(rgbx_to_yuv420sp<3>, , KLEIDICV_BGR_TO_NV12)
    ->Name("bgr_to_nv12");
BENCHMARK_CAPTURE(rgbx_to_yuv420sp<3>, , KLEIDICV_RGB_TO_NV12)
    ->Name("rgb_to_nv12");
BENCHMARK_CAPTURE(rgbx_to_yuv420sp<4>, , KLEIDICV_BGRA_TO_NV12)
    ->Name("bgra_to_nv12");
BENCHMARK_CAPTURE(rgbx_to_yuv420sp<4>, , KLEIDICV_RGBA_TO_NV12)
    ->Name("rgba_to_nv12");
BENCHMARK_CAPTURE(rgbx_to_yuv420sp<3>, , KLEIDICV_BGR_TO_NV21)
    ->Name("bgr_to_nv21");
BENCHMARK_CAPTURE(rgbx_to_yuv420sp<3>, , KLEIDICV_RGB_TO_NV21)
    ->Name("rgb_to_nv21");
BENCHMARK_CAPTURE(rgbx_to_yuv420sp<4>, , KLEIDICV_BGRA_TO_NV21)
    ->Name("bgra_to_nv21");
BENCHMARK_CAPTURE(rgbx_to_yuv420sp<4>, , KLEIDICV_RGBA_TO_NV21)
    ->Name("rgba_to_nv21");

template <size_t Channels>
static void rgbx_to_yuv420p(benchmark::State& state,
                            kleidicv_color_conversion_t color_format) {
  bench_functor(state, [&] {
    (void)kleidicv_rgb_to_yuv_u8(get_source_buffer_a<uint8_t, Channels>(),
                                 Channels * image_width * sizeof(uint8_t),
                                 get_destination_buffer_a<uint8_t, 3>(),
                                 image_width * sizeof(uint8_t), image_width,
                                 image_height, color_format);
  });
}

BENCHMARK_CAPTURE(rgbx_to_yuv420p<3>, , KLEIDICV_BGR_TO_YV12)
    ->Name("bgr_to_yv12");
BENCHMARK_CAPTURE(rgbx_to_yuv420p<3>, , KLEIDICV_RGB_TO_YV12)
    ->Name("rgb_to_yv12");
BENCHMARK_CAPTURE(rgbx_to_yuv420p<4>, , KLEIDICV_BGRA_TO_YV12)
    ->Name("bgra_to_yv12");
BENCHMARK_CAPTURE(rgbx_to_yuv420p<4>, , KLEIDICV_RGBA_TO_YV12)
    ->Name("rgba_to_yv12");
BENCHMARK_CAPTURE(rgbx_to_yuv420p<3>, , KLEIDICV_BGR_TO_IYUV)
    ->Name("bgr_to_iyuv");
BENCHMARK_CAPTURE(rgbx_to_yuv420p<3>, , KLEIDICV_RGB_TO_IYUV)
    ->Name("rgb_to_iyuv");
BENCHMARK_CAPTURE(rgbx_to_yuv420p<4>, , KLEIDICV_BGRA_TO_IYUV)
    ->Name("bgra_to_iyuv");
BENCHMARK_CAPTURE(rgbx_to_yuv420p<4>, , KLEIDICV_RGBA_TO_IYUV)
    ->Name("rgba_to_iyuv");

template <size_t Channels>
static void rgbx_to_yuv444(benchmark::State& state,
                           kleidicv_color_conversion_t color_format) {
  bench_functor(state, [&] {
    (void)kleidicv_rgb_to_yuv_u8(get_source_buffer_a<uint8_t, Channels>(),
                                 image_width * Channels * sizeof(uint8_t),
                                 get_destination_buffer_a<uint8_t, 3>(),
                                 image_width * 3 * sizeof(uint8_t), image_width,
                                 image_height, color_format);
  });
}

BENCHMARK_CAPTURE(rgbx_to_yuv444<3>, , KLEIDICV_BGR_TO_YUV444)
    ->Name("bgr_to_yuv444");
BENCHMARK_CAPTURE(rgbx_to_yuv444<3>, , KLEIDICV_RGB_TO_YUV444)
    ->Name("rgb_to_yuv444");
BENCHMARK_CAPTURE(rgbx_to_yuv444<4>, , KLEIDICV_BGRA_TO_YUV444)
    ->Name("bgra_to_yuv444");
BENCHMARK_CAPTURE(rgbx_to_yuv444<4>, , KLEIDICV_RGBA_TO_YUV444)
    ->Name("rgba_to_yuv444");

template <size_t Channels>
static void rgbx_to_yuv422(benchmark::State& state,
                           kleidicv_color_conversion_t color_format) {
  bench_functor(state, [&]() {
    (void)kleidicv_rgb_to_yuv_u8(get_source_buffer_a<uint8_t, Channels>(),
                                 image_width * Channels * sizeof(uint8_t),
                                 get_destination_buffer_a<uint8_t, 2>(),
                                 image_width * 2 * sizeof(uint8_t), image_width,
                                 image_height, color_format);
  });
}

BENCHMARK_CAPTURE(rgbx_to_yuv422<3>, , KLEIDICV_BGR_TO_YUYV)
    ->Name("bgr_to_yuyv");
BENCHMARK_CAPTURE(rgbx_to_yuv422<3>, , KLEIDICV_BGR_TO_UYVY)
    ->Name("bgr_to_uyvy");
BENCHMARK_CAPTURE(rgbx_to_yuv422<3>, , KLEIDICV_BGR_TO_YVYU)
    ->Name("bgr_to_yvyu");
BENCHMARK_CAPTURE(rgbx_to_yuv422<3>, , KLEIDICV_RGB_TO_YUYV)
    ->Name("rgb_to_yuyv");
BENCHMARK_CAPTURE(rgbx_to_yuv422<3>, , KLEIDICV_RGB_TO_UYVY)
    ->Name("rgb_to_uyvy");
BENCHMARK_CAPTURE(rgbx_to_yuv422<3>, , KLEIDICV_RGB_TO_YVYU)
    ->Name("rgb_to_yvyu");
BENCHMARK_CAPTURE(rgbx_to_yuv422<4>, , KLEIDICV_BGRA_TO_YUYV)
    ->Name("bgra_to_yuyv");
BENCHMARK_CAPTURE(rgbx_to_yuv422<4>, , KLEIDICV_BGRA_TO_UYVY)
    ->Name("bgra_to_uyvy");
BENCHMARK_CAPTURE(rgbx_to_yuv422<4>, , KLEIDICV_BGRA_TO_YVYU)
    ->Name("bgra_to_yvyu");
BENCHMARK_CAPTURE(rgbx_to_yuv422<4>, , KLEIDICV_RGBA_TO_YUYV)
    ->Name("rgba_to_yuyv");
BENCHMARK_CAPTURE(rgbx_to_yuv422<4>, , KLEIDICV_RGBA_TO_UYVY)
    ->Name("rgba_to_uyvy");
BENCHMARK_CAPTURE(rgbx_to_yuv422<4>, , KLEIDICV_RGBA_TO_YVYU)
    ->Name("rgba_to_yvyu");

template <typename T, size_t KernelSize, typename Function>
static void morphology(Function f, benchmark::State& state) {
  const T border_value[4] = {};
  bench_functor(state, [f, &border_value]() {
    (void)f(get_source_buffer_a<T, 1>(), image_width * sizeof(T),
            get_destination_buffer_a<T, 1>(), image_width * sizeof(T),
            image_width, image_height, 1, KernelSize, KernelSize, 0, 0,
            KLEIDICV_BORDER_TYPE_REPLICATE,
            reinterpret_cast<const uint8_t*>(border_value), 1);
  });
}

#define BENCH_MORPHOLOGY(name, kernel_size)                                   \
  static void name##_##kernel_size##x##kernel_size(benchmark::State& state) { \
    morphology<uint8_t, kernel_size>(kleidicv_##name##_u8, state);            \
  }                                                                           \
  BENCHMARK(name##_##kernel_size##x##kernel_size)

BENCH_MORPHOLOGY(dilate, 3);
BENCH_MORPHOLOGY(dilate, 5);
BENCH_MORPHOLOGY(dilate, 17);
BENCH_MORPHOLOGY(erode, 3);
BENCH_MORPHOLOGY(erode, 5);
BENCH_MORPHOLOGY(erode, 17);

template <typename T, typename Function>
static void in_range(Function f, T lower_bound, T upper_bound,
                     benchmark::State& state) {
  bench_functor(state, [f, lower_bound, upper_bound]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T),
            get_destination_buffer_a<uint8_t>(), image_width * sizeof(uint8_t),
            image_width, image_height, lower_bound, upper_bound);
  });
}

#define BENCH_IN_RANGE(benchname, name, lower_bound, upper_bound, type) \
  static void benchname(benchmark::State& state) {                      \
    in_range<type>(kleidicv_##name, lower_bound, upper_bound, state);   \
  }                                                                     \
  BENCHMARK(benchname)

BENCH_IN_RANGE(in_range_u8, in_range_u8, 1, 2, uint8_t);
BENCH_IN_RANGE(in_range_f32, in_range_f32, 1.111, 1.112, float);

// Like kleidicv_scharr_interleaved_s16_u8 but handles edge pixels.
// Edge pixels are handled using KLEIDICV_BORDER_TYPE_REVERSE (or in OpenCV
// terminology REFLECT_101).
static void scharr_xy_border_reverse(const uint8_t* src, size_t src_stride,
                                     int16_t* dst, size_t dst_stride,
                                     size_t width, size_t height) {
  for (size_t y = 0; y < height; y++) {
    const uint8_t* src_0 = src;
    if (y > 0) {
      src_0 += src_stride * (y - 1);
    } else if (height > 1) {
      src_0 += src_stride;
    }

    const uint8_t* src_1 = src + src_stride * y;

    const uint8_t* src_2 = src;
    if (y + 1 < height) {
      src_2 += src_stride * (y + 1);
    } else if (height > 1) {
      src_2 += src_stride * (height - 2);
    }

    int16_t* dst_row = dst + y * dst_stride / sizeof(int16_t);
    for (size_t x = 0; x < width; ++x) {
      size_t x0 = 0;
      if (x > 0) {
        x0 = x - 1;
      } else if (width > 1) {
        x0 = 1;
      }

      size_t x2 = 0;
      if (x + 1 < width) {
        x2 = x + 1;
      } else if (width > 1) {
        x2 = width - 2;
      }
      dst_row[x * 2] = static_cast<int16_t>(
          (src_0[x2] + src_2[x2] - src_0[x0] - src_2[x0]) * 3 +
          (src_1[x2] - src_1[x0]) * 10);
      dst_row[x * 2 + 1] = static_cast<int16_t>(
          (src_2[x0] + src_2[x2] - src_0[x0] - src_0[x2]) * 3 +
          (src_2[x] - src_0[x]) * 10);
    }
  }
}

static constexpr size_t optical_flow_max_window_width = 40;

struct OpticalFlowBenchmarkData {
  std::vector<uint8_t> prev_image;
  std::vector<uint8_t> next_image;
  std::vector<int16_t> deriv;
  std::vector<float> prev_points;
  size_t point_count;
};

static const OpticalFlowBenchmarkData& get_optical_flow_benchmark_data() {
  static const OpticalFlowBenchmarkData data = []() {
    const size_t padded_image_width =
        image_width + optical_flow_max_window_width * 2;
    const size_t padded_image_height =
        image_height + optical_flow_max_window_width * 2;
    std::minstd_rand generator;

    OpticalFlowBenchmarkData result{};

    result.prev_image.resize(padded_image_width * padded_image_height);
    std::generate(result.prev_image.begin(), result.prev_image.end(),
                  generator);

    result.next_image.resize(result.prev_image.size());
    for (size_t i = 0; i != result.prev_image.size(); ++i) {
      // Make the next image the same as the previous image but with some noise
      // added.
      result.next_image[i] = result.prev_image[i] + (generator() & 0xF);
    }

    result.deriv.resize(result.prev_image.size() * 2);
    scharr_xy_border_reverse(result.prev_image.data(), padded_image_width,
                             result.deriv.data(),
                             padded_image_width * 2 * sizeof(int16_t),
                             padded_image_width, padded_image_height);

    const size_t point_count_x =
        std::max<size_t>(1, image_width / optical_flow_max_window_width);
    const size_t point_count_y =
        std::max<size_t>(1, image_height / optical_flow_max_window_width);
    result.point_count = point_count_x * point_count_y;
    result.prev_points.reserve(result.point_count * 2);

    // Generate regularly spaced but jittered points to sample.
    std::uniform_real_distribution<float> point_jitter{0.0F, 1.0F};
    for (size_t j = 0; j != point_count_y; ++j) {
      for (size_t i = 0; i != point_count_x; ++i) {
        float x = (i + point_jitter(generator)) * image_width / point_count_x;
        float y = (j + point_jitter(generator)) * image_height / point_count_y;
        result.prev_points.push_back(x);
        result.prev_points.push_back(y);
      }
    }

    return result;
  }();
  return data;
}

static void standalone_lucas_kanade_alg_u8(benchmark::State& state) {
  const auto& data = get_optical_flow_benchmark_data();
  const size_t padded_image_width =
      image_width + optical_flow_max_window_width * 2;
  std::vector<float> next_points(data.point_count * 2);
  std::vector<uint8_t> status(data.point_count);

  const size_t window_width = state.range(0);
  const size_t window_height = window_width;

  bench_functor(state, [&]() {
    (void)kleidicv_standalone_lucas_kanade_alg_u8(
        data.prev_image.data() + padded_image_width * window_height +
            window_width,
        padded_image_width,
        data.deriv.data() +
            (padded_image_width * window_height + window_width) * 2,
        padded_image_width * 2 * sizeof(int16_t),
        data.next_image.data() + padded_image_width * window_height +
            window_width,
        padded_image_width, image_width, image_height, 1 /*channels*/,
        data.prev_points.data(), next_points.data(), data.point_count,
        status.data(), nullptr /*err*/, window_width, window_height,
        30 /*termination_count*/, 0.0001 /*termination_epsilon*/,
        true /*get_min_eigen_vals*/, 0.0001 /*min_eigen_vals_threshold*/);
  });
}
BENCHMARK(standalone_lucas_kanade_alg_u8)
    ->ArgName("window_width")
    ->Arg(7)
    ->Arg(9)
    ->Arg(11)
    ->Arg(21);

static void optical_flow_pyr_lk_u8_from_pyramid(benchmark::State& state) {
  const auto& data = get_optical_flow_benchmark_data();
  const size_t padded_image_width =
      image_width + optical_flow_max_window_width * 2;
  const size_t image_offset =
      padded_image_width * optical_flow_max_window_width +
      optical_flow_max_window_width;
  const size_t window_width = state.range(0);
  const kleidicv_optflow_lk_context_t context{window_width,
                                              window_width,
                                              2 /*max_level*/,
                                              30 /*termination_count*/,
                                              0.0001F /*termination_epsilon*/,
                                              0.0001F /*min_eig_threshold*/,
                                              KLEIDICV_GET_MIN_EIG};

  kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid = nullptr;
  kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid = nullptr;
  if (kleidicv_build_optical_flow_pyr_lk_pyramid(
          &prev_pyramid, data.prev_image.data() + image_offset,
          padded_image_width, image_width, image_height, 1 /*channels*/,
          context.max_level + 1, context.window_width,
          context.window_height) != KLEIDICV_OK) {
    state.SkipWithError("Failed to build the previous optical flow pyramid");
    return;
  }
  if (kleidicv_build_optical_flow_pyr_lk_pyramid(
          &next_pyramid, data.next_image.data() + image_offset,
          padded_image_width, image_width, image_height, 1 /*channels*/,
          context.max_level + 1, context.window_width,
          context.window_height) != KLEIDICV_OK) {
    (void)kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid);
    state.SkipWithError("Failed to build the next optical flow pyramid");
    return;
  }

  std::vector<float> next_points(data.point_count * 2);
  std::vector<uint8_t> status(data.point_count);
  std::vector<float> err(data.point_count);

  bench_functor(state, [&]() {
    if (kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
            prev_pyramid, next_pyramid, data.prev_points.data(),
            next_points.data(), data.point_count, status.data(), err.data(),
            context) != KLEIDICV_OK) {
      state.SkipWithError(
          "kleidicv_optical_flow_pyr_lk_u8_from_pyramid failed");
      return;
    }
    benchmark::DoNotOptimize(next_points.data());
    benchmark::DoNotOptimize(status.data());
    benchmark::DoNotOptimize(err.data());
  });

  (void)kleidicv_optical_flow_pyr_lk_pyramid_release(next_pyramid);
  (void)kleidicv_optical_flow_pyr_lk_pyramid_release(prev_pyramid);
}
BENCHMARK(optical_flow_pyr_lk_u8_from_pyramid)
    ->ArgName("window_width")
    ->Arg(7)
    ->Arg(9)
    ->Arg(11)
    ->Arg(21);

static void optical_flow_pyr_lk_u8(benchmark::State& state) {
  const auto& data = get_optical_flow_benchmark_data();
  const size_t padded_image_width =
      image_width + optical_flow_max_window_width * 2;
  const size_t image_offset =
      padded_image_width * optical_flow_max_window_width +
      optical_flow_max_window_width;
  const size_t window_width = state.range(0);
  const kleidicv_optflow_lk_context_t context{window_width,
                                              window_width,
                                              2 /*max_level*/,
                                              30 /*termination_count*/,
                                              0.0001F /*termination_epsilon*/,
                                              0.0001F /*min_eig_threshold*/,
                                              KLEIDICV_GET_MIN_EIG};

  std::vector<float> next_points(data.point_count * 2);
  std::vector<uint8_t> status(data.point_count);
  std::vector<float> err(data.point_count);

  bench_functor(state, [&]() {
    if (kleidicv_optical_flow_pyr_lk_u8(
            data.prev_image.data() + image_offset, padded_image_width,
            data.next_image.data() + image_offset, padded_image_width,
            image_width, image_height, 1 /*channels*/, data.prev_points.data(),
            next_points.data(), data.point_count, status.data(), err.data(),
            context) != KLEIDICV_OK) {
      state.SkipWithError("kleidicv_optical_flow_pyr_lk_u8 failed");
      return;
    }
    benchmark::DoNotOptimize(next_points.data());
    benchmark::DoNotOptimize(status.data());
    benchmark::DoNotOptimize(err.data());
  });
}
BENCHMARK(optical_flow_pyr_lk_u8)
    ->ArgName("window_width")
    ->Arg(7)
    ->Arg(9)
    ->Arg(11)
    ->Arg(21);

template <size_t Channels>
static void blur_and_downsample_u8(benchmark::State& state) {
  bench_functor(state, []() {
    (void)kleidicv_blur_and_downsample_u8(
        get_source_buffer_a<uint8_t, Channels>(),
        image_width * sizeof(uint8_t) * Channels, image_width, image_height,
        get_destination_buffer_a<uint8_t, Channels>(),
        ((image_width + 1) / 2) * sizeof(uint8_t) * Channels, Channels,
        KLEIDICV_BORDER_TYPE_REFLECT);
  });
}
BENCHMARK(blur_and_downsample_u8<1>);
BENCHMARK(blur_and_downsample_u8<2>);
BENCHMARK(blur_and_downsample_u8<3>);
BENCHMARK(blur_and_downsample_u8<4>);

template <size_t Channels>
static void scharr_interleaved_s16_u8(benchmark::State& state) {
  bench_functor(state, []() {
    (void)kleidicv_scharr_interleaved_s16_u8(
        get_source_buffer_a<uint8_t, Channels>(),
        image_width * sizeof(uint8_t) * Channels, image_width, image_height,
        Channels, get_destination_buffer_a<int16_t, Channels * 2>(),
        (image_width - 2) * sizeof(int16_t) * Channels * 2);
  });
}
BENCHMARK(scharr_interleaved_s16_u8<1>);
BENCHMARK(scharr_interleaved_s16_u8<2>);
BENCHMARK(scharr_interleaved_s16_u8<3>);
BENCHMARK(scharr_interleaved_s16_u8<4>);

template <size_t Channels>
static void build_optical_flow_pyr_lk_pyramid(benchmark::State& state) {
  const size_t requested_level_count = static_cast<size_t>(state.range(0));
  const size_t window_width = static_cast<size_t>(state.range(1));
  const size_t window_height = window_width;

  bench_functor(state, [&]() {
    kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid = nullptr;
    const kleidicv_error_t err = kleidicv_build_optical_flow_pyr_lk_pyramid(
        &pyramid, get_source_buffer_a<uint8_t, Channels>(),
        image_width * sizeof(uint8_t) * Channels, image_width, image_height,
        Channels, requested_level_count, window_width, window_height);
    benchmark::DoNotOptimize(static_cast<int>(err));

    if (err == KLEIDICV_OK) {
      benchmark::DoNotOptimize(pyramid);
      (void)kleidicv_optical_flow_pyr_lk_pyramid_release(pyramid);
    }
  });
}

static void build_optical_flow_pyr_lk_pyramid_args(
    benchmark::internal::Benchmark* b) {
  b->ArgNames({"level_count", "window_width"});
  for (int level_count : {1, 3, 5}) {
    for (int window_width : {7, 15, 21}) {
      b->Args({level_count, window_width});
    }
  }
}

BENCHMARK(build_optical_flow_pyr_lk_pyramid<1>)
    ->Apply(build_optical_flow_pyr_lk_pyramid_args);
BENCHMARK(build_optical_flow_pyr_lk_pyramid<2>)
    ->Apply(build_optical_flow_pyr_lk_pyramid_args);
BENCHMARK(build_optical_flow_pyr_lk_pyramid<3>)
    ->Apply(build_optical_flow_pyr_lk_pyramid_args);
BENCHMARK(build_optical_flow_pyr_lk_pyramid<4>)
    ->Apply(build_optical_flow_pyr_lk_pyramid_args);

template <class ScalarType>
static const ScalarType* get_random_mapxy() {
  auto generate_mapxy = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 bytes padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe * 2);
    std::mt19937_64 rng;
    std::uniform_int_distribution<ScalarType> dist_x(0, image_width),
        dist_y(0, image_height);
    for (size_t row = 0; row < image_height; ++row) {
      for (size_t column = 0; column < image_width; ++column) {
        size_t index = row * image_stripe + column;
        v[2 * index] = dist_x(rng);
        v[2 * index + 1] = dist_y(rng);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapxy = generate_mapxy();
  return mapxy.data();
}

template <class ScalarType>
static const ScalarType* get_random_mapx() {
  auto generate_mapx = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 elements' padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe);
    std::mt19937_64 rng;
    std::uniform_int_distribution<int> dist_x(0, image_width);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        // Use a second degree function to add a nonlinear blend to the image
        v[index] = dist_x(rng);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapx = generate_mapx();
  return mapx.data();
}

template <class ScalarType>
static const ScalarType* get_random_mapy() {
  auto generate_mapx = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 elements' padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe);
    std::mt19937_64 rng;
    std::uniform_int_distribution<int> dist_y(0, image_height);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        // Use a second degree function to add a nonlinear blend to the image
        v[index] = dist_y(rng);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapy = generate_mapx();
  return mapy.data();
}

template <class ScalarType>
static const ScalarType* get_blend_mapxy() {
  auto generate_mapxy = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 elements' padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe * 2);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        // Use a second degree function to add a nonlinear blend to the image
        v[2 * index] =
            static_cast<int16_t>(column * 2 - column * column / image_width);
        v[2 * index + 1] =
            static_cast<int16_t>(row * (image_width - column) / image_width +
                                 4 * row / image_height);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapxy = generate_mapxy();
  return mapxy.data();
}

template <class ScalarType>
static const ScalarType* get_blend_mapx() {
  auto generate_mapx = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 elements' padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        // Use a second degree function to add a nonlinear blend to the image
        v[index] = static_cast<ScalarType>(
            column * 2 -
            column * column / static_cast<ScalarType>(image_width));
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapx = generate_mapx();
  return mapx.data();
}

template <class ScalarType>
static const ScalarType* get_blend_mapy() {
  auto generate_mapx = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 elements' padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        // Use a second degree function to add a nonlinear blend to the image
        v[index] = static_cast<ScalarType>(
            row * (image_width - column) /
                static_cast<ScalarType>(image_width) +
            4 * row / static_cast<ScalarType>(image_height));
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapy = generate_mapx();
  return mapy.data();
}

template <class ScalarType>
static const ScalarType* get_flip_mapxy() {
  auto generate_mapxy = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 bytes padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe * 2);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        v[2 * index] = static_cast<int16_t>(image_width - column - 1);
        v[2 * index + 1] = static_cast<int16_t>(row);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapxy = generate_mapxy();
  return mapxy.data();
}

template <class ScalarType>
static const ScalarType* get_flip_mapx() {
  auto generate_mapx = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 elements' padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        v[index] = image_width - column - 0.3F;
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapx = generate_mapx();
  return mapx.data();
}

template <class ScalarType>
static const ScalarType* get_flip_mapy() {
  auto generate_mapy = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 elements' padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        v[index] = row + 0.23F;
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapy = generate_mapy();
  return mapy.data();
}

template <class ScalarType>
static const ScalarType* get_identity_mapxy() {
  auto generate_mapxy = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 bytes padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe * 2);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        v[2 * index] = static_cast<int16_t>(column);
        v[2 * index + 1] = static_cast<int16_t>(row);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapxy = generate_mapxy();
  return mapxy.data();
}

static const uint16_t* get_random_mapfrac() {
  static const uint16_t FRAC_BITS = 5;
  static const uint16_t REMAP16POINT5_FRAC_MAX = 1 << FRAC_BITS;

  auto generate_mapfrac = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 bytes padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<uint16_t> v(image_height * image_stripe);
    std::mt19937_64 rng;
    std::uniform_int_distribution<uint16_t> dist_x(0,
                                                   REMAP16POINT5_FRAC_MAX - 1),
        dist_y(0, REMAP16POINT5_FRAC_MAX - 1);
    for (size_t row = 0; row < image_height; ++row) {
      for (size_t column = 0; column < image_width; ++column) {
        v[row * image_stripe + column] =
            dist_x(rng) | (dist_y(rng) << FRAC_BITS);
      }
    }
    return v;
  };
  static std::vector<uint16_t> mapfrac = generate_mapfrac();
  return mapfrac.data();
}

template <typename T, typename Function, typename MapFunc>
static void remap_s16(Function f, MapFunc mf, size_t channels,
                      kleidicv_border_type_t border_type,
                      benchmark::State& state) {
  const T border_value[4] = {};
  bench_functor(state, [f, mf, channels, border_type, border_value]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, get_destination_buffer_a<T>(),
            image_width * sizeof(T), image_width, image_height, channels, mf(),
            image_width * 2 * sizeof(int16_t), border_type, border_value);
  });
}

#define BENCH_REMAP_S16(benchname, name, mapfunc, channels, border_type, type) \
  static void benchname(benchmark::State& state) {                             \
    remap_s16<type>(kleidicv_##name, mapfunc, channels, border_type, state);   \
  }                                                                            \
  BENCHMARK(benchname)

BENCH_REMAP_S16(remap_s16_u8_random, remap_s16_u8, get_random_mapxy<int16_t>, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16(remap_s16_u8_blend, remap_s16_u8, get_blend_mapxy<int16_t>, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16(remap_s16_u8_flip, remap_s16_u8, get_flip_mapxy<int16_t>, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16(remap_s16_u8_identity, remap_s16_u8,
                get_identity_mapxy<int16_t>, 1, KLEIDICV_BORDER_TYPE_REPLICATE,
                uint8_t);

BENCH_REMAP_S16(remap_s16_u16_random, remap_s16_u16, get_random_mapxy<int16_t>,
                1, KLEIDICV_BORDER_TYPE_REPLICATE, uint16_t);

BENCH_REMAP_S16(remap_s16_u16_blend, remap_s16_u16, get_blend_mapxy<int16_t>, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint16_t);

BENCH_REMAP_S16(remap_s16_u16_flip, remap_s16_u16, get_flip_mapxy<int16_t>, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint16_t);

BENCH_REMAP_S16(remap_s16_u16_identity, remap_s16_u16,
                get_identity_mapxy<int16_t>, 1, KLEIDICV_BORDER_TYPE_REPLICATE,
                uint16_t);

template <typename T, typename Function, typename MapFunc>
static void remap_s16point5(Function f, MapFunc mf, size_t channels,
                            kleidicv_border_type_t border_type,
                            benchmark::State& state) {
  const T border_value[4] = {};
  bench_functor(state, [f, mf, channels, border_type, border_value]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, get_destination_buffer_a<T>(),
            image_width * sizeof(T), image_width, image_height, channels, mf(),
            image_width * 2 * sizeof(int16_t), get_random_mapfrac(),
            image_width * sizeof(uint16_t), border_type, border_value);
  });
}

#define BENCH_REMAP_S16POINT5(benchname, name, mapfunc, channels, border_type, \
                              type)                                            \
  static void benchname(benchmark::State& state) {                             \
    remap_s16point5<type>(kleidicv_##name, mapfunc, channels, border_type,     \
                          state);                                              \
  }                                                                            \
  BENCHMARK(benchname)

BENCH_REMAP_S16POINT5(remap_s16point5_u8_random, remap_s16point5_u8,
                      get_random_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u8_blend, remap_s16point5_u8,
                      get_blend_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u8_flip, remap_s16point5_u8,
                      get_flip_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u8_identity, remap_s16point5_u8,
                      get_identity_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u16_random, remap_s16point5_u16,
                      get_random_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint16_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u16_blend, remap_s16point5_u16,
                      get_blend_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint16_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u16_flip, remap_s16point5_u16,
                      get_flip_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint16_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u16_identity, remap_s16point5_u16,
                      get_identity_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint16_t);

template <typename T, typename Function, typename MapFuncX, typename MapFuncY>
static void remap_f32(Function f, MapFuncX mfx, MapFuncY mfy, size_t channels,
                      kleidicv_interpolation_type_t interpolation,
                      kleidicv_border_type_t border_type,
                      benchmark::State& state) {
  const T border_value[4] = {};
  bench_functor(state, [f, mfx, mfy, channels, interpolation, border_type,
                        border_value]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, get_destination_buffer_a<T>(),
            image_width * sizeof(T), image_width, image_height, channels, mfx(),
            image_width * sizeof(float), mfy(), image_width * sizeof(float),
            interpolation, border_type, border_value);
  });
}

#define BENCH_REMAP_F32(benchname, name, mapxfunc, mapyfunc, channels, \
                        interpolation, border_type, type)              \
  static void benchname(benchmark::State& state) {                     \
    remap_f32<type>(kleidicv_##name, mapxfunc, mapyfunc, channels,     \
                    interpolation, border_type, state);                \
  }                                                                    \
  BENCHMARK(benchname)

BENCH_REMAP_F32(remap_f32_u8_linear_random, remap_f32_u8,
                get_random_mapx<float>, get_random_mapy<float>, 1,
                KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_BORDER_TYPE_REPLICATE,
                uint8_t);

BENCH_REMAP_F32(remap_f32_u8_linear_blend, remap_f32_u8, get_blend_mapx<float>,
                get_blend_mapy<float>, 1, KLEIDICV_INTERPOLATION_LINEAR,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_F32(remap_f32_u8_linear_flip, remap_f32_u8, get_flip_mapx<float>,
                get_flip_mapy<float>, 1, KLEIDICV_INTERPOLATION_LINEAR,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_F32(remap_f32_u16_linear_random, remap_f32_u16,
                get_random_mapx<float>, get_random_mapy<float>, 1,
                KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_BORDER_TYPE_REPLICATE,
                uint16_t);

BENCH_REMAP_F32(remap_f32_u16_linear_blend, remap_f32_u16,
                get_blend_mapx<float>, get_blend_mapy<float>, 1,
                KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_BORDER_TYPE_REPLICATE,
                uint16_t);

BENCH_REMAP_F32(remap_f32_u16_linear_flip, remap_f32_u16, get_flip_mapx<float>,
                get_flip_mapy<float>, 1, KLEIDICV_INTERPOLATION_LINEAR,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint16_t);

BENCH_REMAP_F32(remap_f32_u8_nearest_random, remap_f32_u8,
                get_random_mapx<float>, get_random_mapy<float>, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                uint8_t);

BENCH_REMAP_F32(remap_f32_u8_nearest_blend, remap_f32_u8, get_blend_mapx<float>,
                get_blend_mapy<float>, 1, KLEIDICV_INTERPOLATION_NEAREST,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_F32(remap_f32_u8_nearest_flip, remap_f32_u8, get_flip_mapx<float>,
                get_flip_mapy<float>, 1, KLEIDICV_INTERPOLATION_NEAREST,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_F32(remap_f32_u16_nearest_random, remap_f32_u16,
                get_random_mapx<float>, get_random_mapy<float>, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                uint16_t);

BENCH_REMAP_F32(remap_f32_u16_nearest_blend, remap_f32_u16,
                get_blend_mapx<float>, get_blend_mapy<float>, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                uint16_t);

BENCH_REMAP_F32(remap_f32_u16_nearest_flip, remap_f32_u16, get_flip_mapx<float>,
                get_flip_mapy<float>, 1, KLEIDICV_INTERPOLATION_NEAREST,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint16_t);

// clang-format off
static const float transform_identity[] = {
  1.0, 0, 0,
  0, 1.0, 0,
  0, 0, 1.0
};

static const float transform_small[] = {
  0.8, 0.1, 2,
  0.1, 0.8, -2,
  0.001, 0.001, 1.1
};

static const float transform_bend[] = {
  200, 0, 0,
  0, 200, -2,
  0.01, 0.01, 10
};

// designed for source 512 x 512 (rotate center is at [256,256])
// rotate by 30 degrees and upscale to 2.2
static const float transform_rotate[] = {
 0.3685617397559249, -0.2421336247796208, 201.2709623094595,
 0.2158651377567587, 0.3868901230951353, 91.52518789972362,
 -0.0001175503101026296, -6.963609976572184e-05, 0.9431277488336682};

// a near transformation made by OpenCV getPerspectiveTransform
static const float transform_near[] = {
  1.015917119306434, -0.03848938238648505, 2.925193061372864,
 0.04162265799405287, 1.030708451905362, -78.33384234480749,
 -2.763770366739386e-06, -4.019862712379178e-05, 1.003055095661408};
// clang-format on

template <typename T, typename Function>
static void warp_perspective(Function f, const float transform[9],
                             size_t channels,
                             kleidicv_interpolation_type_t interpolation,
                             kleidicv_border_type_t border_type,
                             benchmark::State& state) {
  const T border_value[4] = {};
  bench_functor(state, [f, transform, channels, interpolation, border_type,
                        border_value]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, get_destination_buffer_a<T>(),
            image_width * sizeof(T), image_width, image_height, transform,
            channels, interpolation, border_type, border_value);
  });
}

#define BENCH_WARP_PERSPECTIVE(benchname, name, transform, channels, \
                               interpolation, border_type, type)     \
  static void benchname(benchmark::State& state) {                   \
    warp_perspective<type>(kleidicv_##name, transform, channels,     \
                           interpolation, border_type, state);       \
  }                                                                  \
  BENCHMARK(benchname)

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_nearest_identity,
                       warp_perspective_u8, transform_identity, 1,
                       KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_nearest_small, warp_perspective_u8,
                       transform_small, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_nearest_bend, warp_perspective_u8,
                       transform_bend, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_nearest_rotate, warp_perspective_u8,
                       transform_rotate, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_nearest_near, warp_perspective_u8,
                       transform_near, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

// WarpPerspective Linear

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_linear_identity, warp_perspective_u8,
                       transform_identity, 1, KLEIDICV_INTERPOLATION_LINEAR,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_linear_small, warp_perspective_u8,
                       transform_small, 1, KLEIDICV_INTERPOLATION_LINEAR,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_linear_bend, warp_perspective_u8,
                       transform_bend, 1, KLEIDICV_INTERPOLATION_LINEAR,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_linear_rotate, warp_perspective_u8,
                       transform_rotate, 1, KLEIDICV_INTERPOLATION_LINEAR,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_linear_near, warp_perspective_u8,
                       transform_near, 1, KLEIDICV_INTERPOLATION_LINEAR,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);
