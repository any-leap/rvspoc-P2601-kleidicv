// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SMALL_BUFFER_H
#define KLEIDICV_SMALL_BUFFER_H

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace kleidicv {

// A buffer that goes on the stack if it's small enough, and otherwise is
// allocated on the heap. Similar to the "small vector" idea, but its size is
// fixed at creation time.
template <typename T, size_t SizeOnStack>
class SmallBuffer {
 public:
  static_assert(std::is_trivial_v<T>);

  explicit SmallBuffer(size_t size) : ptr_(buf_), buf_{} {
    if (size > SizeOnStack) {
      size_t allocation_size = 0;
      if (__builtin_mul_overflow(size, sizeof(T), &allocation_size)) {
        ptr_ = nullptr;
      } else {
        ptr_ = reinterpret_cast<T *>(std::malloc(allocation_size));
      }
    }
  }

  // non-copyable
  SmallBuffer(const SmallBuffer &) = delete;
  SmallBuffer &operator=(const SmallBuffer &) = delete;

  SmallBuffer(SmallBuffer &&other) noexcept : ptr_(buf_) {
    if (other.ptr_ == other.buf_) {
      std::memcpy(buf_, other.buf_, sizeof(buf_));
      ptr_ = buf_;
    } else {
      ptr_ = other.ptr_;
      other.ptr_ = other.buf_;
    }
  }

  SmallBuffer &operator=(SmallBuffer &&) = delete;

  ~SmallBuffer() {
    if (ptr_ != buf_) {
      std::free(ptr_);
    }
  }

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }

 private:
  static constexpr size_t kInlineSize = SizeOnStack == 0 ? 1 : SizeOnStack;

  T *ptr_;
  T buf_[kInlineSize];
};

}  // namespace kleidicv

#endif  // KLEIDICV_SMALL_BUFFER_H
