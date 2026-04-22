// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "../../kleidicv/src/transform/add_padding_by_copy_internal.h"
#include "kleidicv/containers/small_buffer.h"

template <typename ElementType>
class SmallBufferTest : public testing::Test {};

using ElementTypes = ::testing::Types<int16_t>;

TYPED_TEST_SUITE(SmallBufferTest, ElementTypes);

// add_padding_by_copy uses one prepared-border SmallBuffer specialization for
// both constant rows and indexed border tables, but the API's dimension guards
// prevent that path from ever reaching SmallBuffer's internal
// multiplication-overflow branch. Cover that branch here with the exact
// specialization used by add_padding_by_copy.
TEST(SmallBufferAddPaddingByCopyTest, PreparedBorderAllocationOverflow) {
  constexpr size_t kElementsToOverflow =
      std::numeric_limits<size_t>::max() / sizeof(size_t) + 1;
  kleidicv::AddPaddingByCopyPreparedBorderBuffer buf{kElementsToOverflow};
  EXPECT_EQ(nullptr, buf.get());
}

TYPED_TEST(SmallBufferTest, Stack) {
  kleidicv::SmallBuffer<TypeParam, 10> buf{1};
  buf.get()[0] = -12345;
  EXPECT_EQ(-12345, buf.get()[0]);
  EXPECT_LT(abs(reinterpret_cast<ptrdiff_t>(buf.get()) -
                reinterpret_cast<ptrdiff_t>(&buf)),
            sizeof(buf));
}

TYPED_TEST(SmallBufferTest, Heap) {
  kleidicv::SmallBuffer<TypeParam, 10> buf{123};
  buf.get()[100] = -12345;
  EXPECT_EQ(-12345, buf.get()[100]);
  EXPECT_GE(abs(reinterpret_cast<ptrdiff_t>(buf.get()) -
                reinterpret_cast<ptrdiff_t>(&buf)),
            sizeof(buf));
}

TYPED_TEST(SmallBufferTest, AllocationOverflow) {
  constexpr size_t kElementsToOverflow =
      std::numeric_limits<size_t>::max() / sizeof(TypeParam) + 1;
  kleidicv::SmallBuffer<TypeParam, 10> buf{kElementsToOverflow};
  EXPECT_EQ(nullptr, buf.get());
}
