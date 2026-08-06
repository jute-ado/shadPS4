// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <concepts>
#include <limits>
#include <utility>

#include <gtest/gtest.h>

#include "core/address_space.h"
#include "core/address_space_backing_size.h"

static_assert(std::same_as<decltype(std::declval<const Core::AddressSpace&>().BackingSize()), u64>);
static_assert(noexcept(std::declval<const Core::AddressSpace&>().BackingSize()));

TEST(AddressSpaceBackingSize, UsesThePhysicalMemoryBaseWithoutExtraDmem) {
    EXPECT_EQ(Core::AddressSpaceBaseBackingSize, 8448_MB);
    EXPECT_EQ(Core::CalculateAddressSpaceBackingSize(0), Core::AddressSpaceBaseBackingSize);
}

TEST(AddressSpaceBackingSize, AddsExtraDmemWithoutMutatingLaterCalculations) {
    constexpr s64 ExtraDmemMBytes = 2048;
    constexpr u64 ExpectedSize = 10496_MB;

    const auto first = Core::CalculateAddressSpaceBackingSize(ExtraDmemMBytes);
    const auto base_after_first = Core::CalculateAddressSpaceBackingSize(0);
    const auto repeated = Core::CalculateAddressSpaceBackingSize(ExtraDmemMBytes);

    EXPECT_EQ(first, ExpectedSize);
    EXPECT_EQ(base_after_first, Core::AddressSpaceBaseBackingSize);
    EXPECT_EQ(repeated, first);
}

TEST(AddressSpaceBackingSize, RejectsInvalidExtraDmemWithoutOverflow) {
    constexpr s64 MaximumExtraDmemMBytes = static_cast<s64>(
        (std::numeric_limits<u64>::max() - Core::AddressSpaceBaseBackingSize) / 1_MB);

    ASSERT_TRUE(Core::CalculateAddressSpaceBackingSize(MaximumExtraDmemMBytes).has_value());
    EXPECT_FALSE(Core::CalculateAddressSpaceBackingSize(MaximumExtraDmemMBytes + 1).has_value());
    EXPECT_FALSE(Core::CalculateAddressSpaceBackingSize(-1).has_value());
}
