// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/device_address_registry.h"

namespace VideoCore {
namespace {

TEST(DeviceAddressRegistry, IdentifiesActiveAndRecentlyRetiredBufferRanges) {
    DeviceAddressRegistry registry;
    registry.Register(0x200000, 0x3000, 0x81000000, 0x1234);

    auto match = registry.Find(0x201234);
    ASSERT_TRUE(match.has_value());
    EXPECT_TRUE(match->active);
    EXPECT_EQ(match->record.device_address, 0x200000U);
    EXPECT_EQ(match->record.guest_address, 0x81000000U);
    EXPECT_EQ(match->offset, 0x1234U);

    registry.Retire(0x200000);
    match = registry.Find(0x201234);
    ASSERT_TRUE(match.has_value());
    EXPECT_FALSE(match->active);
    EXPECT_EQ(match->offset, 0x1234U);
}

TEST(DeviceAddressRegistry, ReportsNearestPrecedingRangeForAnOutOfBoundsAddress) {
    DeviceAddressRegistry registry;
    registry.Register(0x300000, 0x1000, 0x82000000, 0x5678);

    const auto match = registry.FindNearestPreceding(0x301800);
    ASSERT_TRUE(match.has_value());
    EXPECT_TRUE(match->active);
    EXPECT_EQ(match->offset, 0x1800U);
    EXPECT_EQ(match->record.size, 0x1000U);
}

TEST(DeviceAddressRegistry, FindsHostAllocationByGuestAddressAfterDeviceFault) {
    DeviceAddressRegistry registry;
    registry.Register(0x500000, 0x4000, 0x32ca78000, 0x9abc);

    const auto match = registry.FindGuest(0x32ca79000);
    ASSERT_TRUE(match.has_value());
    EXPECT_TRUE(match->active);
    EXPECT_EQ(match->record.device_address, 0x500000U);
    EXPECT_EQ(match->offset, 0x1000U);
}

} // namespace
} // namespace VideoCore
