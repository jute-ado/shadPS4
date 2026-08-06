// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "core/physical_backing_provenance.h"
#include "video_core/buffer_cache/physical_backing_publication_coordinator.h"

namespace {

using Core::PhysicalBackingSpan;
using VideoCore::PhysicalBackingDeviceAddress;
using VideoCore::PhysicalBackingPublicationCoordinator;

constexpr u64 PageSize = 16_KB;
constexpr u64 ImportedBase = 0x2'0000'0000ULL;
constexpr VAddr GuestA = 0x1000'0000ULL;
constexpr VAddr GuestB = 0x2000'0000ULL;
constexpr u64 PhysicalPage = 3 * PageSize;
constexpr u64 OtherPhysicalPage = 5 * PageSize;

TEST(PhysicalBackingPublicationCoordinator, MapsPhysicalAliasesToExactImportedAddresses) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };

    const auto deltas = coordinator.MapSpans(spans);

    ASSERT_TRUE(deltas.has_value());
    ASSERT_EQ(deltas->size(), 2);
    EXPECT_EQ((*deltas)[0].guest_page, GuestA);
    EXPECT_EQ((*deltas)[0].device_address.value, ImportedBase + PhysicalPage);
    EXPECT_EQ((*deltas)[1].guest_page, GuestB);
    EXPECT_EQ((*deltas)[1].device_address.value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationCoordinator, RollsBackWholeBatchWhenOnePageIsRejected) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array conflicting_spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 8},
    };
    EXPECT_FALSE(coordinator.MapSpans(conflicting_spans));

    constexpr std::array retry_spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
    };
    const auto retry_deltas = coordinator.MapSpans(retry_spans);

    ASSERT_TRUE(retry_deltas.has_value());
    ASSERT_EQ(retry_deltas->size(), 1);
    EXPECT_EQ(retry_deltas->front().guest_page, GuestA);
    EXPECT_EQ(retry_deltas->front().device_address.value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationCoordinator, UnmapsOneAliasAndReturnsZeroBdaDelta) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));

    const auto unmap_deltas = coordinator.UnmapRange(GuestA, PageSize);

    ASSERT_TRUE(unmap_deltas.has_value());
    ASSERT_EQ(unmap_deltas->size(), 1);
    EXPECT_EQ(unmap_deltas->front().guest_page, GuestA);
    EXPECT_EQ(unmap_deltas->front().device_address.value, 0);

    constexpr std::array remap_span{
        PhysicalBackingSpan{GuestA, OtherPhysicalPage, PageSize, 8},
    };
    const auto remap_deltas = coordinator.MapSpans(remap_span);
    ASSERT_TRUE(remap_deltas.has_value());
    EXPECT_EQ(remap_deltas->front().device_address.value, ImportedBase + OtherPhysicalPage);
}

} // namespace
