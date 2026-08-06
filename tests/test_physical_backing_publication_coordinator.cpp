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
constexpr u64 OverrideBase = 0x4'0000'0000ULL;
constexpr VAddr GuestA = 0x1000'0000ULL;
constexpr VAddr GuestB = 0x2000'0000ULL;
constexpr VAddr GuestC = 0x3000'0000ULL;
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

TEST(PhysicalBackingPublicationCoordinator,
     GpuWriteSuppressesImportedPublicationForEveryPhysicalAlias) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestC, OtherPhysicalPage, PageSize, 8},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));

    const auto deltas =
        coordinator.SuppressGuestRangeForGpuWrite(GuestB + PageSize - 32, 64);

    ASSERT_TRUE(deltas.has_value());
    ASSERT_EQ(deltas->size(), 2);
    EXPECT_EQ((*deltas)[0].guest_page, GuestA);
    EXPECT_EQ((*deltas)[0].device_address.value, 0);
    EXPECT_EQ((*deltas)[1].guest_page, GuestB);
    EXPECT_EQ((*deltas)[1].device_address.value, 0);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestA)->value, 0);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestB)->value, 0);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestC)->value,
              ImportedBase + OtherPhysicalPage);
    const auto repeated = coordinator.SuppressGuestRangeForGpuWrite(GuestA, PageSize);
    ASSERT_TRUE(repeated.has_value());
    EXPECT_TRUE(repeated->empty());
    const auto unmapped = coordinator.SuppressGuestRangeForGpuWrite(GuestC + PageSize, 32);
    ASSERT_TRUE(unmapped.has_value());
    EXPECT_TRUE(unmapped->empty());

    ASSERT_TRUE(coordinator.UnmapRange(GuestA, PageSize));
    ASSERT_TRUE(coordinator.UnmapRange(GuestB, PageSize));
    constexpr std::array remapped_span{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
    };
    const auto remapped_deltas = coordinator.MapSpans(remapped_span);
    ASSERT_TRUE(remapped_deltas.has_value());
    ASSERT_EQ(remapped_deltas->size(), 1);
    EXPECT_EQ(remapped_deltas->front().guest_page, GuestA);
    EXPECT_EQ(remapped_deltas->front().device_address.value, 0);
}

TEST(PhysicalBackingPublicationCoordinator,
     RetiresSuppressedAllocationBeforePhysicalPageReuse) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array first_span{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(first_span));
    ASSERT_TRUE(coordinator.SuppressGuestRangeForGpuWrite(GuestA, PageSize));
    ASSERT_TRUE(coordinator.UnmapRange(GuestA, PageSize));
    constexpr std::array retirements{
        Core::PhysicalBackingRetirement{PhysicalPage, PageSize, 7},
    };

    ASSERT_TRUE(coordinator.RetirePhysicalAllocations(retirements));

    constexpr std::array reused_span{
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 8},
    };
    const auto reused = coordinator.MapSpans(reused_span);
    ASSERT_TRUE(reused.has_value());
    ASSERT_EQ(reused->size(), 1);
    EXPECT_EQ(reused->front().device_address.value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationCoordinator,
     GpuWriteSuppressionCanPublishSameSubmissionCacheOwnerForEveryAlias) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto suppressed = coordinator.SuppressGuestRangeForGpuWrite(GuestA + 256, 512);
    ASSERT_TRUE(suppressed.has_value());
    ASSERT_EQ(suppressed->size(), 2);
    ASSERT_EQ(coordinator.ResolveGuestPagePublication(GuestA)->value, 0);
    ASSERT_EQ(coordinator.ResolveGuestPagePublication(GuestB)->value, 0);

    const auto writer = coordinator.ActivateCachePageForGuest(
        GuestA, PhysicalBackingDeviceAddress{OverrideBase}, false);

    ASSERT_TRUE(writer.has_value());
    ASSERT_EQ(writer->deltas.size(), 2);
    EXPECT_EQ(writer->deltas[0].guest_page, GuestA);
    EXPECT_EQ(writer->deltas[0].device_address.value, OverrideBase);
    EXPECT_EQ(writer->deltas[1].guest_page, GuestB);
    EXPECT_EQ(writer->deltas[1].device_address.value, OverrideBase);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestA)->value, OverrideBase);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestB)->value, OverrideBase);
}

TEST(PhysicalBackingPublicationCoordinator,
     PublishesSuppressedGpuWriterPagesAsOneAtomicBatch) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestA + PageSize, OtherPhysicalPage, PageSize, 8},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    ASSERT_TRUE(coordinator.SuppressGuestRangeForGpuWrite(GuestA, 2 * PageSize));
    constexpr std::array requests{
        VideoCore::PhysicalBackingCachePageRequest{GuestA,
                                                   PhysicalBackingDeviceAddress{OverrideBase}},
        VideoCore::PhysicalBackingCachePageRequest{
            GuestA + PageSize, PhysicalBackingDeviceAddress{OverrideBase + PageSize}},
    };

    const auto writers = coordinator.ActivateCachePagesForGuests(requests);

    ASSERT_TRUE(writers.has_value());
    ASSERT_EQ(writers->owners.size(), 2);
    ASSERT_EQ(writers->deltas.size(), 3);
    EXPECT_EQ(writers->deltas[0].guest_page, GuestA);
    EXPECT_EQ(writers->deltas[0].device_address.value, OverrideBase);
    EXPECT_EQ(writers->deltas[1].guest_page, GuestA + PageSize);
    EXPECT_EQ(writers->deltas[1].device_address.value, OverrideBase + PageSize);
    EXPECT_EQ(writers->deltas[2].guest_page, GuestB);
    EXPECT_EQ(writers->deltas[2].device_address.value, OverrideBase);
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

TEST(PhysicalBackingPublicationCoordinator, CleanCacheOwnerUpdatesAndRestoresEveryAlias) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));

    const auto owner = coordinator.ActivateCachePage(
        PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, false);

    ASSERT_TRUE(owner.has_value());
    ASSERT_EQ(owner->deltas.size(), 2);
    EXPECT_EQ(owner->deltas[0].device_address.value, OverrideBase);
    EXPECT_EQ(owner->deltas[1].device_address.value, OverrideBase);

    const auto retired_deltas = coordinator.RetireCachePageClean(owner->token);
    ASSERT_TRUE(retired_deltas.has_value());
    ASSERT_EQ(retired_deltas->size(), 2);
    EXPECT_EQ((*retired_deltas)[0].device_address.value, ImportedBase + PhysicalPage);
    EXPECT_EQ((*retired_deltas)[1].device_address.value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationCoordinator, DirtyCacheOwnerZerosAliasesUntilPhysicalWriteback) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto owner = coordinator.ActivateCachePage(
        PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, false);
    ASSERT_TRUE(owner.has_value());

    const auto retirement = coordinator.RetireCachePageGpuDirty(owner->token);

    ASSERT_TRUE(retirement.has_value());
    ASSERT_EQ(retirement->deltas.size(), 2);
    EXPECT_EQ(retirement->deltas[0].device_address.value, 0);
    EXPECT_EQ(retirement->deltas[1].device_address.value, 0);
    EXPECT_EQ(retirement->writeback.physical_offset, PhysicalPage);

    bool wrote_physical_backing = false;
    const auto restored_deltas = coordinator.CommitCachePageWriteback(
        retirement->writeback, [&] {
            wrote_physical_backing = true;
            return true;
        });
    ASSERT_TRUE(restored_deltas.has_value());
    EXPECT_TRUE(wrote_physical_backing);
    ASSERT_EQ(restored_deltas->size(), 2);
    EXPECT_EQ((*restored_deltas)[0].device_address.value, ImportedBase + PhysicalPage);
    EXPECT_EQ((*restored_deltas)[1].device_address.value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationCoordinator, CpuWriteThroughAliasRetiresPhysicalGpuOwner) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto owner = coordinator.ActivateCachePage(
        PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, false);
    ASSERT_TRUE(owner.has_value());
    ASSERT_TRUE(coordinator.MarkCachePageGpuDirty(owner->token, 256, 512));

    const auto retirement = coordinator.RetireOwnerForCpuWrite(GuestB);

    ASSERT_TRUE(retirement.has_value());
    ASSERT_TRUE(retirement->writeback.has_value());
    EXPECT_EQ(retirement->writeback->physical_offset, PhysicalPage);
    ASSERT_EQ(retirement->dirty_slices.size(), 1);
    EXPECT_EQ(retirement->dirty_slices[0].offset, 256);
    EXPECT_EQ(retirement->dirty_slices[0].size, 512);
    ASSERT_EQ(retirement->deltas.size(), 2);
    EXPECT_EQ(retirement->deltas[0].device_address.value, 0);
    EXPECT_EQ(retirement->deltas[1].device_address.value, 0);
}

TEST(PhysicalBackingPublicationCoordinator, TextureOverlapSuppressesEveryPhysicalAlias) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));

    const auto overlap = coordinator.BeginTextureOverlap(GuestB, PageSize);

    ASSERT_TRUE(overlap.has_value());
    ASSERT_EQ(overlap->deltas.size(), 2);
    EXPECT_EQ(overlap->deltas[0].device_address.value, 0);
    EXPECT_EQ(overlap->deltas[1].device_address.value, 0);
    EXPECT_FALSE(coordinator.ActivateCachePage(
        PhysicalPage, PhysicalBackingDeviceAddress{OverrideBase}, false));

    const auto restored_deltas = coordinator.EndTextureOverlap(overlap->token);
    ASSERT_TRUE(restored_deltas.has_value());
    ASSERT_EQ(restored_deltas->size(), 2);
    EXPECT_EQ((*restored_deltas)[0].device_address.value, ImportedBase + PhysicalPage);
    EXPECT_EQ((*restored_deltas)[1].device_address.value, ImportedBase + PhysicalPage);
}

TEST(PhysicalBackingPublicationCoordinator, GuestPageRegistrationUsesMonotonicOwnerTokens) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));

    const auto first = coordinator.ActivateCachePageForGuest(
        GuestA, PhysicalBackingDeviceAddress{OverrideBase}, false);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(coordinator.RetireCachePageClean(first->token));
    const auto second = coordinator.ActivateCachePageForGuest(
        GuestB, PhysicalBackingDeviceAddress{OverrideBase + PageSize}, false);
    ASSERT_TRUE(second.has_value());

    EXPECT_GT(second->token.publication.owner_generation,
              first->token.publication.owner_generation);
    EXPECT_FALSE(coordinator.RetireCachePageClean(first->token));
    ASSERT_TRUE(coordinator.RetireCachePageClean(second->token));
}

TEST(PhysicalBackingPublicationCoordinator, ReportsTrackedGuestPublicationWithoutFallback) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));

    const auto imported = coordinator.ResolveGuestPagePublication(GuestA);
    ASSERT_TRUE(imported.has_value());
    EXPECT_EQ(imported->value, ImportedBase + PhysicalPage);
    EXPECT_FALSE(coordinator.ResolveGuestPagePublication(GuestB));

    const auto overlap = coordinator.BeginTextureOverlap(GuestA, PageSize);
    ASSERT_TRUE(overlap.has_value());
    const auto blocked = coordinator.ResolveGuestPagePublication(GuestA);
    ASSERT_TRUE(blocked.has_value());
    EXPECT_EQ(blocked->value, 0);
}

} // namespace
