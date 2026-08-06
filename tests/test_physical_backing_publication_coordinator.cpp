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

TEST(PhysicalBackingPublicationCoordinator, ReadOnlyTexturesDoNotSuppressImportedBacking) {
    EXPECT_FALSE(VideoCore::ShouldAcquirePhysicalBackingTextureOwnership(false, false));
    EXPECT_TRUE(VideoCore::ShouldAcquirePhysicalBackingTextureOwnership(false, true));
    EXPECT_FALSE(VideoCore::ShouldAcquirePhysicalBackingTextureOwnership(true, true));
}

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
    ASSERT_TRUE(coordinator.MarkCachePageGpuDirty(owner->token, 128, 256));
    ASSERT_TRUE(coordinator.MarkCachePageGpuDirty(owner->token, 256, 512));

    const auto retirement = coordinator.RetireCachePageGpuDirty(owner->token);

    ASSERT_TRUE(retirement.has_value());
    ASSERT_EQ(retirement->deltas.size(), 2);
    EXPECT_EQ(retirement->deltas[0].device_address.value, 0);
    EXPECT_EQ(retirement->deltas[1].device_address.value, 0);
    EXPECT_EQ(retirement->writeback.physical_offset, PhysicalPage);
    ASSERT_EQ(retirement->dirty_slices.size(), 1);
    EXPECT_EQ(retirement->dirty_slices[0].offset, 128);
    EXPECT_EQ(retirement->dirty_slices[0].size, 640);

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

TEST(PhysicalBackingPublicationCoordinator, MarksGpuDirtyThroughAnyPhysicalAlias) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto owner = coordinator.ActivateCachePageForGuest(
        GuestA, PhysicalBackingDeviceAddress{OverrideBase}, false);
    ASSERT_TRUE(owner.has_value());

    ASSERT_TRUE(coordinator.MarkCachePageGpuDirtyForGuest(GuestB, 384, 128));
    const auto retirement = coordinator.RetireCachePageGpuDirty(owner->token);
    ASSERT_TRUE(retirement.has_value());
    ASSERT_EQ(retirement->dirty_slices.size(), 1);
    EXPECT_EQ(retirement->dirty_slices[0].offset, 384);
    EXPECT_EQ(retirement->dirty_slices[0].size, 128);
}

TEST(PhysicalBackingPublicationCoordinator, ResolvesActiveOwnerThroughAnyPhysicalAlias) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto owner = coordinator.ActivateCachePageForGuest(
        GuestA, PhysicalBackingDeviceAddress{OverrideBase}, false);
    ASSERT_TRUE(owner.has_value());

    const auto alias_owner = coordinator.ResolveActiveCachePageForGuest(GuestB);

    ASSERT_TRUE(alias_owner.has_value());
    EXPECT_EQ(*alias_owner, owner->token);
    EXPECT_FALSE(coordinator.ResolveActiveCachePageForGuest(GuestA + PageSize));
}

TEST(PhysicalBackingPublicationCoordinator, ResolvesSharedPhysicalPageAcrossGuestAliases) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));

    EXPECT_EQ(coordinator.ResolvePhysicalPageForGuest(GuestA), PhysicalPage);
    EXPECT_EQ(coordinator.ResolvePhysicalPageForGuest(GuestB), PhysicalPage);
    EXPECT_FALSE(coordinator.ResolvePhysicalPageForGuest(GuestC));
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

TEST(PhysicalBackingPublicationCoordinator, TextureOverlapRetainsExistingDirtyCacheMirror) {
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

    const auto overlap = coordinator.BeginTextureOverlap(GuestB, PageSize);

    ASSERT_TRUE(overlap.has_value());
    ASSERT_EQ(overlap->deltas.size(), 2);
    EXPECT_EQ(overlap->deltas[0].device_address.value, 0);
    EXPECT_EQ(overlap->deltas[1].device_address.value, 0);
    EXPECT_EQ(coordinator.ResolveActiveCachePageForGuest(GuestA), owner->token);
    const auto restored_deltas = coordinator.EndTextureOverlap(overlap->token);
    ASSERT_TRUE(restored_deltas.has_value());
    ASSERT_EQ(restored_deltas->size(), 2);
    EXPECT_EQ((*restored_deltas)[0].device_address.value, OverrideBase);
    EXPECT_EQ((*restored_deltas)[1].device_address.value, OverrideBase);
    const auto dirty_slices = coordinator.ResolveCachePageDirtySlices(owner->token);
    ASSERT_TRUE(dirty_slices.has_value());
    ASSERT_EQ(dirty_slices->size(), 1);
    EXPECT_EQ((*dirty_slices)[0], (VideoCore::PhysicalBackingDirtySlice{256, 512}));
}

TEST(PhysicalBackingPublicationCoordinator, TextureTransitionReusesRetainedCacheOwner) {
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
    const auto overlap = coordinator.BeginTextureOverlap(GuestB, PageSize);
    ASSERT_TRUE(overlap.has_value());
    const std::array requests{VideoCore::PhysicalBackingCachePageRequest{
        GuestA, PhysicalBackingDeviceAddress{OverrideBase}}};

    const auto transition = coordinator.TransitionTexturePagesToDirtyCachePages(
        std::span{&overlap->token, 1}, requests);

    ASSERT_TRUE(transition.has_value());
    ASSERT_EQ(transition->owners.size(), 1);
    EXPECT_EQ(transition->owners[0].token, owner->token);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestA)->value, OverrideBase);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestB)->value, OverrideBase);
    const auto dirty_slices = coordinator.ResolveCachePageDirtySlices(owner->token);
    ASSERT_TRUE(dirty_slices.has_value());
    ASSERT_EQ(dirty_slices->size(), 1);
    EXPECT_EQ((*dirty_slices)[0],
              (VideoCore::PhysicalBackingDirtySlice{0, static_cast<u32>(PageSize)}));
}

TEST(PhysicalBackingPublicationCoordinator,
     TextureTransitionMigratesRetainedCacheOwnerToNewMirrorAddress) {
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
    const auto overlap = coordinator.BeginTextureOverlap(GuestB, PageSize);
    ASSERT_TRUE(overlap.has_value());
    const std::array requests{VideoCore::PhysicalBackingCachePageRequest{
        GuestB, PhysicalBackingDeviceAddress{OverrideBase + PageSize}}};

    const auto transition = coordinator.TransitionTexturePagesToDirtyCachePages(
        std::span{&overlap->token, 1}, requests);

    ASSERT_TRUE(transition.has_value());
    ASSERT_EQ(transition->owners.size(), 1);
    EXPECT_NE(transition->owners[0].token, owner->token);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestA)->value, OverrideBase + PageSize);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestB)->value, OverrideBase + PageSize);
    EXPECT_FALSE(coordinator.ResolveCachePageDirtySlices(owner->token));
    const auto dirty_slices =
        coordinator.ResolveCachePageDirtySlices(transition->owners[0].token);
    ASSERT_TRUE(dirty_slices.has_value());
    ASSERT_EQ(dirty_slices->size(), 1);
    EXPECT_EQ((*dirty_slices)[0],
              (VideoCore::PhysicalBackingDirtySlice{0, static_cast<u32>(PageSize)}));
}

TEST(PhysicalBackingPublicationCoordinator, DmaResidencyAtomicallyPublishesDirtyCacheForAliases) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto overlap = coordinator.BeginTextureOverlap(GuestA, PageSize);
    ASSERT_TRUE(overlap.has_value());
    const std::array requests{VideoCore::PhysicalBackingCachePageRequest{
        GuestB, PhysicalBackingDeviceAddress{OverrideBase}}};

    const auto transition =
        coordinator.TransitionTexturePagesToDirtyCachePages(
            std::span{&overlap->token, 1}, requests);

    ASSERT_TRUE(transition.has_value());
    ASSERT_EQ(transition->owners.size(), 1);
    ASSERT_EQ(transition->deltas.size(), 2);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestA)->value, OverrideBase);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestB)->value, OverrideBase);
    EXPECT_FALSE(coordinator.EndTextureOverlap(overlap->token));
    const auto retired = coordinator.RetireCachePageGpuDirty(transition->owners[0].token);
    ASSERT_TRUE(retired.has_value());
    ASSERT_EQ(retired->dirty_slices.size(), 1);
    EXPECT_EQ(retired->dirty_slices[0],
              (VideoCore::PhysicalBackingDirtySlice{0, static_cast<u32>(PageSize)}));
}

TEST(PhysicalBackingPublicationCoordinator, TextureAuthorityTransfersOnlyAfterPriorOwnerEnds) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto first = coordinator.BeginTextureOverlap(GuestA, PageSize);
    ASSERT_TRUE(first.has_value());
    EXPECT_FALSE(coordinator.BeginTextureOverlap(GuestB, PageSize));
    ASSERT_TRUE(coordinator.EndTextureOverlap(first->token));
    const auto second = coordinator.BeginTextureOverlap(GuestB, PageSize);
    ASSERT_TRUE(second.has_value());
    const std::array requests{VideoCore::PhysicalBackingCachePageRequest{
        GuestA, PhysicalBackingDeviceAddress{OverrideBase}}};

    const auto transition = coordinator.TransitionTexturePagesToDirtyCachePages(
        std::span{&second->token, 1}, requests);

    ASSERT_TRUE(transition.has_value());
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestA)->value, OverrideBase);
    EXPECT_EQ(coordinator.ResolveGuestPagePublication(GuestB)->value, OverrideBase);
}

TEST(PhysicalBackingPublicationCoordinator,
     RejectsSecondTextureOwnerUntilPriorAuthorityIsMaterialized) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto first = coordinator.BeginTextureOverlap(GuestA, PageSize);
    ASSERT_TRUE(first.has_value());

    const auto second = coordinator.BeginTextureOverlap(GuestB, PageSize);

    EXPECT_FALSE(second.has_value());
    if (second) {
        EXPECT_TRUE(coordinator.EndTextureOverlap(second->token));
    }
    EXPECT_TRUE(coordinator.EndTextureOverlap(first->token));
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

TEST(PhysicalBackingPublicationCoordinator, ActivatesBufferPagesAsOneAtomicPublicationBatch) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, OtherPhysicalPage, PageSize, 8},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto existing = coordinator.ActivateCachePageForGuest(
        GuestB, PhysicalBackingDeviceAddress{OverrideBase + PageSize}, false);
    ASSERT_TRUE(existing.has_value());

    constexpr std::array requests{
        VideoCore::PhysicalBackingCachePageRequest{GuestA,
                                                   PhysicalBackingDeviceAddress{OverrideBase}},
        VideoCore::PhysicalBackingCachePageRequest{
            GuestB, PhysicalBackingDeviceAddress{OverrideBase + 2 * PageSize}},
    };
    EXPECT_FALSE(coordinator.ActivateCachePagesForGuests(requests));

    const auto guest_a = coordinator.ResolveGuestPagePublication(GuestA);
    const auto guest_b = coordinator.ResolveGuestPagePublication(GuestB);
    ASSERT_TRUE(guest_a.has_value());
    ASSERT_TRUE(guest_b.has_value());
    EXPECT_EQ(guest_a->value, ImportedBase + PhysicalPage);
    EXPECT_EQ(guest_b->value, OverrideBase + PageSize);
}

TEST(PhysicalBackingPublicationCoordinator, AcquiresExistingAliasOwnerAndNewPagesTogether) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestC, OtherPhysicalPage, PageSize, 8},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    const auto existing = coordinator.ActivateCachePageForGuest(
        GuestA, PhysicalBackingDeviceAddress{OverrideBase}, false);
    ASSERT_TRUE(existing.has_value());
    constexpr std::array requests{
        VideoCore::PhysicalBackingCachePageRequest{
            GuestB, PhysicalBackingDeviceAddress{OverrideBase + PageSize}},
        VideoCore::PhysicalBackingCachePageRequest{
            GuestC, PhysicalBackingDeviceAddress{OverrideBase + 2 * PageSize}},
    };

    const auto acquired = coordinator.AcquireCachePagesForGuests(requests);

    ASSERT_TRUE(acquired.has_value());
    ASSERT_EQ(acquired->owners.size(), 1);
    EXPECT_EQ(acquired->owners[0].guest_page, GuestC);
    ASSERT_EQ(acquired->deltas.size(), 3);
    EXPECT_EQ(acquired->deltas[0].device_address.value, OverrideBase);
    EXPECT_EQ(acquired->deltas[1].device_address.value, OverrideBase);
    EXPECT_EQ(acquired->deltas[2].device_address.value, OverrideBase + 2 * PageSize);
}

TEST(PhysicalBackingPublicationCoordinator, MigratesDirtyOwnerToWritablePhysicalAlias) {
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
    ASSERT_TRUE(coordinator.MarkCachePageGpuDirty(first->token, 64, 128));

    const auto migrated = coordinator.MigrateCachePageForGuest(
        GuestB, PhysicalBackingDeviceAddress{OverrideBase + PageSize});

    ASSERT_TRUE(migrated.has_value());
    EXPECT_EQ(migrated->previous_token, first->token);
    EXPECT_NE(migrated->token, first->token);
    ASSERT_EQ(migrated->deltas.size(), 2);
    EXPECT_EQ(migrated->deltas[0].device_address.value, OverrideBase + PageSize);
    EXPECT_EQ(migrated->deltas[1].device_address.value, OverrideBase + PageSize);
    EXPECT_FALSE(coordinator.RetireCachePageClean(first->token));
    const auto dirty = coordinator.RetireCachePageGpuDirty(migrated->token);
    ASSERT_TRUE(dirty.has_value());
    ASSERT_EQ(dirty->dirty_slices.size(), 1);
    EXPECT_EQ(dirty->dirty_slices[0], (VideoCore::PhysicalBackingDirtySlice{64, 128}));
}

TEST(PhysicalBackingPublicationCoordinator, BatchReturnsOneOwnerForPhysicalAliases) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array spans{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
        PhysicalBackingSpan{GuestB, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(spans));
    constexpr std::array requests{
        VideoCore::PhysicalBackingCachePageRequest{GuestA,
                                                   PhysicalBackingDeviceAddress{OverrideBase}},
        VideoCore::PhysicalBackingCachePageRequest{
            GuestB, PhysicalBackingDeviceAddress{OverrideBase + PageSize}},
    };

    const auto batch = coordinator.ActivateCachePagesForGuests(requests);

    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->owners.size(), 1);
    EXPECT_EQ(batch->owners[0].guest_page, GuestA);
    EXPECT_EQ(batch->owners[0].token.publication.physical_offset, PhysicalPage);
    ASSERT_EQ(batch->deltas.size(), 2);
    EXPECT_EQ(batch->deltas[0].device_address.value, OverrideBase);
    EXPECT_EQ(batch->deltas[1].device_address.value, OverrideBase);
}

TEST(PhysicalBackingPublicationCoordinator, RetiresAllocationBeforePhysicalPageReuse) {
    PhysicalBackingPublicationCoordinator coordinator{PhysicalBackingDeviceAddress{ImportedBase},
                                                      16 * PageSize};
    constexpr std::array first_span{
        PhysicalBackingSpan{GuestA, PhysicalPage, PageSize, 7},
    };
    ASSERT_TRUE(coordinator.MapSpans(first_span));
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
    EXPECT_EQ(reused->front().device_address.value, ImportedBase + PhysicalPage);
}

} // namespace
