// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <optional>
#include <vector>

#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/buffer_cache/physical_backing_writeback_tracker.h"

namespace {

enum class ResidencyCall {
    Synchronize,
    Publish,
};

struct RecordedResidencyCall {
    ResidencyCall operation;
    VAddr address;
    u32 size;
};

class ExpandedDmaBuffer {
public:
    [[nodiscard]] VAddr CpuAddr() const {
        return 0x101E600000;
    }

    [[nodiscard]] size_t SizeBytes() const {
        return 0x200000;
    }
};

} // namespace

TEST(BufferResidency, PublishesExpandedDmaMappingOnlyAfterFullSpanIsResident) {
    ExpandedDmaBuffer buffer;
    std::vector<RecordedResidencyCall> calls;

    VideoCore::PublishDmaBufferAfterSynchronization(
        buffer,
        [&](ExpandedDmaBuffer&, VAddr address, u32 size) {
            calls.push_back({ResidencyCall::Synchronize, address, size});
        },
        [&] { calls.push_back({ResidencyCall::Publish, 0, 0}); });

    ASSERT_EQ(calls.size(), 2);
    EXPECT_EQ(calls[0].operation, ResidencyCall::Synchronize);
    EXPECT_EQ(calls[0].address, buffer.CpuAddr());
    EXPECT_EQ(calls[0].size, buffer.SizeBytes());
    EXPECT_EQ(calls[1].operation, ResidencyCall::Publish);
}

TEST(BufferResidency, DoesNotTouchUnpublishedBufferAfterResidencyUpload) {
    u32 touch_count = 0;

    VideoCore::TouchBufferAfterUploadIfRegistered(false, [&] { ++touch_count; });
    EXPECT_EQ(touch_count, 0);

    VideoCore::TouchBufferAfterUploadIfRegistered(true, [&] { ++touch_count; });
    EXPECT_EQ(touch_count, 1);
}

TEST(BufferResidency, FailedPhysicalOwnershipBatchPreservesCurrentPagePublications) {
    constexpr VAddr PageSize = 16_KB;
    constexpr VAddr FirstPage = 0x1000'0000;
    constexpr u64 ImportedAddress = 0x2'0000'0000;
    std::array<u64, 3> addresses{0x4'0000'0000, 0, 0};
    constexpr std::array physical_pages{FirstPage + PageSize, FirstPage + 2 * PageSize};

    VideoCore::RefreshPhysicalBackingRegistrationAddresses(
        std::span<const VAddr>{physical_pages}, FirstPage, std::span<u64>{addresses},
        [](VAddr page) { return page; },
        [](VAddr page) -> std::optional<u64> {
            if (page == FirstPage + PageSize) {
                return 0x2'0000'0000ULL;
            }
            return 0;
        });

    EXPECT_EQ(addresses[0], 0x4'0000'0000);
    EXPECT_EQ(addresses[1], ImportedAddress);
    EXPECT_EQ(addresses[2], 0);
}

TEST(BufferResidency, PlansFullPageCopyBeforeWritableAliasMigration) {
    constexpr VAddr PageSize = 16_KB;
    const auto copy = VideoCore::PlanPhysicalBackingAliasMigrationCopy(
        0x1000'0000, 4 * PageSize, 0x1000'4000, 0x2000'0000, 4 * PageSize, 0x2000'8000);

    ASSERT_TRUE(copy.has_value());
    EXPECT_EQ(copy->source_offset, PageSize);
    EXPECT_EQ(copy->destination_offset, 2 * PageSize);
    EXPECT_EQ(copy->size, PageSize);

    EXPECT_FALSE(VideoCore::PlanPhysicalBackingAliasMigrationCopy(
        0x1000'0000, PageSize, 0x1000'0001, 0x2000'0000, PageSize, 0x2000'0000));
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingAliasMigrationCopy(
        0x1000'0000, PageSize, 0x1000'4000, 0x2000'0000, PageSize, 0x2000'0000));
}

TEST(BufferResidency, PlansOwnerMigrationWhenAResidentBufferIsReplaced) {
    constexpr VAddr PageSize = 16_KB;
    constexpr VAddr source_base = 0x1000'4000;
    constexpr VAddr target_base = 0x1000'0000;
    constexpr std::array owner_pages{
        source_base,
        source_base + 2 * PageSize,
    };

    const auto migrations = VideoCore::PlanPhysicalBackingOwnerReplacementMigrations(
        source_base, 3 * PageSize, target_base, 5 * PageSize, owner_pages);

    ASSERT_TRUE(migrations.has_value());
    EXPECT_EQ(*migrations,
              (std::vector<VideoCore::PhysicalBackingOwnerReplacementMigration>{
                  {0, source_base, PageSize},
                  {1, source_base + 2 * PageSize, 3 * PageSize},
              }));
}

TEST(BufferResidency, RejectsOwnerMigrationOutsideTheCopiedReplacementSpan) {
    constexpr VAddr PageSize = 16_KB;
    constexpr VAddr source_base = 0x1000'4000;
    constexpr VAddr target_base = 0x1000'0000;
    constexpr std::array owner_pages{source_base + 3 * PageSize};

    EXPECT_FALSE(VideoCore::PlanPhysicalBackingOwnerReplacementMigrations(
                     source_base, 3 * PageSize, target_base, 5 * PageSize, owner_pages)
                     .has_value());
}

TEST(BufferResidency, DefersPhysicalBackingOwnershipUntilGpuWrite) {
    EXPECT_FALSE(VideoCore::ShouldAcquirePhysicalBackingBufferOwnership(false));
    EXPECT_TRUE(VideoCore::ShouldAcquirePhysicalBackingBufferOwnership(true));
}

TEST(BufferResidency, AdvancesTextureAuthorityOnlyAfterAnEncodedGpuWrite) {
    VideoCore::PhysicalBackingTextureWriteOrderTracker tracker;

    EXPECT_TRUE(tracker.Acquire(16));
    EXPECT_TRUE(tracker.Acquire(41));
    EXPECT_EQ(tracker.Get(16), 0);
    EXPECT_EQ(tracker.Get(41), 0);

    constexpr std::array first_command{16U};
    EXPECT_TRUE(tracker.MarkGpuWrites(first_command));
    const u64 image_16_write = tracker.Get(16);
    EXPECT_GT(image_16_write, 0);

    EXPECT_FALSE(tracker.Acquire(41));
    EXPECT_EQ(tracker.Get(41), 0);
    EXPECT_EQ(tracker.Get(16), image_16_write);

    constexpr std::array second_command{41U};
    EXPECT_TRUE(tracker.MarkGpuWrites(second_command));
    EXPECT_GT(tracker.Get(41), image_16_write);

    EXPECT_TRUE(tracker.Acquire(77));
    constexpr std::array simultaneous_outputs{16U, 77U};
    EXPECT_TRUE(tracker.MarkGpuWrites(simultaneous_outputs));
    EXPECT_EQ(tracker.Get(16), tracker.Get(77));

    EXPECT_TRUE(tracker.Release(16));
    EXPECT_FALSE(tracker.MarkGpuWrites(first_command));
}

TEST(BufferResidency, InvalidatesTextureOwnershipBeforeGpuBufferFill) {
    EXPECT_FALSE(VideoCore::ShouldInvalidateTextureCacheBeforeGpuBufferFill(true, true));
    EXPECT_FALSE(VideoCore::ShouldInvalidateTextureCacheBeforeGpuBufferFill(false, false));
    EXPECT_TRUE(VideoCore::ShouldInvalidateTextureCacheBeforeGpuBufferFill(false, true));
}

TEST(BufferResidency, PreservesWholeTextureBeforePartialGpuBufferWrite) {
    const auto transition = VideoCore::PlanPhysicalBackingTextureBufferTransition(
        0x1000'0000, 0x24'0000, 0x1003'0000, 0x870);

    ASSERT_TRUE(transition.has_value());
    EXPECT_EQ(transition->base, 0x1000'0000);
    EXPECT_EQ(transition->size, 0x24'0000);
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingTextureBufferTransition(0x1000'0000, 0x24'0000,
                                                                       0x0fff'f000, 0x2000));
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingTextureBufferTransition(0x1000'0000, 0x24'0000,
                                                                       0x1023'f800, 0x1000));
}

TEST(BufferResidency, ReleasesOnlyTextureTokensConflictingWithBufferWrite) {
    constexpr std::array texture_pages{0x1000ULL, 0x2000ULL};
    constexpr std::array conflicting_pages{0x2000ULL, 0x3000ULL};
    constexpr std::array unrelated_pages{0x3000ULL, 0x4000ULL};

    EXPECT_TRUE(VideoCore::PhysicalBackingPagesIntersect(texture_pages, conflicting_pages));
    EXPECT_FALSE(VideoCore::PhysicalBackingPagesIntersect(texture_pages, unrelated_pages));
}

TEST(BufferResidency, TracksTheActualTextureMirrorProducer) {
    EXPECT_EQ(VideoCore::PhysicalBackingTextureMirrorProducer(true),
              VideoCore::PhysicalBackingTextureProducer::ComputeShader);
    EXPECT_EQ(VideoCore::PhysicalBackingTextureMirrorProducer(false),
              VideoCore::PhysicalBackingTextureProducer::Copy);
}

TEST(BufferResidency, AlignsTheCompleteTextureOwnershipSpan) {
    const auto span = VideoCore::PlanPhysicalBackingTextureOwnershipSpan(0x1000'0123, 0x5000);

    ASSERT_TRUE(span.has_value());
    EXPECT_EQ(span->base, 0x1000'0000);
    EXPECT_EQ(span->size, 0x8000);
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingTextureOwnershipSpan(
        std::numeric_limits<VAddr>::max() - 0x1000, 0x2000));
}

TEST(BufferResidency, PlansTransitiveTextureOwnershipComponentInGpuWriteOrder) {
    const std::array records{
        VideoCore::PhysicalBackingTextureOwnershipRecord{
            .image_index = 16,
            .guest_base = 0x114b'6d00'00,
            .guest_size = 0x24'0000,
            .gpu_write_order = 20,
            .physical_pages = {0xab8d'0000, 0xab8d'4000},
        },
        VideoCore::PhysicalBackingTextureOwnershipRecord{
            .image_index = 41,
            .guest_base = 0x114b'7100'00,
            .guest_size = 0x14'0000,
            .gpu_write_order = 30,
            .physical_pages = {0xab8d'0000},
        },
        VideoCore::PhysicalBackingTextureOwnershipRecord{
            .image_index = 44,
            .guest_base = 0x114b'8500'00,
            .guest_size = 0x2a'9600,
            .gpu_write_order = 40,
            .physical_pages = {0xab8d'4000, 0xab8d'8000},
        },
        VideoCore::PhysicalBackingTextureOwnershipRecord{
            .image_index = 51,
            .guest_base = 0x114b'4c00'00,
            .guest_size = 0x24'0000,
            .gpu_write_order = 10,
            .physical_pages = {0xab8c'c000, 0xab8d'0000},
        },
        VideoCore::PhysicalBackingTextureOwnershipRecord{
            .image_index = 77,
            .guest_base = 0x1200'0000'00,
            .guest_size = 0x4000,
            .gpu_write_order = 50,
            .physical_pages = {0xbeef'0000},
        },
    };
    constexpr std::array seed_pages{0xab8d'0000ULL, 0xdead'0000ULL};

    const auto plan = VideoCore::PlanPhysicalBackingTextureOwnershipComponent(records, seed_pages);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->oldest_to_newest_images,
              (std::vector<VideoCore::PhysicalBackingTextureOwnershipImage>{
                  {51, 10}, {16, 20}, {41, 30}, {44, 40}}));
    EXPECT_EQ(plan->ownership_span.base, 0x114b'4c00'00);
    EXPECT_EQ(plan->ownership_span.size, 0x63'c000);
    EXPECT_EQ(plan->physical_pages,
              (std::vector<u64>{0xab8c'c000, 0xab8d'0000, 0xab8d'4000, 0xab8d'8000}));
}

TEST(BufferResidency, SelectsNewestTextureAliasForEachPhysicalPage) {
    constexpr std::array candidates{
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x114b'6d00'00, 7, 16},
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'4000, 0x114b'6d40'00, 7, 16},
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x214b'7100'00, 8, 41},
    };

    const auto sources = VideoCore::PlanPhysicalBackingTexturePageSources(candidates);

    ASSERT_TRUE(sources.has_value());
    EXPECT_EQ(*sources, (std::vector<VideoCore::PhysicalBackingTexturePageSource>{
                            {0xab8d'0000, 0x214b'7100'00, 8, 41},
                            {0xab8d'4000, 0x114b'6d40'00, 7, 16},
                        }));
}

TEST(BufferResidency, SelectsLastActuallyWrittenAliasInsteadOfLastBoundAlias) {
    constexpr std::array before_second_write{
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x114b'6d00'00, 7, 16},
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x214b'7100'00, 0, 41},
    };
    const auto first_source = VideoCore::PlanPhysicalBackingTexturePageSources(before_second_write);
    ASSERT_TRUE(first_source.has_value());
    EXPECT_EQ(first_source->front(),
              (VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x114b'6d00'00, 7, 16}));

    constexpr std::array after_second_write{
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x114b'6d00'00, 7, 16},
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x214b'7100'00, 8, 41},
    };
    const auto second_source = VideoCore::PlanPhysicalBackingTexturePageSources(after_second_write);
    ASSERT_TRUE(second_source.has_value());
    EXPECT_EQ(second_source->front(),
              (VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x214b'7100'00, 8, 41}));

    constexpr std::array ambiguous_same_command{
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x114b'6d00'00, 9, 16},
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x214b'7100'00, 9, 41},
    };
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingTexturePageSources(ambiguous_same_command));

    constexpr std::array ambiguous_same_guest_page{
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x114b'6d00'00, 10, 16},
        VideoCore::PhysicalBackingTexturePageSource{0xab8d'0000, 0x114b'6d00'00, 10, 41},
    };
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingTexturePageSources(ambiguous_same_guest_page));
}

TEST(BufferResidency, SelectsTheFirstActualTextureUploadConsumer) {
    EXPECT_EQ(VideoCore::PhysicalBackingTextureUploadConsumer(true),
              VideoCore::PhysicalBackingTextureConsumer::ComputeShaderRead);
    EXPECT_EQ(VideoCore::PhysicalBackingTextureUploadConsumer(false),
              VideoCore::PhysicalBackingTextureConsumer::TransferRead);
}

TEST(BufferResidency, PlansOneAliasHandoffForTheWholeGpuCommand) {
    using Kind = VideoCore::PhysicalBackingCommandResourceKind;
    using Resource = VideoCore::PhysicalBackingCommandResource;
    using Access = VideoCore::PhysicalBackingCommandAccess;
    constexpr u64 page = 0xab8d'0000;
    const Resource texture{Kind::Texture, 16};
    const Resource buffer{Kind::Buffer, 41};
    const std::array accesses{
        Access{texture, false, {page}},
        Access{buffer, true, {page}},
        Access{buffer, true, {page}},
        Access{texture, false, {page}},
    };

    const auto plan = VideoCore::PlanPhysicalBackingGpuCommandAliases(accesses);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->read_snapshot_order, (std::vector<Resource>{texture, buffer}));
    EXPECT_EQ(plan->writer_prepare_order, (std::vector<Resource>{buffer}));
    EXPECT_EQ(plan->writer_finalize_order, (std::vector<Resource>{buffer}));
    ASSERT_EQ(plan->pages.size(), 1);
    EXPECT_EQ(plan->pages.front().physical_page, page);
    EXPECT_EQ(plan->pages.front().writers, (std::vector<Resource>{buffer}));
}

TEST(BufferResidency, PlansCompetingBufferAndTextureWritersWithoutDroppingGpuCommand) {
    using Kind = VideoCore::PhysicalBackingCommandResourceKind;
    using Resource = VideoCore::PhysicalBackingCommandResource;
    using Access = VideoCore::PhysicalBackingCommandAccess;
    constexpr u64 page = 0xab8d'0000;
    const Resource buffer{Kind::Buffer, 41};
    const Resource texture{Kind::Texture, 16};
    const std::array accesses{
        Access{buffer, true, {page}},
        Access{texture, true, {page}},
    };

    const auto plan = VideoCore::PlanPhysicalBackingGpuCommandAliases(accesses);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->read_snapshot_order, (std::vector<Resource>{buffer, texture}));
    EXPECT_EQ(plan->writer_prepare_order, (std::vector<Resource>{buffer, texture}));
    EXPECT_EQ(plan->writer_finalize_order, (std::vector<Resource>{buffer, texture}));
    ASSERT_EQ(plan->pages.size(), 1);
    EXPECT_EQ(plan->pages.front().physical_page, page);
    EXPECT_EQ(plan->pages.front().writers, (std::vector<Resource>{buffer, texture}));
}

TEST(BufferResidency, SnapshotsWritableResourceBeforePreparingPartialWrite) {
    using Kind = VideoCore::PhysicalBackingCommandResourceKind;
    using Resource = VideoCore::PhysicalBackingCommandResource;
    using Access = VideoCore::PhysicalBackingCommandAccess;
    constexpr u64 page = 0xab8d'0000;
    const Resource texture{Kind::Texture, 16};
    const std::array accesses{Access{texture, true, {page}}};

    const auto plan = VideoCore::PlanPhysicalBackingGpuCommandAliases(accesses);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->read_snapshot_order, (std::vector<Resource>{texture}));
    EXPECT_EQ(plan->writer_prepare_order, (std::vector<Resource>{texture}));
    EXPECT_EQ(plan->writer_finalize_order, (std::vector<Resource>{texture}));
}

TEST(BufferResidency, LeavesUncontendedReadOnlyPagesOutOfCommandSnapshots) {
    using Kind = VideoCore::PhysicalBackingCommandResourceKind;
    using Resource = VideoCore::PhysicalBackingCommandResource;
    using Access = VideoCore::PhysicalBackingCommandAccess;
    constexpr u64 written_page = 0xab8d'0000;
    constexpr u64 read_only_page = written_page + 16_KB;
    const Resource writer{Kind::Buffer, 41};
    const Resource read_only_texture{Kind::Texture, 16};
    const std::array accesses{
        Access{writer, true, {written_page}},
        Access{read_only_texture, false, {read_only_page}},
    };

    const auto plan = VideoCore::PlanPhysicalBackingGpuCommandAliases(accesses);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->read_snapshot_order, (std::vector<Resource>{writer}));
    EXPECT_EQ(plan->writer_prepare_order, (std::vector<Resource>{writer}));
    EXPECT_EQ(plan->writer_finalize_order, (std::vector<Resource>{writer}));
}

TEST(BufferResidency, RetiresOnlyPhysicalOwnersOverlappedByCpuWrite) {
    using Owner = VideoCore::PhysicalBackingCachePageOwnerLocation;
    constexpr u64 requested_page = 0xab8d'0000;
    constexpr u64 unrelated_page = requested_page + 16_KB;
    constexpr std::array requested{requested_page};
    constexpr std::array owners{
        Owner{41, 0, requested_page},
        Owner{41, 1, unrelated_page},
    };

    const auto retirements = VideoCore::PlanPhysicalBackingCachePageRetirements(requested, owners);

    ASSERT_TRUE(retirements.has_value());
    EXPECT_EQ(*retirements, (std::vector<Owner>{owners.front()}));
}

TEST(BufferResidency, BatchesLargeOwnerRetirementsInsideStagingBudget) {
    using Batch = VideoCore::PhysicalBackingRetirementBatch;
    constexpr size_t owner_count = 4096;
    constexpr u64 staging_budget = 32_MB;

    const auto batches =
        VideoCore::PlanPhysicalBackingRetirementBatches(owner_count, staging_budget, 16_KB);

    ASSERT_TRUE(batches.has_value());
    EXPECT_EQ(*batches, (std::vector<Batch>{{0, 2048}, {2048, 2048}}));
}

TEST(BufferResidency, CoalescesContiguousPhysicalWritebackSlices) {
    using Slice = VideoCore::PhysicalBackingGpuWritebackSlice;
    using Copy = VideoCore::PhysicalBackingGpuWritebackCopy;
    constexpr VAddr buffer_base = 0x100000;
    constexpr u64 backing_size = 0x400000;
    constexpr std::array slices{
        Slice{buffer_base + 0x4000, 0x20000, 0, 0x4000},
        Slice{buffer_base + 0x8000, 0x24000, 0, 0x4000},
        Slice{buffer_base + 0xc000, 0x30000, 0x100, 0x200},
    };

    const auto copies = VideoCore::PlanPhysicalBackingGpuWritebackCopies(buffer_base, 0x10000,
                                                                         backing_size, slices);

    ASSERT_TRUE(copies.has_value());
    EXPECT_EQ(*copies, (std::vector<Copy>{{0x4000, 0x20000, 0x8000}, {0xc100, 0x30100, 0x200}}));
}

TEST(BufferResidency, WaitsOnlyForGpuWritebacksOverlappingHostAccess) {
    constexpr std::array pending_guest_pages{VAddr{0x10000}, VAddr{0x50000}};

    EXPECT_FALSE(VideoCore::PhysicalBackingGpuWritebacksOverlapGuestRange(pending_guest_pages,
                                                                          0x20000, 0x1000));
    EXPECT_TRUE(
        VideoCore::PhysicalBackingGpuWritebacksOverlapGuestRange(pending_guest_pages, 0x13fff, 2));
    EXPECT_FALSE(VideoCore::PhysicalBackingGpuWritebacksOverlapGuestRange(
        pending_guest_pages, std::numeric_limits<VAddr>::max(), 2));
}

TEST(BufferResidency, ExpandsPhysicalWritebackSlicesToVulkanCopyAlignment) {
    using Slice = VideoCore::PhysicalBackingGpuWritebackSlice;
    using Copy = VideoCore::PhysicalBackingGpuWritebackCopy;
    constexpr VAddr buffer_base = 0x100000;
    constexpr std::array slices{
        Slice{buffer_base + 0x4000, 0x20000, 1, 1},
        Slice{buffer_base + 0x8000, 0x24000, 3, 2},
    };

    const auto copies = VideoCore::PlanPhysicalBackingGpuWritebackCopies(
        buffer_base, 0x10000, 0x400000, slices);

    ASSERT_TRUE(copies.has_value());
    EXPECT_EQ(*copies,
              (std::vector<Copy>{{0x4000, 0x20000, 4}, {0x8000, 0x24000, 8}}));
}

TEST(BufferResidency, MergesPhysicalWritebackSlicesThatOverlapAfterAlignment) {
    using Slice = VideoCore::PhysicalBackingGpuWritebackSlice;
    using Copy = VideoCore::PhysicalBackingGpuWritebackCopy;
    constexpr VAddr buffer_base = 0x100000;
    constexpr std::array slices{
        Slice{buffer_base + 0x4000, 0x20000, 1, 1},
        Slice{buffer_base + 0x4000, 0x20000, 3, 1},
    };

    const auto copies = VideoCore::PlanPhysicalBackingGpuWritebackCopies(
        buffer_base, 0x10000, 0x400000, slices);

    ASSERT_TRUE(copies.has_value());
    EXPECT_EQ(*copies, (std::vector<Copy>{{0x4000, 0x20000, 4}}));
}

TEST(BufferResidency, InitializesNewPhysicalOwnerFromTheWholeCanonicalPage) {
    using Copy = VideoCore::PhysicalBackingOwnerInitializationCopy;
    constexpr VAddr buffer_base = 0x100000;
    constexpr VAddr guest_page = buffer_base + 0x4000;

    EXPECT_EQ(VideoCore::PlanPhysicalBackingOwnerInitializationCopy(
                  buffer_base, 0x10000, guest_page),
              (Copy{guest_page, 0x4000, 16_KB}));
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingOwnerInitializationCopy(
                     buffer_base, 0x10000, guest_page + 1)
                     .has_value());
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingOwnerInitializationCopy(
                     buffer_base, 0x4001, guest_page)
                     .has_value());
}

TEST(BufferResidency, WholePageOwnerRoundTripPreservesNeighboringCpuBytes) {
    constexpr VAddr buffer_base = 0x100000;
    constexpr VAddr guest_page = buffer_base + 0x4000;
    constexpr u64 physical_page = 0x20000;
    std::array<u8, 16_KB> canonical{};
    canonical[0x100] = 0x5a;
    std::array<u8, 16_KB> owner{};

    const auto initialization = VideoCore::PlanPhysicalBackingOwnerInitializationCopy(
        buffer_base, 0x10000, guest_page);
    ASSERT_TRUE(initialization.has_value());
    std::copy_n(canonical.begin(), initialization->size, owner.begin());
    owner[1] = 0xa1;
    constexpr std::array dirty{
        VideoCore::PhysicalBackingGpuWritebackSlice{guest_page, physical_page, 1, 1},
    };
    const auto writeback = VideoCore::PlanPhysicalBackingGpuWritebackCopies(
        buffer_base, 0x10000, 0x400000, dirty);
    ASSERT_TRUE(writeback.has_value());
    for (const auto& copy : *writeback) {
        const u64 source_in_page = copy.source_offset - (guest_page - buffer_base);
        const u64 destination_in_page = copy.destination_offset - physical_page;
        std::copy_n(owner.begin() + source_in_page, copy.size,
                    canonical.begin() + destination_in_page);
    }

    EXPECT_EQ(canonical[1], 0xa1);
    EXPECT_EQ(canonical[0x100], 0x5a);
}

TEST(BufferResidency, HashesPhysicalHostMemoryOnlyWhenNoGpuBytesAreAuthoritative) {
    EXPECT_TRUE(VideoCore::ShouldHashPhysicalBackingHostMemory(false));
    EXPECT_FALSE(VideoCore::ShouldHashPhysicalBackingHostMemory(true));
}

TEST(BufferResidency, PhysicalWritebackTrackerDoesNotRequireUnrelatedWait) {
    VideoCore::PhysicalBackingWritebackTracker tracker;
    constexpr std::array first_pages{u64{0x20000}};
    constexpr std::array second_pages{u64{0x24000}};
    constexpr std::array unrelated_pages{u64{0x28000}};

    EXPECT_TRUE(tracker.Record(first_pages, 41));
    EXPECT_TRUE(tracker.Record(second_pages, 47));

    EXPECT_FALSE(tracker.RequiredTick(unrelated_pages).has_value());
    EXPECT_EQ(tracker.RequiredTick(first_pages), 41);
    EXPECT_EQ(tracker.RequiredTick(second_pages), 47);
    EXPECT_EQ(tracker.PendingPageCount(), 2);
}

TEST(BufferResidency, PhysicalWritebackTrackerWaitsForLatestSamePageSubmission) {
    VideoCore::PhysicalBackingWritebackTracker tracker;
    constexpr std::array first_page{u64{0x20000}};
    constexpr std::array second_page{u64{0x24000}};

    EXPECT_TRUE(tracker.Record(first_page, 41));
    EXPECT_TRUE(tracker.Record(second_page, 47));
    EXPECT_TRUE(tracker.Record(first_page, 53));

    EXPECT_EQ(tracker.RequiredTick(first_page), 53);
    EXPECT_EQ(tracker.RequiredTickForAll(), 53);
    EXPECT_EQ(tracker.CompleteThrough(47), (std::vector<u64>{0x24000}));
    EXPECT_EQ(tracker.PendingPageCount(), 1);
    EXPECT_EQ(tracker.RequiredTick(first_page), 53);
    EXPECT_EQ(tracker.CompleteThrough(53), (std::vector<u64>{0x20000}));
    EXPECT_EQ(tracker.PendingPageCount(), 0);
}

TEST(BufferResidency, PhysicalWritebackCompletionPreviewDoesNotConsumeTracker) {
    VideoCore::PhysicalBackingWritebackTracker tracker;
    constexpr std::array first_page{u64{0x20000}};
    constexpr std::array second_page{u64{0x24000}};
    ASSERT_TRUE(tracker.Record(first_page, 41));
    ASSERT_TRUE(tracker.Record(second_page, 47));

    EXPECT_EQ(tracker.PagesCompletingThrough(41), (std::vector<u64>{0x20000}));
    EXPECT_EQ(tracker.PendingPageCount(), 2);
    EXPECT_EQ(tracker.RequiredTick(first_page), 41);
}

TEST(BufferResidency, PhysicalWritebackSynchronizationUsesExactTickWithoutGlobalFinish) {
    VideoCore::PhysicalBackingWritebackTracker tracker;
    constexpr std::array first_page{u64{0x20000}};
    constexpr std::array second_page{u64{0x24000}};
    constexpr std::array unrelated_page{u64{0x28000}};
    ASSERT_TRUE(tracker.Record(first_page, 41));
    ASSERT_TRUE(tracker.Record(second_page, 47));
    std::vector<u64> waited_ticks;

    EXPECT_TRUE(VideoCore::SynchronizePhysicalBackingWritebacks(
                    tracker, unrelated_page,
                    [&](u64 tick) { waited_ticks.push_back(tick); })
                    .empty());
    EXPECT_EQ(VideoCore::SynchronizePhysicalBackingWritebacks(
                  tracker, first_page, [&](u64 tick) { waited_ticks.push_back(tick); }),
              (std::vector<u64>{0x20000}));
    EXPECT_EQ(VideoCore::SynchronizePhysicalBackingWritebacks(
                  tracker, second_page, [&](u64 tick) { waited_ticks.push_back(tick); }),
              (std::vector<u64>{0x24000}));

    EXPECT_EQ(waited_ticks, (std::vector<u64>{41, 47}));
    EXPECT_EQ(tracker.PendingPageCount(), 0);
}

TEST(BufferResidency, EmptyPhysicalWritebackTrackerSkipsGuestRangeResolution) {
    VideoCore::PhysicalBackingWritebackTracker tracker;
    bool resolved{};
    bool synchronized{};

    EXPECT_TRUE(VideoCore::SynchronizePhysicalBackingHostAccessIfPending(
        tracker,
        [&]() -> std::optional<std::vector<u64>> {
            resolved = true;
            return std::vector<u64>{0x20000};
        },
        [&](std::span<const u64>) {
            synchronized = true;
            return true;
        }));

    EXPECT_FALSE(resolved);
    EXPECT_FALSE(synchronized);
}

TEST(BufferResidency, PendingPhysicalWritebackResolvesAndSynchronizesGuestRange) {
    VideoCore::PhysicalBackingWritebackTracker tracker;
    constexpr std::array page{u64{0x20000}};
    ASSERT_TRUE(tracker.Record(page, 41));
    bool resolved{};
    bool synchronized{};

    EXPECT_TRUE(VideoCore::SynchronizePhysicalBackingHostAccessIfPending(
        tracker,
        [&]() -> std::optional<std::vector<u64>> {
            resolved = true;
            return std::vector<u64>{page.begin(), page.end()};
        },
        [&](std::span<const u64> resolved_pages) {
            synchronized = true;
            EXPECT_TRUE(std::ranges::equal(resolved_pages, page));
            return true;
        }));

    EXPECT_TRUE(resolved);
    EXPECT_TRUE(synchronized);
}
