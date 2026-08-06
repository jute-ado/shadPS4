// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <optional>
#include <vector>

#include "video_core/buffer_cache/buffer_residency.h"

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
    EXPECT_EQ(plan->read_snapshot_order, (std::vector<Resource>{texture}));
    EXPECT_EQ(plan->writer_prepare_order, (std::vector<Resource>{buffer}));
    EXPECT_EQ(plan->writer_finalize_order, (std::vector<Resource>{buffer}));
    ASSERT_EQ(plan->pages.size(), 1);
    EXPECT_EQ(plan->pages.front().physical_page, page);
    ASSERT_TRUE(plan->pages.front().writer.has_value());
    EXPECT_EQ(*plan->pages.front().writer, buffer);
}

TEST(BufferResidency, RejectsCompetingBufferAndTextureWritersInOneGpuCommand) {
    using Kind = VideoCore::PhysicalBackingCommandResourceKind;
    using Resource = VideoCore::PhysicalBackingCommandResource;
    using Access = VideoCore::PhysicalBackingCommandAccess;
    constexpr u64 page = 0xab8d'0000;
    const std::array accesses{
        Access{Resource{Kind::Buffer, 41}, true, {page}},
        Access{Resource{Kind::Texture, 16}, true, {page}},
    };

    EXPECT_FALSE(VideoCore::PlanPhysicalBackingGpuCommandAliases(accesses));
}
