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
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingTextureBufferTransition(
        0x1000'0000, 0x24'0000, 0x0fff'f000, 0x2000));
    EXPECT_FALSE(VideoCore::PlanPhysicalBackingTextureBufferTransition(
        0x1000'0000, 0x24'0000, 0x1023'f800, 0x1000));
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
            .write_order = 20,
            .physical_pages = {0xab8d'0000, 0xab8d'4000},
        },
        VideoCore::PhysicalBackingTextureOwnershipRecord{
            .image_index = 41,
            .guest_base = 0x114b'7100'00,
            .guest_size = 0x14'0000,
            .write_order = 30,
            .physical_pages = {0xab8d'0000},
        },
        VideoCore::PhysicalBackingTextureOwnershipRecord{
            .image_index = 44,
            .guest_base = 0x114b'8500'00,
            .guest_size = 0x2a'9600,
            .write_order = 40,
            .physical_pages = {0xab8d'4000, 0xab8d'8000},
        },
        VideoCore::PhysicalBackingTextureOwnershipRecord{
            .image_index = 51,
            .guest_base = 0x114b'4c00'00,
            .guest_size = 0x24'0000,
            .write_order = 10,
            .physical_pages = {0xab8c'c000, 0xab8d'0000},
        },
        VideoCore::PhysicalBackingTextureOwnershipRecord{
            .image_index = 77,
            .guest_base = 0x1200'0000'00,
            .guest_size = 0x4000,
            .write_order = 50,
            .physical_pages = {0xbeef'0000},
        },
    };
    constexpr std::array seed_pages{0xab8d'0000ULL};

    const auto plan =
        VideoCore::PlanPhysicalBackingTextureOwnershipComponent(records, seed_pages);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->ordered_image_indices, (std::vector<u32>{51, 16, 41, 44}));
    EXPECT_EQ(plan->ownership_span.base, 0x114b'4c00'00);
    EXPECT_EQ(plan->ownership_span.size, 0x63'c000);
    EXPECT_EQ(plan->physical_pages,
              (std::vector<u64>{0xab8c'c000, 0xab8d'0000, 0xab8d'4000, 0xab8d'8000}));
}
