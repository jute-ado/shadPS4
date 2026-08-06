// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <tuple>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_fault_admission.h"
#include "video_core/buffer_cache/region_manager.h"

TEST(BufferFaultAdmission, TracksCpuFaultWhileReplacementIsUnpublished) {
    int invalidation_count = 0;
    int dma_mark_count = 0;

    const bool tracked = VideoCore::ProcessTrackedBufferFault(
        [&] {
            ++invalidation_count;
            return true;
        },
        [&] { ++dma_mark_count; });

    EXPECT_TRUE(tracked);
    EXPECT_EQ(invalidation_count, 1);
    EXPECT_EQ(dma_mark_count, 1);
}

TEST(BufferFaultAdmission, LeavesUnknownUntrackedPageUnchanged) {
    int invalidation_count = 0;
    int dma_mark_count = 0;

    const bool tracked = VideoCore::ProcessTrackedBufferFault(
        [&] {
            ++invalidation_count;
            return false;
        },
        [&] { ++dma_mark_count; });

    EXPECT_FALSE(tracked);
    EXPECT_EQ(invalidation_count, 1);
    EXPECT_EQ(dma_mark_count, 0);
}

TEST(BufferFaultAdmission, RequiresRegistrationOrExactPageTrackerOwnership) {
    static constexpr VAddr ManagerBase = 4_MB;
    static constexpr VAddr RegisteredPage = ManagerBase;
    static constexpr VAddr TransientProtectedPage = ManagerBase + VideoCore::TRACKER_BYTES_PER_PAGE;
    static constexpr VAddr UnrelatedDirtyPage = ManagerBase + 2 * VideoCore::TRACKER_BYTES_PER_PAGE;

    VideoCore::RegionManager manager;
    manager.SetCpuAddress(ManagerBase);
    manager.GetRegionBits<VideoCore::Type::CPU>().Fill();
    manager.GetRegionBits<VideoCore::Type::CPU>().Unset(1);

    const auto process_fault = [&](VAddr address, bool is_registered) {
        int invalidation_count = 0;
        int dma_mark_count = 0;
        const bool admitted = VideoCore::ProcessTrackedBufferFault(
            [&] {
                ++invalidation_count;
                return is_registered ||
                       manager.IsRegionCpuTracked(address - manager.GetCpuAddr(), 1);
            },
            [&] { ++dma_mark_count; });
        return std::tuple{admitted, invalidation_count, dma_mark_count};
    };

    EXPECT_EQ(process_fault(RegisteredPage, true), std::tuple(true, 1, 1));
    EXPECT_EQ(process_fault(TransientProtectedPage, false), std::tuple(true, 1, 1));
    EXPECT_EQ(process_fault(UnrelatedDirtyPage, false), std::tuple(false, 1, 0));
}

TEST(BufferFaultAdmission, RetainsExactOwnershipSampleWhileWaitingForTransactionLock) {
    static constexpr VAddr ManagerBase = 8_MB;
    static constexpr VAddr SnapshotPage = ManagerBase + VideoCore::TRACKER_BYTES_PER_PAGE;

    VideoCore::RegionManager manager;
    manager.SetCpuAddress(ManagerBase);
    manager.GetRegionBits<VideoCore::Type::CPU>().Fill();
    manager.GetRegionBits<VideoCore::Type::CPU>().Unset(1);

    const bool owned_before_wait =
        manager.IsRegionCpuTracked(SnapshotPage - manager.GetCpuAddr(), 1);
    manager.GetRegionBits<VideoCore::Type::CPU>().Set(1);
    const bool owned_after_wait =
        manager.IsRegionCpuTracked(SnapshotPage - manager.GetCpuAddr(), 1);

    EXPECT_TRUE(owned_before_wait);
    EXPECT_FALSE(owned_after_wait);
    EXPECT_TRUE(VideoCore::IsBufferFaultOwned(false, owned_before_wait, owned_after_wait));
    EXPECT_FALSE(VideoCore::IsBufferFaultOwned(false, false, false));
}
