// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "video_core/mapped_page_protection.h"

namespace {

constexpr VAddr PageSize = 0x1000;

enum class TestPermission {
    None,
    Read,
};

using Protection = std::tuple<VAddr, u64, TestPermission>;

TEST(MappedPageProtectionState, RejectsEmptyAndWrappedRangesWithoutCallbacks) {
    VideoCore::MappedPageProtectionState state;
    u32 callback_count{};
    const auto callback = [&](VAddr, u64) { ++callback_count; };

    EXPECT_FALSE(state.PublishMappedRange(0x1000, 0, callback, callback));
    EXPECT_FALSE(state.PublishMappedRange(std::numeric_limits<VAddr>::max() - 0x7ff, PageSize,
                                          callback, callback));
    EXPECT_FALSE(state.RetireMappedRange(0x1000, 0, callback));
    EXPECT_FALSE(state.Contains(0x1000, 0));
    EXPECT_EQ(callback_count, 0);
}

TEST(MappedPageProtectionState, PartialRetireSplitsMappedRange) {
    VideoCore::MappedPageProtectionState state;
    ASSERT_TRUE(state.PublishMappedRange(0x1000, 0x4000, [](VAddr, u64) {}, [](VAddr, u64) {}));

    ASSERT_TRUE(state.RetireMappedRange(0x2000, PageSize, [](VAddr, u64) {}));

    EXPECT_TRUE(state.Contains(0x1000, PageSize));
    EXPECT_FALSE(state.Contains(0x2000, PageSize));
    EXPECT_TRUE(state.Contains(0x3000, 0x2000));
    EXPECT_FALSE(state.Contains(0x1000, 0x4000));
}

TEST(MappedPageProtectionState, RetireBeforeFirstPublishIsNoOp) {
    VideoCore::MappedPageProtectionState state;
    bool unregistered{};

    EXPECT_TRUE(
        state.RetireMappedRange(0x1000, PageSize, [&](VAddr, u64) { unregistered = true; }));
    EXPECT_FALSE(unregistered);
    EXPECT_FALSE(state.Contains(0x1000, PageSize));
}

TEST(MappedPageProtectionState, PartialCoverageRetireOnlyRemovesCoveredSpan) {
    VideoCore::MappedPageProtectionState state;
    ASSERT_TRUE(state.PublishMappedRange(0x2000, PageSize, [](VAddr, u64) {}, [](VAddr, u64) {}));
    std::vector<std::pair<VAddr, u64>> unregistered;

    ASSERT_TRUE(state.RetireMappedRange(0x1000, 0x3000, [&](VAddr address, u64 size) {
        unregistered.emplace_back(address, size);
    }));

    EXPECT_EQ(unregistered, (std::vector<std::pair<VAddr, u64>>{{0x2000, PageSize}}));
    EXPECT_FALSE(state.Contains(0x2000, PageSize));
}

TEST(MappedPageProtectionState, OverlappingOwnersRetainCoverageUntilLastRetire) {
    VideoCore::MappedPageProtectionState state;
    std::vector<std::pair<VAddr, u64>> registered;
    std::vector<std::pair<VAddr, u64>> unregistered;
    const auto register_range = [&](VAddr address, u64 size) {
        registered.emplace_back(address, size);
    };

    ASSERT_TRUE(state.PublishMappedRange(0x1000, 0x4000, register_range, [](VAddr, u64) {}));
    ASSERT_TRUE(state.PublishMappedRange(0x2000, PageSize, register_range, [](VAddr, u64) {}));
    EXPECT_EQ(registered, (std::vector<std::pair<VAddr, u64>>{{0x1000, 0x4000}}));

    ASSERT_TRUE(state.RetireMappedRange(0x2000, PageSize, [&](VAddr address, u64 size) {
        unregistered.emplace_back(address, size);
    }));
    EXPECT_TRUE(state.Contains(0x1000, 0x4000));
    EXPECT_TRUE(unregistered.empty());

    ASSERT_TRUE(state.RetireMappedRange(0x1000, 0x4000, [&](VAddr address, u64 size) {
        unregistered.emplace_back(address, size);
    }));
    EXPECT_EQ(unregistered, (std::vector<std::pair<VAddr, u64>>{{0x1000, 0x4000}}));
    EXPECT_FALSE(state.Contains(0x1000, PageSize));
}

TEST(MappedPageProtectionState, OverlapCallbacksOnlyCoverVisibilityTransitions) {
    VideoCore::MappedPageProtectionState state;
    ASSERT_TRUE(state.PublishMappedRange(0x2000, PageSize, [](VAddr, u64) {}, [](VAddr, u64) {}));
    std::vector<std::pair<VAddr, u64>> registered;

    ASSERT_TRUE(state.PublishMappedRange(
        0x1000, 0x3000, [&](VAddr address, u64 size) { registered.emplace_back(address, size); },
        [](VAddr, u64) {}));

    EXPECT_EQ(registered,
              (std::vector<std::pair<VAddr, u64>>{{0x1000, PageSize}, {0x3000, PageSize}}));
}

TEST(MappedPageProtectionState, ProtectionAcrossMappedHolesEmitsSeparateRuns) {
    VideoCore::MappedPageProtectionState state;
    ASSERT_TRUE(state.PublishMappedRange(0x1000, PageSize, [](VAddr, u64) {}, [](VAddr, u64) {}));
    ASSERT_TRUE(state.PublishMappedRange(0x3000, 0x2000, [](VAddr, u64) {}, [](VAddr, u64) {}));
    std::vector<std::pair<VAddr, u64>> ranges;

    ASSERT_TRUE(state.ForEachMappedSubrange(
        0x0800, 0x5000, [&](VAddr address, u64 size) { ranges.emplace_back(address, size); }));

    EXPECT_EQ(ranges, (std::vector<std::pair<VAddr, u64>>{{0x1000, PageSize}, {0x3000, 0x2000}}));
}

TEST(MappedPageProtectionState, LiveLowAddressHoleNeverInvokesProtection) {
    VideoCore::MappedPageProtectionState state;
    bool protected_unmapped_memory{};

    ASSERT_TRUE(state.ForEachMappedSubrange(0x7fc000, 0x4000,
                                            [&](VAddr, u64) { protected_unmapped_memory = true; }));

    EXPECT_FALSE(protected_unmapped_memory);
}

TEST(MappedPageProtectionState, PublishOccursAfterRegistrationAndReplay) {
    VideoCore::MappedPageProtectionState state;
    std::vector<std::string> events;

    ASSERT_TRUE(state.PublishMappedRange(
        0x1000, PageSize,
        [&](VAddr, u64) {
            EXPECT_FALSE(state.Contains(0x1000, PageSize));
            events.emplace_back("register");
        },
        [&](VAddr, u64) {
            EXPECT_FALSE(state.Contains(0x1000, PageSize));
            events.emplace_back("replay");
        }));

    EXPECT_EQ(events, (std::vector<std::string>{"register", "replay"}));
    EXPECT_TRUE(state.Contains(0x1000, PageSize));
}

TEST(MappedPageProtectionState, RetireOccursBeforeUnregister) {
    VideoCore::MappedPageProtectionState state;
    ASSERT_TRUE(state.PublishMappedRange(0x1000, PageSize, [](VAddr, u64) {}, [](VAddr, u64) {}));
    bool unregistered{};

    ASSERT_TRUE(state.RetireMappedRange(0x1000, PageSize, [&](VAddr, u64) {
        EXPECT_FALSE(state.Contains(0x1000, PageSize));
        unregistered = true;
    }));

    EXPECT_TRUE(unregistered);
}

TEST(MappedPageProtectionReplay, EqualAdjacentWatcherPermissionsCoalesce) {
    const std::vector<std::optional<TestPermission>> pages{
        TestPermission::None, TestPermission::None, TestPermission::Read, TestPermission::Read};
    std::vector<Protection> protections;

    ASSERT_TRUE(VideoCore::ReplayActivePageProtectionRuns(
        0x1000, pages.size() * PageSize, PageSize,
        [&](VAddr page_address) { return pages[(page_address - 0x1000) / PageSize]; },
        [&](VAddr address, u64 size, TestPermission permission) {
            protections.emplace_back(address, size, permission);
        }));

    EXPECT_EQ(protections, (std::vector<Protection>{{0x1000, 0x2000, TestPermission::None},
                                                    {0x3000, 0x2000, TestPermission::Read}}));
}

TEST(MappedPageProtectionReplay, UntrackedPagesAreSkippedDuringReplay) {
    const std::vector<std::optional<TestPermission>> pages{TestPermission::None, std::nullopt,
                                                           TestPermission::None};
    std::vector<Protection> protections;

    ASSERT_TRUE(VideoCore::ReplayActivePageProtectionRuns(
        0x1000, pages.size() * PageSize, PageSize,
        [&](VAddr page_address) { return pages[(page_address - 0x1000) / PageSize]; },
        [&](VAddr address, u64 size, TestPermission permission) {
            protections.emplace_back(address, size, permission);
        }));

    EXPECT_EQ(protections, (std::vector<Protection>{{0x1000, PageSize, TestPermission::None},
                                                    {0x3000, PageSize, TestPermission::None}}));
}

TEST(MappedPageProtectionReplay, WatcherArmedWhileUnmappedReplaysOnPublish) {
    VideoCore::MappedPageProtectionState state;
    const std::vector<std::optional<TestPermission>> pages{TestPermission::None,
                                                           TestPermission::None};
    std::vector<Protection> protections;
    bool registered{};

    ASSERT_TRUE(state.PublishMappedRange(
        0x1000, 0x2000, [&](VAddr, u64) { registered = true; },
        [&](VAddr address, u64 size) {
            EXPECT_TRUE(registered);
            EXPECT_FALSE(state.Contains(address, size));
            EXPECT_TRUE(VideoCore::ReplayActivePageProtectionRuns(
                address, size, PageSize,
                [&](VAddr page_address) { return pages[(page_address - address) / PageSize]; },
                [&](VAddr run_address, u64 run_size, TestPermission permission) {
                    protections.emplace_back(run_address, run_size, permission);
                }));
        }));

    EXPECT_EQ(protections, (std::vector<Protection>{{0x1000, 0x2000, TestPermission::None}}));
    EXPECT_TRUE(state.Contains(0x1000, 0x2000));
}

TEST(MappedPageProtectionReplay, RejectsUnalignedAndWrappedRanges) {
    u32 emitted{};
    const auto state_at_page = [](VAddr) { return std::optional{TestPermission::None}; };
    const auto emit = [&](VAddr, u64, TestPermission) { ++emitted; };

    EXPECT_FALSE(
        VideoCore::ReplayActivePageProtectionRuns(0x1001, PageSize, PageSize, state_at_page, emit));
    EXPECT_FALSE(VideoCore::ReplayActivePageProtectionRuns(0x1000, PageSize - 1, PageSize,
                                                           state_at_page, emit));
    EXPECT_FALSE(VideoCore::ReplayActivePageProtectionRuns(
        std::numeric_limits<VAddr>::max() - 0x7ff, PageSize, PageSize, state_at_page, emit));
    EXPECT_EQ(emitted, 0);
}

TEST(AddressRangeChunking, SplitsAtLockGranularityWithoutRecursion) {
    std::vector<std::pair<VAddr, u64>> chunks;

    ASSERT_TRUE(VideoCore::ForEachAddressRangeChunk(
        0x3000, 0x801000, 0x400000,
        [&](VAddr address, u64 size) { chunks.emplace_back(address, size); }));

    EXPECT_EQ(chunks, (std::vector<std::pair<VAddr, u64>>{
                          {0x3000, 0x3fd000}, {0x400000, 0x400000}, {0x800000, 0x4000}}));
}

TEST(LogicalGpuMappingState, SparseOwnerSurvivesUnmatchedAndPairedHostLifecycle) {
    VideoCore::LogicalGpuMappingState state;
    ASSERT_TRUE(state.PublishSparsePlaceholderRange(0x1000, 0x4000));

    EXPECT_TRUE(state.RetireHostMappedRange(0x2000, PageSize));
    EXPECT_TRUE(state.Contains(0x1000, 0x4000));

    ASSERT_TRUE(state.PublishHostMappedRange(0x2000, PageSize));
    ASSERT_TRUE(state.RetireHostMappedRange(0x2000, PageSize));
    EXPECT_TRUE(state.Contains(0x1000, 0x4000));

    ASSERT_TRUE(state.RetireSparsePlaceholderRange(0x1000, 0x4000));
    EXPECT_FALSE(state.Contains(0x1000, PageSize));
}

TEST(LogicalGpuMappingState, AdjacentOwnerKindsFormOneLogicalGpuRange) {
    VideoCore::LogicalGpuMappingState state;
    ASSERT_TRUE(state.PublishSparsePlaceholderRange(0x1000, PageSize));
    ASSERT_TRUE(state.PublishHostMappedRange(0x2000, PageSize));

    EXPECT_TRUE(state.Contains(0x1000, 0x2000));
}

TEST(LogicalGpuMappingState, HostOwnerSurvivesUnmatchedSparseRetire) {
    VideoCore::LogicalGpuMappingState state;
    ASSERT_TRUE(state.PublishHostMappedRange(0x1000, 0x2000));

    EXPECT_TRUE(state.RetireSparsePlaceholderRange(0x1000, 0x2000));
    EXPECT_TRUE(state.Contains(0x1000, 0x2000));

    ASSERT_TRUE(state.RetireHostMappedRange(0x1000, 0x2000));
    EXPECT_FALSE(state.Contains(0x1000, PageSize));
}

TEST(LogicalGpuMappingState, SparsePlaceholderPreservesAbiByteGranularity) {
    VideoCore::LogicalGpuMappingState state;
    ASSERT_TRUE(state.PublishSparsePlaceholderRange(0x1000, 0x1234));

    EXPECT_TRUE(state.Contains(0x1000, 0x1234));
    EXPECT_FALSE(state.Contains(0x1000, 0x1235));
    EXPECT_TRUE(state.RetireSparsePlaceholderRange(0x1000, 0x1234));
}

} // namespace
