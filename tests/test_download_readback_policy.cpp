// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/download_readback_policy.h"

namespace VideoCore {
namespace {

std::string ReadText(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(DownloadReadbackPolicy, NonCoherentReservationsAreAtomIsolated) {
    const auto reservation = PlanDownloadReservation(65, 16, 64, false);

    EXPECT_EQ(reservation.size, 128U);
    EXPECT_EQ(reservation.alignment, 64U);
}

TEST(DownloadReadbackPolicy, CoherentReservationsKeepRequestedShape) {
    const auto reservation = PlanDownloadReservation(65, 16, 64, true);

    EXPECT_EQ(reservation.size, 65U);
    EXPECT_EQ(reservation.alignment, 16U);
}

TEST(DownloadReadbackPolicy, ExactReadbackClipsTrackerPageToRequestedBytes) {
    const auto range = ClipDownloadRange(0x242710000ULL, 0x1000, 0x242710318ULL, 8);

    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->address, 0x242710318ULL);
    EXPECT_EQ(range->size, 8U);
}

TEST(DownloadReadbackPolicy, ExactReadbackRejectsDisjointTrackerRange) {
    EXPECT_FALSE(ClipDownloadRange(0x1000, 0x1000, 0x3000, 8).has_value());
}

TEST(DownloadReadbackPolicy, CpuConsumptionFollowsFinishThenInvalidation) {
    std::vector<std::string_view> operations;

    const bool completed = CompleteDownloadReadback([&] { operations.emplace_back("finish"); },
                                                    [&] {
                                                        operations.emplace_back("invalidate");
                                                        return true;
                                                    },
                                                    [&] { operations.emplace_back("consume"); });

    EXPECT_TRUE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"finish", "invalidate", "consume"}));
}

TEST(DownloadReadbackPolicy, FailedInvalidationPreventsCpuConsumption) {
    std::vector<std::string_view> operations;

    const bool completed = CompleteDownloadReadback([&] { operations.emplace_back("finish"); },
                                                    [&] {
                                                        operations.emplace_back("invalidate");
                                                        return false;
                                                    },
                                                    [&] { operations.emplace_back("consume"); });

    EXPECT_FALSE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"finish", "invalidate"}));
}

TEST(DownloadReadbackPolicy, PrivateSnapshotComposesCpuAndExactGpuBytesAfterCompletion) {
    std::array<u8, 8> output{};
    const std::array<u8, 8> cpu{0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17};
    const std::array<u8, 4> gpu{0xa1, 0xa2, 0xb1, 0xb2};
    const std::array overlays{
        MemorySnapshotOverlay{.output_offset = 1, .download_offset = 0, .size = 2},
        MemorySnapshotOverlay{.output_offset = 5, .download_offset = 2, .size = 2},
    };
    std::vector<std::string_view> operations;

    const bool completed = CompletePrivateSnapshotReadback(
        [&] { operations.emplace_back("finish"); },
        [&] {
            operations.emplace_back("invalidate");
            return true;
        },
        [&] {
            operations.emplace_back("cpu");
            output = cpu;
            return true;
        },
        [&] {
            operations.emplace_back("gpu");
            return ApplyMemorySnapshotOverlays(output, gpu, overlays);
        });

    EXPECT_TRUE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"finish", "invalidate", "cpu", "gpu"}));
    EXPECT_EQ(output, (std::array<u8, 8>{0x10, 0xa1, 0xa2, 0x13, 0x14, 0xb1, 0xb2, 0x17}));
}

TEST(DownloadReadbackPolicy, InvalidOverlayLeavesDestinationUnchanged) {
    std::array<u8, 4> output{1, 2, 3, 4};
    const std::array<u8, 2> gpu{5, 6};
    const std::array overlays{
        MemorySnapshotOverlay{.output_offset = 3, .download_offset = 0, .size = 2},
    };

    EXPECT_FALSE(ApplyMemorySnapshotOverlays(output, gpu, overlays));
    EXPECT_EQ(output, (std::array<u8, 4>{1, 2, 3, 4}));
}

TEST(DownloadReadbackPolicy, ExactGpuWritesUsePrivateGpuReadback) {
    std::vector<std::string_view> operations;
    const bool completed = ReadAuthoritativeMemorySnapshot(
        true, true,
        [&] {
            operations.emplace_back("cpu");
            return true;
        },
        [&] {
            operations.emplace_back("gpu");
            return true;
        });

    EXPECT_TRUE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"gpu"}));
}

TEST(DownloadReadbackPolicy, PendingGpuOwnershipRefusesStaleCpuBacking) {
    std::vector<std::string_view> operations;
    const bool completed = ReadAuthoritativeMemorySnapshot(
        false, true,
        [&] {
            operations.emplace_back("cpu");
            return true;
        },
        [&] {
            operations.emplace_back("gpu");
            return true;
        });

    EXPECT_FALSE(completed);
    EXPECT_TRUE(operations.empty());
}

TEST(DownloadReadbackPolicy, CpuOwnedBytesUseGuestBacking) {
    std::vector<std::string_view> operations;
    const bool completed = ReadAuthoritativeMemorySnapshot(
        false, false,
        [&] {
            operations.emplace_back("cpu");
            return true;
        },
        [&] {
            operations.emplace_back("gpu");
            return true;
        });

    EXPECT_TRUE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"cpu"}));
}

TEST(DownloadReadbackPolicy, ProductionInvalidatesAfterCompletionBeforeCpuConsumption) {
    const auto buffer = ReadText(SHADPS4_BUFFER_SOURCE_PATH);
    const auto cache = ReadText(SHADPS4_BUFFER_CACHE_SOURCE_PATH);

    EXPECT_NE(buffer.find("PlanDownloadReservation(size, alignment"), std::string::npos);
    EXPECT_NE(buffer.find("bool Buffer::InvalidateMappedRange("), std::string::npos);

    const auto write_data = cache.find("const auto write_data =");
    const auto finish = cache.find("scheduler.Finish()", write_data);
    const auto invalidate = cache.find("download_buffer.InvalidateMappedRange", finish);
    const auto consume = cache.find("write_data", invalidate);
    ASSERT_NE(write_data, std::string::npos);
    ASSERT_NE(finish, std::string::npos);
    ASSERT_NE(invalidate, std::string::npos);
    ASSERT_NE(consume, std::string::npos);
    EXPECT_LT(finish, invalidate);
    EXPECT_LT(invalidate, consume);
}

} // namespace
} // namespace VideoCore
