// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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

    const bool completed = CompleteDownloadReadback(
        [&] { operations.emplace_back("finish"); },
        [&] {
            operations.emplace_back("invalidate");
            return true;
        },
        [&] { operations.emplace_back("consume"); });

    EXPECT_TRUE(completed);
    EXPECT_EQ(operations,
              (std::vector<std::string_view>{"finish", "invalidate", "consume"}));
}

TEST(DownloadReadbackPolicy, FailedInvalidationPreventsCpuConsumption) {
    std::vector<std::string_view> operations;

    const bool completed = CompleteDownloadReadback(
        [&] { operations.emplace_back("finish"); },
        [&] {
            operations.emplace_back("invalidate");
            return false;
        },
        [&] { operations.emplace_back("consume"); });

    EXPECT_FALSE(completed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"finish", "invalidate"}));
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
