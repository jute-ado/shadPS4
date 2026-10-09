// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/download_readback_policy.h"

namespace VideoCore {
namespace {

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

} // namespace
} // namespace VideoCore
