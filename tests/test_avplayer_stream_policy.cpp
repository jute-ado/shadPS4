// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "core/libraries/avplayer/avplayer_stream_policy.h"

namespace Libraries::AvPlayer {
namespace {

TEST(AvPlayerStreamPolicy, DiscoveryRequiresAtLeastOneSupportedStream) {
    EXPECT_FALSE(HasDiscoveredStreams(0));
    EXPECT_TRUE(HasDiscoveredStreams(1));
}

TEST(AvPlayerStreamPolicy, SelectedStreamIndexMustAddressTheFilteredList) {
    EXPECT_FALSE(IsValidSelectedStreamIndex(-1, 2));
    EXPECT_TRUE(IsValidSelectedStreamIndex(1, 2));
    EXPECT_FALSE(IsValidSelectedStreamIndex(2, 2));
}

TEST(AvPlayerStreamPolicy, FfmpegIndexUsesTheUnfilteredFormatStreamCount) {
    EXPECT_TRUE(IsValidFfmpegStreamIndex(2, 3));
    EXPECT_FALSE(IsValidFfmpegStreamIndex(3, 3));
}

TEST(AvPlayerStreamPolicy, FileReadRequestRejectsNegativeSizesAndStopsAtEndOfFile) {
    EXPECT_FALSE(ResolveFileReadRequest(4, 8, -1).has_value());
    EXPECT_EQ(ResolveFileReadRequest(4, 8, 8), 4u);
    EXPECT_EQ(ResolveFileReadRequest(8, 8, 8), 0u);
}

TEST(AvPlayerStreamPolicy, FailedFileReadDoesNotCorruptTheStreamPosition) {
    const auto result = ResolveFileReadResult(7, 4, -5);

    EXPECT_EQ(result.return_value, -5);
    EXPECT_EQ(result.next_position, 7u);
}

TEST(AvPlayerStreamPolicy, OverreportedFileReadIsLimitedToTheRequestedBytes) {
    const auto result = ResolveFileReadResult(7, 4, 9);

    EXPECT_EQ(result.return_value, 4);
    EXPECT_EQ(result.next_position, 11u);
}

TEST(AvPlayerStreamPolicy, FileSeekSaturatesWithoutSignedOverflow) {
    constexpr auto largest_offset = std::numeric_limits<s64>::max();
    constexpr auto smallest_offset = std::numeric_limits<s64>::min();
    constexpr u64 seek_limit = static_cast<u64>(largest_offset);

    EXPECT_EQ(ResolveFileSeek(seek_limit - 1, seek_limit, largest_offset), seek_limit);
    EXPECT_EQ(ResolveFileSeek(seek_limit, seek_limit, smallest_offset), 0u);
    EXPECT_EQ(ResolveFileSeek(7, 10, -3), 4u);
}

} // namespace
} // namespace Libraries::AvPlayer
