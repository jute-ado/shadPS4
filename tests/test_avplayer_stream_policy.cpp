// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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

} // namespace
} // namespace Libraries::AvPlayer
