// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <span>

#include <gtest/gtest.h>

#include "common/backing_copy.h"

TEST(BackingCopy, PreservesSourceOrderAcrossDestinationSpans) {
    const std::array<u8, 8> source{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<u8, 3> first{};
    std::array<u8, 2> second{};
    std::array<u8, 3> third{};
    const std::array<std::span<u8>, 3> destinations{first, second, third};

    ASSERT_TRUE(Common::CopyToWritableSpans(source, destinations));
    EXPECT_EQ(first, (std::array<u8, 3>{0, 1, 2}));
    EXPECT_EQ(second, (std::array<u8, 2>{3, 4}));
    EXPECT_EQ(third, (std::array<u8, 3>{5, 6, 7}));
}

TEST(BackingCopy, RejectsInsufficientCapacityBeforeWriting) {
    const std::array<u8, 5> source{1, 2, 3, 4, 5};
    std::array<u8, 2> first{0xaa, 0xaa};
    std::array<u8, 2> second{0xbb, 0xbb};
    const std::array<std::span<u8>, 2> destinations{first, second};

    EXPECT_FALSE(Common::CopyToWritableSpans(source, destinations));
    EXPECT_EQ(first, (std::array<u8, 2>{0xaa, 0xaa}));
    EXPECT_EQ(second, (std::array<u8, 2>{0xbb, 0xbb}));
}
