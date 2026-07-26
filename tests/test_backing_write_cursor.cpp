// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <span>

#include <gtest/gtest.h>

#include "core/backing_write_cursor.h"

TEST(BackingWriteCursor, AdvancesSourceAcrossFragmentedBackingRanges) {
    constexpr std::array<u8, 10> Source{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<u8, 3> first{};
    std::array<u8, 4> second{};
    std::array<u8, 3> third{};
    std::array<std::span<u8>, 3> destinations{first, second, third};

    Core::BackingWriteCursor cursor{Source};
    EXPECT_TRUE(cursor.TryWrite(destinations));

    EXPECT_EQ(first, (std::array<u8, 3>{0, 1, 2}));
    EXPECT_EQ(second, (std::array<u8, 4>{3, 4, 5, 6}));
    EXPECT_EQ(third, (std::array<u8, 3>{7, 8, 9}));
    EXPECT_TRUE(cursor.IsComplete());
}

TEST(BackingWriteCursor, StopsAtTheEndOfTheSource) {
    constexpr std::array<u8, 3> Source{0x12, 0x34, 0x56};
    std::array<u8, 5> destination{};

    Core::BackingWriteCursor cursor{Source};
    EXPECT_EQ(cursor.Write(destination), Source.size());

    EXPECT_EQ(destination, (std::array<u8, 5>{0x12, 0x34, 0x56, 0, 0}));
    EXPECT_EQ(cursor.Remaining(), 0u);
    EXPECT_TRUE(cursor.IsComplete());
}

TEST(BackingWriteCursor, RejectsIncompleteBackingWithoutModifyingAValidPrefix) {
    constexpr std::array<u8, 6> Source{0, 1, 2, 3, 4, 5};
    std::array<u8, 2> first{0xAA, 0xAA};
    std::array<u8, 3> second{0xBB, 0xBB, 0xBB};
    std::array<std::span<u8>, 2> destinations{first, second};

    Core::BackingWriteCursor cursor{Source};
    EXPECT_FALSE(cursor.TryWrite(destinations));

    EXPECT_EQ(first, (std::array<u8, 2>{0xAA, 0xAA}));
    EXPECT_EQ(second, (std::array<u8, 3>{0xBB, 0xBB, 0xBB}));
    EXPECT_EQ(cursor.Remaining(), Source.size());
}
