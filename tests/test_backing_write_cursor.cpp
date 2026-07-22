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

    Core::BackingWriteCursor cursor{Source};
    EXPECT_EQ(cursor.Write(first), first.size());
    EXPECT_EQ(cursor.Write(second), second.size());
    EXPECT_EQ(cursor.Write(third), third.size());

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
