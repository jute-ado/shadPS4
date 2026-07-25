// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <string_view>

#include <gtest/gtest.h>

#include "core/loader/dynamic_table_policy.h"
#include "core/loader/elf.h"

using namespace Core::Loader;

TEST(DynamicTablePolicy, FindsTerminatorInsideLoadedSegment) {
    const std::array entries{
        elf_dynamic{.d_tag = DT_INIT, .d_un = {.d_val = 0x1234}},
        elf_dynamic{.d_tag = DT_NULL, .d_un = {.d_val = 0}},
        elf_dynamic{.d_tag = DT_FINI, .d_un = {.d_val = 0x5678}},
    };

    EXPECT_EQ(FindDynamicTerminator(entries), 1u);
}

TEST(DynamicTablePolicy, AcceptsAnEmptyTableTerminatedByItsFirstEntry) {
    const std::array entries{
        elf_dynamic{.d_tag = DT_NULL, .d_un = {.d_val = 0}},
    };

    EXPECT_EQ(FindDynamicTerminator(entries), 0u);
}

TEST(DynamicTablePolicy, RejectsAnUnterminatedTable) {
    const std::array entries{
        elf_dynamic{.d_tag = DT_INIT, .d_un = {.d_val = 0x1234}},
        elf_dynamic{.d_tag = DT_FINI, .d_un = {.d_val = 0x5678}},
    };

    EXPECT_FALSE(FindDynamicTerminator(entries).has_value());
}

TEST(DynamicTablePolicy, RequiresWholeDynamicEntries) {
    EXPECT_TRUE(IsDynamicTableSizeAligned(0));
    EXPECT_TRUE(IsDynamicTableSizeAligned(sizeof(elf_dynamic) * 3));
    EXPECT_FALSE(IsDynamicTableSizeAligned(sizeof(elf_dynamic) - 1));
    EXPECT_FALSE(IsDynamicTableSizeAligned(sizeof(elf_dynamic) + 1));
}

TEST(DynamicTablePolicy, FindsStringTableIndependentOfDynamicTagOrder) {
    const std::array entries{
        elf_dynamic{.d_tag = DT_NEEDED, .d_un = {.d_val = 1}},
        elf_dynamic{.d_tag = DT_SCE_STRSZ, .d_un = {.d_val = 5}},
        elf_dynamic{.d_tag = DT_SCE_STRTAB, .d_un = {.d_ptr = 2}},
    };
    const std::array<u8, 8> data{'x', 'x', '\0', 'l', 'i', 'b', '\0', 'x'};

    const auto table = FindDynamicStringTable(entries, data);

    ASSERT_TRUE(table.has_value());
    ASSERT_EQ(table->size(), 5u);
    EXPECT_EQ(ReadDynamicString(*table, 1), std::optional<std::string_view>{"lib"});
}

TEST(DynamicTablePolicy, RejectsStringTableOutsideDynamicData) {
    const std::array<u8, 8> data{};
    const std::array offset_outside{
        elf_dynamic{.d_tag = DT_SCE_STRTAB, .d_un = {.d_ptr = 9}},
        elf_dynamic{.d_tag = DT_SCE_STRSZ, .d_un = {.d_val = 0}},
    };
    const std::array size_outside{
        elf_dynamic{.d_tag = DT_SCE_STRTAB, .d_un = {.d_ptr = 4}},
        elf_dynamic{.d_tag = DT_SCE_STRSZ, .d_un = {.d_val = 5}},
    };

    EXPECT_FALSE(FindDynamicStringTable(offset_outside, data).has_value());
    EXPECT_FALSE(FindDynamicStringTable(size_outside, data).has_value());
}

TEST(DynamicTablePolicy, RequiresDynamicStringsToBeBoundedAndTerminated) {
    const std::array<char, 5> terminated{'\0', 'o', 'k', '\0', 'x'};
    const std::array<char, 3> unterminated{'b', 'a', 'd'};

    EXPECT_EQ(ReadDynamicString(terminated, 1), std::optional<std::string_view>{"ok"});
    EXPECT_FALSE(ReadDynamicString(terminated, terminated.size()).has_value());
    EXPECT_FALSE(ReadDynamicString(unterminated, 0).has_value());
}
