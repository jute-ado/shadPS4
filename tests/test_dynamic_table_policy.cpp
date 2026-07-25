// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

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
