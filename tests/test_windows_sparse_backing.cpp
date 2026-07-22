// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h>
#include "core/windows_sparse_backing.h"

TEST(WindowsSparseBacking, CommittedSectionRangeIsVisibleThroughAlias) {
    constexpr u64 Granularity = 64_KB;
    constexpr u64 SectionSize = 2 * Granularity;

    const HANDLE process = GetCurrentProcess();
    const HANDLE section = CreateFileMapping2(
        INVALID_HANDLE_VALUE, nullptr, FILE_MAP_ALL_ACCESS, PAGE_EXECUTE_READWRITE,
        Core::WindowsSparseBacking::SectionAllocationType, SectionSize, nullptr, nullptr, 0);
    ASSERT_NE(section, nullptr);

    void* canonical_placeholder =
        VirtualAlloc2(process, nullptr, SectionSize, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                      PAGE_NOACCESS, nullptr, 0);
    ASSERT_NE(canonical_placeholder, nullptr);
    void* canonical = MapViewOfFile3(section, process, canonical_placeholder, 0, SectionSize,
                                     MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0);
    ASSERT_EQ(canonical, canonical_placeholder);

    void* alias_placeholder =
        VirtualAlloc2(process, nullptr, Granularity, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                      PAGE_NOACCESS, nullptr, 0);
    ASSERT_NE(alias_placeholder, nullptr);
    void* alias = MapViewOfFile3(section, process, alias_placeholder, Granularity, Granularity,
                                 MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
    ASSERT_EQ(alias, alias_placeholder);

    void* committed = Core::WindowsSparseBacking::CommitRange(canonical, Granularity, Granularity,
                                                              PAGE_EXECUTE_READWRITE);
    ASSERT_EQ(committed, static_cast<u8*>(canonical) + Granularity);

    auto* canonical_bytes = static_cast<u8*>(canonical);
    auto* alias_bytes = static_cast<u8*>(alias);
    canonical_bytes[Granularity + 17] = 0xA5;
    EXPECT_EQ(alias_bytes[17], 0xA5);
    alias_bytes[31] = 0x5A;
    EXPECT_EQ(canonical_bytes[Granularity + 31], 0x5A);

    EXPECT_TRUE(UnmapViewOfFile2(process, alias, MEM_PRESERVE_PLACEHOLDER));
    EXPECT_TRUE(VirtualFreeEx(process, alias, 0, MEM_RELEASE));
    EXPECT_TRUE(UnmapViewOfFile2(process, canonical, MEM_PRESERVE_PLACEHOLDER));
    EXPECT_TRUE(VirtualFreeEx(process, canonical, 0, MEM_RELEASE));
    EXPECT_TRUE(CloseHandle(section));
}
#endif
