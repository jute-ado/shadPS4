// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <limits>

#include <gtest/gtest.h>

#include "core/loader/elf.h"
#include "core/loader/program_header_policy.h"

using namespace Core::Loader;

TEST(ElfProgramHeaderPolicy, RecognizesNonLoadableMetadata) {
    EXPECT_EQ(ClassifyProgramHeader(PT_NULL), ProgramHeaderAction::Ignore);
    EXPECT_EQ(ClassifyProgramHeader(PT_INTERP), ProgramHeaderAction::Ignore);
    EXPECT_EQ(ClassifyProgramHeader(PT_NOTE), ProgramHeaderAction::Ignore);
    EXPECT_EQ(ClassifyProgramHeader(PT_PHDR), ProgramHeaderAction::Ignore);
    EXPECT_EQ(ClassifyProgramHeader(PT_GNU_STACK), ProgramHeaderAction::Ignore);
    EXPECT_EQ(ClassifyProgramHeader(PT_GNU_RELRO), ProgramHeaderAction::Ignore);
    EXPECT_EQ(ClassifyProgramHeader(PT_SCE_MODULE_PARAM), ProgramHeaderAction::Ignore);
    EXPECT_EQ(ClassifyProgramHeader(PT_SCE_COMMENT), ProgramHeaderAction::Ignore);
    EXPECT_EQ(ClassifyProgramHeader(PT_SCE_LIBVERSION), ProgramHeaderAction::Ignore);
}

TEST(ElfProgramHeaderPolicy, KeepsImplementedHeadersOnProcessingPath) {
    EXPECT_EQ(ClassifyProgramHeader(PT_LOAD), ProgramHeaderAction::Process);
    EXPECT_EQ(ClassifyProgramHeader(PT_DYNAMIC), ProgramHeaderAction::Process);
    EXPECT_EQ(ClassifyProgramHeader(PT_TLS), ProgramHeaderAction::Process);
    EXPECT_EQ(ClassifyProgramHeader(PT_SCE_DYNLIBDATA), ProgramHeaderAction::Process);
    EXPECT_EQ(ClassifyProgramHeader(PT_SCE_PROCPARAM), ProgramHeaderAction::Process);
    EXPECT_EQ(ClassifyProgramHeader(PT_SCE_RELRO), ProgramHeaderAction::Process);
    EXPECT_EQ(ClassifyProgramHeader(PT_GNU_EH_FRAME), ProgramHeaderAction::Process);
}

TEST(ElfProgramHeaderPolicy, PreservesDiagnosticsForUnknownHeaders) {
    EXPECT_EQ(ClassifyProgramHeader(0x6abcdeff), ProgramHeaderAction::Unsupported);
}

TEST(ElfProgramHeaderPolicy, AlignsSegmentMemorySizeWithValidElfAlignment) {
    elf_program_header header{.p_memsz = 0x1001, .p_align = 0x1000};

    EXPECT_EQ(GetAlignedSegmentSize(header), 0x2000u);

    header.p_align = 0;
    EXPECT_EQ(GetAlignedSegmentSize(header), 0x1001u);
}

TEST(ElfProgramHeaderPolicy, RejectsInvalidOrOverflowingSegmentAlignment) {
    elf_program_header header{.p_memsz = 0x1001, .p_align = 3};
    EXPECT_FALSE(GetAlignedSegmentSize(header).has_value());

    header.p_memsz = std::numeric_limits<u64>::max();
    header.p_align = 0x1000;
    EXPECT_FALSE(GetAlignedSegmentSize(header).has_value());
}

TEST(ElfProgramHeaderPolicy, CalculatesMaximumLoadImageEnd) {
    const std::array headers{
        elf_program_header{.p_type = PT_LOAD, .p_vaddr = 0x1000, .p_memsz = 0x2000},
        elf_program_header{.p_type = PT_DYNAMIC, .p_vaddr = 0xf000, .p_memsz = 0x1000},
        elf_program_header{.p_type = PT_SCE_RELRO, .p_vaddr = 0x5000, .p_memsz = 0x1000},
    };

    EXPECT_EQ(CalculateLoadImageSize(headers), 0x6000u);
}

TEST(ElfProgramHeaderPolicy, RejectsOverflowingLoadImageEnd) {
    const std::array headers{
        elf_program_header{.p_type = PT_LOAD,
                           .p_vaddr = std::numeric_limits<u64>::max() - 3,
                           .p_memsz = 4},
    };

    EXPECT_FALSE(CalculateLoadImageSize(headers).has_value());
}
