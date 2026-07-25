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

TEST(ElfProgramHeaderPolicy, ValidatesFileBackedRangesWithoutOverflow) {
    EXPECT_TRUE(IsFileRangeValid(0x1000, 0x200, 0x300));
    EXPECT_TRUE(IsFileRangeValid(0x1000, 0x1000, 0));

    EXPECT_FALSE(IsFileRangeValid(0x1000, 0x1001, 0));
    EXPECT_FALSE(IsFileRangeValid(0x1000, 0xf00, 0x101));
    EXPECT_FALSE(
        IsFileRangeValid(std::numeric_limits<u64>::max(), std::numeric_limits<u64>::max() - 1, 2));
}

TEST(ElfProgramHeaderPolicy, ValidatesNestedSegmentRangesWithoutOverflow) {
    EXPECT_TRUE(IsRangeContained(0x200, 0x500, 0x300, 0x200));
    EXPECT_TRUE(IsRangeContained(0x200, 0x500, 0x700, 0));

    EXPECT_FALSE(IsRangeContained(0x200, 0x500, 0x1ff, 1));
    EXPECT_FALSE(IsRangeContained(0x200, 0x500, 0x600, 0x101));
    EXPECT_FALSE(IsRangeContained(std::numeric_limits<u64>::max() - 1, 2,
                                  std::numeric_limits<u64>::max(), 1));
}

TEST(ElfProgramHeaderPolicy, ResolvesBoundedSelfSegmentFileOffsets) {
    const std::array segments{
        self_segment_header{.flags = 0x800, .file_offset = 0x200, .file_size = 0x400},
    };
    const std::array headers{
        elf_program_header{.p_offset = 0x1000, .p_filesz = 0x400},
    };

    EXPECT_EQ(ResolveSelfSegmentFileOffset(segments, headers, 0x1000, 0x1100, 0x80), 0x300u);
    EXPECT_EQ(ResolveSelfSegmentFileOffset(segments, headers, 0x1000, 0x1400, 0), 0x600u);
}

TEST(ElfProgramHeaderPolicy, RejectsUnmappedOrTruncatedSelfSegmentRanges) {
    std::array segments{
        self_segment_header{.flags = 0x800, .file_offset = 0x200, .file_size = 0x400},
    };
    const std::array headers{
        elf_program_header{.p_offset = 0x1000, .p_filesz = 0x400},
    };

    EXPECT_FALSE(
        ResolveSelfSegmentFileOffset(segments, headers, 0x1000, 0x13f0, 0x20).has_value());
    EXPECT_FALSE(
        ResolveSelfSegmentFileOffset(segments, headers, 0x500, 0x1100, 0x80).has_value());

    segments[0].flags |= u64{1} << 20;
    EXPECT_FALSE(
        ResolveSelfSegmentFileOffset(segments, headers, 0x1000, 0x1100, 0x80).has_value());
}

TEST(ElfProgramHeaderPolicy, RejectsLoadSegmentsLargerThanTheirMemoryImage) {
    elf_program_header header{
        .p_type = PT_LOAD,
        .p_filesz = 0x1001,
        .p_memsz = 0x1000,
    };
    EXPECT_FALSE(IsProgramHeaderFileSizeValid(header));

    header.p_filesz = header.p_memsz;
    EXPECT_TRUE(IsProgramHeaderFileSizeValid(header));

    header.p_type = PT_DYNAMIC;
    header.p_memsz = 0;
    EXPECT_TRUE(IsProgramHeaderFileSizeValid(header));
}

TEST(ElfProgramHeaderPolicy, IdentifiesHeadersReadDirectlyFromTheFile) {
    EXPECT_TRUE(IsDirectlyLoadedFromFile(PT_LOAD));
    EXPECT_TRUE(IsDirectlyLoadedFromFile(PT_SCE_RELRO));
    EXPECT_TRUE(IsDirectlyLoadedFromFile(PT_DYNAMIC));
    EXPECT_TRUE(IsDirectlyLoadedFromFile(PT_SCE_DYNLIBDATA));

    EXPECT_FALSE(IsDirectlyLoadedFromFile(PT_TLS));
    EXPECT_FALSE(IsDirectlyLoadedFromFile(PT_SCE_PROCPARAM));
    EXPECT_FALSE(IsDirectlyLoadedFromFile(PT_GNU_EH_FRAME));
}

TEST(ElfProgramHeaderPolicy, ResolvesBoundedFileTableOffsets) {
    EXPECT_EQ(ResolveFileTableOffset(0x1000, 0x40, 0x80, 0x20, 4), 0xc0u);
    EXPECT_EQ(ResolveFileTableOffset(0x1000, 0x40, 0x80, 0x20, 0), 0xc0u);

    EXPECT_FALSE(ResolveFileTableOffset(0x100, 0x40, 0x80, 0x20, 3).has_value());
    EXPECT_FALSE(ResolveFileTableOffset(std::numeric_limits<u64>::max(),
                                        std::numeric_limits<u64>::max() - 1, 2, 1, 1)
                     .has_value());
    EXPECT_FALSE(ResolveFileTableOffset(std::numeric_limits<u64>::max(), 0, 0,
                                        std::numeric_limits<u64>::max(), 2)
                     .has_value());
}

TEST(ElfProgramHeaderPolicy, RequiresMatchingElfHeaderEntrySizes) {
    EXPECT_EQ(ResolveElfHeaderTableOffset(0x1000, 0x40, 0x80, sizeof(elf_program_header),
                                          sizeof(elf_program_header), 2),
              0xc0u);

    EXPECT_FALSE(ResolveElfHeaderTableOffset(0x1000, 0x40, 0x80,
                                             sizeof(elf_program_header) - 1,
                                             sizeof(elf_program_header), 2)
                     .has_value());
    EXPECT_FALSE(ResolveElfHeaderTableOffset(0x100, 0x40, 0x80,
                                             sizeof(elf_program_header),
                                             sizeof(elf_program_header), 2)
                     .has_value());
}

TEST(ElfProgramHeaderPolicy, BoundsOptionalSelfProgramIdInsideDeclaredHeader) {
    EXPECT_EQ(ResolveSelfProgramIdOffset(0x1000, 0x200, 0x180, 0x40), 0x180u);

    EXPECT_FALSE(ResolveSelfProgramIdOffset(0x1000, 0x17f, 0x180, 0x40).has_value());
    EXPECT_FALSE(ResolveSelfProgramIdOffset(0x1000, 0x1a0, 0x180, 0x40).has_value());
    EXPECT_FALSE(ResolveSelfProgramIdOffset(0x190, 0x200, 0x180, 0x40).has_value());
}
