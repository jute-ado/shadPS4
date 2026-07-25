// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
