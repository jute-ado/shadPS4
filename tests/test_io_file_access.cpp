// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "common/io_file.h"

using Common::FS::FileAccessMode;
using Common::FS::IOFile;

TEST(IOFileAccess, ClassifiesEverySupportedMode) {
    struct Expectation {
        FileAccessMode mode;
        bool readable;
        bool writable;
    };
    constexpr Expectation expectations[] = {
        {FileAccessMode::Read, true, false},
        {FileAccessMode::Write, false, true},
        {FileAccessMode::ReadWrite, true, true},
        {FileAccessMode::Append, false, true},
        {FileAccessMode::ReadAppend, true, true},
        {FileAccessMode::Create, false, true},
    };

    for (const auto& expectation : expectations) {
        EXPECT_EQ(IOFile::IsReadableMode(expectation.mode), expectation.readable);
        EXPECT_EQ(IOFile::IsWritableMode(expectation.mode), expectation.writable);
    }
}

TEST(IOFileAccess, WriteOnlyIncludesCreateAndAppendModes) {
    EXPECT_TRUE(IOFile::IsWriteOnlyMode(FileAccessMode::Write));
    EXPECT_TRUE(IOFile::IsWriteOnlyMode(FileAccessMode::Append));
    EXPECT_TRUE(IOFile::IsWriteOnlyMode(FileAccessMode::Create));
    EXPECT_FALSE(IOFile::IsWriteOnlyMode(FileAccessMode::Read));
    EXPECT_FALSE(IOFile::IsWriteOnlyMode(FileAccessMode::ReadWrite));
    EXPECT_FALSE(IOFile::IsWriteOnlyMode(FileAccessMode::ReadAppend));
}
