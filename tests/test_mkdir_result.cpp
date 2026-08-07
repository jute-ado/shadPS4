// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <system_error>

#include <gtest/gtest.h>

#include "core/libraries/kernel/mkdir_result.h"
#include "core/libraries/kernel/posix_error.h"

using Libraries::Kernel::ClassifyMkdirResult;

TEST(MkdirResult, ReportsConcurrentCreatorAsExistingDirectory) {
    const std::error_code no_error;

    EXPECT_EQ(ClassifyMkdirResult(false, no_error), POSIX_EEXIST);
}

TEST(MkdirResult, ReportsConcurrentCreatorFileExistsErrorAsExistingDirectory) {
    const auto file_exists = std::make_error_code(std::errc::file_exists);

    EXPECT_EQ(ClassifyMkdirResult(false, file_exists), POSIX_EEXIST);
}

TEST(MkdirResult, PreservesFilesystemErrorWhenDirectoryExistsAfterFailure) {
    const std::error_code io_error = std::make_error_code(std::errc::io_error);

    EXPECT_EQ(ClassifyMkdirResult(false, io_error), POSIX_EIO);
}

TEST(MkdirResult, ReportsSuccessfulCreationAtomically) {
    const std::error_code no_error;

    EXPECT_EQ(ClassifyMkdirResult(true, no_error), 0);
}
