// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <filesystem>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "common/types.h"
#include "core/libraries/save_data/savedata_filesystem_boundary.h"

namespace Libraries::SaveData {
namespace {

TEST(SaveDataFilesystemBoundary, ReturnsOkWithoutCallingFailureHandler) {
    bool operation_called = false;
    bool failure_called = false;

    const auto result = RunFilesystemOperation(
        [&] { operation_called = true; },
        [&](const std::filesystem::filesystem_error&) { failure_called = true; });

    EXPECT_EQ(result, Error::OK);
    EXPECT_TRUE(operation_called);
    EXPECT_FALSE(failure_called);
}

TEST(SaveDataFilesystemBoundary, ConvertsFilesystemFailureToGuestInternalError) {
    const std::filesystem::path failed_path{"synthetic-save/sce_sys/param.sfo"};
    std::string reported_message;
    std::filesystem::path reported_path;

    EXPECT_NO_THROW({
        const auto result = RunFilesystemOperation(
            [&] {
                throw std::filesystem::filesystem_error(
                    "synthetic write failure", failed_path,
                    std::make_error_code(std::errc::permission_denied));
            },
            [&](const std::filesystem::filesystem_error& error) {
                reported_message = error.what();
                reported_path = error.path1();
            });

        EXPECT_EQ(result, Error::INTERNAL);
    });
    EXPECT_EQ(reported_path, failed_path);
    EXPECT_NE(reported_message.find("synthetic write failure"), std::string::npos);
}

TEST(SaveDataFilesystemBoundary, PreservesGuestResultReturnedByOperation) {
    const auto result = RunFilesystemOperation(
        [] { return Error::BUSY; }, [](const std::filesystem::filesystem_error&) {});

    EXPECT_EQ(result, Error::BUSY);
}

TEST(SaveDataFilesystemBoundary, PreservesGuestResultMappedByFailureHandler) {
    const auto result = RunFilesystemOperation(
        [] {
            throw std::filesystem::filesystem_error(
                "synthetic capacity failure", std::make_error_code(std::errc::no_space_on_device));
        },
        [](const std::filesystem::filesystem_error&) { return Error::NO_SPACE_FS; });

    EXPECT_EQ(result, Error::NO_SPACE_FS);
}

TEST(SaveDataFilesystemBoundary, DoesNotHideProgrammingErrors) {
    EXPECT_THROW(
        RunFilesystemOperation(
            [] { throw std::logic_error{"synthetic programming error"}; },
            [](const std::filesystem::filesystem_error&) {}),
        std::logic_error);
}

} // namespace
} // namespace Libraries::SaveData
