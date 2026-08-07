// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

namespace fs = std::filesystem;

namespace {

class TemporaryLongPath {
public:
    TemporaryLongPath() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        root = fs::temp_directory_path() / ("shadps4-long-path-" + std::to_string(nonce));
        target = root;
        while (target.native().size() <= 280) {
            target /= "0123456789abcdef";
        }
    }

    ~TemporaryLongPath() {
        std::error_code error;
        fs::remove_all(ExtendedLengthPath(root), error);
    }

    static fs::path ExtendedLengthPath(const fs::path& path) {
        const auto absolute = fs::absolute(path).native();
        return fs::path{L"\\\\?\\" + absolute};
    }

    fs::path root;
    fs::path target;
};

} // namespace

TEST(WindowsLongPath, CreatesAndQueriesDirectoryBeyondLegacyMaxPath) {
    TemporaryLongPath path;
    ASSERT_GT(path.target.native().size(), 260u);

    std::error_code create_error;
    ASSERT_TRUE(fs::create_directories(path.target, create_error)) << create_error.message();
    ASSERT_FALSE(create_error) << create_error.message();

    std::error_code exists_error;
    EXPECT_TRUE(fs::exists(path.target, exists_error));
    EXPECT_FALSE(exists_error) << exists_error.message();
}
