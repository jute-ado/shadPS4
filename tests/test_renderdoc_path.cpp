// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "video_core/renderdoc_path.h"

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("shadps4-renderdoc-path-test-" + std::to_string(suffix));
        std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

TEST(RenderDocPath, RejectsEmptyAndMissingPaths) {
    EXPECT_FALSE(VideoCore::ResolveRenderDocModulePath({}));
    EXPECT_FALSE(VideoCore::ResolveRenderDocModulePath("missing-renderdoc-directory"));
}

TEST(RenderDocPath, AcceptsAnExplicitLibraryFile) {
    TemporaryDirectory temporary;
    const auto library = temporary.path / "custom-renderdoc-library";
    std::ofstream{library}.put('\0');

    EXPECT_EQ(VideoCore::ResolveRenderDocModulePath(library), library);
}

TEST(RenderDocPath, ResolvesThePlatformLibraryInsideADirectory) {
    TemporaryDirectory temporary;
    const auto library = temporary.path / VideoCore::RenderDocModuleFilename();
    std::ofstream{library}.put('\0');

    EXPECT_EQ(VideoCore::ResolveRenderDocModulePath(temporary.path), library);
}

} // namespace
