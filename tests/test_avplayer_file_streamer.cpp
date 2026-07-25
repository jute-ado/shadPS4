// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "core/libraries/avplayer/avplayer_file_streamer.h"

namespace Libraries::AvPlayer {
namespace {

struct FileCallbacks {
    std::string opened_path;
    bool closed{};
};

s32 PS4_SYSV_ABI OpenFile(void* opaque, const char* path) {
    auto* callbacks = static_cast<FileCallbacks*>(opaque);
    callbacks->opened_path = path;
    return 1;
}

s32 PS4_SYSV_ABI CloseFile(void* opaque) {
    auto* callbacks = static_cast<FileCallbacks*>(opaque);
    callbacks->closed = true;
    return 0;
}

s32 PS4_SYSV_ABI ReadFile(void*, u8*, u64, u32) {
    return 0;
}

u64 PS4_SYSV_ABI FileSize(void*) {
    return 0;
}

TEST(AvPlayerFileStreamer, OpenReceivesOnlyTheBoundedPathView) {
    FileCallbacks callbacks;
    const AvPlayerFileReplacement replacement{
        .object_ptr = &callbacks,
        .open = OpenFile,
        .close = CloseFile,
        .read_offset = ReadFile,
        .size = FileSize,
    };
    constexpr std::array path_storage{'g', 'a', 'm', 'e', '.', 'p', 'm', 'f', 'X', '\0'};
    const std::string_view path{path_storage.data(), path_storage.size() - 2};

    {
        AvPlayerFileStreamer streamer{replacement};
        ASSERT_TRUE(streamer.Init(path));
        EXPECT_EQ(callbacks.opened_path, "game.pmf");
    }
    EXPECT_TRUE(callbacks.closed);
}

} // namespace
} // namespace Libraries::AvPlayer
