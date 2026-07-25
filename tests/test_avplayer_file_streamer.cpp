// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <climits>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "core/libraries/avplayer/avplayer_file_streamer.h"

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/mem.h>
}

namespace Libraries::AvPlayer {
namespace {

struct FileCallbacks {
    std::string opened_path;
    u32 open_count{};
    u32 close_count{};
    u64 file_size{};
    bool closed{};
};

s32 PS4_SYSV_ABI OpenFile(void* opaque, const char* path) {
    auto* callbacks = static_cast<FileCallbacks*>(opaque);
    callbacks->opened_path = path;
    ++callbacks->open_count;
    return 1;
}

s32 PS4_SYSV_ABI CloseFile(void* opaque) {
    auto* callbacks = static_cast<FileCallbacks*>(opaque);
    callbacks->closed = true;
    ++callbacks->close_count;
    return 0;
}

s32 PS4_SYSV_ABI ReadFile(void*, u8*, u64, u32) {
    return 0;
}

u64 PS4_SYSV_ABI FileSize(void* opaque) {
    return static_cast<FileCallbacks*>(opaque)->file_size;
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

TEST(AvPlayerFileStreamer, ReinitializationIsRejectedWithoutReopeningTheFile) {
    FileCallbacks callbacks;
    const AvPlayerFileReplacement replacement{
        .object_ptr = &callbacks,
        .open = OpenFile,
        .close = CloseFile,
        .read_offset = ReadFile,
        .size = FileSize,
    };

    AvPlayerFileStreamer streamer{replacement};
    ASSERT_TRUE(streamer.Init("first.pmf"));
    EXPECT_FALSE(streamer.Init("second.pmf"));
    EXPECT_EQ(callbacks.open_count, 1u);
    EXPECT_EQ(callbacks.opened_path, "first.pmf");
}

TEST(AvPlayerFileStreamer, IncompleteReplacementIsRejectedBeforeOpeningTheFile) {
    FileCallbacks callbacks;
    const AvPlayerFileReplacement replacement{
        .object_ptr = &callbacks,
        .open = OpenFile,
        .close = CloseFile,
        .read_offset = nullptr,
        .size = FileSize,
    };

    AvPlayerFileStreamer streamer{replacement};
    EXPECT_FALSE(streamer.Init("game.pmf"));
    EXPECT_EQ(callbacks.open_count, 0u);
    EXPECT_FALSE(callbacks.closed);
}

TEST(AvPlayerFileStreamer, AllocationFailureClosesTheFileAndAllowsRetry) {
    FileCallbacks callbacks;
    const AvPlayerFileReplacement replacement{
        .object_ptr = &callbacks,
        .open = OpenFile,
        .close = CloseFile,
        .read_offset = ReadFile,
        .size = FileSize,
    };

    {
        AvPlayerFileStreamer streamer{replacement};
        av_max_alloc(1);
        EXPECT_FALSE(streamer.Init("game.pmf"));
        av_max_alloc(INT_MAX);

        EXPECT_EQ(callbacks.open_count, 1u);
        EXPECT_EQ(callbacks.close_count, 1u);
        EXPECT_TRUE(streamer.Init("game.pmf"));
        EXPECT_EQ(callbacks.open_count, 2u);
    }
    EXPECT_EQ(callbacks.close_count, 2u);
}

TEST(AvPlayerFileStreamer, ForcedSeekPreservesTheRequestedOrigin) {
    FileCallbacks callbacks{.file_size = 16};
    const AvPlayerFileReplacement replacement{
        .object_ptr = &callbacks,
        .open = OpenFile,
        .close = CloseFile,
        .read_offset = ReadFile,
        .size = FileSize,
    };

    AvPlayerFileStreamer streamer{replacement};
    ASSERT_TRUE(streamer.Init("game.pmf"));
    auto* context = streamer.GetContext();
    ASSERT_NE(context->seek, nullptr);
    EXPECT_EQ(context->seek(context->opaque, 7, SEEK_SET | AVSEEK_FORCE), 7);
}

} // namespace
} // namespace Libraries::AvPlayer
