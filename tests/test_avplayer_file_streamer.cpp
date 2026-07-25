// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <climits>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

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
    std::vector<u8> data;
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

s32 PS4_SYSV_ABI ReadFile(void* opaque, u8* buffer, u64 offset, u32 size) {
    auto* callbacks = static_cast<FileCallbacks*>(opaque);
    if (offset >= callbacks->data.size()) {
        return 0;
    }
    const auto read_size =
        std::min<std::size_t>(size, callbacks->data.size() - static_cast<std::size_t>(offset));
    std::memcpy(buffer, callbacks->data.data() + offset, read_size);
    return static_cast<s32>(read_size);
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

TEST(AvPlayerFileStreamer, ResetDiscardsPrefetchedInput) {
    FileCallbacks callbacks{
        .file_size = 5000,
        .data = std::vector<u8>(5000),
    };
    for (std::size_t index = 0; index < callbacks.data.size(); ++index) {
        callbacks.data[index] = static_cast<u8>(index % 251);
    }
    const AvPlayerFileReplacement replacement{
        .object_ptr = &callbacks,
        .open = OpenFile,
        .close = CloseFile,
        .read_offset = ReadFile,
        .size = FileSize,
    };

    AvPlayerFileStreamer streamer{replacement};
    ASSERT_TRUE(streamer.Init("game.pmf"));
    u8 byte{};
    ASSERT_EQ(avio_read(streamer.GetContext(), &byte, 1), 1);
    ASSERT_EQ(byte, callbacks.data[0]);

    streamer.Reset();

    std::vector<u8> replayed(callbacks.data.size());
    ASSERT_EQ(avio_read(streamer.GetContext(), replayed.data(), static_cast<int>(replayed.size())),
              static_cast<int>(replayed.size()));
    EXPECT_EQ(replayed, callbacks.data);
}

} // namespace
} // namespace Libraries::AvPlayer
