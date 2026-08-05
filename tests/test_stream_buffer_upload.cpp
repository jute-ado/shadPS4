// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <utility>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/stream_buffer_upload.h"

namespace {

class FakeStreamBuffer {
public:
    FakeStreamBuffer() {
        storage.fill(0xff);
    }

    std::pair<u8*, u64> Map(u64 size, u64 alignment) {
        const u64 aligned_offset =
            alignment == 0 ? offset : (offset + alignment - 1) / alignment * alignment;
        mapped_offset = aligned_offset;
        mapped_size = size;
        return {storage.data() + aligned_offset, aligned_offset};
    }

    void Commit() {
        offset = mapped_offset + mapped_size;
        ++commit_count;
    }

    std::array<u8, 64> storage;
    u64 commit_count{};

private:
    u64 offset{};
    u64 mapped_offset{};
    u64 mapped_size{};
};

} // namespace

TEST(StreamBufferUpload, ZeroedAllocationsAreCommittedBeforeReuse) {
    FakeStreamBuffer buffer;

    const u64 first_offset = VideoCore::AllocateZeroedStreamRegion(buffer, 8, 16);
    const u64 second_offset = VideoCore::AllocateZeroedStreamRegion(buffer, 8, 16);

    EXPECT_EQ(first_offset, 0);
    EXPECT_EQ(second_offset, 16);
    EXPECT_EQ(buffer.commit_count, 2);
    for (u64 i = 0; i < 8; ++i) {
        EXPECT_EQ(buffer.storage[first_offset + i], 0);
        EXPECT_EQ(buffer.storage[second_offset + i], 0);
    }
}
