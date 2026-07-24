// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdint>

#include <gtest/gtest.h>

#include "core/libraries/videodec/videodec_frame_handle.h"

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/mem.h>
}

namespace {

void CountBufferRelease(void* opaque, std::uint8_t* data) {
    static_cast<std::atomic_int*>(opaque)->fetch_add(1);
    av_free(data);
}

} // namespace

TEST(VideodecFrameHandle, ReleasesFrameBuffersOnEarlyScopeExit) {
    std::atomic_int releases{0};
    auto* frame = av_frame_alloc();
    ASSERT_NE(frame, nullptr);
    auto* data = static_cast<std::uint8_t*>(av_malloc(1));
    ASSERT_NE(data, nullptr);
    frame->buf[0] = av_buffer_create(data, 1, CountBufferRelease, &releases, 0);
    ASSERT_NE(frame->buf[0], nullptr);
    frame->data[0] = data;

    {
        auto handle = Libraries::Videodec::AdoptFrame(frame);
        ASSERT_NE(handle, nullptr);
    }

    EXPECT_EQ(releases.load(), 1);
}
