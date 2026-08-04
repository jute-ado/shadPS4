// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdint>

#include <gtest/gtest.h>

#include "core/libraries/videodec/videodec_packet_handle.h"

extern "C" {
#include <libavutil/mem.h>
}

namespace {

void CountBufferRelease(void* opaque, std::uint8_t* data) {
    static_cast<std::atomic_int*>(opaque)->fetch_add(1);
    av_free(data);
}

} // namespace

TEST(VideodecPacketHandle, ReleasesPacketOnEarlyScopeExit) {
    std::atomic_int releases{0};
    auto* packet = av_packet_alloc();
    ASSERT_NE(packet, nullptr);
    auto* data = static_cast<std::uint8_t*>(av_malloc(1));
    ASSERT_NE(data, nullptr);
    packet->buf = av_buffer_create(data, 1, CountBufferRelease, &releases, 0);
    ASSERT_NE(packet->buf, nullptr);
    packet->data = data;
    packet->size = 1;

    {
        auto handle = Libraries::Videodec::AdoptPacket(packet);
        ASSERT_NE(handle, nullptr);
    }

    EXPECT_EQ(releases.load(), 1);
}
