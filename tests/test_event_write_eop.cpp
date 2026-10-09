// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/types.h"
#include "video_core/amdgpu/eop_completion.h"

namespace {

struct TestEopPacket {
    u32 data;

    void SignalFence(auto&& write_memory, auto&& signal_interrupt) const {
        write_memory(data);
        signal_interrupt();
    }
};

} // namespace

TEST(EventWriteEop, SubmitsGpuWorkBeforeFenceWriteAndInterrupt) {
    u32 fence = 0;
    bool interrupt_signalled = false;
    std::vector<std::string> operations;

    AmdGpu::SubmitEop(
        TestEopPacket{.data = 0x12345678}, [&] { operations.emplace_back("submit"); },
        [&](u32 data) {
            operations.emplace_back("fence");
            fence = data;
        },
        [&] {
            operations.emplace_back("interrupt");
            interrupt_signalled = true;
        });

    EXPECT_EQ(fence, 0x12345678u);
    EXPECT_TRUE(interrupt_signalled);
    EXPECT_EQ(operations, (std::vector<std::string>{"submit", "fence", "interrupt"}));
}
