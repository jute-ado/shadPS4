// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "video_core/amdgpu/pm4_cmds.h"

namespace AmdGpu {
namespace {

TEST(Pm4CondExec, DecodesTheFourDwordPayload) {
    alignas(u32) volatile u32 condition = 1;
    const auto condition_address = reinterpret_cast<std::uintptr_t>(&condition);

    std::array<u32, 5> words{};
    words[1] = static_cast<u32>(condition_address) & 0xfffffffcu;
    words[2] = static_cast<u32>(condition_address >> 32u);
    words[3] = 0;
    words[4] = 0x123u;

    const auto& packet = *reinterpret_cast<const PM4CmdCondExec*>(words.data());

    EXPECT_EQ(sizeof(PM4CmdCondExec), sizeof(words));
    EXPECT_EQ(packet.exec_count.Value(), 0x123u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(packet.Address()), condition_address);
}

TEST(Pm4CondExec, TreatsAnyNonzeroConditionDwordAsTrue) {
    alignas(u32) volatile u32 condition = 0x100u;
    const auto condition_address = reinterpret_cast<std::uintptr_t>(&condition);

    std::array<u32, 5> words{};
    words[1] = static_cast<u32>(condition_address) & 0xfffffffcu;
    words[2] = static_cast<u32>(condition_address >> 32u);

    const auto& packet = *reinterpret_cast<const PM4CmdCondExec*>(words.data());

    EXPECT_FALSE(packet.ShouldSkip());
    condition = 0;
    EXPECT_TRUE(packet.ShouldSkip());
}

} // namespace
} // namespace AmdGpu
