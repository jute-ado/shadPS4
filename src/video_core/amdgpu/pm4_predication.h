// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "common/types.h"

namespace AmdGpu {

struct DecodedPredication {
    u64 address{};
    u32 condition{};
    bool wait_for_results{};
    u32 operation{};
};

[[nodiscard]] constexpr std::optional<DecodedPredication> DecodeSetPredication(
    std::span<const u32> payload) {
    if (payload.size() < 2) {
        return std::nullopt;
    }

    constexpr u32 FlagsMask = 0x00071100u;
    u32 flags;
    u64 address;
    if (payload.size() >= 3 && (payload[0] & ~FlagsMask) == 0 && payload[2] <= 0xffffu) {
        flags = payload[0];
        address = static_cast<u64>(payload[2]) << 32 | (payload[1] & 0xfffffff0u);
    } else {
        flags = payload[1];
        address = static_cast<u64>(payload[1] & 0xffu) << 32 | (payload[0] & 0xfffffff0u);
    }

    return DecodedPredication{
        .address = address,
        .condition = (flags >> 8) & 1u,
        .wait_for_results = ((flags >> 12) & 1u) != 0,
        .operation = (flags >> 16) & 7u,
    };
}

[[nodiscard]] constexpr std::optional<bool> EvaluatePredication(const DecodedPredication& packet,
                                                                u64 value) {
    if (packet.operation == 0) {
        return false;
    }
    if (packet.operation != 3 || packet.condition > 1) {
        return std::nullopt;
    }
    return packet.condition == 0 ? value != 0 : value == 0;
}

[[nodiscard]] constexpr bool ShouldSkipPm4Packet(bool predicate_enabled, bool skip_state) {
    return predicate_enabled && skip_state;
}

} // namespace AmdGpu
