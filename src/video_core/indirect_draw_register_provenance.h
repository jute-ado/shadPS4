// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>

namespace VideoCore {

// SH-relative dword offsets, in the same physical-stage order as Regs::ProgramForStage.
// Production verifies these against the actual register layout with static_asserts.
inline constexpr std::array<std::uint32_t, 6> IndirectUserDataBankOffsets{0x0c, 0x4c,  0x8c,
                                                                          0xcc, 0x10c, 0x14c};

struct IndirectUserDataLocation {
    std::uint32_t stage{};
    std::uint32_t sgpr{};
};

[[nodiscard]] constexpr std::optional<IndirectUserDataLocation> MapIndirectUserDataLocation(
    std::uint32_t location) noexcept {
    if (location == 0 || location > 0xffffU) {
        return std::nullopt;
    }
    for (std::uint32_t stage = 0; stage < IndirectUserDataBankOffsets.size(); ++stage) {
        const auto base = IndirectUserDataBankOffsets[stage];
        if (location >= base && location - base < 16) {
            return IndirectUserDataLocation{stage, location - base};
        }
    }
    return std::nullopt;
}

struct IndirectUserDataObservation {
    bool value_known{};
    std::uint32_t stage{};
    std::uint32_t sgpr{};
    std::uint32_t value{};
};

template <typename ReadRegister>
[[nodiscard]] IndirectUserDataObservation ObserveIndirectUserDataRegister(
    std::uint32_t location, bool enabled, ReadRegister&& read_register) {
    const auto mapped = enabled ? MapIndirectUserDataLocation(location) : std::nullopt;
    if (!mapped) {
        return {};
    }
    return {true, mapped->stage, mapped->sgpr,
            std::invoke(read_register, mapped->stage, mapped->sgpr)};
}

// Receipt only: these fields do not write a CP register or specialize a shader.
struct IndirectDrawRegisterLocations {
    bool present{};
    std::uint32_t base_vtx_loc{};
    std::uint32_t start_inst_loc{};
    bool draw_index_present{};
    bool draw_index_enabled{};
    std::uint32_t draw_index_loc{};
};

template <typename Packet>
[[nodiscard]] IndirectDrawRegisterLocations CaptureIndirectDrawRegisterLocations(
    const Packet& packet) noexcept {
    IndirectDrawRegisterLocations result{
        .present = true,
        .base_vtx_loc = packet.base_vtx_loc.Value(),
        .start_inst_loc = packet.start_inst_loc.Value(),
    };
    if constexpr (requires {
                      packet.draw_index_loc;
                      packet.draw_index_enable;
                  }) {
        result.draw_index_present = true;
        result.draw_index_enabled = packet.draw_index_enable.Value() != 0;
        result.draw_index_loc = packet.draw_index_loc.Value();
    }
    return result;
}

} // namespace VideoCore
