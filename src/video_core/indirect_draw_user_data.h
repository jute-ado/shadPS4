// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include "shader_recompiler/draw_user_data.h"
#include "video_core/indirect_draw_register_provenance.h"

namespace VideoCore {

// Conventional physical VS only. The renderer resolves one immutable context before
// shader selection; this helper never reads or writes registers.
[[nodiscard]] constexpr std::optional<Shader::DrawUserDataLocations> ResolveVertexDrawUserData(
    const IndirectDrawRegisterLocations& packet) noexcept {
    Shader::DrawUserDataLocations result{};
    if (!packet.present) {
        return result;
    }
    const auto decode = [](std::uint32_t offset, u8& destination) {
        const auto mapped = MapIndirectUserDataLocation(offset);
        if (!mapped || mapped->stage != 1) {
            return false;
        }
        destination = static_cast<u8>(mapped->sgpr);
        return true;
    };
    // Base/start locations have a zero sentinel; draw-index uses its explicit enable bit.
    if ((packet.base_vtx_loc != 0 && !decode(packet.base_vtx_loc, result.base_vertex)) ||
        (packet.start_inst_loc != 0 && !decode(packet.start_inst_loc, result.base_instance)) ||
        (packet.draw_index_present && packet.draw_index_enabled &&
         !decode(packet.draw_index_loc, result.draw_index))) {
        return std::nullopt;
    }
    const std::array locations{result.base_vertex, result.base_instance, result.draw_index};
    for (std::size_t first = 0; first < locations.size(); ++first) {
        for (std::size_t second = first + 1; second < locations.size(); ++second) {
            if (locations[first] != Shader::DrawUserDataLocations::Disabled &&
                locations[first] == locations[second]) {
                return std::nullopt;
            }
        }
    }
    return result;
}

} // namespace VideoCore
