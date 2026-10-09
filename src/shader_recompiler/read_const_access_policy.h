// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Shader {

enum class ReadConstAccess {
    FlatBuffer,
    DmaWithFlatBufferFallback,
    DmaOnly,
};

constexpr ReadConstAccess SelectReadConstAccess(bool has_dynamic_offset,
                                                bool direct_memory_access_enabled) {
    if (has_dynamic_offset) {
        return ReadConstAccess::DmaOnly;
    }
    return direct_memory_access_enabled ? ReadConstAccess::DmaWithFlatBufferFallback
                                        : ReadConstAccess::FlatBuffer;
}

constexpr bool RequiresReadConstDma(bool has_dynamic_offset, bool direct_memory_access_enabled) {
    return has_dynamic_offset || direct_memory_access_enabled;
}

} // namespace Shader
