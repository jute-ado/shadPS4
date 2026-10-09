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
    if (!direct_memory_access_enabled) {
        return ReadConstAccess::FlatBuffer;
    }
    if (has_dynamic_offset) {
        return ReadConstAccess::DmaOnly;
    }
    return ReadConstAccess::DmaWithFlatBufferFallback;
}

constexpr bool RequiresReadConstDma([[maybe_unused]] bool has_dynamic_offset,
                                    bool direct_memory_access_enabled) {
    return direct_memory_access_enabled;
}

constexpr bool RetainsReadConstDescriptorMetadata(bool direct_memory_access_enabled) {
    return direct_memory_access_enabled;
}

} // namespace Shader
