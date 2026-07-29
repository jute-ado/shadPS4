// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan {

enum class ImageDescriptorKind {
    Sampled,
    Storage,
};

[[nodiscard]] constexpr ImageDescriptorKind ImageDescriptorKindForShaderAccess(
    bool shader_writes) {
    return shader_writes ? ImageDescriptorKind::Storage : ImageDescriptorKind::Sampled;
}

} // namespace Vulkan
