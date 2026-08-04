// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan {

constexpr bool PipelineCompileTimingRequested(const char* value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

} // namespace Vulkan
