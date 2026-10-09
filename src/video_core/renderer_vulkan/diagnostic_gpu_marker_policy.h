// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan {

constexpr bool ShouldEnableDiagnosticGpuMarkers(bool extension_supported,
                                                bool diagnostic_opt_in) {
    return extension_supported && diagnostic_opt_in;
}

} // namespace Vulkan
