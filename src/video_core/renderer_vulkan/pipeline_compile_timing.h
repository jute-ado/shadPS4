// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Vulkan {

constexpr bool PipelineCompileTimingRequested(const char* value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

constexpr bool PipelineCompileTimingShouldReport(const std::int64_t duration_nanoseconds) {
    constexpr std::int64_t SlowCompileNanoseconds = 5'000'000;
    return duration_nanoseconds >= SlowCompileNanoseconds;
}

} // namespace Vulkan
