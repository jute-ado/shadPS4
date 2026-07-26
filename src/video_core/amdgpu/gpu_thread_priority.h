// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/thread.h"

namespace AmdGpu {

[[nodiscard]] constexpr Common::ThreadPriority GpuCommandProcessorPriority() noexcept {
    return Common::ThreadPriority::VeryHigh;
}

} // namespace AmdGpu
