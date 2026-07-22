// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::Kernel {

[[nodiscard]] constexpr bool IsValidSemaphoreWaitCount(s32 need_count, s32 max_count) {
    return need_count > 0 && need_count <= max_count;
}

[[nodiscard]] constexpr bool IsValidSemaphoreSignalCount(s32 signal_count, s32 token_count,
                                                         s32 max_count) {
    return signal_count > 0 && token_count <= max_count && signal_count <= max_count - token_count;
}

[[nodiscard]] constexpr bool IsValidSemaphoreCancelCount(s32 set_count, s32 max_count) {
    return set_count <= max_count;
}

[[nodiscard]] constexpr u32 RemainingSemaphoreTimeout(u32 original_timeout, u64 elapsed) {
    return elapsed >= original_timeout ? 0 : original_timeout - static_cast<u32>(elapsed);
}

} // namespace Libraries::Kernel
