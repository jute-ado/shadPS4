// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

[[nodiscard]] constexpr bool ShouldUseStreamBuffer(bool is_written, u64 size, u64 max_stream_size,
                                                   bool is_gpu_modified) {
    return !is_written && size <= max_stream_size && !is_gpu_modified;
}

} // namespace VideoCore
