// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Shader::Optimization {

[[nodiscard]] constexpr bool ShouldApplyBufferThreadId(bool add_thread_id,
                                                       bool is_compute_stage) {
    return add_thread_id && is_compute_stage;
}

} // namespace Shader::Optimization
