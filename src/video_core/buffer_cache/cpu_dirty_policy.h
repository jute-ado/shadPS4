// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace VideoCore {

[[nodiscard]] constexpr bool ShouldClearCpuDirtyAfterUpload(bool has_alternate_fault_stack,
                                                            bool is_stack_mapping,
                                                            bool gpu_writable) noexcept {
    return has_alternate_fault_stack || !is_stack_mapping || gpu_writable;
}

} // namespace VideoCore
