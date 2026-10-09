// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Shader::Backend::SPIRV {

constexpr bool NeedsStorageImageFormat(bool is_written, bool is_atomic) {
    return is_written || is_atomic;
}

} // namespace Shader::Backend::SPIRV
