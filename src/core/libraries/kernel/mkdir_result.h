// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <system_error>

#include "core/libraries/kernel/posix_error.h"

namespace Libraries::Kernel {

constexpr int ClassifyMkdirResult(bool created, const std::error_code& error) {
    if (created) {
        return 0;
    }
    if (!error || error == std::errc::file_exists) {
        return POSIX_EEXIST;
    }
    return POSIX_EIO;
}

} // namespace Libraries::Kernel
