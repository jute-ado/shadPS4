// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <system_error>

#include "core/libraries/kernel/posix_error.h"

namespace Libraries::Kernel {

constexpr int ClassifyMkdirResult(bool created, const std::error_code& error,
                                  bool exists_after_create) {
    if (!created && exists_after_create &&
        (!error || error == std::errc::file_exists)) {
        return POSIX_EEXIST;
    }
    if (error) {
        return POSIX_EIO;
    }
    if (!created) {
        return POSIX_EIO;
    }
    return exists_after_create ? 0 : POSIX_ENOENT;
}

} // namespace Libraries::Kernel
