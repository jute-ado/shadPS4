// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>

#include "common/logging/log.h"

namespace Common {

std::string GetCurrentThreadName() {
    return "shadPS4::PpSampleTeeTest";
}

namespace Log {

std::unordered_map<std::string_view, std::shared_ptr<spdlog::logger>> ALL_LOGGERS{};

} // namespace Log
} // namespace Common

void assert_fail_impl() {
    std::abort();
}

[[noreturn]] void unreachable_impl() {
    std::abort();
}
