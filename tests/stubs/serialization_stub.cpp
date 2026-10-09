// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>

#include "common/logging/log.h"

namespace Common {
std::string GetCurrentThreadName() {
    return "shadPS4::SerializationTest";
}
} // namespace Common

namespace Common::Log {
std::unordered_map<std::string_view, std::shared_ptr<spdlog::logger>> ALL_LOGGERS{};
} // namespace Common::Log

void assert_fail_impl() {
    std::abort();
}

void assert_fail_debug_msg(const char*) {
    std::abort();
}

[[noreturn]] void unreachable_impl() {
    std::abort();
}
