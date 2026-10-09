// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <stdexcept>

#include "common/assert.h"

namespace Common::Log {

bool g_should_append = false;
std::unordered_map<std::string_view, std::shared_ptr<spdlog::logger>> ALL_LOGGERS;

} // namespace Common::Log

void assert_fail_impl() {
    throw std::runtime_error{"Assertion failed in unit test"};
}

[[noreturn]] void unreachable_impl() {
    throw std::runtime_error{"Unreachable code in unit test"};
}
