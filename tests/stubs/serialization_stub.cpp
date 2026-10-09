// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>

#include "common/logging/log.h"
#include "shader_recompiler/ir/passes/srt.h"

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

namespace Shader {
namespace {
void PS4_SYSV_ABI TestWalker(const u32*, u32*) {}

size_t walker_registration_count{};
size_t walker_registered_bytes{};
} // namespace

PFN_SrtWalker RegisterWalkerCode(const u8*, size_t size) {
    ++walker_registration_count;
    walker_registered_bytes += size;
    return size == 0 ? nullptr : &TestWalker;
}

void ResetWalkerRegistrationStatsForTesting() {
    walker_registration_count = 0;
    walker_registered_bytes = 0;
}

size_t WalkerRegistrationCountForTesting() {
    return walker_registration_count;
}

size_t WalkerRegisteredBytesForTesting() {
    return walker_registered_bytes;
}
} // namespace Shader
