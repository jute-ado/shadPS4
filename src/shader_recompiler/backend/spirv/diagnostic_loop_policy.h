// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <charconv>
#include <limits>
#include <string_view>

#include "common/types.h"

namespace Shader::Backend::SPIRV {

struct DiagnosticLoopBudget {
    u64 shader_hash{};
    u32 loop_index{};
    u32 iteration_limit{};

    constexpr bool operator==(const DiagnosticLoopBudget&) const noexcept = default;
};

namespace Detail {

template <typename T>
[[nodiscard]] inline bool ParseUnsigned(const char* text, T& result) noexcept {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    std::string_view value{text};
    int base = 10;
    if (value.starts_with("0x") || value.starts_with("0X")) {
        value.remove_prefix(2);
        base = 16;
    }
    if (value.empty()) {
        return false;
    }
    T parsed{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed, base);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return false;
    }
    result = parsed;
    return true;
}

} // namespace Detail

[[nodiscard]] inline DiagnosticLoopBudget ParseDiagnosticLoopBudget(
    const char* shader_hash, const char* loop_index, const char* iteration_limit) noexcept {
    DiagnosticLoopBudget budget{};
    if (!Detail::ParseUnsigned(shader_hash, budget.shader_hash) || budget.shader_hash == 0 ||
        !Detail::ParseUnsigned(loop_index, budget.loop_index) ||
        !Detail::ParseUnsigned(iteration_limit, budget.iteration_limit) ||
        budget.iteration_limit == 0) {
        return {};
    }
    return budget;
}

[[nodiscard]] constexpr u32 DiagnosticLoopLimitForShader(
    const DiagnosticLoopBudget& budget, const u64 shader_hash, const u32 loop_index) noexcept {
    if (budget.shader_hash == 0 || budget.iteration_limit == 0 ||
        budget.shader_hash != shader_hash || budget.loop_index != loop_index) {
        return 0;
    }
    return budget.iteration_limit;
}

} // namespace Shader::Backend::SPIRV
