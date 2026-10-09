// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <charconv>
#include <cstdlib>
#include <string_view>
#include <utility>

#include "common/types.h"

namespace VideoCore {

// An exact, default-off diagnostic selector with a budget independent from broad SRT failure
// logging. This prevents unrelated early failures from starving one requested shader.
class SrtFailureFlatDiagnostics {
public:
    static constexpr unsigned MaxRows = 8;

    SrtFailureFlatDiagnostics() = default;

    explicit SrtFailureFlatDiagnostics(const char* configured)
        : configured_{configured != nullptr} {
        if (!configured) {
            return;
        }
        auto text = std::string_view{configured};
        if (text.starts_with("0x") || text.starts_with("0X")) {
            text.remove_prefix(2);
        }
        valid_ = false;
        if (text.empty() || text.size() > 16) {
            return;
        }
        const auto [end, error] =
            std::from_chars(text.data(), text.data() + text.size(), shader_hash_, 16);
        valid_ = error == std::errc{} && end == text.data() + text.size();
    }

    [[nodiscard]] bool Configured() const noexcept {
        return configured_;
    }

    [[nodiscard]] bool Valid() const noexcept {
        return valid_;
    }

    [[nodiscard]] bool Enabled() const noexcept {
        return configured_ && valid_;
    }

    [[nodiscard]] bool Matches(u64 shader_hash) const noexcept {
        return Enabled() && shader_hash == shader_hash_;
    }

    // Emits a successful source/composition diagnostic row. Success rows have their own
    // bounded budget so they cannot hide a later failure row for the selected shader.
    template <typename Record>
    void Emit(u64 shader_hash, Record&& record) noexcept {
        EmitWithBudget(count_, shader_hash, std::forward<Record>(record));
    }

    // Emits a failure snapshot row using a budget independent from successful source rows.
    // This is intentionally exact-hash and bounded just like Emit().
    template <typename Record>
    void EmitFailure(u64 shader_hash, Record&& record) noexcept {
        EmitWithBudget(failure_count_, shader_hash, std::forward<Record>(record));
    }

private:
    template <typename Record>
    void EmitWithBudget(std::atomic_uint& budget, u64 shader_hash, Record&& record) noexcept {
        if (!Matches(shader_hash)) {
            return;
        }
        auto row = budget.load(std::memory_order_relaxed);
        while (row < MaxRows) {
            if (budget.compare_exchange_weak(row, row + 1, std::memory_order_relaxed)) {
                try {
                    std::forward<Record>(record)(row);
                } catch (...) {
                    // A failed diagnostic sink consumes its slot, never a shader attempt.
                }
                return;
            }
        }
    }

    u64 shader_hash_{};
    bool configured_{};
    bool valid_{true};
    std::atomic_uint count_{};
    std::atomic_uint failure_count_{};
};

inline SrtFailureFlatDiagnostics& GetSrtFailureFlatDiagnostics() {
    static SrtFailureFlatDiagnostics diagnostics{
        std::getenv("SHADPS4_SRT_FAILURE_FLAT_SHADER_HASH")};
    return diagnostics;
}

} // namespace VideoCore
