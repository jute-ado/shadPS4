// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <string_view>
#include <utility>

namespace VideoCore {

[[nodiscard]] constexpr bool ParseSrtReadFailureDiagnostics(std::string_view value) noexcept {
    return value == "1";
}

class SrtReadFailureDiagnostics {
public:
    static constexpr unsigned MaxRows = 32;
    explicit SrtReadFailureDiagnostics(bool enabled_) : enabled{enabled_} {}

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    template <typename Record>
    void Emit(Record&& record) noexcept {
        if (!enabled) {
            return;
        }
        auto row = count.load(std::memory_order_relaxed);
        while (row < MaxRows) {
            if (count.compare_exchange_weak(row, row + 1, std::memory_order_relaxed)) {
                try {
                    std::forward<Record>(record)(row);
                } catch (...) {
                    // A failed diagnostic sink consumes its slot, never a shader attempt.
                }
                return;
            }
        }
    }

private:
    const bool enabled;
    std::atomic_uint count{};
};

inline SrtReadFailureDiagnostics& GetSrtReadFailureDiagnostics() {
    static SrtReadFailureDiagnostics diagnostics{[] {
        const char* value = std::getenv("SHADPS4_SRT_READ_FAILURE_DIAGNOSTICS");
        return value && ParseSrtReadFailureDiagnostics(value);
    }()};
    return diagnostics;
}

} // namespace VideoCore
