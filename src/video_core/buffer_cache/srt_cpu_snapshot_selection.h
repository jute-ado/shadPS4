// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>
#include <utility>

#include "common/types.h"

namespace VideoCore {

[[nodiscard]] constexpr bool ParseSrtCpuSnapshotSelection(std::string_view value) noexcept {
    return value == "1";
}

class SrtCpuSnapshotSelection {
public:
    explicit SrtCpuSnapshotSelection(bool enabled_ = false) : enabled{enabled_} {}

    template <typename Log>
    [[nodiscard]] bool Complete(bool normal_completed, std::span<const u8> cpu_capture,
                                size_t capture_offset, std::span<u8> output, Log&& log) {
        if (!normal_completed || !enabled) {
            return normal_completed;
        }
        if (capture_offset > cpu_capture.size() ||
            output.size() > cpu_capture.size() - capture_offset) {
            return false;
        }
        if (output.empty()) {
            return true;
        }
        // Diagnostic data selection only: the caller has already completed normal GPU readback
        // and CPU-write reconciliation. Reuse that same owned capture, never reread guest memory.
        std::memmove(output.data(), cpu_capture.data() + capture_offset, output.size());
        if (!reported.exchange(true, std::memory_order_relaxed)) {
            try {
                std::forward<Log>(log)();
            } catch (...) {
                // A failed diagnostic sink cannot change the completed data selection.
            }
        }
        return true;
    }

private:
    const bool enabled;
    std::atomic_bool reported{};
};

inline SrtCpuSnapshotSelection& GetSrtCpuSnapshotSelection() {
    static SrtCpuSnapshotSelection diagnostic{[] {
        const char* value = std::getenv("SHADPS4_SRT_CPU_SNAPSHOT_SELECTION_DIAGNOSTIC");
        return value && ParseSrtCpuSnapshotSelection(value);
    }()};
    return diagnostic;
}

} // namespace VideoCore
