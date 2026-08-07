// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>

#include "common/types.h"

namespace VideoCore {

struct DiagnosticDeleteDecision {
    bool logical_delete{};
    bool erase_now{};
};

class DiagnosticReadbackPin {
public:
    void Acquire() noexcept {
        ++count;
    }

    bool Release() noexcept {
        if (count == 0) {
            return false;
        }
        --count;
        return count == 0 && delete_requested;
    }

    DiagnosticDeleteDecision RequestDelete() noexcept {
        if (delete_requested) {
            return {};
        }
        delete_requested = true;
        return {.logical_delete = true, .erase_now = count == 0};
    }

    [[nodiscard]] bool IsPinned() const noexcept {
        return count != 0;
    }

    [[nodiscard]] bool IsDeletePending() const noexcept {
        return delete_requested && count != 0;
    }

private:
    u32 count{};
    bool delete_requested{};
};

class DiagnosticWriteGeneration {
public:
    void MarkWrite() noexcept {
        serial += serial != std::numeric_limits<u64>::max();
    }

    [[nodiscard]] u64 Serial() const noexcept {
        return serial;
    }

private:
    u64 serial{};
};

} // namespace VideoCore
