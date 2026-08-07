// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

struct DiagnosticDeleteDecision {
    bool logical_delete{};
    bool erase_now{};
};

class DiagnosticReadbackPin {
public:
    void Acquire() {
        ++count;
    }

    bool Release() {
        if (count == 0) {
            return false;
        }
        --count;
        return count == 0 && delete_requested;
    }

    DiagnosticDeleteDecision RequestDelete() {
        if (delete_requested) {
            return {};
        }
        delete_requested = true;
        return {.logical_delete = true, .erase_now = count == 0};
    }

    bool IsPinned() const {
        return count != 0;
    }

    bool IsDeletePending() const {
        return delete_requested && count != 0;
    }

private:
    u32 count{};
    bool delete_requested{};
};

} // namespace VideoCore
