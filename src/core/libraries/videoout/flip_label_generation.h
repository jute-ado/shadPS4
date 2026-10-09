// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::VideoOut {

// A display request may be reserved after an older request for the same buffer. Keep reservation,
// completion, and presentation identities separate so the older completed presentation can
// release its label without treating an uncompleted newer reservation as displayed.
class FlipLabelGeneration {
public:
    [[nodiscard]] constexpr u64 Reserve() {
        return ++latest_reservation;
    }

    constexpr void Complete(u64 generation) {
        if (generation > latest_completion) {
            latest_completion = generation;
        }
    }

    constexpr void Display(u64 generation) {
        if (generation > displayed_reservation) {
            displayed_reservation = generation;
        }
    }

    [[nodiscard]] constexpr bool CanReleaseDisplayed() const {
        return displayed_reservation == latest_completion;
    }

private:
    u64 latest_reservation{};
    u64 latest_completion{};
    u64 displayed_reservation{};
};

} // namespace Libraries::VideoOut
