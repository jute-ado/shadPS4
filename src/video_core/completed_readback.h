// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

namespace VideoCore {

enum class CompletedReadbackResult {
    Consumed,
    InvalidationFailed,
};

template <typename Invalidate, typename Consume>
[[nodiscard]] CompletedReadbackResult ConsumeCompletedReadback(const bool is_coherent,
                                                               Invalidate&& invalidate,
                                                               Consume&& consume) {
    if (!is_coherent && !std::invoke(std::forward<Invalidate>(invalidate))) {
        return CompletedReadbackResult::InvalidationFailed;
    }

    std::invoke(std::forward<Consume>(consume));
    return CompletedReadbackResult::Consumed;
}

} // namespace VideoCore
