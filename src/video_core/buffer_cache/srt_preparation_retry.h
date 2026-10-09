// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/buffer_cache/srt_memory_sync.h"

namespace VideoCore {

// Limit contention work; exhausting the budget retains the original failed-preparation result.
// This is recovery from an invalidated private capture, not a licence to accept mixed epochs.
inline constexpr u32 MaxSrtPreparationAttempts = 3;

template <typename Binding, typename Attempt, typename IsReady>
auto RetrySrtPreparation(Binding& binding, Attempt&& attempt, IsReady&& is_ready) {
    for (u32 index = 0;; ++index) {
        auto attempted_binding = binding;
        auto failure = SrtMemoryReadFailure::None;
        bool translation_rejected = false;
        // The attempt owns its reader/epoch and staged specialization. They must be destroyed
        // before returning here, so the next attempt neither nests epochs nor reuses bytes.
        auto result = attempt(attempted_binding, failure, translation_rejected);
        if (is_ready(result)) {
            binding = attempted_binding;
            return result;
        }
        if (failure != SrtMemoryReadFailure::EpochInvalidated || translation_rejected ||
            index + 1 == MaxSrtPreparationAttempts) {
            return result;
        }
    }
}

} // namespace VideoCore
