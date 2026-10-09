// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "common/types.h"

namespace VideoCore {

// Diagnostic metadata only: these observations must never choose CPU/GPU authority. A zero
// changed bit cannot distinguish an untouched word from an unobserved same-value CPU store.
struct CpuWriteFocusDiagnostic {
    static constexpr u32 MaximumWords = 8;

    bool available{};
    VAddr word_address{};
    u32 word_count{};
    u32 snapshot_mask{};
    u32 forced_word_mask{};
    u32 changed_word_mask{};
    // Any forced-mapping bytes intersecting each word; not proof that the whole word is forced.
    u32 mapping_overlap_mask{};
    std::array<u32, MaximumWords> before{};
    std::array<u32, MaximumWords> captured{};
};

} // namespace VideoCore
