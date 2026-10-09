// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <optional>
#include <string_view>

#include "common/types.h"
#include "video_core/buffer_cache/srt_failure_flat_snapshot.h"
#include "video_core/buffer_cache/srt_memory_sync.h"

namespace VideoCore {

// The generated walker currently materializes 48-bit pointer values. Keep the exact raw pair in
// diagnostics, but derive the attempted address with the same mask used by the walker. This is a
// best-effort match against the retained flat prefix; it is evidence for debugging, not proof of
// which guest-side operation produced the pair.
inline constexpr u64 SrtFlatPointerAddressMask = 0x0000FFFFFFFFFFFFULL;

struct SrtFailureFlatProvenance {
    u64 shader_hash{};
    std::string_view phase{};
    u32 low_flat_index{};
    u32 high_flat_index{};
    u32 raw_low{};
    u32 raw_high{};
    u64 raw_address{};
    u64 masked_address{};
    u64 pointer_size{};
    u32 invalid_reasons{};
    SrtMemoryReadFailure pointer_failure{SrtMemoryReadFailure::None};
};

// Finds the retained flat dword pair that produced the first failed reader address. The helper
// intentionally does not read guest memory, alter the flat buffer, or change reader admission.
inline std::optional<SrtFailureFlatProvenance> CaptureSrtFailureFlatProvenance(
    u64 shader_hash, std::string_view phase, const SrtFailureFlatSnapshot& snapshot,
    const SrtReadFailureRequest& request, SrtMemoryReadFailure failure) noexcept {
    if (!request.known || request.output_null || request.size == 0 ||
        request.address > SrtFlatPointerAddressMask) {
        return std::nullopt;
    }

    const u32 available_dwords = std::min<u32>(
        snapshot.copied_dwords, static_cast<u32>(SrtFailureFlatSnapshot::PrefixDwords));
    if (available_dwords < 2) {
        return std::nullopt;
    }

    for (u32 low_index = 0; low_index + 1 < available_dwords; ++low_index) {
        const u32 raw_low = snapshot.words[low_index];
        const u32 raw_high = snapshot.words[low_index + 1];
        const u64 raw_address = (static_cast<u64>(raw_high) << 32) | raw_low;
        const u64 masked_address = raw_address & SrtFlatPointerAddressMask;
        if (masked_address != request.address) {
            continue;
        }
        return SrtFailureFlatProvenance{
            .shader_hash = shader_hash,
            .phase = phase,
            .low_flat_index = low_index,
            .high_flat_index = low_index + 1,
            .raw_low = raw_low,
            .raw_high = raw_high,
            .raw_address = raw_address,
            .masked_address = masked_address,
            .pointer_size = request.size,
            .invalid_reasons = request.invalid_reasons,
            .pointer_failure = failure,
        };
    }
    return std::nullopt;
}

} // namespace VideoCore
