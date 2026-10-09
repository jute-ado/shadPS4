// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <optional>

#include "common/types.h"

namespace Vulkan::UtilityBuffer {

struct Reservation {
    u64 total_size{};
    u64 base_alignment{1};
};

/// Adds two byte counts without allowing a u64 wraparound.
[[nodiscard]] constexpr std::optional<u64> CheckedAdd(const u64 lhs, const u64 rhs) noexcept {
    if (rhs > std::numeric_limits<u64>::max() - lhs) {
        return std::nullopt;
    }
    return lhs + rhs;
}

/// Rounds a byte offset up without allowing either the padding or the result to wrap.
[[nodiscard]] constexpr std::optional<u64> CheckedAlignUp(const u64 value,
                                                          const u64 alignment) noexcept {
    if (alignment == 0) {
        return std::nullopt;
    }
    const u64 remainder = value % alignment;
    const u64 padding = remainder == 0 ? 0 : alignment - remainder;
    return CheckedAdd(value, padding);
}

/// Multiplies two guest-controlled counts without allowing a u64 wraparound.
[[nodiscard]] constexpr std::optional<u64> CheckedMultiply(const u64 lhs,
                                                           const u64 rhs) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<u64>::max() / lhs) {
        return std::nullopt;
    }
    return lhs * rhs;
}

/// Computes the backing bytes needed by a shared-memory descriptor for one dispatch.
///
/// ComputeProgram::NumWorkgroups() uses u32 arithmetic, so callers must multiply the dispatch
/// dimensions here before narrowing them to the register helper's historical return type.
[[nodiscard]] constexpr std::optional<u64> CheckedSharedMemorySize(
    const u64 shared_memory_size, const u64 dim_x, const u64 dim_y, const u64 dim_z) noexcept {
    const auto xy = CheckedMultiply(dim_x, dim_y);
    if (!xy) {
        return std::nullopt;
    }
    const auto workgroups = CheckedMultiply(*xy, dim_z);
    if (!workgroups) {
        return std::nullopt;
    }
    return CheckedMultiply(shared_memory_size, *workgroups);
}

/// Plans one contiguous stream reservation for utility descriptors.
///
/// Payloads must expose an unsigned `alignment` and a `.bytes.size()` count. The stream capacity
/// check happens before mapping so overflow and oversized commands cannot publish a partial set of
/// descriptors or consume a ring-buffer reservation.
template <typename PayloadRange>
[[nodiscard]] std::optional<Reservation> PlanReservation(const PayloadRange& payloads,
                                                          const u64 stream_capacity) {
    u64 total_size{};
    u64 base_alignment{1};
    for (const auto& payload : payloads) {
        const u64 alignment = static_cast<u64>(payload.alignment);
        // StreamBuffer::Map aligns its current ring cursor as well as this aggregate. An
        // alignment larger than the allocation cannot be guaranteed safe for a non-zero cursor.
        if (alignment == 0 || alignment > stream_capacity) {
            return std::nullopt;
        }
        const auto aligned = CheckedAlignUp(total_size, alignment);
        if (!aligned) {
            return std::nullopt;
        }

        const auto byte_count = payload.bytes.size();
        if constexpr (sizeof(byte_count) > sizeof(u64)) {
            if (byte_count > std::numeric_limits<u64>::max()) {
                return std::nullopt;
            }
        }
        const auto next_total = CheckedAdd(*aligned, static_cast<u64>(byte_count));
        if (!next_total) {
            return std::nullopt;
        }
        total_size = *next_total;
        base_alignment = std::max(base_alignment, alignment);
    }

    if (total_size > stream_capacity) {
        return std::nullopt;
    }
    return Reservation{total_size, base_alignment};
}

} // namespace Vulkan::UtilityBuffer
