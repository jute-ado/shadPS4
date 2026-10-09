// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>

#include "common/types.h"

namespace Vulkan {

// Guest GPU addresses are 40 bits wide on the PS4.  BufferCache::ObtainBuffer also takes a
// u32 byte count, so a range admitted here can be passed to it without narrowing or wrapping.
inline constexpr u64 MaxGpuAddress = 1ull << 40;
inline constexpr u64 MaxGpuBufferSize = std::numeric_limits<u32>::max();

enum class GpuSpanAdmissionReason {
    Accepted,
    AddressOverflow,
    SizeOverflow,
    Unmapped,
};

struct GpuSpan {
    VAddr address{};
    u32 size{};

    bool operator==(const GpuSpan&) const = default;
};

[[nodiscard]] constexpr std::optional<u64> CheckedGpuProduct(u64 lhs, u64 rhs) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<u64>::max() / lhs) {
        return std::nullopt;
    }
    return lhs * rhs;
}

// Validate and narrow a complete guest range exactly once.  Callers must perform any mapping
// probe with the returned address and size, never with base + offset computed beforehand.
[[nodiscard]] constexpr std::optional<GpuSpan> TryMakeGpuSpan(VAddr base, u64 offset,
                                                               u64 size) noexcept {
    const u64 base_address = static_cast<u64>(base);
    if (base_address >= MaxGpuAddress || offset > MaxGpuAddress - base_address) {
        return std::nullopt;
    }

    const u64 address = base_address + offset;
    // Match MemoryManager::IsValidGpuMapping: the address-space limit is an exclusive
    // endpoint, and a zero-length range at that endpoint is not a usable GPU address.
    if (address >= MaxGpuAddress || size >= MaxGpuAddress - address ||
        size > MaxGpuBufferSize) {
        return std::nullopt;
    }
    return GpuSpan{static_cast<VAddr>(address), static_cast<u32>(size)};
}

[[nodiscard]] constexpr GpuSpanAdmissionReason CheckGpuSpanAdmission(VAddr base, u64 offset,
                                                                      u64 size,
                                                                      bool mapped) noexcept {
    const u64 base_address = static_cast<u64>(base);
    if (base_address >= MaxGpuAddress || offset > MaxGpuAddress - base_address) {
        return GpuSpanAdmissionReason::AddressOverflow;
    }
    const u64 address = base_address + offset;
    if (address >= MaxGpuAddress || size >= MaxGpuAddress - address ||
        size > MaxGpuBufferSize) {
        return GpuSpanAdmissionReason::SizeOverflow;
    }
    return mapped ? GpuSpanAdmissionReason::Accepted : GpuSpanAdmissionReason::Unmapped;
}

[[nodiscard]] constexpr bool IsGpuSpanAdmissible(VAddr base, u64 offset, u64 size,
                                                 bool mapped) noexcept {
    return CheckGpuSpanAdmission(base, offset, size, mapped) ==
           GpuSpanAdmissionReason::Accepted;
}

} // namespace Vulkan
