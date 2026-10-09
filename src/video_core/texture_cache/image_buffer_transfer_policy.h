// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>

#include "common/types.h"

namespace VideoCore {

/// Computes the byte range described by a tightly packed buffer-image transfer without allowing
/// guest-controlled dimensions to wrap. `row_length` and `image_height` use the units required by
/// the Vulkan copy command (texels, except where the caller has already converted a block layout),
/// while `bits_per_element` is the corresponding element/block width.
///
/// This helper deliberately returns a u64: callers can compare the result with their backing
/// buffer/range before reserving or recording a copy. A null result means a zero field, a malformed
/// bit width, or an overflowing product and must be treated as an admission failure.
[[nodiscard]] constexpr std::optional<u64> CheckedImageBufferTransferSize(
    const u64 row_length, const u64 image_height, const u64 depth, const u64 layers,
    const u64 bits_per_element) noexcept {
    if (row_length == 0 || image_height == 0 || depth == 0 || layers == 0 ||
        bits_per_element == 0 || bits_per_element % 8 != 0) {
        return std::nullopt;
    }

    u64 bytes = bits_per_element / 8;
    for (const u64 factor : {row_length, image_height, depth, layers}) {
        if (factor > std::numeric_limits<u64>::max() / bytes) {
            return std::nullopt;
        }
        bytes *= factor;
    }
    return bytes;
}

/// Whether one buffer-image copy region remains inside the host buffer allocation.  Keep the
/// subtraction form so an offset near UINT64_MAX cannot wrap when the transfer size is added.
[[nodiscard]] constexpr bool IsImageBufferTransferRangeWithinBuffer(
    const u64 buffer_offset, const u64 transfer_size, const u64 buffer_size) noexcept {
    return buffer_offset <= buffer_size && transfer_size <= buffer_size - buffer_offset;
}

/// Vulkan buffer-image copy commands only accept single-sampled images.
///
/// Resolving an MSAA image is a separate operation with format, sample-count,
/// and (for depth images) aspect-specific rules.  Callers must not silently
/// turn a buffer copy into a resolve, so the generic buffer-transfer policy is
/// deliberately fail-closed.
[[nodiscard]] constexpr bool SupportsImageBufferTransfer(u32 num_samples) noexcept {
    return num_samples == 1;
}

/// Whether an image-to-buffer-to-image round trip is representable by Vulkan
/// buffer-image copy commands without an explicit resolve operation.
[[nodiscard]] constexpr bool SupportsImageBufferRoundTrip(u32 source_samples,
                                                          u32 destination_samples) noexcept {
    return SupportsImageBufferTransfer(source_samples) &&
           SupportsImageBufferTransfer(destination_samples);
}

} // namespace VideoCore
