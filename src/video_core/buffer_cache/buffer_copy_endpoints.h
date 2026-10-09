// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

#include "common/types.h"

namespace VideoCore {

template <typename Handle>
struct BufferCopyEndpoint {
    Handle handle;
    u64 offset;
};

template <typename Handle>
[[nodiscard]] bool BufferCopyRangesOverlap(const BufferCopyEndpoint<Handle>& source,
                                           u64 source_size,
                                           const BufferCopyEndpoint<Handle>& destination,
                                           u64 destination_size) noexcept {
    if (source.handle != destination.handle) {
        return false;
    }
    return source.offset < destination.offset + destination_size &&
           destination.offset < source.offset + source_size;
}

template <typename PrepareDestination, typename SettleResidency, typename PrepareSource,
          typename Publish>
[[nodiscard]] bool PrepareBufferCopyBatchThenPublish(
    PrepareDestination&& prepare_destination, SettleResidency&& settle_residency,
    PrepareSource&& prepare_source, Publish&& publish) {
    // Destination acquisition can wait, merge cache allocations, and submit the current stream
    // tick. Keep it logical, settle completed fault work, and only then materialize the source.
    // The source may be a stream-ring slice, so no later wait-capable acquisition may occur before
    // the consuming copy is recorded.
    auto destination = std::invoke(std::forward<PrepareDestination>(prepare_destination));
    std::invoke(std::forward<SettleResidency>(settle_residency));
    auto source = std::invoke(std::forward<PrepareSource>(prepare_source));
    return std::invoke(std::forward<Publish>(publish), std::move(source),
                       std::move(destination));
}

template <typename PrepareSource, typename PrepareDestination,
          typename ResolveSource, typename ResolveDestination>
auto PrepareBufferCopyEndpoints(VAddr source_address, VAddr destination_address,
                                PrepareSource&& prepare_source,
                                PrepareDestination&& prepare_destination,
                                ResolveSource&& resolve_source,
                                ResolveDestination&& resolve_destination) {
    // Preparation may relocate slot storage, merge either buffer, and wait for a
    // submission that retires the superseded Vulkan handles. Resolve the current
    // registered owners only after both preparations, without allocating or waiting.
    prepare_source();
    prepare_destination();
    const auto& source = resolve_source();
    const auto source_endpoint = BufferCopyEndpoint{source.Handle(), source.Offset(source_address)};
    const auto& destination = resolve_destination();
    return std::pair{
        source_endpoint,
        BufferCopyEndpoint{destination.Handle(), destination.Offset(destination_address)}};
}

} // namespace VideoCore
