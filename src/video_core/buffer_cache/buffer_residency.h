// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <utility>

#include "common/types.h"

namespace VideoCore {

template <typename Endpoint>
struct PreparedBufferBinding {
    VAddr address{};
    u32 size{};
    bool requires_final_resolution{};
    std::optional<Endpoint> stable_endpoint{};
    // Stream-backed endpoints are only valid for the reservation generation in which they were
    // allocated. A generation is optional so cache-backed and test-defined endpoints retain the
    // old two-phase behavior without being coupled to StreamBuffer.
    std::optional<u64> stable_generation{};
    // Preparation can fail before a cache lookup (for example when a guest range crosses an
    // unmapped VMA). Keep the two-phase value type while making that failure fail closed at the
    // finalization boundary.
    bool valid{true};
};

template <typename Endpoint, typename Resolve>
[[nodiscard]] std::optional<Endpoint> FinalizePreparedBufferBinding(
    const PreparedBufferBinding<Endpoint>& prepared, Resolve&& resolve) {
    if (!prepared.valid) {
        return std::nullopt;
    }
    if (prepared.requires_final_resolution) {
        return std::forward<Resolve>(resolve)(prepared.address, prepared.size);
    }
    return prepared.stable_endpoint;
}

// Finalizes a prepared endpoint while allowing a caller that owns a recyclable backing store to
// validate its reservation token. If the stable endpoint is stale, reacquire must return a
// non-recyclable (or otherwise newly valid) endpoint for the same logical guest range.
template <typename Endpoint, typename Resolve, typename StableValid, typename Reacquire>
[[nodiscard]] std::optional<Endpoint> FinalizePreparedBufferBinding(
    const PreparedBufferBinding<Endpoint>& prepared, Resolve&& resolve,
    StableValid&& stable_valid, Reacquire&& reacquire) {
    if (!prepared.valid) {
        return std::nullopt;
    }
    if (prepared.requires_final_resolution) {
        return std::forward<Resolve>(resolve)(prepared.address, prepared.size);
    }
    if (!prepared.stable_endpoint) {
        return std::nullopt;
    }
    if (std::forward<StableValid>(stable_valid)(prepared)) {
        return prepared.stable_endpoint;
    }
    return std::forward<Reacquire>(reacquire)(prepared.address, prepared.size);
}

template <typename Buffer, typename Synchronize, typename Publish>
void PublishDmaBufferAfterSynchronization(Buffer& buffer, Synchronize&& synchronize,
                                          Publish&& publish) {
    synchronize(buffer, buffer.CpuAddr(), static_cast<u32>(buffer.SizeBytes()));
    publish();
}

template <typename Touch>
void TouchBufferAfterUploadIfRegistered(bool is_registered, Touch&& touch) {
    if (is_registered) {
        touch();
    }
}

template <typename PrepareVertexBuffers, typename PrepareIndexBuffer,
          typename PrepareIndirectBuffers, typename SettleResidency, typename FinalizeBuffers,
          typename BindVertexBuffers, typename BindIndexBuffer>
[[nodiscard]] bool PrepareDrawBuffersThenBindCommandState(
    PrepareVertexBuffers&& prepare_vertex_buffers, PrepareIndexBuffer&& prepare_index_buffer,
    PrepareIndirectBuffers&& prepare_indirect_buffers, SettleResidency&& settle_residency,
    FinalizeBuffers&& finalize_buffers, BindVertexBuffers&& bind_vertex_buffers,
    BindIndexBuffer&& bind_index_buffer) {
    // Any preparation step can wait for a stream-buffer region and replace the current command
    // buffer. Publish command-buffer-local draw state only after every such step has completed.
    auto prepared_vertex_state = prepare_vertex_buffers();
    auto prepared_index_state = prepare_index_buffer();
    prepare_indirect_buffers();
    // A preparation wait can make deferred fault records ready. Apply them before resolving any
    // cache-backed token into the one canonical allocation published by this command.
    settle_residency();
    if (!finalize_buffers(prepared_vertex_state, prepared_index_state)) {
        return false;
    }
    bind_vertex_buffers(prepared_vertex_state);
    bind_index_buffer(prepared_index_state);
    return true;
}

} // namespace VideoCore
