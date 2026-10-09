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
};

template <typename Endpoint, typename Resolve>
[[nodiscard]] std::optional<Endpoint> FinalizePreparedBufferBinding(
    const PreparedBufferBinding<Endpoint>& prepared, Resolve&& resolve) {
    if (prepared.requires_final_resolution) {
        return std::forward<Resolve>(resolve)(prepared.address, prepared.size);
    }
    return prepared.stable_endpoint;
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
          typename PrepareIndirectBuffers, typename FinalizeBuffers, typename BindVertexBuffers,
          typename BindIndexBuffer>
[[nodiscard]] bool PrepareDrawBuffersThenBindCommandState(
    PrepareVertexBuffers&& prepare_vertex_buffers, PrepareIndexBuffer&& prepare_index_buffer,
    PrepareIndirectBuffers&& prepare_indirect_buffers, FinalizeBuffers&& finalize_buffers,
    BindVertexBuffers&& bind_vertex_buffers, BindIndexBuffer&& bind_index_buffer) {
    // Any preparation step can wait for a stream-buffer region and replace the current command
    // buffer. Publish command-buffer-local draw state only after every such step has completed.
    auto prepared_vertex_state = prepare_vertex_buffers();
    auto prepared_index_state = prepare_index_buffer();
    prepare_indirect_buffers();
    if (!finalize_buffers(prepared_vertex_state, prepared_index_state)) {
        return false;
    }
    bind_vertex_buffers(prepared_vertex_state);
    bind_index_buffer(prepared_index_state);
    return true;
}

} // namespace VideoCore
