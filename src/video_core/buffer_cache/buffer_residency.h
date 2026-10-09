// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <unordered_set>

#include "common/types.h"

namespace VideoCore {

template <size_t Capacity>
class BoundedDiagnosticDeduplicator {
public:
    [[nodiscard]] bool ShouldLog(u64 key) {
        if (keys.contains(key) || keys.size() >= Capacity) {
            return false;
        }
        keys.insert(key);
        return true;
    }

private:
    std::unordered_set<u64> keys;
};

template <typename PageManager>
[[nodiscard]] bool IsGuestBufferBindingMapped(PageManager& page_manager, bool is_special,
                                              VAddr address, u64 size) {
    if (is_special) {
        return true;
    }
    return address != 0 && size != 0 && page_manager.IsGpuMapped(address, size);
}

template <typename PageManager>
[[nodiscard]] bool IsGuestImageBindingMapped(PageManager& page_manager, VAddr address, u64 size) {
    return address != 0 && size != 0 && page_manager.IsGpuMapped(address, size);
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
          typename PrepareIndirectBuffers, typename BindVertexBuffers, typename BindIndexBuffer>
void PrepareDrawBuffersThenBindCommandState(PrepareVertexBuffers&& prepare_vertex_buffers,
                                            PrepareIndexBuffer&& prepare_index_buffer,
                                            PrepareIndirectBuffers&& prepare_indirect_buffers,
                                            BindVertexBuffers&& bind_vertex_buffers,
                                            BindIndexBuffer&& bind_index_buffer) {
    // Any preparation step can wait for a stream-buffer region and replace the current command
    // buffer. Publish command-buffer-local draw state only after every such step has completed.
    auto prepared_vertex_state = prepare_vertex_buffers();
    auto prepared_index_state = prepare_index_buffer();
    prepare_indirect_buffers();
    bind_vertex_buffers(prepared_vertex_state);
    bind_index_buffer(prepared_index_state);
}

template <typename PrepareFixedBuffers, typename RefreshResourceBuffers,
          typename PublishResourceBuffers, typename PublishFixedBuffers>
void RefreshDrawBufferStateBeforePublication(PrepareFixedBuffers&& prepare_fixed_buffers,
                                             RefreshResourceBuffers&& refresh_resource_buffers,
                                             PublishResourceBuffers&& publish_resource_buffers,
                                             PublishFixedBuffers&& publish_fixed_buffers) {
    // Resource and fixed-function ranges can overlap. Acquire both sets before resolving their
    // Vulkan handles, then prepare the fixed-function state once more after resource refresh. The
    // second fixed pass operates on already-known ranges, so neither published set can refer to a
    // buffer replaced by a later topology-changing acquisition.
    prepare_fixed_buffers();
    refresh_resource_buffers();
    prepare_fixed_buffers();
    publish_resource_buffers();
    publish_fixed_buffers();
}

} // namespace VideoCore
