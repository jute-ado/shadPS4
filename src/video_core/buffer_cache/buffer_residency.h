// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <utility>

#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace VideoCore {

[[nodiscard]] constexpr bool IsGuestBufferDescriptorBindable(
    bool is_special, VAddr address, u64 size, bool is_gpu_address_valid,
    [[maybe_unused]] bool has_cpu_mapped_backing) noexcept {
    // GPU-written buffers can remain resident in the buffer cache while their guest virtual
    // range is reserved rather than CPU mapped. CPU mapping state therefore cannot decide
    // descriptor validity; FindBuffer/ObtainBuffer owns the backing and sparse-copy behavior.
    return !is_special && address != 0 && size != 0 && is_gpu_address_valid;
}

template <typename Buffer>
void InitializeDmaPageTable(Buffer& page_table, u32 size) {
    // A zero entry is the shader-side sentinel for an unmapped guest page. Device-local
    // allocations have undefined initial contents, so publish that invariant explicitly before
    // any shader can look up a guest address.
    page_table.Fill(0, size, 0);
}

[[nodiscard]] constexpr bool CanUseCpuStreamCopy(bool is_written, u32 size, u32 maximum_size,
                                                 bool has_exact_gpu_modification) noexcept {
    return !is_written && size <= maximum_size && !has_exact_gpu_modification;
}

[[nodiscard]] constexpr bool ShouldWidenCpuUploadToDescriptor(
    u32 descriptor_size, u32 maximum_size, bool has_detected_cpu_change) noexcept {
    return descriptor_size <= maximum_size && has_detected_cpu_change;
}

template <typename Discard, typename IsRegistered>
[[nodiscard]] bool DiscardCpuSnapshotsBeforeBufferLookup(Discard&& discard,
                                                         IsRegistered&& is_registered) {
    // A guest unmap can remove the cache registration before invalidation reaches this cache.
    // Snapshot bytes belong to the guest mapping, not to a host buffer registration, so discard
    // them first and never retain a span that may subsequently be unmapped.
    std::forward<Discard>(discard)();
    return std::forward<IsRegistered>(is_registered)();
}

template <typename IsReadable, typename Capture, typename Discard>
[[nodiscard]] bool CaptureCpuSnapshotIfReadable(VAddr page_addr, IsReadable&& is_readable,
                                                Capture&& capture, Discard&& discard,
                                                size_t write_offset, size_t write_size) {
    if (!std::forward<IsReadable>(is_readable)(page_addr, TRACKER_BYTES_PER_PAGE)) {
        std::forward<Discard>(discard)(page_addr, TRACKER_BYTES_PER_PAGE);
        return false;
    }
    const auto page = std::span<const u8, TRACKER_BYTES_PER_PAGE>{
        std::bit_cast<const u8*>(page_addr), TRACKER_BYTES_PER_PAGE};
    return std::forward<Capture>(capture)(
        page_addr, page, write_offset, write_size);
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

template <typename Record, typename Retire>
void RecordThenRetireTransientResource(Record&& record, Retire&& retire) {
    std::forward<Record>(record)();
    std::forward<Retire>(retire)();
}

template <typename Scheduler, typename Retire>
void RetirePublishedDmaBuffer(Scheduler& scheduler, Retire&& retire) {
    // Shader device addresses are consumed indirectly through the GPU page table, so retaining a
    // VkBuffer handle while recording a draw is not observable here. Leave one complete command
    // submission between unpublishing the address and destroying its backing allocation.
    scheduler.DeferOperationAfterNextSubmit(std::forward<Retire>(retire));
}

template <typename Prepare, typename BindIndex, typename BindVertex>
void PrepareDrawBuffersThenBindDynamicVertexState(Prepare&& prepare, BindIndex&& bind_index,
                                                  BindVertex&& bind_vertex) {
    std::forward<Prepare>(prepare)();
    std::forward<BindIndex>(bind_index)();
    std::forward<BindVertex>(bind_vertex)();
}

} // namespace VideoCore
