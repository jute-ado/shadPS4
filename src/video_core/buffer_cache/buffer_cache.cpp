// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <magic_enum/magic_enum.hpp>
#include "common/alignment.h"
#include "common/debug.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/buffer_copy_endpoints.h"
#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/buffer_cache/download_readback_policy.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/srt_cpu_snapshot_selection.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {

static constexpr size_t DataShareBufferSize = 64_KB;
static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      fault_manager{instance, scheduler, *this, tracker, CACHING_PAGEBITS, CACHING_NUMPAGES},
      staging_buffer{instance, scheduler, MemoryUsage::Upload, StagingBufferSize},
      stream_buffer{instance, scheduler, MemoryUsage::Stream, UboStreamBufferSize},
      download_buffer{instance, scheduler, MemoryUsage::Download, DownloadBufferSize},
      device_buffer{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize},
      gds_buffer{instance, scheduler, MemoryUsage::Stream, 0, AllFlags, DataShareBufferSize},
      bda_pagetable_buffer{instance, scheduler, MemoryUsage::DeviceLocal,
                           0,        AllFlags,  BDA_PAGETABLE_SIZE} {
    Vulkan::SetObjectName(instance.GetDevice(), gds_buffer.Handle(), "GDS Buffer");
    Vulkan::SetObjectName(instance.GetDevice(), bda_pagetable_buffer.Handle(),
                          "BDA Page Table Buffer");

    memory_tracker = std::make_unique<MemoryTracker>(tracker);

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);

    // Set up garbage collection parameters
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = DEFAULT_TRIGGER_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    trigger_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_TRIGGER_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
}

BufferCache::~BufferCache() = default;

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, MemoryInvalidationReason reason) {
    if (reason == MemoryInvalidationReason::MappingChange) {
        // A cache unregister need not have removed an old CPU before-image. Never
        // let it survive into a new mapping, even when no buffer is registered now.
        cpu_page_write_tracker.Discard(device_addr, size);
    }
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    memory_tracker->InvalidateRegion(
        device_addr, size,
        [this, device_addr, size] {
            cpu_page_write_tracker.Discard(device_addr, size);
            ReadMemory(device_addr, size, true);
        },
        [this](VAddr page_addr, size_t write_offset, size_t write_size) {
            const auto page = std::span<const u8, TRACKER_BYTES_PER_PAGE>{
                std::bit_cast<const u8*>(page_addr), TRACKER_BYTES_PER_PAGE};
            return cpu_page_write_tracker.Capture(page_addr, page, write_offset, write_size);
        },
        [this](VAddr address, u64 length) {
            cpu_page_write_tracker.ForceMappingUpload(address, length);
        },
        reason);
    dma_dirty_ranges.Mark(device_addr, size);
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write) {
    liverpool->SendCommand<true>([this, device_addr, size, is_write] {
        Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
        // GPU-modified ranges come as many small scattered islands, so the download
        // is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr buf_start = buffer.CpuAddr();
        const VAddr buf_end = buf_start + buffer.SizeBytes();
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), buf_start);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), buf_end);
        DownloadBufferMemory<false>(buffer, window_start, window_end - window_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    });
}

bool BufferCache::ReadMemoryExact(VAddr device_addr, u64 size) {
    ASSERT_MSG(size <= std::numeric_limits<u32>::max(), "Exact readback range is too large: {}",
               size);
    bool readback_completed = false;
    liverpool->SendCommand<true>([this, device_addr, size, &readback_completed] {
        const BufferId buffer_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (IsBufferInvalid(buffer_id)) {
            return;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        if (!buffer.IsInBounds(device_addr, size)) {
            return;
        }
        DownloadBufferMemory<false, true>(buffer, device_addr, size);
        readback_completed = true;
    });
    return readback_completed;
}

bool BufferCache::ReadMemorySnapshot(VAddr device_addr, void* output, u64 size) {
    if (output == nullptr || size == 0 || size > std::numeric_limits<u32>::max() ||
        device_addr > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }

    bool snapshot_completed = false;
    liverpool->SendCommand<true>([this, device_addr, output, size, &snapshot_completed] {
        const bool has_exact_gpu_modifications = gpu_modified_ranges.Intersects(device_addr, size);
        const bool is_gpu_page_owned = memory_tracker->IsRegionGpuModified(device_addr, size);
        const auto read_cpu_backing = [this, device_addr, output, size] {
            memory->CopySparseMemory(device_addr, static_cast<u8*>(output), size);
            return true;
        };
        const auto read_gpu_buffer = [this, device_addr, output, size] {
            return DownloadBufferMemorySnapshot(device_addr, output, size);
        };
        snapshot_completed = ReadAuthoritativeMemorySnapshot(
            has_exact_gpu_modifications, is_gpu_page_owned, read_cpu_backing, read_gpu_buffer);
    });
    return snapshot_completed;
}

template <bool async, bool exact>
void BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    boost::container::small_vector<std::pair<VAddr, u64>, 4> downloaded_ranges;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            if constexpr (exact) {
                const auto clipped =
                    ClipDownloadRange(device_addr_out, range_size, device_addr, size);
                if (!clipped) {
                    return;
                }
                device_addr_out = clipped->address;
                range_size = clipped->size;
            }
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            downloaded_ranges.emplace_back(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    std::optional<Core::DeferredBackingWrite> publication;
    if constexpr (async) {
        boost::container::small_vector<Core::ReadbackWriteRange, 4> destinations;
        for (const auto& copy : copies) {
            destinations.push_back({buffer.CpuAddr() + copy.srcOffset, copy.size, copy.dstOffset});
        }
        // Capture before recording the copy (and before staging Map can drain older callbacks).
        // A refused capture must not consume dirty metadata or queue a partial publication.
        publication = memory->CaptureReadbackDestinations(destinations, total_size_bytes);
        if (!publication) {
            return;
        }
    }
    for (const auto& [address, range_size] : downloaded_ranges) {
        gpu_modified_ranges.Subtract(address, range_size);
    }
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    // Synchronize prior GPU writes to this buffer before the transfer read
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = buffer.buffer,
        .offset = 0,
        .size = buffer.SizeBytes(),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
    const vk::BufferMemoryBarrier2 host_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .buffer = download_buffer.Handle(),
        .offset = offset,
        .size = total_size_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });

    const auto write_data = [this, device_addr, size, copies, offset, download,
                             buffer_cpu_addr = buffer.CpuAddr()]() {
        auto* memory = Core::Memory::Instance();
        cpu_page_write_tracker.Discard(device_addr, size);
        for (const auto& copy : copies) {
            const VAddr copy_device_addr = buffer_cpu_addr + copy.srcOffset;
            const u64 dst_offset = copy.dstOffset - offset;
            memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download + dst_offset,
                                    copy.size);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
    };
    if constexpr (async) {
        scheduler.DeferOperation([this, token = std::move(*publication), download, offset,
                                  total_size_bytes] {
            ASSERT_MSG(download_buffer.InvalidateMappedRange(offset, total_size_bytes),
                       "Failed to invalidate asynchronous download range {:#x}:{:#x}", offset,
                       total_size_bytes);
            // Still-owned destinations and their cleanup share one mapping lock. Never clear
            // CPU/GPU ownership of a replacement, or drop unrelated live bytes with it.
            memory->TryPublishReadback(token, {download, total_size_bytes}, [this](auto ranges) {
                for (const auto& range : ranges) {
                    cpu_page_write_tracker.Discard(range.address, range.size);
                    memory_tracker->UnmarkRegionAsGpuModified(range.address, range.size);
                }
            });
        });
    } else {
        const bool completed = CompleteDownloadReadback(
            [this] { scheduler.Finish(); },
            [this, offset, total_size_bytes] {
                return download_buffer.InvalidateMappedRange(offset, total_size_bytes);
            },
            write_data);
        ASSERT_MSG(completed, "Failed to invalidate download range {:#x}:{:#x}", offset,
                   total_size_bytes);
    }
}

bool BufferCache::DownloadBufferMemorySnapshot(VAddr device_addr, void* output, u64 size) {
    const auto cpu_capture_range = PlanCpuSnapshotCapture(device_addr, size);
    if (!cpu_capture_range) {
        return false;
    }
    struct SnapshotCopy {
        Buffer* buffer;
        vk::BufferCopy region;
    };

    boost::container::small_vector<SnapshotCopy, 4> copies;
    boost::container::small_vector<MemorySnapshotOverlay, 4> overlays;
    boost::container::small_vector<Buffer*, 4> source_buffers;
    u64 total_size_bytes = 0;
    bool valid = true;
    gpu_modified_ranges.ForEachInRange(device_addr, size, [&](VAddr range_start, VAddr range_end) {
        while (valid && range_start < range_end) {
            const BufferId buffer_id = page_table[range_start >> CACHING_PAGEBITS].buffer_id;
            if (IsBufferInvalid(buffer_id)) {
                valid = false;
                return;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            const VAddr buffer_start = buffer.CpuAddr();
            if (buffer.SizeBytes() > std::numeric_limits<VAddr>::max() - buffer_start) {
                valid = false;
                return;
            }
            const VAddr buffer_end = buffer_start + buffer.SizeBytes();
            if (range_start < buffer_start || range_start >= buffer_end) {
                valid = false;
                return;
            }
            const VAddr copy_end = std::min(range_end, buffer_end);
            const u64 copy_size = copy_end - range_start;
            const u64 aligned_size = Common::AlignUp(copy_size, 64ULL);
            if (aligned_size > DownloadBufferSize - total_size_bytes) {
                valid = false;
                return;
            }
            copies.push_back(SnapshotCopy{
                .buffer = &buffer,
                .region =
                    vk::BufferCopy{
                        .srcOffset = range_start - buffer_start,
                        .dstOffset = total_size_bytes,
                        .size = copy_size,
                    },
            });
            overlays.push_back(MemorySnapshotOverlay{
                .output_offset = range_start - device_addr,
                .download_offset = total_size_bytes,
                .size = copy_size,
            });
            if (std::ranges::find(source_buffers, &buffer) == source_buffers.end()) {
                source_buffers.push_back(&buffer);
            }
            total_size_bytes += aligned_size;
            range_start = copy_end;
        }
    });
    if (!valid || copies.empty()) {
        return false;
    }

    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    if (download == nullptr) {
        return false;
    }
    for (auto& copy : copies) {
        copy.region.dstOffset += offset;
    }
    download_buffer.Commit();

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    boost::container::small_vector<vk::BufferMemoryBarrier2, 4> pre_barriers;
    pre_barriers.reserve(source_buffers.size());
    for (const Buffer* buffer : source_buffers) {
        pre_barriers.push_back(vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = buffer->buffer,
            .offset = 0,
            .size = buffer->SizeBytes(),
        });
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });
    for (const auto& copy : copies) {
        cmdbuf.copyBuffer(copy.buffer->buffer, download_buffer.Handle(), copy.region);
    }
    const vk::BufferMemoryBarrier2 host_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .buffer = download_buffer.Handle(),
        .offset = offset,
        .size = total_size_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });

    // Keep the same CPU capture for both the base copy and pending-write reconciliation.
    // Boundary words are needed to match the CPU tracker's word-granular upload policy, but
    // only requested bytes are published into the private output. SRT epochs cover the mapped
    // guest pages throughout this copy; no live CPU memory is read after applying GPU bytes.
    const VAddr cpu_capture_address = cpu_capture_range->address;
    std::vector<u8> cpu_capture(cpu_capture_range->size);
    return CompletePrivateSnapshotReadback(
        [this] { scheduler.Finish(); },
        [this, offset, total_size_bytes] {
            return download_buffer.InvalidateMappedRange(offset, total_size_bytes);
        },
        [this, device_addr, output, size, cpu_capture_address, &cpu_capture] {
            memory->CopySparseMemory(cpu_capture_address, cpu_capture.data(), cpu_capture.size());
            std::memcpy(output, cpu_capture.data() + (device_addr - cpu_capture_address), size);
            return true;
        },
        [this, device_addr, output, size, download, total_size_bytes, &overlays,
         cpu_capture_address, &cpu_capture] {
            const std::span<u8> destination{static_cast<u8*>(output), static_cast<size_t>(size)};
            const bool normal_completed =
                ApplyMemorySnapshotOverlays(
                    destination,
                    std::span<const u8>{download, static_cast<size_t>(total_size_bytes)},
                    std::span<const MemorySnapshotOverlay>{overlays.data(), overlays.size()}) &&
                cpu_page_write_tracker.ApplyPendingWrites(cpu_capture_address, cpu_capture,
                                                          device_addr, destination);
            return GetSrtCpuSnapshotSelection().Complete(
                normal_completed, cpu_capture, device_addr - cpu_capture_address, destination, [] {
                    LOG_WARNING(Render_Vulkan,
                                "DIAGNOSTIC SRT CPU capture selection active: normal GPU readback "
                                "and CPU-write reconciliation completed; selecting the same CPU "
                                "capture instead of the authoritative composition");
                });
        });
}

BufferCache::PreparedVertexBuffers BufferCache::PrepareVertexBuffers(
    const Vulkan::GraphicsPipeline& pipeline,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& /*barriers*/) {
    const auto& regs = liverpool->regs;
    PreparedVertexBuffers prepared;
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(prepared.attributes, prepared.bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (prepared.bindings.empty()) {
        return prepared;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    Vulkan::VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    Vulkan::VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (const auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        prepared.ranges.push_back({
            .base_address = range.base_address,
            .end_address = range.end_address,
            .binding = PrepareBufferBinding(range.base_address, static_cast<u32>(size), false),
        });
    }

    // Retain logical range indices. Host handles are resolved only after every command acquisition.
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(prepared.ranges, [&](const PreparedVertexBufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != prepared.ranges.cend());
            prepared.host_buffers.emplace_back(VK_NULL_HANDLE);
            prepared.host_offsets.push_back(0);
            prepared.host_range_indices.push_back(
                static_cast<u32>(std::distance(prepared.ranges.begin(), host_buffer_info)));
            prepared.guest_addresses.push_back(buffer.base_address);
        } else {
            prepared.host_buffers.emplace_back(VK_NULL_HANDLE);
            prepared.host_offsets.push_back(0);
            prepared.host_range_indices.push_back(std::numeric_limits<u32>::max());
            prepared.guest_addresses.push_back(0);
        }
        prepared.host_sizes.push_back(buffer.GetSize());
        prepared.host_strides.push_back(buffer.GetStride());
    }

    return prepared;
}

bool BufferCache::FinalizeVertexBuffers(
    PreparedVertexBuffers& prepared,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    Vulkan::VertexInputs<BufferBindingEndpoint> endpoints;
    endpoints.reserve(prepared.ranges.size());
    for (const auto& range : prepared.ranges) {
        const auto endpoint = FinalizeBufferBinding(
            range.binding, vk::AccessFlagBits2::eVertexAttributeRead,
            vk::PipelineStageFlagBits2::eVertexAttributeInput, barriers);
        if (!endpoint) {
            return false;
        }
        endpoints.push_back(*endpoint);
    }
    for (u32 index = 0; index < prepared.host_buffers.size(); ++index) {
        const u32 range_index = prepared.host_range_indices[index];
        if (range_index == std::numeric_limits<u32>::max()) {
            continue;
        }
        ASSERT(range_index < prepared.ranges.size());
        const auto& range = prepared.ranges[range_index];
        const auto& endpoint = endpoints[range_index];
        prepared.host_buffers[index] = endpoint.buffer;
        prepared.host_offsets[index] =
            endpoint.offset + prepared.guest_addresses[index] - range.base_address;
    }
    return true;
}

void BufferCache::BindVertexBuffers(const PreparedVertexBuffers& prepared) {
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = prepared.host_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        if (num_buffers != 0) {
            cmdbuf.bindVertexBuffers(0, num_buffers, prepared.host_buffers.data(),
                                     prepared.host_offsets.data());
        }
        cmdbuf.setVertexInputEXT(prepared.bindings, prepared.attributes);
    } else if (num_buffers != 0) {
        cmdbuf.bindVertexBuffers2(0, num_buffers, prepared.host_buffers.data(),
                                  prepared.host_offsets.data(), prepared.host_sizes.data(),
                                  prepared.host_strides.data());
    }
}

BufferCache::PreparedIndexBuffer BufferCache::PrepareIndexBuffer(
    u32 index_offset,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& /*barriers*/) {
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Acquire the index buffer. Binding is deferred until all potentially waiting acquisitions
    // for the draw have completed.
    const u32 index_buffer_size = regs.num_indices * index_size;
    return PreparedIndexBuffer{
        .binding = PrepareBufferBinding(index_address, index_buffer_size, false),
        .buffer = VK_NULL_HANDLE,
        .offset = 0,
        .type = index_type,
    };
}

bool BufferCache::FinalizeIndexBuffer(
    PreparedIndexBuffer& prepared,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto endpoint = FinalizeBufferBinding(prepared.binding, vk::AccessFlagBits2::eIndexRead,
                                                vk::PipelineStageFlagBits2::eIndexInput, barriers);
    if (!endpoint) {
        return false;
    }
    prepared.buffer = endpoint->buffer;
    prepared.offset = endpoint->offset;
    return true;
}

void BufferCache::BindIndexBuffer(const PreparedIndexBuffer& prepared) {
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0, "GDS offset must be dword aligned");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + num_bytes / sizeof(u32), value);
            return;
        }
    }
    Buffer* buffer = [&] {
        if (is_gds) {
            return &gds_buffer;
        }
        const auto [buffer, offset] = ObtainBuffer(address, num_bytes, true);
        return buffer;
    }();
    buffer->Fill(buffer->Offset(address), num_bytes, value);
}

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
        // Without a readback there's nothing we can do with this
        // Fallback to creating dst buffer on GPU to at least have this data there
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    auto prepare_source = [&] {
        if (src_gds) {
            return;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true);
    };
    auto prepare_destination = [&] {
        if (dst_gds) {
            return;
        }
        const auto buffer_id = FindBuffer(dst, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, dst, num_bytes, true, true);
        gpu_modified_ranges.Add(dst, num_bytes);
    };
    const auto resolve_current = [&](VAddr address, bool is_gds) -> const Buffer& {
        if (is_gds) {
            return gds_buffer;
        }
        // Both ranges were prepared above. A merge can replace a buffer but retains
        // its covered addresses. Do not allocate or wait after taking these handles.
        const auto id = page_table[address >> CACHING_PAGEBITS].buffer_id;
        ASSERT(id);
        const auto& buffer = slot_buffers[id];
        ASSERT(buffer.IsInBounds(address, num_bytes));
        return buffer;
    };
    const auto [src_buffer, dst_buffer] = PrepareBufferCopyEndpoints(
        src, dst, prepare_source, prepare_destination,
        [&]() -> const Buffer& { return resolve_current(src, src_gds); },
        [&]() -> const Buffer& { return resolve_current(dst, dst_gds); });
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.offset,
        .dstOffset = dst_buffer.offset,
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.handle,
            .offset = dst_buffer.offset,
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.handle,
            .offset = src_buffer.offset,
            .size = num_bytes,
        },
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_before,
    });
    cmdbuf.copyBuffer(src_buffer.handle, dst_buffer.handle, region);
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.handle,
            .offset = dst_buffer.offset,
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.handle,
            .offset = src_buffer.offset,
            .size = num_bytes,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_after,
    });
}

std::pair<Buffer*, u32> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written,
                                                   bool is_texel_buffer, BufferId buffer_id) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= CACHING_PAGESIZE && !IsRegionGpuModified(device_addr, size)) {
        const u64 offset = stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
        return {&stream_buffer, offset};
    }
    if (IsBufferInvalid(buffer_id)) {
        buffer_id = FindBuffer(device_addr, size);
    }
    Buffer& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
    }
    return {&buffer, buffer.Offset(device_addr)};
}

BufferCache::PreparedBufferBinding BufferCache::PrepareBufferBinding(
    VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer, BufferId buffer_id) {
    const auto [buffer, offset] =
        ObtainBuffer(device_addr, size, is_written, is_texel_buffer, buffer_id);
    const bool cache_backed = buffer != &stream_buffer;
    return PreparedBufferBinding{
        .address = device_addr,
        .size = size,
        .requires_final_resolution = cache_backed,
        .stable_endpoint = cache_backed
                               ? std::nullopt
                               : std::optional{BufferBindingEndpoint{buffer->Handle(), offset}},
    };
}

std::optional<BufferCache::BufferBindingEndpoint> BufferCache::FinalizeBufferBinding(
    const PreparedBufferBinding& prepared, vk::AccessFlags2 access,
    vk::PipelineStageFlagBits2 stage,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    Buffer* resolved_buffer{};
    auto endpoint = FinalizePreparedBufferBinding(
        prepared, [&](VAddr address, u32 size) -> std::optional<BufferBindingEndpoint> {
            const BufferId buffer_id = page_table[address >> CACHING_PAGEBITS].buffer_id;
            if (IsBufferInvalid(buffer_id)) {
                return std::nullopt;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            if (!buffer.IsInBounds(address, size)) {
                return std::nullopt;
            }
            resolved_buffer = &buffer;
            return BufferBindingEndpoint{buffer.Handle(), buffer.Offset(address)};
        });
    if (!endpoint) {
        return std::nullopt;
    }
    Buffer& barrier_buffer = prepared.requires_final_resolution ? *resolved_buffer : stream_buffer;
    if (auto barrier = barrier_buffer.GetBarrier(access, stage, endpoint->offset)) {
        barriers.emplace_back(*barrier);
    }
    return endpoint;
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
    // Check if any buffer contains the full requested range.
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            SynchronizeBuffer(buffer, gpu_addr, size, false, false);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    // If some buffer within was GPU modified create a full buffer to avoid losing GPU data.
    if (IsRegionGpuModified(gpu_addr, size)) {
        return ObtainBuffer(gpu_addr, size, false, false);
    }
    // In all other cases, just do a CPU copy to the staging buffer.
    const auto [data, offset] = staging_buffer.Map(size, 16);
    memory->CopySparseMemory(gpu_addr, data, size);
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
    // Check if we are missing some edge case here
    return buffer_ranges.Intersects(addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModifiedExact(VAddr addr, size_t size) const {
    return gpu_modified_ranges.Intersects(addr, size);
}

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size) {
    ASSERT(device_addr != 0);
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    if (!buffer_id) {
        return CreateBuffer(device_addr, size);
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        return buffer_id;
    }
    return CreateBuffer(device_addr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(VAddr device_addr, u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    VAddr begin = device_addr;
    VAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    const auto expand_begin = [&](VAddr add_value) {
        static constexpr VAddr min_page = CACHING_PAGESIZE + DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    const auto expand_end = [&](VAddr add_value) {
        static constexpr VAddr max_page = 1ULL << MemoryTracker::MAX_CPU_PAGE_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.is_picked) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.is_picked = true;
        const VAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const VAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_end(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

void BufferCache::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                              bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    const vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = dst_base_offset,
        .size = overlap.SizeBytes(),
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
    if (auto src_barrier = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer)) {
        pre_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier =
            new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTransfer, dst_base_offset)) {
        pre_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    cmdbuf.copyBuffer(overlap.Handle(), new_buffer.Handle(), copy);

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
    if (auto src_barrier =
            overlap.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllCommands)) {
        post_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier = new_buffer.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, dst_base_offset)) {
        post_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pBufferMemoryBarriers = post_barriers.data(),
    });
    DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(VAddr device_addr, u32 wanted_size) {
    const VAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, overlap.begin,
                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
    auto& new_buffer = slot_buffers[new_buffer_id];
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    PublishDmaBufferAfterSynchronization(
        new_buffer,
        [this](Buffer& buffer, VAddr address, u32 size) {
            SynchronizeBuffer(buffer, address, size, false, false, false);
        },
        [this, new_buffer_id] { Register(new_buffer_id); });
    return new_buffer_id;
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager.ProcessFaultBuffer();
}

void BufferCache::DrainReadyFaults() {
    fault_manager.DrainReadyFaults();
}

void BufferCache::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

void BufferCache::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const VAddr device_addr_begin = buffer.CpuAddr();
    const VAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page].buffer_id = buffer_id;
        } else {
            page_table[page].buffer_id = BufferId{};
        }
    }
    if constexpr (insert) {
        total_used_memory += Common::AlignUp(size, CACHING_PAGESIZE);
        buffer.SetLRUId(lru_cache.Insert(buffer_id, gc_tick));
        boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
        bda_addrs.reserve(size_pages);
        for (u64 i = 0; i < size_pages; ++i) {
            vk::DeviceAddress addr = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
            bda_addrs.push_back(addr);
        }
        WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                        bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
    } else {
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        lru_cache.Free(buffer.LRUId());
        const u64 offset = bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
        bda_pagetable_buffer.Fill(offset, size_pages * sizeof(vk::DeviceAddress), 0);
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer, bool is_registered) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    boost::container::small_vector<std::pair<VAddr, u64>, 16> uploaded_cpu_ranges;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](VAddr device_addr_out, u64 range_size) {
            const auto add_upload = [&](VAddr upload_addr, u64 upload_size) {
                copies.emplace_back(total_size_bytes, upload_addr - buffer_start, upload_size);
                uploaded_cpu_ranges.emplace_back(upload_addr, upload_size);
                total_size_bytes += upload_size;
            };

            const VAddr range_end = device_addr_out + range_size;
            VAddr page_addr = device_addr_out & ~(TRACKER_BYTES_PER_PAGE - 1);
            while (page_addr < range_end) {
                const VAddr upload_begin = std::max(device_addr_out, page_addr);
                const VAddr upload_end = std::min(range_end, page_addr + TRACKER_BYTES_PER_PAGE);
                const auto current_page = std::span<const u8, TRACKER_BYTES_PER_PAGE>{
                    std::bit_cast<const u8*>(page_addr), TRACKER_BYTES_PER_PAGE};
                const bool consumed = cpu_page_write_tracker.Consume(
                    page_addr, current_page, upload_begin - page_addr, upload_end - upload_begin,
                    [&](CpuPageUploadRange range) {
                        add_upload(page_addr + range.offset, range.size);
                    });
                if (!consumed) {
                    add_upload(upload_begin, upload_end - upload_begin);
                }
                page_addr += TRACKER_BYTES_PER_PAGE;
            }
        },
        [&] { src_buffer = UploadCopies(buffer, copies, total_size_bytes); });

    boost::container::small_vector<VAddr, 16> uploaded_pages;
    for (const auto& [upload_addr, upload_size] : uploaded_cpu_ranges) {
        gpu_modified_ranges.Subtract(upload_addr, upload_size);
        const VAddr upload_end = upload_addr + upload_size;
        for (VAddr page = upload_addr & ~(TRACKER_BYTES_PER_PAGE - 1); page < upload_end;
             page += TRACKER_BYTES_PER_PAGE) {
            uploaded_pages.push_back(page);
        }
    }
    if (!is_written) {
        std::ranges::sort(uploaded_pages);
        const auto unique_end = std::ranges::unique(uploaded_pages).begin();
        uploaded_pages.erase(unique_end, uploaded_pages.end());
        for (const VAddr page : uploaded_pages) {
            if (!gpu_modified_ranges.Intersects(page, TRACKER_BYTES_PER_PAGE)) {
                memory_tracker->UnmarkRegionAsGpuModified(page, TRACKER_BYTES_PER_PAGE);
            }
        }
    }

    if (src_buffer) {
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eTransferRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        const vk::BufferMemoryBarrier2 post_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
        cmdbuf.copyBuffer(src_buffer, buffer.buffer, copies);
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &post_barrier,
        });
        TouchBufferAfterUploadIfRegistered(is_registered, [&] { TouchBuffer(buffer); });
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeBufferFromImage(buffer, device_addr, size);
    }
    return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    if (staging) {
        for (auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            // Apply the staging offset
            copy.srcOffset += offset;
        }
        staging_buffer.Commit();
        return staging_buffer.Handle();
    } else {
        // For large one time transfers use a temporary host buffer.
        auto temp_buffer =
            std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
        const vk::Buffer src_buffer = temp_buffer->Handle();
        u8* const staging = temp_buffer->mapped_data.data();
        for (const auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
        }
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable { buffer.reset(); });
        return src_buffer;
    }
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            buffer.Fill(buffer.Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset > buffer.SizeBytes() ||
            mip_info.offset > buffer.SizeBytes() - buf_offset ||
            mip_info.size > buffer.SizeBytes() - buf_offset - mip_info.offset) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    return tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset,
                                  buffer.SizeBytes());
}

void BufferCache::SynchronizeBuffersInRange(VAddr device_addr, u64 size) {
    const VAddr device_addr_end = device_addr + size;
    ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
        RENDERER_TRACE;
        VAddr start = std::max(buffer.CpuAddr(), device_addr);
        VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), device_addr_end);
        u32 size = static_cast<u32>(end - start);
        SynchronizeBuffer(buffer, start, size, false, false);
    });
}

void BufferCache::SynchronizeDmaBuffers() {
    for (const auto& range : dma_dirty_ranges.Take()) {
        SynchronizeBuffersInRange(range.address, range.size);
    }
}

void BufferCache::WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes) {
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        // For large one time transfers use a temporary host buffer.
        // RenderDoc can lag quite a bit if the stream buffer is too large.
        Buffer temp_buffer{
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes};
        src_buffer = temp_buffer.Handle();
        u8* const staging = temp_buffer.mapped_data.data();
        std::memcpy(staging, value, num_bytes);
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable {});
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.Handle(), copy);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

void BufferCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    const bool aggressive = total_used_memory >= critical_gc_memory;
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 80 : 160, gc_tick);
    int max_deletions = aggressive ? 64 : 32;
    const auto clean_up = [&](BufferId buffer_id) {
        if (max_deletions == 0) {
            return;
        }
        --max_deletions;
        Buffer& buffer = slot_buffers[buffer_id];
        // InvalidateMemory(buffer.CpuAddr(), buffer.SizeBytes());
        DownloadBufferMemory<true>(buffer, buffer.CpuAddr(), buffer.SizeBytes());
        memory_tracker->MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
        DeleteBuffer(buffer_id);
    };
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeleteBuffer(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    Unregister(buffer_id);
    scheduler.DeferOperation([this, buffer_id] { slot_buffers.erase(buffer_id); });
    buffer.is_deleted = true;
}

} // namespace VideoCore
