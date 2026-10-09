// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/small_vector.hpp>
#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/buffer_cache/buffer_registration_index.h"
#include "video_core/buffer_cache/cpu_page_write_snapshot.h"
#include "video_core/buffer_cache/dma_dirty_ranges.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/memory_invalidation_reason.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/multi_level_page_table.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
}

namespace VideoCore {

using BufferId = Common::SlotId;

class TextureCache;
class MemoryTracker;
class PageManager;

class BufferCache {
public:
    static constexpr u32 CACHING_PAGEBITS = 14;
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;
    static constexpr u64 DEVICE_PAGESIZE = 16_KB;
    static constexpr u64 CACHING_NUMPAGES = u64{1} << (40 - CACHING_PAGEBITS);
    static constexpr u64 BDA_PAGETABLE_SIZE = CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

    // Default values for garbage collection
    static constexpr s64 DEFAULT_TRIGGER_GC_MEMORY = 1_GB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 2_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    struct PageData {
        BufferId buffer_id{};
    };

    struct Traits {
        using Entry = PageData;
        static constexpr size_t AddressSpaceBits = 40;
        static constexpr size_t FirstLevelBits = 16;
        static constexpr size_t PageBits = CACHING_PAGEBITS;
    };
    using PageTable = MultiLevelPageTable<Traits>;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        VAddr begin;
        VAddr end;
        bool has_stream_leap = false;
    };

    struct BufferBindingEndpoint {
        vk::Buffer buffer;
        vk::DeviceSize offset;
    };

    // Collects all uses of final buffer allocations for one guest command. Resolution is
    // side-effect free; CommitBufferBindings publishes one whole-allocation dependency per
    // allocation only after every logical binding has resolved successfully.
    struct BufferBindingBatch {
        struct Access {
            Buffer* buffer;
            vk::AccessFlags2 access_mask;
            vk::PipelineStageFlags2 stage_mask;
        };

        boost::container::small_vector<Access, 16> accesses;
        bool committed{};
    };

    using PreparedBufferBinding = VideoCore::PreparedBufferBinding<BufferBindingEndpoint>;

    struct PreparedVertexBufferRange {
        VAddr base_address;
        VAddr end_address;
        PreparedBufferBinding binding;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    struct PreparedVertexBuffers {
        Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
        Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
        Vulkan::VertexInputs<vk::Buffer> host_buffers;
        Vulkan::VertexInputs<vk::DeviceSize> host_offsets;
        Vulkan::VertexInputs<vk::DeviceSize> host_sizes;
        Vulkan::VertexInputs<vk::DeviceSize> host_strides;
        Vulkan::VertexInputs<PreparedVertexBufferRange> ranges;
        Vulkan::VertexInputs<u32> host_range_indices;
        Vulkan::VertexInputs<VAddr> guest_addresses;
    };

    struct PreparedIndexBuffer {
        PreparedBufferBinding binding;
        vk::Buffer buffer;
        vk::DeviceSize offset;
        vk::IndexType type;
    };

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return &bda_pagetable_buffer;
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager.GetFaultBuffer();
    }

    /// Retrieves the buffer with the specified id.
    [[nodiscard]] Buffer& GetBuffer(BufferId id) {
        return slot_buffers[id];
    }

    /// Retrieves a utility buffer optimized for specified memory usage.
    StreamBuffer& GetUtilityBuffer(MemoryUsage usage) noexcept {
        if (usage == MemoryUsage::Stream) {
            return stream_buffer;
        } else if (usage == MemoryUsage::Download) {
            return download_buffer;
        } else if (usage == MemoryUsage::DeviceLocal) {
            return device_buffer;
        } else {
            return staging_buffer;
        }
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size,
                          MemoryInvalidationReason reason = MemoryInvalidationReason::CpuWrite);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false);

    /// Flushes exactly the requested GPU-modified range from an existing containing buffer.
    [[nodiscard]] bool ReadMemoryExact(VAddr device_addr, u64 size);

    /// Reads authoritative bytes into caller-owned memory without publishing them into mutable
    /// guest memory or changing GPU/CPU ownership tracking.
    [[nodiscard]] bool ReadMemorySnapshot(VAddr device_addr, void* output, u64 size);

    /// Acquires host vertex buffers without recording command-buffer-local state.
    [[nodiscard]] PreparedVertexBuffers PrepareVertexBuffers(
        const Vulkan::GraphicsPipeline& pipeline,
        boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Resolves every cache-backed vertex endpoint after all command acquisitions finish.
    [[nodiscard]] bool FinalizeVertexBuffers(
        PreparedVertexBuffers& prepared, BufferBindingBatch& binding_batch);

    /// Records prepared vertex buffer state on the current command buffer.
    void BindVertexBuffers(const PreparedVertexBuffers& prepared);

    /// Acquires the host index buffer without recording command-buffer-local state.
    [[nodiscard]] PreparedIndexBuffer PrepareIndexBuffer(
        u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Resolves the cache-backed index endpoint after all command acquisitions finish.
    [[nodiscard]] bool FinalizeIndexBuffer(
        PreparedIndexBuffer& prepared, BufferBindingBatch& binding_batch);

    /// Records prepared index buffer state on the current command buffer.
    void BindIndexBuffer(const PreparedIndexBuffer& prepared);

    /// Writes a value to GPU buffer. (uses command buffer to temporarily store the data)
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);

    /// Performs buffer to buffer data copy on the GPU.
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);

    /// Obtains a buffer for the specified region.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(VAddr gpu_addr, u32 size, bool is_written,
                                                       bool is_texel_buffer = false,
                                                       BufferId buffer_id = {});

    /// Acquires a logical range while retaining only stable stream state or a cache lookup token.
    [[nodiscard]] PreparedBufferBinding PrepareBufferBinding(VAddr gpu_addr, u32 size,
                                                              bool is_written,
                                                              bool is_texel_buffer = false,
                                                              BufferId buffer_id = {});

    /// Resolves a prepared range from the final cache allocation and records its actual use.
    [[nodiscard]] std::optional<BufferBindingEndpoint> FinalizeBufferBinding(
        const PreparedBufferBinding& prepared, vk::AccessFlags2 access,
        vk::PipelineStageFlags2 stage, BufferBindingBatch& binding_batch);

    /// Atomically publishes the aggregate access plan after every logical binding resolves.
    void CommitBufferBindings(
        BufferBindingBatch& binding_batch,
        boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBufferForImage(VAddr gpu_addr, u32 size);

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Return true when the exact byte range has been written by the GPU.
    [[nodiscard]] bool IsRegionGpuModifiedExact(VAddr addr, size_t size) const;

    /// Return buffer id for the specified region
    BufferId FindBuffer(VAddr device_addr, u32 size);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Applies completed fault records at an outer GPU-owner command boundary.
    void DrainReadyFaults();

    /// Synchronizes all buffers in the specified range.
    void SynchronizeBuffersInRange(VAddr device_addr, u64 size);

    /// Synchronizes all buffers neede for DMA.
    void SynchronizeDmaBuffers();

    /// Runs the garbage collector.
    void RunGarbageCollector();

private:
    template <typename Func>
    void ForEachBufferInRange(VAddr device_addr, u64 size, Func&& func) {
        buffer_ranges.OwnerForEachInRange(device_addr, size,
                                         [&](u64 page_start, u64 page_end, BufferId id) {
                                             Buffer& buffer = slot_buffers[id];
                                             func(id, buffer);
                                         });
    }

    inline bool IsBufferInvalid(BufferId buffer_id) const {
        return !buffer_id || slot_buffers[buffer_id].is_deleted;
    }

    template <bool async, bool exact = false>
    void DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size);

    [[nodiscard]] bool DownloadBufferMemorySnapshot(VAddr device_addr, void* output, u64 size);

    [[nodiscard]] OverlapResult ResolveOverlaps(VAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    BufferId CreateBuffer(VAddr device_addr, u32 wanted_size);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    bool SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer, bool is_registered = true);

    vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                            size_t total_size_bytes);

    bool SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size);

    void WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes);

    void TouchBuffer(const Buffer& buffer);

    void DeleteBuffer(BufferId buffer_id);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    FaultManager fault_manager;
    std::unique_ptr<MemoryTracker> memory_tracker;
    StreamBuffer staging_buffer;
    StreamBuffer stream_buffer;
    StreamBuffer download_buffer;
    StreamBuffer device_buffer;
    Buffer gds_buffer;
    Buffer bda_pagetable_buffer;
    CpuPageWriteTracker cpu_page_write_tracker;
    DmaDirtyRangeTracker dma_dirty_ranges;
    Common::SlotVector<Buffer> slot_buffers;
    u64 total_used_memory = 0;
    u64 trigger_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 gc_tick = 0;
    Common::LeastRecentlyUsedCache<BufferId, u64> lru_cache;
    RangeSet gpu_modified_ranges;
    BufferRegistrationIndex<BufferId> buffer_ranges;
    PageTable page_table;
};

} // namespace VideoCore
