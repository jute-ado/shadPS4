// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
#include "common/types.h"
#include "video_core/buffer_cache/buffer_barrier_policy.h"
#include "video_core/buffer_cache/buffer_write_range_tracker.h"
#include "core/memory.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

struct VmaAllocationInfo;

namespace VideoCore {

/// Hints and requirements for the backing memory type of a commit
enum class MemoryUsage {
    DeviceLocal, ///< Requests device local buffer.
    Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
    Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
    Stream,      ///< Requests device local host visible buffer, falling back host memory.
};

constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

struct UniqueBuffer {
    explicit UniqueBuffer(vk::Device device, VmaAllocator allocator);
    ~UniqueBuffer();

    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    UniqueBuffer(UniqueBuffer&& other)
        : allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          buffer{std::exchange(other.buffer, VK_NULL_HANDLE)} {}
    UniqueBuffer& operator=(UniqueBuffer&& other) {
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        return *this;
    }

    void Create(const vk::BufferCreateInfo& image_ci, MemoryUsage usage,
                VmaAllocationInfo* out_alloc_info);

    operator vk::Buffer() const {
        return buffer;
    }

    vk::Device device;
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Buffer buffer{};
    vk::DeviceAddress bda_addr = 0;
};

class Buffer {
public:
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    MemoryUsage usage, VAddr cpu_addr_, vk::BufferUsageFlags flags,
                    u64 size_bytes_);

    Buffer& operator=(const Buffer&) = delete;
    Buffer(const Buffer&) = delete;

    Buffer& operator=(Buffer&&) = default;
    Buffer(Buffer&&) = default;

    void IncreaseStreamScore(int score) noexcept {
        stream_score += score;
    }

    [[nodiscard]] int StreamScore() const noexcept {
        return stream_score;
    }

    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= cpu_addr && addr + size <= cpu_addr + SizeBytes();
    }

    [[nodiscard]] VAddr CpuAddr() const noexcept {
        return cpu_addr;
    }

    [[nodiscard]] u64 Offset(VAddr other_cpu_addr) const noexcept {
        return other_cpu_addr - cpu_addr;
    }

    size_t SizeBytes() const {
        return size_bytes;
    }

    void SetLRUId(u64 id) noexcept {
        lru_id = id;
    }

    u64 LRUId() const noexcept {
        return lru_id;
    }

    vk::Buffer Handle() const noexcept;

    vk::DeviceAddress BufferDeviceAddress() const noexcept;

    [[nodiscard]] u64 LastUseTick() const noexcept {
        return last_use_tick;
    }

    std::optional<vk::BufferMemoryBarrier2> GetBarrier(vk::AccessFlags2 dst_acess_mask,
                                                       vk::PipelineStageFlagBits2 dst_stage,
                                                       u32 offset = 0, u64 requested_size = 0) {
        constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite |
                                     vk::AccessFlagBits2::eShaderWrite |
                                     vk::AccessFlagBits2::eMemoryWrite;
        const bool source_access_writes = static_cast<bool>(access_mask & write_flags);
        const bool destination_access_writes = static_cast<bool>(dst_acess_mask & write_flags);
        if (offset >= size_bytes) {
            return {};
        }
        const u64 barrier_size =
            requested_size == 0 ? size_bytes - offset
                                : std::min<u64>(requested_size, size_bytes - offset);
        const bool overlaps_pending_write =
            pending_write_ranges.Transition(destination_access_writes, offset, barrier_size);
        if (!NeedsBufferBarrier(dst_acess_mask == access_mask, stage == dst_stage,
                                source_access_writes) &&
            !overlaps_pending_write) {
            return {};
        }

        const vk::PipelineStageFlags2 source_stage =
            overlaps_pending_write ? vk::PipelineStageFlagBits2::eAllCommands : stage;
        const vk::AccessFlags2 source_access =
            overlaps_pending_write ? access_mask | vk::AccessFlagBits2::eMemoryWrite : access_mask;

        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = source_stage,
            .srcAccessMask = source_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_acess_mask,
            .buffer = Handle(),
            .offset = offset,
            .size = barrier_size,
        };
        access_mask = dst_acess_mask;
        stage = dst_stage;
        return barrier;
    }

    void Fill(u64 offset, u32 num_bytes, u32 value);

public:
    VAddr cpu_addr = 0;
    bool is_picked{};
    bool is_coherent{};
    bool is_deleted{};
    int stream_score = 0;
    size_t size_bytes = 0;
    u64 lru_id = 0;
    mutable u64 last_use_tick = 0;
    BufferWriteRangeTracker pending_write_ranges;
    std::span<u8> mapped_data;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    MemoryUsage usage;
    UniqueBuffer buffer;
    vk::Flags<vk::AccessFlagBits2> access_mask{
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
    vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
};

class StreamBuffer : public Buffer {
public:
    explicit StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          MemoryUsage usage, u64 size_bytes_);

    /// Reserves a region of memory from the stream buffer.
    std::pair<u8*, u64> Map(u64 size, u64 alignment = 0, bool allow_wait = true);

    /// Ensures that reserved bytes of memory are available to the GPU.
    void Commit();

    /// Maps and commits a memory region with user provided data
    u64 Copy(auto src, size_t size, size_t alignment = 0) {
        const auto [data, offset] = Map(size, alignment);
        auto* memory = Core::Memory::Instance();
        const VAddr src_vaddr = reinterpret_cast<const VAddr>(src);
        if (memory->IsValidMapping(src_vaddr)) {
            memory->CopySparseMemory(src_vaddr, data, size);
        } else {
            std::memcpy(data, reinterpret_cast<const void*>(src), size);
        }
        Commit();
        return offset;
    }

private:
    struct Watch {
        u64 tick{};
        u64 upper_bound{};
    };

    /// Increases the amount of watches available.
    void ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size);

    /// Waits pending watches until requested upper bound.
    bool WaitPendingOperations(u64 requested_upper_bound, bool allow_wait);

private:
    u64 offset{};
    u64 mapped_size{};
    std::vector<Watch> current_watches;
    std::size_t current_watch_cursor{};
    std::optional<size_t> invalidation_mark;
    std::vector<Watch> previous_watches;
    std::size_t wait_cursor{};
    u64 wait_bound{};
};

} // namespace VideoCore
