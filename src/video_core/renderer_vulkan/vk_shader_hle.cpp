// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>

#include "shader_recompiler/info.h"
#include "video_core/buffer_cache/buffer_copy_endpoints.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"

extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Vulkan {

static constexpr u64 COPY_SHADER_HASH = 0xfefebf9f;

static bool ExecuteCopyShaderHLE(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program,
                                 Rasterizer& rasterizer) {
    auto& scheduler = rasterizer.GetScheduler();
    auto& buffer_cache = rasterizer.GetBufferCache();

    // Copy shader defines three formatted buffers as inputs: control, source, and destination.
    const auto ctl_buf_sharp = info.buffers[0].GetSharp(info);
    const auto src_buf_sharp = info.buffers[1].GetSharp(info);
    const auto dst_buf_sharp = info.buffers[2].GetSharp(info);
    const auto buf_stride = src_buf_sharp.GetStride();
    ASSERT(buf_stride == dst_buf_sharp.GetStride());

    struct CopyShaderControl {
        u32 dst_idx;
        u32 src_idx;
        u32 end;
    };
    static_assert(sizeof(CopyShaderControl) == 12);
    ASSERT(ctl_buf_sharp.GetStride() == sizeof(CopyShaderControl));
    const auto ctl_buf = reinterpret_cast<const CopyShaderControl*>(ctl_buf_sharp.base_address);

    static std::vector<vk::BufferCopy> copies;
    copies.clear();
    copies.reserve(cs_program.dim_x);

    for (u32 i = 0; i < cs_program.dim_x; i++) {
        const auto& [dst_idx, src_idx, end] = ctl_buf[i];
        const u32 local_dst_offset = dst_idx * buf_stride;
        const u32 local_src_offset = src_idx * buf_stride;
        const u32 local_size = (end + 1) * buf_stride;
        copies.emplace_back(local_src_offset, local_dst_offset, local_size);
    }

    static constexpr vk::DeviceSize MaxDistanceForMerge = 64_MB;
    struct CopyBatch {
        u32 begin;
        u32 end;
        vk::DeviceSize src_min;
        vk::DeviceSize src_max;
        vk::DeviceSize dst_min;
        vk::DeviceSize dst_max;
    };
    std::vector<CopyBatch> batches;
    batches.reserve(copies.size());

    u32 batch_start{};
    u32 batch_end{};

    while (batch_end < copies.size()) {
        // Place first copy into the current batch
        const auto& copy = copies[batch_start];
        auto src_offset_min = copy.srcOffset;
        auto src_offset_max = copy.srcOffset + copy.size;
        auto dst_offset_min = copy.dstOffset;
        auto dst_offset_max = copy.dstOffset + copy.size;

        for (++batch_end; batch_end < copies.size(); batch_end++) {
            // Compute new src and dst bounds if we were to batch this copy
            const auto& [src_offset, dst_offset, size] = copies[batch_end];
            auto new_src_offset_min = std::min(src_offset_min, src_offset);
            auto new_src_offset_max = std::max(src_offset_max, src_offset + size);
            if (new_src_offset_max - new_src_offset_min > MaxDistanceForMerge) {
                break;
            }

            auto new_dst_offset_min = std::min(dst_offset_min, dst_offset);
            auto new_dst_offset_max = std::max(dst_offset_max, dst_offset + size);
            if (new_dst_offset_max - new_dst_offset_min > MaxDistanceForMerge) {
                break;
            }

            // We can batch this copy
            src_offset_min = new_src_offset_min;
            src_offset_max = new_src_offset_max;
            dst_offset_min = new_dst_offset_min;
            dst_offset_max = new_dst_offset_max;
        }

        const VAddr source_address = src_buf_sharp.base_address + src_offset_min;
        const VAddr destination_address = dst_buf_sharp.base_address + dst_offset_min;
        const u64 source_size = src_offset_max - src_offset_min;
        const u64 destination_size = dst_offset_max - dst_offset_min;
        if (source_address < destination_address + destination_size &&
            destination_address < source_address + source_size) {
            // vkCmdCopyBuffer forbids overlapping source and destination ranges. Let the guest
            // compute shader handle this command before any HLE batch has been published.
            return false;
        }
        batches.push_back({batch_start, batch_end, src_offset_min, src_offset_max, dst_offset_min,
                           dst_offset_max});
        batch_start = batch_end;
    }

    for (const auto& batch : batches) {
        const VAddr source_address = src_buf_sharp.base_address + batch.src_min;
        const VAddr destination_address = dst_buf_sharp.base_address + batch.dst_min;
        const u32 source_size = static_cast<u32>(batch.src_max - batch.src_min);
        const u32 destination_size = static_cast<u32>(batch.dst_max - batch.dst_min);

        const bool published = VideoCore::PrepareBufferCopyBatchThenPublish(
            [&] { return buffer_cache.PrepareBufferBinding(destination_address, destination_size,
                                                           true); },
            [&] { buffer_cache.DrainReadyFaults(); },
            [&] { return buffer_cache.PrepareBufferBinding(source_address, source_size, false); },
            [&](const VideoCore::BufferCache::PreparedBufferBinding& prepared_source,
                const VideoCore::BufferCache::PreparedBufferBinding& prepared_destination) {
                VideoCore::BufferCache::BufferBindingBatch binding_batch;
                const auto source = buffer_cache.FinalizeBufferBinding(
                    prepared_source, vk::AccessFlagBits2::eTransferRead,
                    vk::PipelineStageFlagBits2::eTransfer, binding_batch);
                const auto destination = buffer_cache.FinalizeBufferBinding(
                    prepared_destination, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eTransfer, binding_batch);
                if (!source || !destination) {
                    return false;
                }

                const VideoCore::BufferCopyEndpoint source_endpoint{source->buffer,
                                                                    source->offset};
                const VideoCore::BufferCopyEndpoint destination_endpoint{destination->buffer,
                                                                         destination->offset};
                if (VideoCore::BufferCopyRangesOverlap(source_endpoint, source_size,
                                                       destination_endpoint, destination_size)) {
                    return false;
                }

                boost::container::small_vector<vk::BufferMemoryBarrier2, 16> buffer_barriers;
                buffer_cache.CommitBufferBindings(binding_batch, buffer_barriers);
                scheduler.EndRendering();

                static constexpr vk::MemoryBarrier2 READ_BARRIER{
                    .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                    .dstAccessMask = vk::AccessFlagBits2::eTransferRead |
                                     vk::AccessFlagBits2::eTransferWrite,
                };
                scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
                    .memoryBarrierCount = 1,
                    .pMemoryBarriers = &READ_BARRIER,
                    .bufferMemoryBarrierCount = static_cast<u32>(buffer_barriers.size()),
                    .pBufferMemoryBarriers = buffer_barriers.data(),
                });

                boost::container::small_vector<vk::BufferCopy, 16> vk_copies;
                vk_copies.reserve(batch.end - batch.begin);
                for (u32 index = batch.begin; index < batch.end; ++index) {
                    const auto& copy = copies[index];
                    vk_copies.emplace_back(copy.srcOffset - batch.src_min + source->offset,
                                           copy.dstOffset - batch.dst_min + destination->offset,
                                           copy.size);
                }

                LOG_TRACE(Render_Vulkan, "HLE buffer copy: src_size = {}, dst_size = {}",
                          source_size, destination_size);
                scheduler.CommandBuffer().copyBuffer(source->buffer, destination->buffer,
                                                     vk_copies);
                return true;
            });
        if (!published) {
            // No wait-capable operation follows source preparation. A failure here therefore
            // indicates a broken cache invariant rather than a recoverable HLE decline.
            UNREACHABLE_MSG("Failed to publish a prepared HLE copy batch");
        }
    }

    static constexpr vk::MemoryBarrier2 WRITE_BARRIER{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    scheduler.CommandBuffer().pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &WRITE_BARRIER});

    return true;
}

bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer) {
    switch (info.pgm_hash) {
    case COPY_SHADER_HASH:
        // Diagnostic escape hatch for comparing the guest shader against the optional fast path.
        // The fallback is the already compiled guest compute pipeline, not a title-specific path.
        if (std::getenv("SHADPS4_DISABLE_COPY_SHADER_HLE") != nullptr) {
            return false;
        }
        return ExecuteCopyShaderHLE(info, cs_program, rasterizer);
    default:
        return false;
    }
}

} // namespace Vulkan
