// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "common/alignment.h"
#include "common/debug.h"
#include "common/logging/log.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/vk_external_address_space_backing.h"
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

void BufferCache::ApplyPhysicalBackingBdaDeltas(std::span<const PhysicalBackingBdaDelta> deltas) {
    if (deltas.empty()) {
        return;
    }
    std::vector<PhysicalBackingBdaDelta> ordered{deltas.begin(), deltas.end()};
    std::ranges::sort(ordered, {}, &PhysicalBackingBdaDelta::guest_page);
    for (size_t index = 0; index < ordered.size(); ++index) {
        if ((ordered[index].guest_page & (CACHING_PAGESIZE - 1)) != 0 ||
            ordered[index].guest_page >= (1ULL << 40) ||
            (index != 0 && ordered[index - 1].guest_page == ordered[index].guest_page)) {
            return;
        }
    }

    size_t run_begin = 0;
    while (run_begin < ordered.size()) {
        size_t run_end = run_begin + 1;
        while (run_end < ordered.size() &&
               ordered[run_end].guest_page == ordered[run_end - 1].guest_page + CACHING_PAGESIZE) {
            ++run_end;
        }
        std::vector<vk::DeviceAddress> addresses;
        addresses.reserve(run_end - run_begin);
        for (size_t index = run_begin; index < run_end; ++index) {
            addresses.push_back(ordered[index].device_address.value);
        }
        const u64 page_begin = ordered[run_begin].guest_page >> CACHING_PAGEBITS;
        WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                        addresses.data(),
                        static_cast<u32>(addresses.size() * sizeof(vk::DeviceAddress)));
        run_begin = run_end;
    }
}

void BufferCache::MarkPhysicalBackingGpuDirty(VAddr device_addr, u64 size) {
    if (!physical_backing_coordinator || size == 0) {
        return;
    }
    const VAddr end = device_addr + size;
    while (device_addr < end) {
        const VAddr page = Common::AlignDown(device_addr, CACHING_PAGESIZE);
        const VAddr slice_end = std::min(end, page + CACHING_PAGESIZE);
        static_cast<void>(physical_backing_coordinator->MarkCachePageGpuDirtyForGuest(
            page, static_cast<u32>(device_addr - page), static_cast<u32>(slice_end - device_addr)));
        device_addr = slice_end;
    }
}

bool BufferCache::TransitionPhysicalBackingTexturesForBufferAccess(VAddr device_addr, u64 size) {
    if (!physical_backing_coordinator || size == 0) {
        return true;
    }
    if (device_addr > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }
    const VAddr end = device_addr + size;
    std::vector<u64> physical_pages;
    for (VAddr guest_page = Common::AlignDown(device_addr, CACHING_PAGESIZE); guest_page < end;
         guest_page += CACHING_PAGESIZE) {
        if (const auto physical =
                physical_backing_coordinator->ResolvePhysicalPageForGuest(guest_page)) {
            physical_pages.push_back(*physical);
        }
    }
    std::ranges::sort(physical_pages);
    physical_pages.erase(std::ranges::unique(physical_pages).begin(), physical_pages.end());
    if (physical_pages.empty()) {
        return true;
    }

    const auto images = texture_cache.FindPhysicalBackingImagesForPages(physical_pages);
    if (images.empty()) {
        return true;
    }
    const auto component =
        texture_cache.PlanPhysicalBackingTextureOwnershipComponent(physical_pages);
    if (!component) {
        return false;
    }
    const auto& ownership_span = component->ownership_span;
    const BufferId image_buffer_id = FindBuffer(ownership_span.base, ownership_span.size);
    Buffer& image_buffer = slot_buffers[image_buffer_id];
    if (!image_buffer.IsInBounds(ownership_span.base, ownership_span.size)) {
        return false;
    }
    std::vector<PhysicalBackingTextureBufferTransition> image_spans;
    image_spans.reserve(component->oldest_to_newest_images.size());
    std::vector<PhysicalBackingTexturePageSource> page_source_candidates;
    const std::unordered_set<u64> component_physical_pages(component->physical_pages.begin(),
                                                           component->physical_pages.end());
    for (const auto& ownership_image : component->oldest_to_newest_images) {
        const u32 image_index = ownership_image.image_index;
        const ImageId image_id{image_index};
        const Image& image = texture_cache.GetImage(image_id);
        const auto image_span = PlanPhysicalBackingTextureOwnershipSpan(image.info.guest_address,
                                                                        image.info.guest_size);
        if (!image_span || !SynchronizeBufferFromImage(image_buffer, image_id)) {
            return false;
        }
        image_spans.push_back(*image_span);
        const VAddr image_end = image_span->base + image_span->size;
        for (VAddr guest_page = image_span->base; guest_page < image_end;
             guest_page += CACHING_PAGESIZE) {
            const auto physical =
                physical_backing_coordinator->ResolvePhysicalPageForGuest(guest_page);
            if (physical && component_physical_pages.contains(*physical)) {
                page_source_candidates.push_back(
                    {*physical, guest_page, ownership_image.gpu_write_order, image_index});
            }
        }
    }

    const auto page_sources = PlanPhysicalBackingTexturePageSources(page_source_candidates);
    if (!page_sources || page_sources->size() != component->physical_pages.size()) {
        return false;
    }
    std::vector<PhysicalBackingCachePageRequest> requests;
    requests.reserve(page_sources->size());
    for (const auto& source : *page_sources) {
        const u64 target_offset = image_buffer.Offset(source.guest_page);
        if (image_buffer.BufferDeviceAddress() > std::numeric_limits<u64>::max() - target_offset) {
            return false;
        }
        requests.push_back({
            .guest_page = source.guest_page,
            .override_page_address =
                PhysicalBackingDeviceAddress{image_buffer.BufferDeviceAddress() + target_offset},
        });
    }

    std::vector<PhysicalBackingBdaDelta> publication_deltas;
    const bool transitioned =
        texture_cache.TransitionPhysicalBackingTextureOwnershipComponentForBufferAccess(
            component->oldest_to_newest_images,
            [&](std::span<const PhysicalBackingTextureToken> tokens) {
                size_t token_page_count = 0;
                for (const auto& token : tokens) {
                    if (token.physical_pages.size() >
                        std::numeric_limits<size_t>::max() - token_page_count) {
                        return false;
                    }
                    token_page_count += token.physical_pages.size();
                    for (const u64 physical_page : token.physical_pages) {
                        if (physical_backing_owner_buffers.contains(physical_page)) {
                            return false;
                        }
                    }
                }
                auto [owners_it, inserted] =
                    physical_backing_cache_pages.try_emplace(image_buffer_id);
                auto& target_owners = owners_it->second;
                target_owners.reserve(target_owners.size() + token_page_count);
                physical_backing_owner_buffers.reserve(physical_backing_owner_buffers.size() +
                                                       token_page_count);
                const auto publication =
                    physical_backing_coordinator->TransitionTexturePagesToDirtyCachePages(tokens,
                                                                                          requests);
                if (!publication) {
                    if (inserted && target_owners.empty()) {
                        physical_backing_cache_pages.erase(owners_it);
                    }
                    return false;
                }
                for (const auto& owner : publication->owners) {
                    target_owners.push_back({.guest_page = owner.guest_page, .token = owner.token});
                    const auto [owner_it, owner_inserted] = physical_backing_owner_buffers.emplace(
                        owner.token.publication.physical_offset, image_buffer_id);
                    ASSERT_MSG(owner_inserted && owner_it->second == image_buffer_id,
                               "Prevalidated texture mirror owner insertion failed");
                }
                publication_deltas = std::move(publication->deltas);
                return true;
            });
    if (!transitioned) {
        return false;
    }
    for (const auto& image_span : image_spans) {
        gpu_modified_ranges.Add(image_span.base, image_span.size);
        memory_tracker->MarkRegionAsGpuModified(image_span.base, image_span.size);
    }
    ApplyPhysicalBackingBdaDeltas(publication_deltas);
    return true;
}

bool BufferCache::TransitionAuthoritativeTextureForDmaRead(VAddr device_addr, u32 size) {
    return TransitionPhysicalBackingTexturesForBufferAccess(device_addr, size);
}

bool BufferCache::AcquirePhysicalBackingOwnersForGpuWrite(BufferId target_buffer_id,
                                                          Buffer& target_buffer, VAddr device_addr,
                                                          u64 size) {
    if (!physical_backing_coordinator || !ShouldAcquirePhysicalBackingBufferOwnership(size != 0)) {
        return true;
    }
    if (device_addr > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }
    const VAddr end = device_addr + size;
    std::vector<PhysicalBackingCachePageRequest> requests;
    for (VAddr target_page = Common::AlignDown(device_addr, CACHING_PAGESIZE); target_page < end;
         target_page += CACHING_PAGESIZE) {
        if (!physical_backing_coordinator->ResolveGuestPagePublication(target_page)) {
            continue;
        }
        requests.push_back({
            .guest_page = target_page,
            .override_page_address =
                PhysicalBackingDeviceAddress{target_buffer.BufferDeviceAddress() +
                                             target_buffer.Offset(target_page)},
        });
    }
    if (requests.empty()) {
        return true;
    }
    const auto publication = physical_backing_coordinator->AcquireCachePagesForGuests(requests);
    if (!publication) {
        return false;
    }
    auto& target_owners = physical_backing_cache_pages[target_buffer_id];
    target_owners.reserve(target_owners.size() + publication->owners.size());
    for (const auto& owner : publication->owners) {
        target_owners.push_back({.guest_page = owner.guest_page, .token = owner.token});
        const auto [owner_it, inserted] = physical_backing_owner_buffers.emplace(
            owner.token.publication.physical_offset, target_buffer_id);
        if (!inserted && owner_it->second != target_buffer_id) {
            return false;
        }
    }
    ApplyPhysicalBackingBdaDeltas(publication->deltas);

    for (VAddr target_page = Common::AlignDown(device_addr, CACHING_PAGESIZE); target_page < end;
         target_page += CACHING_PAGESIZE) {
        const auto active =
            physical_backing_coordinator->ResolveActiveCachePageForGuest(target_page);
        if (!active) {
            continue;
        }
        const u64 physical_offset = active->publication.physical_offset;
        const auto source_id_it = physical_backing_owner_buffers.find(physical_offset);
        if (source_id_it == physical_backing_owner_buffers.end()) {
            return false;
        }
        const BufferId source_buffer_id = source_id_it->second;
        const PhysicalBackingDeviceAddress target_address{target_buffer.BufferDeviceAddress() +
                                                          target_buffer.Offset(target_page)};
        const auto current_address =
            physical_backing_coordinator->ResolveGuestPagePublication(target_page);
        if (source_buffer_id == target_buffer_id && current_address == target_address) {
            continue;
        }
        if (IsBufferInvalid(source_buffer_id)) {
            return false;
        }
        auto source_owners_it = physical_backing_cache_pages.find(source_buffer_id);
        if (source_owners_it == physical_backing_cache_pages.end()) {
            return false;
        }
        auto& source_owners = source_owners_it->second;
        const auto source_owner_it =
            std::ranges::find(source_owners, *active, &PhysicalBackingCachePageOwner::token);
        if (source_owner_it == source_owners.end()) {
            return false;
        }

        Buffer& source_buffer = slot_buffers[source_buffer_id];
        const auto copy = PlanPhysicalBackingAliasMigrationCopy(
            source_buffer.CpuAddr(), source_buffer.SizeBytes(), source_owner_it->guest_page,
            target_buffer.CpuAddr(), target_buffer.SizeBytes(), target_page);
        if (!copy) {
            return false;
        }

        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
        if (auto barrier = source_buffer.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                                    vk::PipelineStageFlagBits2::eTransfer,
                                                    copy->source_offset)) {
            pre_barriers.push_back(*barrier);
        }
        if (auto barrier = target_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                                    vk::PipelineStageFlagBits2::eTransfer,
                                                    copy->destination_offset)) {
            pre_barriers.push_back(*barrier);
        }
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
            .pBufferMemoryBarriers = pre_barriers.data(),
        });
        cmdbuf.copyBuffer(source_buffer.Handle(), target_buffer.Handle(),
                          vk::BufferCopy{
                              .srcOffset = copy->source_offset,
                              .dstOffset = copy->destination_offset,
                              .size = copy->size,
                          });
        boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
        if (auto barrier = source_buffer.GetBarrier(
                vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                vk::PipelineStageFlagBits2::eAllCommands, copy->source_offset)) {
            post_barriers.push_back(*barrier);
        }
        if (auto barrier = target_buffer.GetBarrier(
                vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                vk::PipelineStageFlagBits2::eAllCommands, copy->destination_offset)) {
            post_barriers.push_back(*barrier);
        }
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
            .pBufferMemoryBarriers = post_barriers.data(),
        });

        const auto migration =
            physical_backing_coordinator->MigrateCachePageForGuest(target_page, target_address);
        if (!migration || migration->previous_token != *active) {
            return false;
        }
        source_owners.erase(source_owner_it);
        if (source_owners.empty()) {
            physical_backing_cache_pages.erase(source_owners_it);
        }
        physical_backing_cache_pages[target_buffer_id].push_back(
            {.guest_page = target_page, .token = migration->token});
        source_id_it->second = target_buffer_id;
        ApplyPhysicalBackingBdaDeltas(migration->deltas);
    }
    return true;
}

std::optional<std::vector<PhysicalBackingTextureToken>>
BufferCache::BeginPhysicalBackingTextureOverlap(VAddr device_addr, u64 size) {
    std::vector<PhysicalBackingTextureToken> tokens;
    if (!physical_backing_coordinator || size == 0) {
        return tokens;
    }
    if (!TransitionPhysicalBackingTexturesForBufferAccess(device_addr, size)) {
        return std::nullopt;
    }
    if (!RetirePhysicalBackingOwnersForCpuWrite(device_addr, size)) {
        return std::nullopt;
    }
    const VAddr end = device_addr + size;
    for (VAddr page = Common::AlignDown(device_addr, CACHING_PAGESIZE); page < end;
         page += CACHING_PAGESIZE) {
        if (!physical_backing_coordinator->ResolveGuestPagePublication(page)) {
            continue;
        }
        const auto publication =
            physical_backing_coordinator->BeginTextureOverlap(page, CACHING_PAGESIZE);
        if (!publication) {
            for (auto token_it = tokens.rbegin(); token_it != tokens.rend(); ++token_it) {
                if (const auto deltas =
                        physical_backing_coordinator->EndTextureOverlap(*token_it)) {
                    ApplyPhysicalBackingBdaDeltas(*deltas);
                }
            }
            return std::nullopt;
        }
        tokens.push_back(publication->token);
        ApplyPhysicalBackingBdaDeltas(publication->deltas);
    }
    return tokens;
}

bool BufferCache::EndPhysicalBackingTextureOverlap(
    std::span<const PhysicalBackingTextureToken> tokens) {
    if (!physical_backing_coordinator) {
        return tokens.empty();
    }
    for (auto token_it = tokens.rbegin(); token_it != tokens.rend(); ++token_it) {
        const auto deltas = physical_backing_coordinator->EndTextureOverlap(*token_it);
        if (!deltas) {
            return false;
        }
        ApplyPhysicalBackingBdaDeltas(*deltas);
    }
    return true;
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size) {
    if (physical_backing_coordinator && size != 0) {
        bool retired = false;
        liverpool->SendCommand<true>([this, device_addr, size, &retired] {
            retired = RetirePhysicalBackingOwnersForCpuWrite(device_addr, size);
        });
        if (!retired) {
            LOG_ERROR(Render_Vulkan,
                      "Physical backing owner retirement failed before CPU invalidation");
            return;
        }
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
        });
    dma_dirty_ranges.Mark(device_addr, size);
}

bool BufferCache::RetirePhysicalBackingOwnersForCpuWrite(VAddr device_addr, u64 size) {
    const VAddr end = device_addr + size;
    std::vector<BufferId> buffers;
    for (VAddr page = Common::AlignDown(device_addr, CACHING_PAGESIZE); page < end;
         page += CACHING_PAGESIZE) {
        const auto owner = physical_backing_coordinator->ResolveActiveCachePageForGuest(page);
        if (!owner) {
            continue;
        }
        const auto owner_buffer =
            physical_backing_owner_buffers.find(owner->publication.physical_offset);
        if (owner_buffer == physical_backing_owner_buffers.end()) {
            return false;
        }
        const auto cache_pages = physical_backing_cache_pages.find(owner_buffer->second);
        if (cache_pages == physical_backing_cache_pages.end() ||
            !std::ranges::any_of(cache_pages->second, [&](const auto& cache_page) {
                return cache_page.token == *owner;
            })) {
            return false;
        }
        buffers.push_back(owner_buffer->second);
    }
    std::ranges::sort(buffers);
    buffers.erase(std::unique(buffers.begin(), buffers.end()), buffers.end());
    for (const BufferId buffer_id : buffers) {
        if (!DeleteBuffer(buffer_id)) {
            return false;
        }
    }
    return true;
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write) {
    liverpool->SendCommand<true>([this, device_addr, size, is_write] {
        Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
        DownloadBufferMemory<false>(buffer, device_addr, size, is_write);
    });
}

template <bool async>
void BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
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
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
    const auto write_data = [&]() {
        auto* memory = Core::Memory::Instance();
        cpu_page_write_tracker.Discard(device_addr, size);
        for (const auto& copy : copies) {
            const VAddr copy_device_addr = buffer.CpuAddr() + copy.srcOffset;
            const u64 dst_offset = copy.dstOffset - offset;
            memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download + dst_offset,
                                    copy.size);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if constexpr (async) {
        scheduler.DeferOperation(write_data);
    } else {
        scheduler.Finish();
        write_data();
    }
}

void BufferCache::BindVertexBuffers(
    const Vulkan::GraphicsPipeline& pipeline,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->regs;
    Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.setVertexInputEXT(bindings, attributes);
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        vk::Buffer vk_buffer;
        u64 offset;

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
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        const auto [buffer, offset] = ObtainBuffer(range.base_address, size, false);
        range.vk_buffer = buffer->buffer;
        range.offset = offset;
        if (IsRegionGpuModified(range.base_address, size)) {
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }

    // Bind vertex buffers
    Vulkan::VertexInputs<vk::Buffer> host_buffers;
    Vulkan::VertexInputs<vk::DeviceSize> host_offsets;
    Vulkan::VertexInputs<vk::DeviceSize> host_sizes;
    Vulkan::VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->vk_buffer);
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = guest_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

void BufferCache::BindIndexBuffer(
    u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [vk_buffer, offset] = ObtainBuffer(index_address, index_buffer_size, false);
    if (IsRegionGpuModified(index_address, index_buffer_size)) {
        if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                 vk::PipelineStageFlagBits2::eIndexInput)) {
            barriers.emplace_back(*barrier);
        }
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindIndexBuffer(vk_buffer->Handle(), offset, index_type);
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
    if (ShouldInvalidateTextureCacheBeforeGpuBufferFill(is_gds, true)) {
        texture_cache.InvalidateMemoryFromGPU(address, num_bytes);
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
    auto& src_buffer = [&] -> const Buffer& {
        if (src_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true);
        return buffer;
    }();
    BufferId dst_buffer_id{};
    auto& dst_buffer = [&] -> Buffer& {
        if (dst_gds) {
            return gds_buffer;
        }
        ASSERT_MSG(TransitionPhysicalBackingTexturesForBufferAccess(dst, num_bytes),
                   "Failed to preserve physical texture ownership before GPU copy");
        dst_buffer_id = FindBuffer(dst, num_bytes);
        auto& buffer = slot_buffers[dst_buffer_id];
        SynchronizeBuffer(buffer, dst, num_bytes, true, true);
        ASSERT_MSG(AcquirePhysicalBackingOwnersForGpuWrite(dst_buffer_id, buffer, dst, num_bytes),
                   "Failed to acquire physical backing ownership before GPU copy");
        gpu_modified_ranges.Add(dst, num_bytes);
        MarkPhysicalBackingGpuDirty(dst, num_bytes);
        return buffer;
    }();
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.Offset(src),
        .dstOffset = dst_buffer.Offset(dst),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
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
    cmdbuf.copyBuffer(src_buffer.Handle(), dst_buffer.Handle(), region);
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
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
    if (is_written) {
        ASSERT_MSG(TransitionPhysicalBackingTexturesForBufferAccess(device_addr, size),
                   "Failed to preserve physical texture ownership before GPU write");
    }
    if (IsBufferInvalid(buffer_id)) {
        buffer_id = FindBuffer(device_addr, size);
    }
    Buffer& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        ASSERT_MSG(AcquirePhysicalBackingOwnersForGpuWrite(buffer_id, buffer, device_addr, size),
                   "Failed to acquire physical backing ownership before GPU write");
        gpu_modified_ranges.Add(device_addr, size);
        MarkPhysicalBackingGpuDirty(device_addr, size);
    }
    return {&buffer, buffer.Offset(device_addr)};
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
    if (!DeleteBuffer(overlap_id)) {
        LOG_ERROR(Render_Vulkan,
                  "Deferred overlapping buffer deletion after physical writeback failure");
    }
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

void BufferCache::Register(BufferId buffer_id) {
    static_cast<void>(ChangeRegister<true>(buffer_id));
}

bool BufferCache::Unregister(BufferId buffer_id) {
    return ChangeRegister<false>(buffer_id);
}

template <bool insert>
bool BufferCache::ChangeRegister(BufferId buffer_id) {
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
            const VAddr guest_page = (page_begin + i) << CACHING_PAGEBITS;
            const PhysicalBackingDeviceAddress buffer_page{buffer.BufferDeviceAddress() +
                                                           (i << CACHING_PAGEBITS)};
            const auto resolved =
                physical_backing_coordinator
                    ? physical_backing_coordinator->ResolveGuestPagePublication(guest_page)
                    : std::nullopt;
            bda_addrs.push_back(resolved ? resolved->value : buffer_page.value);
        }
        WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                        bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
    } else {
        const auto physical_it = physical_backing_cache_pages.find(buffer_id);
        if (physical_it != physical_backing_cache_pages.end()) {
            auto& owners = physical_it->second;
            for (auto& owner : owners) {
                if (owner.pending_writeback) {
                    continue;
                }
                const auto retirement =
                    physical_backing_coordinator->RetireOwnerForCpuWrite(owner.guest_page);
                if (!retirement) {
                    return false;
                }
                physical_backing_owner_buffers.erase(owner.token.publication.physical_offset);
                ApplyPhysicalBackingBdaDeltas(retirement->deltas);
                if (retirement->writeback) {
                    owner.pending_writeback = retirement->writeback;
                    owner.dirty_slices = retirement->dirty_slices;
                }
            }

            struct PendingCopy {
                size_t owner_index{};
                size_t destination_offset{};
                u32 size{};
            };
            std::vector<vk::BufferCopy> copies;
            std::vector<PendingCopy> pending_copies;
            u64 total_size_bytes = 0;
            for (size_t owner_index = 0; owner_index < owners.size(); ++owner_index) {
                const auto& owner = owners[owner_index];
                if (!owner.pending_writeback) {
                    continue;
                }
                for (const auto& slice : owner.dirty_slices) {
                    copies.push_back({
                        .srcOffset = owner.guest_page - buffer.CpuAddr() + slice.offset,
                        .dstOffset = total_size_bytes,
                        .size = slice.size,
                    });
                    pending_copies.push_back(
                        {owner_index, static_cast<size_t>(total_size_bytes), slice.size});
                    total_size_bytes = Common::AlignUp(total_size_bytes + slice.size, 64ULL);
                }
            }

            u8* download{};
            if (total_size_bytes != 0) {
                u64 download_offset{};
                std::tie(download, download_offset) = download_buffer.Map(total_size_bytes);
                for (auto& copy : copies) {
                    copy.dstOffset += download_offset;
                }
                download_buffer.Commit();
                scheduler.EndRendering();
                const auto cmdbuf = scheduler.CommandBuffer();
                if (const auto barrier = buffer.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                                           vk::PipelineStageFlagBits2::eTransfer)) {
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                        .bufferMemoryBarrierCount = 1,
                        .pBufferMemoryBarriers = &*barrier,
                    });
                }
                cmdbuf.copyBuffer(buffer.Handle(), download_buffer.Handle(), copies);
                scheduler.Finish();
            }

            bool all_committed = true;
            bool wrote_back = false;
            std::vector<PhysicalBackingBdaDelta> restored_deltas;
            for (size_t owner_index = 0; owner_index < owners.size(); ++owner_index) {
                auto& owner = owners[owner_index];
                if (!owner.pending_writeback) {
                    continue;
                }
                const auto restored = physical_backing_coordinator->CommitCachePageWriteback(
                    *owner.pending_writeback, [&] {
                        if (!external_address_space_backing || !download) {
                            return false;
                        }
                        size_t slice_index = 0;
                        for (const auto& copy : pending_copies) {
                            if (copy.owner_index != owner_index) {
                                continue;
                            }
                            const auto& slice = owner.dirty_slices[slice_index++];
                            if (!external_address_space_backing->TryWritePhysical(
                                    owner.pending_writeback->physical_offset + slice.offset,
                                    std::span<const u8>{download + copy.destination_offset,
                                                        copy.size})) {
                                return false;
                            }
                        }
                        return slice_index == owner.dirty_slices.size();
                    });
                if (!restored) {
                    all_committed = false;
                    continue;
                }
                wrote_back = true;
                for (const auto& delta : *restored) {
                    gpu_modified_ranges.Subtract(delta.guest_page, CACHING_PAGESIZE);
                    memory_tracker->UnmarkRegionAsGpuModified(delta.guest_page, CACHING_PAGESIZE);
                    cpu_page_write_tracker.Discard(delta.guest_page, CACHING_PAGESIZE);
                }
                restored_deltas.insert(restored_deltas.end(), restored->begin(), restored->end());
                owner.pending_writeback.reset();
                owner.dirty_slices.clear();
            }
            if (!all_committed) {
                return false;
            }
            if (wrote_back) {
                const vk::MemoryBarrier2 barrier{
                    .srcStageMask = vk::PipelineStageFlagBits2::eHost,
                    .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .dstAccessMask =
                        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                };
                scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
                    .memoryBarrierCount = 1,
                    .pMemoryBarriers = &barrier,
                });
                ApplyPhysicalBackingBdaDeltas(restored_deltas);
            }
            physical_backing_cache_pages.erase(physical_it);
        }
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        lru_cache.Free(buffer.LRUId());
        boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
        bda_addrs.reserve(size_pages);
        for (u64 page = page_begin; page != page_end; ++page) {
            const VAddr guest_page = page << CACHING_PAGEBITS;
            const auto resolved =
                physical_backing_coordinator
                    ? physical_backing_coordinator->ResolveGuestPagePublication(guest_page)
                    : std::nullopt;
            bda_addrs.push_back(resolved ? resolved->value : 0);
        }
        WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                        bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
    return true;
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
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    return SynchronizeBufferFromImage(buffer, image_id);
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, ImageId image_id) {
    Image& image = texture_cache.GetImage(image_id);
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u32 copy_size = 0;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset + mip_info.offset + mip_info.size > buffer.SizeBytes()) {
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
        copy_size += mip_info.size;
    }
    if (copy_size == 0) {
        return false;
    }
    const auto producer = PhysicalBackingTextureMirrorProducer(image.info.props.is_tiled);
    const auto producer_stage = producer == PhysicalBackingTextureProducer::ComputeShader
                                    ? vk::PipelineStageFlagBits2::eComputeShader
                                    : vk::PipelineStageFlagBits2::eCopy;
    const auto producer_access = producer == PhysicalBackingTextureProducer::ComputeShader
                                     ? vk::AccessFlagBits2::eShaderWrite
                                     : vk::AccessFlagBits2::eTransferWrite;
    scheduler.EndRendering();
    auto cmdbuf = scheduler.CommandBuffer();
    if (const auto barrier = buffer.GetBarrier(producer_access, producer_stage, 0)) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &*barrier,
        });
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset, copy_size);
    if (const auto barrier =
            buffer.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                              vk::PipelineStageFlagBits2::eAllCommands, 0)) {
        scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &*barrier,
        });
    }
    return true;
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
        if (!physical_backing_cache_pages.contains(buffer_id)) {
            DownloadBufferMemory<true>(buffer, buffer.CpuAddr(), buffer.SizeBytes(), true);
        }
        if (!DeleteBuffer(buffer_id)) {
            LOG_ERROR(
                Render_Vulkan,
                "Deferred garbage-collected buffer deletion after physical writeback failure");
        }
    };
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

bool BufferCache::DeleteBuffer(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    if (!Unregister(buffer_id)) {
        return false;
    }
    scheduler.DeferOperation([this, buffer_id] { slot_buffers.erase(buffer_id); });
    buffer.is_deleted = true;
    return true;
}

} // namespace VideoCore
