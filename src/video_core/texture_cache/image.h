// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/enum.h"
#include "common/incremental_id.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_color_scope_producer.h"
#include "video_core/texture_cache/image_depth_stencil_scope_producer.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_producer.h"
#include "video_core/texture_cache/image_subresource_lookup.h"
#include "video_core/texture_cache/image_view.h"

#include <bit>
#include <deque>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <utility>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

namespace VideoCore {

[[nodiscard]] constexpr bool IsImageWriteAccess(vk::AccessFlags2 access) noexcept {
    constexpr vk::AccessFlags2 WriteAccessMask =
        vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite |
        vk::AccessFlagBits2::eColorAttachmentWrite |
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eMemoryWrite;
    return static_cast<bool>(access & WriteAccessMask);
}

enum ImageFlagBits : u32 {
    Empty = 0,
    MaybeCpuDirty = 1 << 0, ///< The page this image is in was touched before the image address
    CpuDirty = 1 << 1,      ///< Contents have been modified from the CPU
    GpuDirty = 1 << 2, ///< Contents have been modified from the GPU (valid data in buffer cache)
    Dirty = MaybeCpuDirty | CpuDirty | GpuDirty,
    GpuModified = 1 << 3, ///< Contents have been modified from the GPU
    Registered = 1 << 6,  ///< True when the image is registered
    Picked = 1 << 7,      ///< Temporary flag to mark the image as picked
};
DECLARE_ENUM_FLAG_OPERATORS(ImageFlagBits)

struct UniqueImage {
    explicit UniqueImage() = default;
    explicit UniqueImage(vk::Device device, VmaAllocator allocator)
        : device{device}, allocator{allocator} {}
    ~UniqueImage();

    UniqueImage(const UniqueImage&) = delete;
    UniqueImage& operator=(const UniqueImage&) = delete;

    UniqueImage(UniqueImage&& other)
        : allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          image{std::exchange(other.image, VK_NULL_HANDLE)}, image_ci{std::move(other.image_ci)} {}
    UniqueImage& operator=(UniqueImage&& other) {
        image = std::exchange(other.image, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        image_ci = std::move(other.image_ci);
        return *this;
    }

    void Create(const vk::ImageCreateInfo& image_ci);

    void Destroy();

    operator vk::Image() const {
        return image;
    }

    operator bool() const {
        return image;
    }

public:
    vk::Device device{};
    VmaAllocator allocator{};
    VmaAllocation allocation{};
    vk::Image image{};
    vk::ImageCreateInfo image_ci{};
};

class BlitHelper;

enum class ImageTransferKind : u8 {
    None,
    CopyImage,
    CopyViaBuffer,
    CopyMip,
    Resolve,
    Clear,
    BackingSampleCopy,
};

enum class ImageTransferOrigin : u8 {
    Unknown,
    DepthOverlap,
    ExpandImage,
    ComputeCopy,
};

struct ImageDepthStencilProducerObservation {
    ImageColorScopeDrawKind kind{ImageColorScopeDrawKind::Unknown};
    ImageColorScopePrivateLink link{};
    SubresourceRange range{};
    u64 mutation_epoch{};
    u64 backing_generation{};
    u64 rendering_serial{};
    u64 host_rendering_instance_id{};
    u64 fragment_pgm_hash{};
    u32 user_clip_plane_mask{};
    ImageColorScopeDrawKind scope_first_depth_kind{ImageColorScopeDrawKind::Unknown};
    u64 scope_first_depth_fragment_pgm_hash{};
    u64 upstream_candidate_generation{};
    u32 scope_first_depth_element_count{};
    u32 scope_first_depth_instance_count{};
    u32 scope_first_depth_sampled_images{};
    u32 scope_first_depth_storage_writes{};
    u32 scope_depth_writer_count{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_images{};
    u32 storage_writes{};
    u32 upstream_candidate_index{};
    u32 attachment_aspect_mask{};
    u8 fragment_mrt_mask{};
    u8 guest_primitive_type{};
    u8 effective_primitive_type{};
    u8 cull_mode{};
    u8 front_face{};
    u8 depth_compare_op{};
    u8 scope_first_depth_guest_primitive_type{};
    u8 scope_first_depth_effective_primitive_type{};
    u8 scope_first_depth_cull_mode{};
    u8 scope_first_depth_front_face{};
    bool scope_first_depth_indexed{};
    bool scope_first_depth_quad_list_converted{};
    bool scope_first_depth_valid{};
    bool scope_depth_writer_count_overflow{};
    bool scope_depth_history_valid{};
    bool scope_depth_clear{};
    bool scope_stencil_clear{};
    bool indexed{};
    bool quad_list_converted{};
    bool fragment_writes_depth{};
    bool fragment_has_discard{};
    bool depth_test_enabled{};
    bool depth_bounds_test_enabled{};
    bool depth_bias_enabled{};
    bool stencil_test_enabled{};
    bool depth_write{};
    bool stencil_write{};
    bool sampled_id_alias{};
    bool sampled_backing_alias{};
    bool sampled_aspect_alias{};
    bool sampled_range_alias{};
    bool sampled_feedback_alias{};
    bool after_diagnostic_resume{};
    bool upstream_candidate_valid{};
    bool valid{};
};

// A transition receipt is a stack-local adapter over the one real GetBarriers result. Callers must
// sanitize its range and private writer fields into bounded scalar facts before retaining it past
// the sampled-image binding loop.
inline constexpr u32 ImageTransitBarrierReceiptMaxSubresources = 64;

inline constexpr u32 ImageDepthStencilAspectMask =
    static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
inline constexpr vk::PipelineStageFlags2 ImageShaderReadStages =
    vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;

[[nodiscard]] constexpr bool IsCanonicalDepthStencilReadOnlyLayout(
    u32 request_aspect_mask, vk::ImageLayout layout) noexcept {
    const bool depth = (request_aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    const bool stencil = (request_aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    if (!depth && !stencil) {
        return false;
    }
    if ((request_aspect_mask & ~ImageDepthStencilAspectMask) != 0) {
        return false;
    }
    if (depth && stencil) {
        return layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    }
    if (depth) {
        return layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
               layout == vk::ImageLayout::eDepthReadOnlyOptimal;
    }
    return layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
           layout == vk::ImageLayout::eStencilReadOnlyOptimal;
}

[[nodiscard]] constexpr bool IsCanonicalDepthStencilAttachmentLayout(
    u32 request_aspect_mask, vk::ImageLayout layout) noexcept {
    const bool depth = (request_aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    const bool stencil = (request_aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    if (!depth && !stencil || (request_aspect_mask & ~ImageDepthStencilAspectMask) != 0) {
        return false;
    }
    if (depth && stencil) {
        return layout == vk::ImageLayout::eDepthStencilAttachmentOptimal;
    }
    if (depth) {
        return layout == vk::ImageLayout::eDepthAttachmentOptimal ||
               layout == vk::ImageLayout::eDepthStencilAttachmentOptimal;
    }
    return layout == vk::ImageLayout::eStencilAttachmentOptimal ||
           layout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
           layout == vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal;
}

[[nodiscard]] constexpr bool IsCanonicalImageShaderReadRequest(
    u32 request_aspect_mask, vk::ImageLayout layout, vk::AccessFlags2 access,
    vk::PipelineStageFlags2 stages) noexcept {
    return IsCanonicalDepthStencilReadOnlyLayout(request_aspect_mask, layout) &&
           access == vk::AccessFlagBits2::eShaderRead && stages == ImageShaderReadStages;
}

struct ImageReadStateObservationSeed {
    SubresourceRange request_range{};
    u32 resource_levels{};
    u32 resource_layers{};
    u32 request_aspect_mask{};
    vk::ImageLayout destination_layout{vk::ImageLayout::eUndefined};
    vk::AccessFlags2 destination_access{vk::AccessFlagBits2::eNone};
    vk::PipelineStageFlags2 destination_stages{vk::PipelineStageFlagBits2::eNone};
    bool uniform{};
};

struct ImageReadStateSample {
    vk::PipelineStageFlags2 stages{vk::PipelineStageFlagBits2::eNone};
    vk::AccessFlags2 access{vk::AccessFlagBits2::eNone};
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
};

struct ImageReadStateObservation {
    u64 selected_subresource_mask{};
    u64 layout_readable_mask{};
    u64 access_readable_mask{};
    u64 stage_readable_mask{};
    u64 readable_mask{};
    u32 selected_subresource_count{};
    u32 layout_readable_count{};
    u32 access_readable_count{};
    u32 stage_readable_count{};
    u32 readable_count{};
    bool destination_supported{};
    bool uniform{};
    bool observed{};
    bool malformed{};
    bool overflowed{};
};

[[nodiscard]] inline ImageReadStateObservation BuildImageReadStateObservation(
    const ImageReadStateObservationSeed& seed,
    std::span<const ImageReadStateSample> states) noexcept {
    ImageReadStateObservation observation{
        .destination_supported =
            IsCanonicalImageShaderReadRequest(seed.request_aspect_mask,
                                              seed.destination_layout,
                                              seed.destination_access,
                                              seed.destination_stages),
        .uniform = seed.uniform,
        .observed = true,
    };

    const auto& request = seed.request_range;
    const bool request_base_valid = request.base.level < seed.resource_levels &&
                                    request.base.layer < seed.resource_layers;
    const bool request_extent_valid = request.extent.levels != 0 &&
                                      request.extent.layers != 0 && request_base_valid &&
                                      request.extent.levels <=
                                          seed.resource_levels - request.base.level &&
                                      request.extent.layers <=
                                          seed.resource_layers - request.base.layer;
    if (!request_extent_valid || seed.request_aspect_mask == 0) {
        observation.malformed = true;
        return observation;
    }

    const u64 selected_count = static_cast<u64>(request.extent.levels) *
                               static_cast<u64>(request.extent.layers);
    if (selected_count > ImageTransitBarrierReceiptMaxSubresources) {
        observation.selected_subresource_count = ImageTransitBarrierReceiptMaxSubresources;
        observation.selected_subresource_mask = std::numeric_limits<u64>::max();
        observation.malformed = true;
        observation.overflowed = true;
        return observation;
    }

    observation.selected_subresource_count = static_cast<u32>(selected_count);
    observation.selected_subresource_mask =
        selected_count == ImageTransitBarrierReceiptMaxSubresources
            ? std::numeric_limits<u64>::max()
            : (u64{1} << selected_count) - 1;

    const size_t expected_state_count = seed.uniform ? 1 : static_cast<size_t>(selected_count);
    if (states.size() != expected_state_count) {
        observation.malformed = true;
        return observation;
    }

    constexpr auto any_write = vk::AccessFlagBits2::eTransferWrite |
                               vk::AccessFlagBits2::eShaderWrite |
                               vk::AccessFlagBits2::eMemoryWrite |
                               vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eHostWrite;
    for (u32 index = 0; index < observation.selected_subresource_count; ++index) {
        const auto& state = states[seed.uniform ? 0 : index];
        const u64 bit = u64{1} << index;
        const bool layout_readable = state.layout == seed.destination_layout;
        const bool access_readable =
            seed.destination_access != vk::AccessFlagBits2::eNone &&
            (state.access & seed.destination_access) == seed.destination_access &&
            !static_cast<bool>(state.access & any_write);
        const bool stage_readable =
            seed.destination_stages != vk::PipelineStageFlagBits2::eNone &&
            (state.stages & seed.destination_stages) == seed.destination_stages;
        if (layout_readable) {
            observation.layout_readable_mask |= bit;
        }
        if (access_readable) {
            observation.access_readable_mask |= bit;
        }
        if (stage_readable) {
            observation.stage_readable_mask |= bit;
        }
        if (layout_readable && access_readable && stage_readable) {
            observation.readable_mask |= bit;
        }
    }

    observation.layout_readable_count =
        static_cast<u32>(std::popcount(observation.layout_readable_mask));
    observation.access_readable_count =
        static_cast<u32>(std::popcount(observation.access_readable_mask));
    observation.stage_readable_count =
        static_cast<u32>(std::popcount(observation.stage_readable_mask));
    observation.readable_count = static_cast<u32>(std::popcount(observation.readable_mask));
    return observation;
}

struct ImageTransitBarrierReceiptSeed {
    SubresourceRange request_range{};
    u32 resource_levels{};
    u32 resource_layers{};
    u32 request_aspect_mask{};
    vk::ImageLayout destination_layout{vk::ImageLayout::eUndefined};
    vk::AccessFlags2 destination_access{vk::AccessFlagBits2::eNone};
    vk::PipelineStageFlags2 destination_stages{vk::PipelineStageFlagBits2::eNone};
    ImageDepthStencilProducerObservation transition_writer{};
    u64 prior_depth_stencil_write_mask{};
    u64 already_readable_mask{};
    u64 post_readable_mask{};
    bool uniform_before{};
    bool uniform_after{};
    bool transit_invoked{};
    bool malformed{};
    bool overflowed{};
};

struct ImageTransitBarrierReceipt {
    SubresourceRange request_range{};
    ImageDepthStencilProducerObservation transition_writer{};
    u64 selected_subresource_mask{};
    u64 prior_depth_stencil_write_mask{};
    u64 already_readable_mask{};
    u64 covering_barrier_mask{};
    u64 source_depth_stencil_layout_cover_mask{};
    u64 source_depth_stencil_access_cover_mask{};
    u64 source_depth_stencil_stage_cover_mask{};
    u64 destination_layout_cover_mask{};
    u64 destination_access_cover_mask{};
    u64 destination_stage_cover_mask{};
    // Kept as derived aliases while downstream diagnostics migrate to the independent facts.
    u64 source_depth_stencil_write_cover_mask{};
    u64 destination_shader_read_cover_mask{};
    u64 post_readable_mask{};
    u32 request_aspect_mask{};
    u32 selected_subresource_count{};
    u32 prior_depth_stencil_write_count{};
    u32 already_readable_count{};
    u32 covering_barrier_count{};
    u32 source_depth_stencil_layout_cover_count{};
    u32 source_depth_stencil_access_cover_count{};
    u32 source_depth_stencil_stage_cover_count{};
    u32 destination_layout_cover_count{};
    u32 destination_access_cover_count{};
    u32 destination_stage_cover_count{};
    u32 source_depth_stencil_write_cover_count{};
    u32 destination_shader_read_cover_count{};
    u32 post_readable_count{};
    u32 barrier_count{};
    bool destination_depth_read_only{};
    bool destination_shader_read{};
    bool destination_supported{};
    bool uniform_before{};
    bool uniform_after{};
    bool transit_invoked{};
    bool submitted{};
    bool malformed{};
    bool overflowed{};
};

[[nodiscard]] inline ImageTransitBarrierReceipt BuildImageTransitBarrierReceipt(
    const ImageTransitBarrierReceiptSeed& seed,
    std::span<const vk::ImageMemoryBarrier2> barriers) noexcept {
    const bool destination_depth_read_only =
        IsCanonicalDepthStencilReadOnlyLayout(seed.request_aspect_mask,
                                              seed.destination_layout);
    const bool destination_shader_read =
        seed.destination_access == vk::AccessFlagBits2::eShaderRead;
    const bool destination_shader_stages = seed.destination_stages == ImageShaderReadStages;
    ImageTransitBarrierReceipt receipt{
        .request_range = seed.request_range,
        .transition_writer = seed.transition_writer,
        .request_aspect_mask = seed.request_aspect_mask,
        .destination_depth_read_only = destination_depth_read_only,
        .destination_shader_read = destination_shader_read,
        .destination_supported = destination_depth_read_only && destination_shader_read &&
                                 destination_shader_stages,
        .uniform_before = seed.uniform_before,
        .uniform_after = seed.uniform_after,
        .transit_invoked = seed.transit_invoked,
        .submitted = !barriers.empty(),
        .malformed = seed.malformed,
        .overflowed = seed.overflowed,
    };

    const auto& request = seed.request_range;
    const bool request_base_valid = request.base.level < seed.resource_levels &&
                                    request.base.layer < seed.resource_layers;
    const bool request_extent_valid = request.extent.levels != 0 &&
                                      request.extent.layers != 0 && request_base_valid &&
                                      request.extent.levels <=
                                          seed.resource_levels - request.base.level &&
                                      request.extent.layers <=
                                          seed.resource_layers - request.base.layer;
    if (!request_extent_valid || seed.request_aspect_mask == 0) {
        receipt.malformed = true;
        return receipt;
    }

    const u64 selected_count = static_cast<u64>(request.extent.levels) *
                               static_cast<u64>(request.extent.layers);
    if (selected_count > ImageTransitBarrierReceiptMaxSubresources) {
        receipt.selected_subresource_count = ImageTransitBarrierReceiptMaxSubresources;
        receipt.selected_subresource_mask = std::numeric_limits<u64>::max();
        receipt.malformed = true;
        receipt.overflowed = true;
        return receipt;
    }

    receipt.selected_subresource_count = static_cast<u32>(selected_count);
    receipt.selected_subresource_mask =
        selected_count == ImageTransitBarrierReceiptMaxSubresources
            ? std::numeric_limits<u64>::max()
            : (u64{1} << selected_count) - 1;

    const u64 outside_selected = (seed.prior_depth_stencil_write_mask |
                                  seed.already_readable_mask | seed.post_readable_mask) &
                                 ~receipt.selected_subresource_mask;
    receipt.prior_depth_stencil_write_mask =
        seed.prior_depth_stencil_write_mask & receipt.selected_subresource_mask;
    receipt.already_readable_mask =
        seed.already_readable_mask & receipt.selected_subresource_mask;
    receipt.post_readable_mask = seed.post_readable_mask & receipt.selected_subresource_mask;
    if (outside_selected != 0 ||
        (receipt.prior_depth_stencil_write_mask & receipt.already_readable_mask) != 0) {
        receipt.malformed = true;
    }

    if (barriers.size() > std::numeric_limits<u32>::max()) {
        receipt.barrier_count = std::numeric_limits<u32>::max();
        receipt.malformed = true;
        receipt.overflowed = true;
    } else {
        receipt.barrier_count = static_cast<u32>(barriers.size());
    }

    const auto resolve_end = [](u32 base, u32 count, u32 limit, u32 remaining,
                                u32& end) noexcept {
        if (base >= limit || count == 0) {
            return false;
        }
        if (count == remaining) {
            end = limit;
            return true;
        }
        if (count > limit - base) {
            return false;
        }
        end = base + count;
        return true;
    };

    for (const auto& barrier : barriers) {
        const auto& range = barrier.subresourceRange;
        u32 barrier_mip_end{};
        u32 barrier_layer_end{};
        if (!resolve_end(range.baseMipLevel, range.levelCount, seed.resource_levels,
                         VK_REMAINING_MIP_LEVELS, barrier_mip_end) ||
            !resolve_end(range.baseArrayLayer, range.layerCount, seed.resource_layers,
                         VK_REMAINING_ARRAY_LAYERS, barrier_layer_end)) {
            receipt.malformed = true;
            continue;
        }

        const u32 barrier_aspects =
            static_cast<u32>(static_cast<VkImageAspectFlags>(range.aspectMask));
        if (barrier_aspects == 0) {
            receipt.malformed = true;
            continue;
        }
        if ((barrier_aspects & seed.request_aspect_mask) != seed.request_aspect_mask) {
            continue;
        }

        u64 covered{};
        for (u32 mip_offset = 0; mip_offset < request.extent.levels; ++mip_offset) {
            const u32 mip = request.base.level + mip_offset;
            if (mip < range.baseMipLevel || mip >= barrier_mip_end) {
                continue;
            }
            for (u32 layer_offset = 0; layer_offset < request.extent.layers; ++layer_offset) {
                const u32 layer = request.base.layer + layer_offset;
                if (layer < range.baseArrayLayer || layer >= barrier_layer_end) {
                    continue;
                }
                const u32 dense_index = mip_offset * request.extent.layers + layer_offset;
                covered |= u64{1} << dense_index;
            }
        }
        if (covered == 0) {
            continue;
        }

        receipt.covering_barrier_mask |= covered;
        if (IsCanonicalDepthStencilAttachmentLayout(seed.request_aspect_mask,
                                                     barrier.oldLayout)) {
            receipt.source_depth_stencil_layout_cover_mask |= covered;
        }
        constexpr auto source_depth_stencil_access =
            vk::AccessFlagBits2::eDepthStencilAttachmentRead |
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        const bool source_access_covers =
            static_cast<bool>(barrier.srcAccessMask &
                              vk::AccessFlagBits2::eDepthStencilAttachmentWrite) &&
            !static_cast<bool>(barrier.srcAccessMask & ~source_depth_stencil_access);
        if (source_access_covers) {
            receipt.source_depth_stencil_access_cover_mask |= covered;
        }
        const bool source_stages_cover =
            (barrier.srcStageMask & ImageShaderReadStages) == ImageShaderReadStages;
        if (source_stages_cover) {
            receipt.source_depth_stencil_stage_cover_mask |= covered;
        }
        const bool destination_access_covers =
            barrier.dstAccessMask == seed.destination_access;
        const bool destination_stages_cover =
            (barrier.dstStageMask & seed.destination_stages) == seed.destination_stages;
        if (barrier.newLayout == seed.destination_layout) {
            receipt.destination_layout_cover_mask |= covered;
        }
        if (destination_access_covers) {
            receipt.destination_access_cover_mask |= covered;
        }
        if (destination_stages_cover) {
            receipt.destination_stage_cover_mask |= covered;
        }

        // These compatibility aliases remain the conjunctions consumed by the older report path.
        if (source_access_covers) {
            receipt.source_depth_stencil_write_cover_mask |= covered;
        }
        if (seed.destination_access != vk::AccessFlagBits2::eNone &&
            seed.destination_stages != vk::PipelineStageFlagBits2::eNone &&
            destination_access_covers && destination_stages_cover) {
            receipt.destination_shader_read_cover_mask |= covered;
        }
    }

    receipt.prior_depth_stencil_write_count =
        static_cast<u32>(std::popcount(receipt.prior_depth_stencil_write_mask));
    receipt.already_readable_count =
        static_cast<u32>(std::popcount(receipt.already_readable_mask));
    receipt.covering_barrier_count =
        static_cast<u32>(std::popcount(receipt.covering_barrier_mask));
    receipt.source_depth_stencil_layout_cover_count =
        static_cast<u32>(std::popcount(receipt.source_depth_stencil_layout_cover_mask));
    receipt.source_depth_stencil_access_cover_count =
        static_cast<u32>(std::popcount(receipt.source_depth_stencil_access_cover_mask));
    receipt.source_depth_stencil_stage_cover_count =
        static_cast<u32>(std::popcount(receipt.source_depth_stencil_stage_cover_mask));
    receipt.destination_layout_cover_count =
        static_cast<u32>(std::popcount(receipt.destination_layout_cover_mask));
    receipt.destination_access_cover_count =
        static_cast<u32>(std::popcount(receipt.destination_access_cover_mask));
    receipt.destination_stage_cover_count =
        static_cast<u32>(std::popcount(receipt.destination_stage_cover_mask));
    receipt.source_depth_stencil_write_cover_count =
        static_cast<u32>(std::popcount(receipt.source_depth_stencil_write_cover_mask));
    receipt.destination_shader_read_cover_count =
        static_cast<u32>(std::popcount(receipt.destination_shader_read_cover_mask));
    receipt.post_readable_count = static_cast<u32>(std::popcount(receipt.post_readable_mask));
    return receipt;
}

template <typename Query, typename PostObservation>
[[nodiscard]] inline auto QueryImageTransitBarriersAndBuildReceipt(
    ImageTransitBarrierReceiptSeed seed, Query&& query,
    PostObservation&& post_observation) noexcept(
    noexcept(std::forward<Query>(query)()) &&
    noexcept(std::forward<PostObservation>(post_observation)(seed))) {
    auto barriers = std::forward<Query>(query)();
    std::forward<PostObservation>(post_observation)(seed);
    const auto receipt = BuildImageTransitBarrierReceipt(
        seed, std::span<const vk::ImageMemoryBarrier2>{barriers.data(), barriers.size()});
    return std::pair{std::move(barriers), receipt};
}

struct ImageTransferObservation {
    ImageTransferKind kind{ImageTransferKind::None};
    ImageTransferOrigin origin{ImageTransferOrigin::Unknown};
    ImageProducerClass source_producer{ImageProducerClass::Unknown};
    ImageTransferKind source_transfer_kind{ImageTransferKind::None};
    ImageTransferOrigin source_transfer_origin{ImageTransferOrigin::Unknown};
    ImageProducerClass nested_source_producer{ImageProducerClass::Unknown};
    ImageTransferKind nested_source_transfer_kind{ImageTransferKind::None};
    ImageTransferOrigin nested_source_transfer_origin{ImageTransferOrigin::Unknown};
    ImageColorScopeDrawKind source_depth_stencil_kind{ImageColorScopeDrawKind::Unknown};
    u64 mutation_epoch{};
    u64 backing_generation{};
    u32 aspect_mask{};
    u32 base_mip{};
    u32 mip_count{};
    u32 base_layer{};
    u32 layer_count{};
    u32 source_samples{};
    u32 old_samples{};
    u32 new_samples{};
    u32 source_depth_stencil_element_count{};
    u32 source_depth_stencil_instance_count{};
    u32 source_depth_stencil_sampled_images{};
    u32 source_depth_stencil_storage_writes{};
    u32 source_depth_stencil_upstream_candidate_index{};
    u64 source_depth_stencil_upstream_candidate_generation{};
    bool range_exact{};
    bool request_matches_encoded{};
    bool full_xy{};
    bool source_transfer_exact{};
    bool source_identity_comparable{};
    bool source_image_changed{};
    bool source_backing_changed{};
    bool source_mutation_comparable{};
    bool source_mutation_changed{};
    bool source_is_depth{};
    bool nested_source_transfer_exact{};
    bool nested_source_is_depth{};
    bool source_depth_stencil_exact{};
    bool source_depth_stencil_indexed{};
    bool source_depth_write{};
    bool source_stencil_write{};
    bool source_depth_stencil_upstream_candidate_valid{};
    bool valid{};
};

struct ImageTransferSourceToken {
    ImageColorScopePrivateLink link{};
    u64 backing_generation{};
    u64 mutation_epoch{};

    [[nodiscard]] constexpr bool Valid() const noexcept {
        return link.Valid() && backing_generation != 0 && mutation_epoch != 0;
    }
};

[[nodiscard]] constexpr ImageTransferSourceToken RetainDiagnosticTransferSourceToken(
    ImageTransferKind kind, ImageTransferSourceToken token) noexcept {
    return kind == ImageTransferKind::CopyImage ? token : ImageTransferSourceToken{};
}

struct Image {
    Image(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler, BlitHelper& blit_helper,
          Common::SlotVector<ImageView>& slot_image_views, const ImageInfo& info);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    bool Overlaps(VAddr overlap_cpu_addr, size_t overlap_size) const noexcept {
        const VAddr overlap_end = overlap_cpu_addr + overlap_size;
        const auto image_addr = info.guest_address;
        const auto image_end = info.guest_address + info.guest_size;
        return image_addr < overlap_end && overlap_cpu_addr < image_end;
    }

    vk::Image GetImage() const {
        return backing->image.image;
    }

    bool IsTracked() {
        return track_addr != 0 && track_addr_end != 0;
    }

    bool SafeToDownload() const {
        return True(flags & ImageFlagBits::GpuModified) && False(flags & (ImageFlagBits::Dirty));
    }

    void AssociateDepth(ImageId depth_image_id, u64 depth_image_uid) {
        depth_id = depth_image_id;
        depth_uid = depth_image_uid;
    }

    void DisassociateDepth() {
        depth_id = {};
        depth_uid = {};
    }

    ImageView& FindView(const ImageViewInfo& view_info, bool ensure_guest_samples = true,
                        bool* view_owner_exact = nullptr);

    using Barriers = boost::container::small_vector<vk::ImageMemoryBarrier2, 32>;
    Barriers GetBarriers(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                         vk::PipelineStageFlags2 dst_stage,
                         std::optional<SubresourceRange> subres_range);
    [[nodiscard]] ImageReadStateObservation ObserveReadState(
        vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
        std::optional<SubresourceRange> subres_range) const noexcept;
    void Transit(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                 std::optional<SubresourceRange> range, vk::CommandBuffer cmdbuf = {},
                 ImageTransitBarrierReceipt* receipt = nullptr);
    void Upload(std::span<const vk::BufferImageCopy> upload_copies, vk::Buffer buffer, u64 offset);
    void Download(std::span<const vk::BufferImageCopy> download_copies, vk::Buffer buffer,
                  u64 offset, u64 download_size);

    void CopyImage(Image& src_image, ImageTransferOrigin origin = ImageTransferOrigin::Unknown,
                   ImageId source_id = {});
    void CopyImageWithBuffer(Image& src_image, vk::Buffer buffer, u64 offset);
    void CopyMip(Image& src_image, u32 mip, u32 slice);

    void Resolve(Image& src_image, const VideoCore::SubresourceRange& mrt0_range,
                 const VideoCore::SubresourceRange& mrt1_range);
    void Clear(const vk::ClearValue& clear_value, const VideoCore::SubresourceRange& range);

    void SetBackingSamples(u32 num_samples, bool copy_backing = true);

    [[nodiscard]] u64 EnsureDiagnosticBackingGeneration() noexcept {
        if (backing->diagnostic_generation == 0) {
            backing->diagnostic_generation = global_backing_generation.Next();
        }
        return backing->diagnostic_generation;
    }

    void MarkDiagnosticProducer(ImageProducerClass classification) noexcept {
        if (IsPpSourceProducerTrackingEnabled()) {
            diagnostic_depth_stencil = {};
            diagnostic_depth_writer_scope =
                InvalidateImageDepthWriterScope(diagnostic_depth_writer_scope);
            if (classification != ImageProducerClass::Transfer) {
                diagnostic_transfer = {};
            }
            diagnostic_producer.Mark(classification);
        }
    }

    void MarkDiagnosticDepthStencilProducer(ImageColorScopeDrawDescriptor draw,
                                              u64 rendering_serial,
                                              u64 host_rendering_instance_id,
                                              bool after_diagnostic_resume,
                                              ImageColorScopePrivateLink link,
                                              SubresourceRange range,
                                              u32 attachment_aspect_mask,
                                              bool depth_clear, bool stencil_clear,
                                              bool depth_write, bool stencil_write,
                                              bool sampled_id_alias,
                                              bool sampled_backing_alias,
                                              bool sampled_aspect_alias,
                                              bool sampled_range_alias) noexcept {
        if (!IsPpSourceProducerTrackingEnabled() || (!depth_write && !stencil_write)) {
            return;
        }
        const u64 previous_mutation_epoch = diagnostic_producer.MutationEpoch();
        const u64 backing_generation = backing ? EnsureDiagnosticBackingGeneration() : 0;
        diagnostic_transfer = {};
        diagnostic_producer.Mark(ImageProducerClass::Unknown);
        const u64 mutation_epoch = diagnostic_producer.MutationEpoch();
        diagnostic_depth_writer_scope =
            AdvanceImageDepthWriterScope(
                diagnostic_depth_writer_scope,
                {
                    .rendering_serial = rendering_serial,
                    .backing_generation = backing_generation,
                    .previous_mutation_epoch = previous_mutation_epoch,
                    .next_mutation_epoch = mutation_epoch,
                    .shape = MakeImageDepthWriterShape(draw),
                    .depth_clear = depth_clear,
                    .stencil_clear = stencil_clear,
                    .depth_write = depth_write,
                    .stencil_write = stencil_write,
                })
                .state;
        const auto& scope = diagnostic_depth_writer_scope;
        diagnostic_depth_stencil = {
            .kind = draw.kind,
            .link = link,
            .range = range,
            .mutation_epoch = mutation_epoch,
            .backing_generation = backing_generation,
            .rendering_serial = rendering_serial,
            .host_rendering_instance_id = host_rendering_instance_id,
            .fragment_pgm_hash = draw.fragment_pgm_hash,
            .user_clip_plane_mask = draw.user_clip_plane_mask,
            .scope_first_depth_kind = scope.first_shape.kind,
            .scope_first_depth_fragment_pgm_hash = scope.first_shape.fragment_pgm_hash,
            .scope_first_depth_element_count = scope.first_shape.element_count,
            .scope_first_depth_instance_count = scope.first_shape.instance_count,
            .scope_first_depth_sampled_images = scope.first_shape.sampled_images,
            .scope_first_depth_storage_writes = scope.first_shape.storage_writes,
            .scope_depth_writer_count = scope.writer_count,
            .element_count = draw.element_count,
            .instance_count = draw.instance_count,
            .sampled_images = draw.sampled_images,
            .storage_writes = draw.storage_writes,
            .attachment_aspect_mask = attachment_aspect_mask,
            .fragment_mrt_mask = draw.fragment_mrt_mask,
            .guest_primitive_type = draw.guest_primitive_type,
            .effective_primitive_type = draw.effective_primitive_type,
            .cull_mode = draw.cull_mode,
            .front_face = draw.front_face,
            .depth_compare_op = draw.depth_compare_op,
            .scope_first_depth_guest_primitive_type = scope.first_shape.guest_primitive_type,
            .scope_first_depth_effective_primitive_type =
                scope.first_shape.effective_primitive_type,
            .scope_first_depth_cull_mode = scope.first_shape.cull_mode,
            .scope_first_depth_front_face = scope.first_shape.front_face,
            .scope_first_depth_indexed = scope.first_shape.indexed,
            .scope_first_depth_quad_list_converted = scope.first_shape.quad_list_converted,
            .scope_first_depth_valid = scope.first_valid,
            .scope_depth_writer_count_overflow = scope.writer_count_overflow,
            .scope_depth_history_valid = scope.valid,
            .scope_depth_clear = scope.depth_clear,
            .scope_stencil_clear = scope.stencil_clear,
            .indexed = draw.indexed,
            .quad_list_converted = draw.quad_list_converted,
            .fragment_writes_depth = draw.fragment_writes_depth,
            .fragment_has_discard = draw.fragment_has_discard,
            .depth_test_enabled = draw.depth_test_enabled,
            .depth_bounds_test_enabled = draw.depth_bounds_test_enabled,
            .depth_bias_enabled = draw.depth_bias_enabled,
            .stencil_test_enabled = draw.stencil_test_enabled,
            .depth_write = depth_write,
            .stencil_write = stencil_write,
            .sampled_id_alias = sampled_id_alias,
            .sampled_backing_alias = sampled_backing_alias,
            .sampled_aspect_alias = sampled_aspect_alias,
            .sampled_range_alias = sampled_range_alias,
            .sampled_feedback_alias = sampled_id_alias && sampled_backing_alias &&
                                      sampled_aspect_alias && sampled_range_alias,
            .after_diagnostic_resume = after_diagnostic_resume,
            .valid = draw.kind != ImageColorScopeDrawKind::Unknown && link.Valid() &&
                     range.extent.levels != 0 && range.extent.layers != 0 &&
                     mutation_epoch != 0 && backing_generation != 0 &&
                     rendering_serial != 0 && host_rendering_instance_id != 0 &&
                     attachment_aspect_mask != 0,
        };
    }

    void TagDiagnosticDepthStencilUpstreamCandidate(u64 generation, u32 candidate_index) noexcept {
        if (!diagnostic_depth_stencil.valid || !backing ||
            generation == 0 ||
            diagnostic_depth_stencil.mutation_epoch != DiagnosticMutationEpoch() ||
            diagnostic_depth_stencil.backing_generation != EnsureDiagnosticBackingGeneration()) {
            return;
        }
        diagnostic_depth_stencil.upstream_candidate_generation = generation;
        diagnostic_depth_stencil.upstream_candidate_index = candidate_index;
        diagnostic_depth_stencil.upstream_candidate_valid = true;
    }

    void MarkDiagnosticTransfer(ImageTransferKind kind, vk::ImageAspectFlags aspect,
                                SubresourceRange range, u32 old_samples, u32 new_samples,
                                u32 source_samples = 0, bool range_exact = true,
                                bool request_matches_encoded = true,
                                bool full_xy = true) noexcept;

    [[nodiscard]] ImageTransferObservation PeekDiagnosticTransfer() const noexcept {
        return diagnostic_transfer;
    }

    [[nodiscard]] ImageTransferSourceToken PeekDiagnosticTransferSourceToken() const noexcept {
        return diagnostic_transfer_source_token;
    }

    [[nodiscard]] ImageSubresourceLookupFact
    PeekDiagnosticSubresourceLookupProvenance() const noexcept {
        return diagnostic_subresource_lookup_provenance;
    }

    void SetDiagnosticSubresourceLookupProvenance(
        const ImageSubresourceLookupFact provenance) noexcept {
        diagnostic_subresource_lookup_provenance = provenance;
    }

    [[nodiscard]] ImageDepthStencilProducerObservation
    PeekDiagnosticDepthStencilProducer() const noexcept {
        return diagnostic_depth_stencil;
    }

    [[nodiscard]] ImageProducerObservation ObserveDiagnosticProducer() noexcept {
        return diagnostic_producer.Observe();
    }

    [[nodiscard]] ImageProducerObservation ObserveDiagnosticSampledInputProducer() noexcept {
        return diagnostic_producer.ObserveSampledInput();
    }

    [[nodiscard]] ImageProducerObservation PeekDiagnosticProducer() const noexcept {
        return diagnostic_producer.Peek();
    }

    [[nodiscard]] u64 DiagnosticMutationEpoch() const noexcept {
        return diagnostic_producer.MutationEpoch();
    }

    void BeginDiagnosticColorScope(u64 scope_serial, bool clear_at_begin) noexcept {
        if (IsPpSourceProducerTrackingEnabled()) {
            diagnostic_color_scope.BeginScope(scope_serial, clear_at_begin);
        }
    }

    void MarkDiagnosticColorDraw(u64 scope_serial,
                                 ImageColorScopeDrawDescriptor descriptor) noexcept {
        if (IsPpSourceProducerTrackingEnabled()) {
            diagnostic_color_scope.MarkDraw(scope_serial, descriptor);
        }
    }

    [[nodiscard]] ImageColorScopeProducerObservation ObserveDiagnosticColorScope() const noexcept {
        return diagnostic_color_scope.Observe();
    }

    void ResetDiagnosticProducer() noexcept {
        if (IsPpSourceProducerTrackingEnabled()) {
            diagnostic_producer.Reset();
            diagnostic_color_scope.Reset();
            diagnostic_transfer = {};
            diagnostic_depth_stencil = {};
            diagnostic_depth_writer_scope =
                InvalidateImageDepthWriterScope(diagnostic_depth_writer_scope);
            diagnostic_transfer_source_uid = 0;
            diagnostic_transfer_source_backing_generation = 0;
            diagnostic_transfer_source_mutation_epoch = 0;
            diagnostic_transfer_source_token = {};
        }
    }

public:
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    BlitHelper* blit_helper;
    Common::SlotVector<ImageView>* slot_image_views;
    ImageInfo info;
    vk::ImageAspectFlags aspect_mask = vk::ImageAspectFlagBits::eColor;
    vk::SampleCountFlags supported_samples = vk::SampleCountFlagBits::e1;
    ImageFlagBits flags = ImageFlagBits::Dirty;
    VAddr track_addr = 0;
    VAddr track_addr_end = 0;
    ImageId depth_id{};
    u64 depth_uid{};

    // Resource state tracking
    vk::ImageUsageFlags usage_flags;
    vk::FormatFeatureFlags2 format_features;
    struct State {
        vk::PipelineStageFlags2 pl_stage = vk::PipelineStageFlagBits2::eAllCommands;
        vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eNone;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };
    struct BackingImage {
        UniqueImage image;
        State state;
        std::vector<State> subresource_states;
        boost::container::small_vector<ImageViewInfo, 4> image_view_infos;
        boost::container::small_vector<ImageViewId, 4> image_view_ids;
        u32 num_samples;
        u64 diagnostic_generation{};
    };
    std::deque<BackingImage> backing_images;
    BackingImage* backing{};
    boost::container::static_vector<u64, 16> mip_hashes{};
    u64 image_uid{};
    u64 lru_id{};
    u64 tick_accessed_last{};
    u64 hash{};
    ImageProducerState diagnostic_producer{};
    ImageColorScopeProducerState diagnostic_color_scope{};
    ImageTransferObservation diagnostic_transfer{};
    ImageDepthStencilProducerObservation diagnostic_depth_stencil{};
    ImageDepthWriterScopeState diagnostic_depth_writer_scope{};
    ImageSubresourceLookupFact diagnostic_subresource_lookup_provenance{};
    u64 diagnostic_transfer_source_uid{};
    u64 diagnostic_transfer_source_backing_generation{};
    u64 diagnostic_transfer_source_mutation_epoch{};
    ImageTransferSourceToken diagnostic_transfer_source_token{};

    struct {
        u32 texture : 1;
        u32 storage : 1;
        u32 render_target : 1;
        u32 depth_target : 1;
        u32 vo_surface : 1;
    } usage{};

    struct {
        u32 is_bound : 1;
        u32 is_target : 1;
        u32 needs_rebind : 1;
        u32 force_general : 1;
    } binding{};

private:
    static Common::IncrementalIdProvider<u64> global_image_uid;
    static Common::IncrementalIdProvider<u64> global_backing_generation;
};

} // namespace VideoCore
