// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <ranges>
#include "common/assert.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/texture_rejection_diagnostics.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/depth_copy_policy.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_buffer_transfer_policy.h"
#include "video_core/texture_cache/image_copy_layer_policy.h"
#include "video_core/texture_cache/image_subresource_layout.h"
#include "video_core/texture_cache/image_usage_policy.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

using namespace Vulkan;

Common::IncrementalIdProvider<u64> Image::global_image_uid{};

static std::optional<vk::ImageUsageFlags> ImageUsageFlags(const Vulkan::Instance* instance,
                                                          const ImageInfo& info) {
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc |
                                vk::ImageUsageFlagBits::eTransferDst |
                                vk::ImageUsageFlagBits::eSampled;
    if (!info.props.is_block) {
        if (info.props.is_depth) {
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
        } else {
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
            if (instance->IsAttachmentFeedbackLoopLayoutSupported()) {
                usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
            }
            // Prefer creating images with storage usage to avoid needing re-creation in case of
            // e.g. compute clears. The sample-count policy below removes this optional usage when
            // multisampled storage is unavailable. ExtendedUsage is also used.
            usage |= vk::ImageUsageFlagBits::eStorage;
        }
    } else {
        // Similarly to above, we specify storage usage. This is typically not supported by
        // compressed formats, but may be used for uncompressed views. In order to satisfy this,
        // we will also specify the extended usage bit.
        usage |= vk::ImageUsageFlagBits::eStorage;
    }

    const auto selected_usage = SelectImageUsageForSamples(
        usage, info.num_samples, info.storage_required,
        instance->IsShaderStorageImageMultisampleSupported());
    if (!selected_usage) {
        return std::nullopt;
    }
    return selected_usage->usage;
}

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return vk::ImageType::e1D;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DMsaaArray:
    case AmdGpu::ImageType::Color2DArray:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE();
    }
}

static vk::FormatFeatureFlags2 FormatFeatureFlags(const vk::ImageUsageFlags usage_flags) {
    vk::FormatFeatureFlags2 feature_flags{};
    if (usage_flags & vk::ImageUsageFlagBits::eTransferSrc) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferSrc;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eTransferDst) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferDst;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eSampled) {
        feature_flags |= vk::FormatFeatureFlagBits2::eSampledImage;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eColorAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eColorAttachment;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    }
    // Note: StorageImage is intentionally ignored for now since it is an optional broad usage and
    // can mess up compatibility checks.
    return feature_flags;
}

UniqueImage::~UniqueImage() {
    if (image) {
        vmaDestroyImage(allocator, image, allocation);
    }
}

void UniqueImage::Destroy() {
    if (image) {
        vmaDestroyImage(allocator, image, allocation);
        image = vk::Image{};
        allocation = {};
    }
}

vk::Result UniqueImage::TryCreate(const vk::ImageCreateInfo& image_ci,
                                   const Vulkan::Instance* diagnostic_instance) {
    this->image_ci = image_ci;
    ASSERT(!image);
    allocation = {};
    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkImageCreateInfo image_ci_unsafe = static_cast<VkImageCreateInfo>(image_ci);
    VkImage unsafe_image{};
    VkResult result = vmaCreateImage(allocator, &image_ci_unsafe, &alloc_info, &unsafe_image,
                                     &allocation, nullptr);
    if (result != VK_SUCCESS && diagnostic_instance != nullptr) {
        diagnostic_instance->LogDeviceMemoryBudget("image_allocation", vk::Result{result});
    }
    if (result != VK_SUCCESS && TextureRejectionDiagnosticsEnabled()) {
        // The failed allocation has no valid image handle to query for memory requirements.
        LOG_ERROR(Render_Vulkan,
                  "Image allocation diagnostic result={} extent={}x{}x{} mips={} layers={} "
                  "format={} type={} tiling={} samples={} usage={:#x} flags={:#x}; "
                  "host_allocation_bytes=unavailable",
                  vk::to_string(vk::Result{result}), image_ci.extent.width,
                  image_ci.extent.height, image_ci.extent.depth, image_ci.mipLevels,
                  image_ci.arrayLayers, vk::to_string(image_ci.format),
                  vk::to_string(image_ci.imageType), vk::to_string(image_ci.tiling),
                  static_cast<u32>(image_ci.samples), static_cast<VkImageUsageFlags>(image_ci.usage),
                  static_cast<VkImageCreateFlags>(image_ci.flags));
        if (const auto* context = ScopedTextureDiagnosticContext::Current()) {
            const auto& words = context->words;
            LOG_ERROR(Render_Vulkan,
                      "Image allocation T# context stage={} logical_stage={} shader={:#x} "
                      "sharp={} entry={} count={} stride_dw={} r128={} reasons={:#x} "
                      "base_byte_logical_gpu_mapped={} raw=[{:08x} {:08x} {:08x} {:08x} "
                      "{:08x} {:08x} {:08x} {:08x}]",
                      context->stage, context->logical_stage, context->shader_hash,
                      context->sharp_index, context->array_index, context->descriptor_count,
                      context->descriptor_stride_dw, context->is_r128, context->reasons,
                      context->base_byte_logical_gpu_mapped, words[0], words[1], words[2], words[3],
                      words[4], words[5], words[6], words[7]);
        }
    }
    if (result != VK_SUCCESS) {
        return vk::Result{result};
    }
    image = vk::Image{unsafe_image};
    return vk::Result::eSuccess;
}

void UniqueImage::Create(const vk::ImageCreateInfo& image_ci,
                         const Vulkan::Instance* diagnostic_instance) {
    const auto result = TryCreate(image_ci, diagnostic_instance);
    ASSERT_MSG(result == vk::Result::eSuccess, "Failed allocating image with error {}",
               vk::to_string(result));
}

Image::Image(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
             BlitHelper& blit_helper_, Common::SlotVector<ImageView>& slot_image_views_,
             const ImageInfo& info_)
    : instance{&instance_}, scheduler{&scheduler_}, blit_helper{&blit_helper_},
      slot_image_views{&slot_image_views_}, info{info_} {
    // Stencil placeholder images intentionally have an undefined Vulkan format, but they still
    // participate in the registered cache and therefore need a generation identity for deferred
    // unmap retirement. Allocate the UID before the early return so a reused ImageId slot cannot
    // make two placeholder generations look identical to an old callback.
    image_uid = global_image_uid.Next();
    if (info.pixel_format == vk::Format::eUndefined) {
        return;
    }
    mip_hashes.resize(info.resources.levels);
    // Here we force `eExtendedUsage` as don't know all image usage cases beforehand. In normal case
    // the texture cache should re-create the resource with the usage requested
    vk::ImageCreateFlags flags{vk::ImageCreateFlagBits::eMutableFormat |
                               vk::ImageCreateFlagBits::eExtendedUsage};
    if (info.props.is_volume) {
        flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
        if (instance->Is2dViewOf3dSupported()) {
            flags |= vk::ImageCreateFlagBits::e2DViewCompatibleEXT;
        }
    }
    if (info.props.is_block && instance->IsBlockTexelViewSupported()) {
        flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
    }

    const auto preferred_usage = ImageUsageFlags(instance, info);
    if (!preferred_usage) {
        LOG_ERROR(Render_Vulkan,
                  "Rejecting image requiring unsupported multisampled storage usage: format={} "
                  "type={:#x} samples={}",
                  vk::to_string(info.pixel_format), static_cast<u32>(info.type), info.num_samples);
        usage_flags = {};
        supported_samples = {};
        return;
    }
    format_features = FormatFeatureFlags(*preferred_usage);
    if (info.props.is_depth) {
        aspect_mask = vk::ImageAspectFlagBits::eDepth;
        if (info.props.has_stencil) {
            aspect_mask |= vk::ImageAspectFlagBits::eStencil;
        }
    }

    constexpr auto tiling = vk::ImageTiling::eOptimal;
    const auto supported_format = instance->GetSupportedFormat(info.pixel_format, format_features);
    vk::PhysicalDeviceImageFormatInfo2 format_info{
        .format = supported_format,
        .type = ConvertImageType(info.type),
        .tiling = tiling,
        .usage = *preferred_usage,
        .flags = flags,
    };
    vk::ImageFormatProperties2 supported_properties{};
    const auto selected_usage = SelectSupportedImageUsage(
        info.props.is_block && !info.storage_required, *preferred_usage,
        [&](const vk::ImageUsageFlags candidate_usage) {
            format_info.usage = candidate_usage;
            const auto properties =
                instance->GetPhysicalDevice().getImageFormatProperties2(format_info);
            if (properties.result != vk::Result::eSuccess) {
                return false;
            }
            supported_properties = properties.value;
            return true;
        });
    if (!selected_usage) {
        LOG_ERROR(Render_Vulkan, "image format {} type {} is not supported (flags {}, usage {})",
                  vk::to_string(supported_format), vk::to_string(format_info.type),
                  vk::to_string(format_info.flags), vk::to_string(*preferred_usage));
        // Do not pass an unsupported format/usage combination to vkCreateImage.  A failed
        // capability query is a guest-descriptor admission failure, not permission to construct
        // an image with the rejected usage.  TextureCache removes this backing-less candidate and
        // callers bind a null descriptor or drop the command at their boundary.
        usage_flags = {};
        supported_samples = {};
        return;
    } else {
        usage_flags = selected_usage->usage;
        supported_samples = supported_properties.imageFormatProperties.sampleCounts;
        if (selected_usage->dropped_optional_storage) {
            LOG_WARNING(Render_Vulkan,
                        "Image format {} does not support optional storage usage; using sampled "
                        "compressed image compatibility path",
                        vk::to_string(supported_format));
        }
    }

    const u32 max_array_layers =
        supported_properties.imageFormatProperties.maxArrayLayers;
    if (!IsHostImageArrayLayerCountSupported(info.resources.layers, max_array_layers)) {
        LOG_ERROR(Render_Vulkan,
                  "image format {} cannot represent {} array layers (host maximum {}); "
                  "discarding guest image before creation",
                  vk::to_string(supported_format), info.resources.layers, max_array_layers);
        usage_flags = {};
        supported_samples = {};
        return;
    }

    const auto& host_limits = supported_properties.imageFormatProperties;
    const Extent3D max_extent{
        .width = host_limits.maxExtent.width,
        .height = host_limits.maxExtent.height,
        .depth = host_limits.maxExtent.depth,
    };
    if (!IsHostImageGeometrySupported(info.size, info.resources.levels, max_extent,
                                      host_limits.maxMipLevels)) {
        LOG_ERROR(Render_Vulkan,
                  "image format {} cannot represent extent {}x{}x{} with {} mips (host maximum "
                  "{}x{}x{}, {} mips); discarding guest image before creation",
                  vk::to_string(supported_format), info.size.width, info.size.height,
                  info.size.depth, info.resources.levels, max_extent.width, max_extent.height,
                  max_extent.depth, host_limits.maxMipLevels);
        usage_flags = {};
        supported_samples = {};
        return;
    }

    const vk::ImageCreateInfo image_ci = {
        .flags = flags,
        .imageType = ConvertImageType(info.type),
        .format = supported_format,
        .extent{
            .width = info.size.width,
            .height = info.size.height,
            .depth = info.size.depth,
        },
        .mipLevels = static_cast<u32>(info.resources.levels),
        .arrayLayers = static_cast<u32>(info.resources.layers),
        .samples = LiverpoolToVK::NumSamples(info.num_samples, supported_samples),
        .tiling = tiling,
        .usage = usage_flags,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    backing = &backing_images.emplace_back();
    backing->num_samples = info.num_samples;
    backing->image = UniqueImage{instance->GetDevice(), instance->GetAllocator()};
    if (backing->image.TryCreate(image_ci, instance) != vk::Result::eSuccess) {
        backing_images.pop_back();
        backing = nullptr;
        usage_flags = {};
        supported_samples = {};
        return;
    }

    Vulkan::SetObjectName(instance->GetDevice(), GetImage(),
                          "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{}", info.size.width,
                          info.size.height, info.size.depth, AmdGpu::NameOf(info.tile_mode),
                          vk::to_string(info.pixel_format), info.guest_address, info.guest_size,
                          info.resources.layers, info.resources.levels, info.num_samples);
}

Image::~Image() = default;

bool Image::CanHostScale(const u32 scale_percent, const u32 display_width,
                         const u32 display_height) const {
    if (!IsValid()) {
        return false;
    }
    const bool supports_blit = instance->IsFormatSupported(
        backing->image.image_ci.format,
        vk::FormatFeatureFlagBits2::eBlitSrc | vk::FormatFeatureFlagBits2::eBlitDst);
    const auto plan = PlanHostImageScale({
        .width = info.size.width,
        .height = info.size.height,
        .display_width = display_width,
        .display_height = display_height,
        .scale_percent = scale_percent,
        .levels = info.resources.levels,
        .layers = info.resources.layers,
        .samples = info.num_samples,
        .is_2d = ConvertImageType(info.type) == vk::ImageType::e2D,
        .is_block_compressed = bool(info.props.is_block),
        .supports_blit = supports_blit,
        .is_render_target = bool(usage.render_target),
        .is_depth_target = bool(usage.depth_target),
    });
    return plan.disposition == HostImageScaleDisposition::Identity || plan.IsScaled();
}

bool Image::SetHostScale(const u32 scale_percent, const u32 display_width,
                         const u32 display_height) {
    if (!IsValid()) {
        return false;
    }
    const bool supports_blit = instance->IsFormatSupported(
        backing->image.image_ci.format,
        vk::FormatFeatureFlagBits2::eBlitSrc | vk::FormatFeatureFlagBits2::eBlitDst);
    const auto plan = PlanHostImageScale({
        .width = info.size.width,
        .height = info.size.height,
        .display_width = display_width,
        .display_height = display_height,
        .scale_percent = scale_percent,
        .levels = info.resources.levels,
        .layers = info.resources.layers,
        .samples = info.num_samples,
        .is_2d = ConvertImageType(info.type) == vk::ImageType::e2D,
        .is_block_compressed = bool(info.props.is_block),
        .supports_blit = supports_blit,
        .is_render_target = bool(usage.render_target),
        .is_depth_target = bool(usage.depth_target),
    });
    if (plan.disposition != HostImageScaleDisposition::Identity && !plan.IsScaled()) {
        return false;
    }
    const u32 requested_scale = plan.IsScaled() ? plan.numerator : 1u;
    if (GetHostScale() == requested_scale) {
        return true;
    }

    scheduler->EndRendering();
    auto new_image_ci = backing->image.image_ci;
    new_image_ci.extent.width = plan.host_width;
    new_image_ci.extent.height = plan.host_height;

    auto* old_backing = backing;
    const auto matching_backing = std::ranges::find_if(backing_images, [&](const auto& candidate) {
        return std::addressof(candidate) != old_backing &&
               candidate.num_samples == old_backing->num_samples &&
               candidate.image.image_ci.extent == new_image_ci.extent;
    });
    BackingImage* new_backing{};
    if (matching_backing != backing_images.end()) {
        new_backing = std::addressof(*matching_backing);
    } else {
        new_backing = &backing_images.emplace_back();
        new_backing->num_samples = old_backing->num_samples;
        new_backing->image = UniqueImage{instance->GetDevice(), instance->GetAllocator()};
        if (new_backing->image.TryCreate(new_image_ci, instance) != vk::Result::eSuccess) {
            backing_images.pop_back();
            return false;
        }
    }

    const auto old_extent = old_backing->image.image_ci.extent;
    const auto new_extent = new_backing->image.image_ci.extent;
    auto barriers =
        GetBarriers(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                    vk::PipelineStageFlagBits2::eCopy, std::nullopt);
    backing = new_backing;
    const auto destination_barriers =
        GetBarriers(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eCopy, std::nullopt);
    backing = old_backing;
    barriers.insert(barriers.end(), destination_barriers.begin(), destination_barriers.end());
    const auto cmdbuf = scheduler->CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });

    boost::container::small_vector<vk::ImageBlit, 2> blits;
    const auto add_blit = [&](const vk::ImageAspectFlagBits aspect) {
        if (aspect_mask & aspect) {
            blits.push_back(vk::ImageBlit{
                .srcSubresource{
                    .aspectMask = aspect,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = info.resources.layers,
                },
                .srcOffsets =
                    std::array{
                        vk::Offset3D{0, 0, 0},
                        vk::Offset3D{static_cast<s32>(old_extent.width),
                                     static_cast<s32>(old_extent.height), 1},
                    },
                .dstSubresource{
                    .aspectMask = aspect,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = info.resources.layers,
                },
                .dstOffsets =
                    std::array{
                        vk::Offset3D{0, 0, 0},
                        vk::Offset3D{static_cast<s32>(new_extent.width),
                                     static_cast<s32>(new_extent.height), 1},
                    },
            });
        }
    };
    add_blit(vk::ImageAspectFlagBits::eColor);
    add_blit(vk::ImageAspectFlagBits::eDepth);
    add_blit(vk::ImageAspectFlagBits::eStencil);
    cmdbuf.blitImage(old_backing->image.image, vk::ImageLayout::eTransferSrcOptimal,
                     new_backing->image.image, vk::ImageLayout::eTransferDstOptimal, blits,
                     vk::Filter::eNearest);

    backing = new_backing;
    host_scale_plan = plan;
    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
    if (plan.IsScaled() && !host_scale_logged) {
        LOG_INFO(Render_Vulkan, "Internal resolution promoted {}x{} to {}x{} ({})", info.size.width,
                 info.size.height, new_extent.width, new_extent.height,
                 vk::to_string(new_image_ci.format));
        host_scale_logged = true;
    }
    Vulkan::SetObjectName(instance->GetDevice(), GetImage(),
                          "Image host-scaled {}x{} from {}x{} {:#x}:{:#x}", new_extent.width,
                          new_extent.height, info.size.width, info.size.height, info.guest_address,
                          info.guest_size);
    return true;
}

ImageView& Image::FindView(const ImageViewInfo& view_info, bool ensure_guest_samples) {
    if (ensure_guest_samples && backing->num_samples > 1 != info.num_samples > 1) {
        SetBackingSamples(info.num_samples);
    }
    const auto& view_infos = backing->image_view_infos;
    const auto it = std::ranges::find(view_infos, view_info);
    if (it != view_infos.end()) {
        const auto view_id = backing->image_view_ids[std::distance(view_infos.begin(), it)];
        return (*slot_image_views)[view_id];
    }
    const auto view_id = slot_image_views->insert(*instance, view_info, *this);
    backing->image_view_infos.emplace_back(view_info);
    backing->image_view_ids.emplace_back(view_id);
    return (*slot_image_views)[view_id];
}

Image::Barriers Image::GetBarriers(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                                   vk::PipelineStageFlags2 dst_stage,
                                   std::optional<SubresourceRange> subres_range) {
    if (!IsValid()) {
        return {};
    }
    if (subres_range && !IsValidSubresourceRange(*subres_range, info.resources)) {
        LOG_ERROR(Render_Vulkan,
                  "Rejecting out-of-range image barrier: base={}+{} extent={}x{} available={}x{}",
                  subres_range->base.level, subres_range->base.layer,
                  subres_range->extent.levels, subres_range->extent.layers,
                  info.resources.levels, info.resources.layers);
        return {};
    }
    auto& last_state = backing->state;
    auto& subresource_states = backing->subresource_states;

    const bool needs_partial_transition =
        subres_range &&
        (subres_range->base != SubresourceBase{} || subres_range->extent != info.resources);
    const bool partially_transited = !subresource_states.empty();

    Barriers barriers;
    if (needs_partial_transition || partially_transited) {
        if (!partially_transited) {
            subresource_states.resize(info.resources.levels * info.resources.layers);
            std::fill(subresource_states.begin(), subresource_states.end(), last_state);
        }

        // In case of partial transition, we need to change the specified subresources only.
        // Otherwise all subresources need to be set to the same state so we can use a full
        // resource transition for the next time.
        const auto mips =
            needs_partial_transition
                ? std::ranges::views::iota(subres_range->base.level,
                                           subres_range->base.level + subres_range->extent.levels)
                : std::views::iota(0u, info.resources.levels);
        const auto layers =
            needs_partial_transition
                ? std::ranges::views::iota(subres_range->base.layer,
                                           subres_range->base.layer + subres_range->extent.layers)
                : std::views::iota(0u, info.resources.layers);

        for (u32 mip : mips) {
            for (u32 layer : layers) {
                // NOTE: these loops may produce a lot of small barriers.
                // If this becomes a problem, we can optimize it by merging adjacent barriers.
                const auto subres_idx = mip * info.resources.layers + layer;
                ASSERT(subres_idx < subresource_states.size());
                auto& state = subresource_states[subres_idx];

                constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite |
                                             vk::AccessFlagBits2::eShaderWrite |
                                             vk::AccessFlagBits2::eMemoryWrite;
                const bool is_write = static_cast<bool>(state.access_mask & write_flags);
                if (state.layout != dst_layout || state.access_mask != dst_mask || is_write) {
                    barriers.emplace_back(vk::ImageMemoryBarrier2{
                        .srcStageMask = state.pl_stage,
                        .srcAccessMask = state.access_mask,
                        .dstStageMask = dst_stage,
                        .dstAccessMask = dst_mask,
                        .oldLayout = state.layout,
                        .newLayout = dst_layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = GetImage(),
                        .subresourceRange{
                            .aspectMask = aspect_mask,
                            .baseMipLevel = mip,
                            .levelCount = 1,
                            .baseArrayLayer = layer,
                            .layerCount = 1,
                        },
                    });
                    state.layout = dst_layout;
                    state.access_mask = dst_mask;
                    state.pl_stage = dst_stage;
                }
            }
        }

        if (!needs_partial_transition) {
            subresource_states.clear();
        }
    } else { // Full resource transition
        constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite |
                                     vk::AccessFlagBits2::eShaderWrite |
                                     vk::AccessFlagBits2::eMemoryWrite;
        const bool is_write = static_cast<bool>(last_state.access_mask & write_flags);
        if (last_state.layout == dst_layout && last_state.access_mask == dst_mask && !is_write) {
            return {};
        }

        barriers.emplace_back(vk::ImageMemoryBarrier2{
            .srcStageMask = last_state.pl_stage,
            .srcAccessMask = last_state.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = last_state.layout,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = GetImage(),
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        });
    }

    last_state.layout = dst_layout;
    last_state.access_mask = dst_mask;
    last_state.pl_stage = dst_stage;

    return barriers;
}

std::optional<vk::ImageLayout> Image::GetSubresourceLayout(const SubresourceRange& range) const {
    if (!IsValid()) {
        return std::nullopt;
    }
    return ResolveSubresourceLayout(backing->state.layout,
                                    std::span<const State>{backing->subresource_states},
                                    info.resources, range);
}

void Image::Transit(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                    std::optional<SubresourceRange> range, vk::CommandBuffer cmdbuf /*= {}*/) {
    // Adjust pipeline stage
    const vk::PipelineStageFlags2 dst_pl_stage =
        (dst_mask == vk::AccessFlagBits2::eTransferRead ||
         dst_mask == vk::AccessFlagBits2::eTransferWrite)
            ? vk::PipelineStageFlagBits2::eTransfer
            : vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;

    const auto barriers = GetBarriers(dst_layout, dst_mask, dst_pl_stage, range);
    if (barriers.empty()) {
        return;
    }

    if (!cmdbuf) {
        // When using external cmdbuf you are responsible for ending rp.
        scheduler->EndRendering();
        cmdbuf = scheduler->CommandBuffer();
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });
}

void Image::Upload(std::span<const vk::BufferImageCopy> upload_copies, vk::Buffer buffer,
                   u64 offset) {
    if (!SupportsImageBufferTransfer(info.num_samples)) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping image buffer upload for multisampled image ({} samples)",
                    info.num_samples);
        return;
    }
    if (!IsValid()) {
        LOG_WARNING(Render_Vulkan, "Skipping image buffer upload for invalid image backing");
        return;
    }
    if (backing->num_samples != info.num_samples && info.props.is_depth) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping image buffer upload because the depth backing is multisampled "
                    "({} samples)",
                    backing->num_samples);
        return;
    }
    ASSERT(SetHostScale(100));
    SetBackingSamples(info.num_samples, false);
    if (!backing || backing->num_samples != 1) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping image buffer upload because the backing remains multisampled "
                    "({} samples)",
                    backing ? backing->num_samples : 0);
        return;
    }
    scheduler->EndRendering();

    const vk::BufferMemoryBarrier2 pre_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = buffer,
        .offset = offset,
        .size = info.guest_size,
    };
    const vk::BufferMemoryBarrier2 post_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer,
        .offset = offset,
        .size = info.guest_size,
    };
    const auto image_barriers =
        GetBarriers(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eCopy, {});
    const auto cmdbuf = scheduler->CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
        .imageMemoryBarrierCount = static_cast<u32>(image_barriers.size()),
        .pImageMemoryBarriers = image_barriers.data(),
    });
    cmdbuf.copyBufferToImage(buffer, GetImage(), vk::ImageLayout::eTransferDstOptimal,
                             upload_copies);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
    flags &= ~ImageFlagBits::Dirty;
}

void Image::Download(std::span<const vk::BufferImageCopy> download_copies, vk::Buffer buffer,
                     u64 offset, u64 download_size) {
    if (!SupportsImageBufferTransfer(info.num_samples)) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping image buffer download for multisampled image ({} samples)",
                    info.num_samples);
        return;
    }
    if (!IsValid()) {
        LOG_WARNING(Render_Vulkan, "Skipping image buffer download for invalid image backing");
        return;
    }
    if (backing->num_samples != info.num_samples &&
        (info.props.is_depth || info.resources.levels != 1 || info.resources.layers != 1)) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping image buffer download because the backing cannot be resolved for "
                    "buffer transfer (samples={}, levels={}, layers={})",
                    backing->num_samples, info.resources.levels, info.resources.layers);
        return;
    }
    ASSERT(SetHostScale(100));
    SetBackingSamples(info.num_samples);
    if (!backing || backing->num_samples != 1) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping image buffer download because the backing remains multisampled "
                    "({} samples)",
                    backing ? backing->num_samples : 0);
        return;
    }
    scheduler->EndRendering();

    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer,
        .offset = offset,
        .size = download_size,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .buffer = buffer,
        .offset = offset,
        .size = download_size,
    };
    const auto image_barriers =
        GetBarriers(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                    vk::PipelineStageFlagBits2::eCopy, {});
    auto cmdbuf = scheduler->CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
        .imageMemoryBarrierCount = static_cast<u32>(image_barriers.size()),
        .pImageMemoryBarriers = image_barriers.data(),
    });
    cmdbuf.copyImageToBuffer(GetImage(), vk::ImageLayout::eTransferSrcOptimal, buffer,
                             download_copies);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

static std::pair<u32, u32> SanitizeCopyLayers(const ImageInfo& src_info, const ImageInfo& dst_info,
                                              const u32 depth) {
    const auto vk_src_type = ConvertImageType(src_info.type);
    const auto vk_dst_type = ConvertImageType(dst_info.type);

    u32 src_layers = src_info.resources.layers;
    u32 dst_layers = dst_info.resources.layers;

    // 3D images can only use 1 layer.
    if (vk_src_type == vk::ImageType::e3D && src_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D source layers {} to 1.", src_layers);
        src_layers = 1;
    }
    if (vk_dst_type == vk::ImageType::e3D && dst_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D destination layers {} to 1.", dst_layers);
        dst_layers = 1;
    }

    // If the image type is equal, layer count must match. Take the minimum of both.
    if (vk_src_type == vk_dst_type) {
        if (src_layers != dst_layers) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy source layers {} and destination layers {} to minimum.",
                        src_layers, dst_layers);
            src_layers = dst_layers = std::min(src_layers, dst_layers);
        }
    } else {
        // For 2D <-> 3D copies, 2D layer count must equal 3D depth.
        if (vk_src_type == vk::ImageType::e2D && vk_dst_type == vk::ImageType::e3D &&
            src_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D source layers {} to 3D destination depth {}", src_layers,
                        depth);
            src_layers = depth;
        }
        if (vk_src_type == vk::ImageType::e3D && vk_dst_type == vk::ImageType::e2D &&
            dst_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D destination layers {} to 3D source depth {}", dst_layers,
                        depth);
            dst_layers = depth;
        }
    }

    return std::make_pair(src_layers, dst_layers);
}

static bool CopyDepthRegionsWithBuffer(Image& destination, Image& source,
                                       std::span<const vk::ImageCopy> regions) {
    if (!source.IsValid() || !destination.IsValid() ||
        !SupportsImageBufferRoundTrip(source.backing->num_samples,
                                      destination.backing->num_samples)) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping depth image buffer round trip because an image is invalid or "
                    "multisampled");
        return false;
    }
    const auto plan = BuildDepthCopyBufferPlan(source.backing->image.image_ci.format,
                                               destination.backing->image.image_ci.format, regions);
    if (!plan) {
        LOG_ERROR(Render_Vulkan, "Rejecting invalid depth-plane buffer copy geometry");
        return false;
    }

    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = plan->byte_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    const VmaAllocationCreateInfo allocation_ci{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    const auto allocator = destination.instance->GetAllocator();
    VkBuffer buffer{};
    VmaAllocation allocation{};
    const auto result =
        vmaCreateBuffer(allocator, &buffer_ci, &allocation_ci, &buffer, &allocation, nullptr);
    if (result != VK_SUCCESS) {
        destination.instance->LogDeviceMemoryBudget("depth_copy_allocation", vk::Result{result});
        LOG_ERROR(Render_Vulkan, "Failed to allocate depth-plane copy buffer: {}",
                  vk::to_string(vk::Result{result}));
        return false;
    }

    auto& scheduler = *destination.scheduler;
    scheduler.EndRendering();
    source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    destination.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        {});
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImageToBuffer(source.GetImage(), vk::ImageLayout::eTransferSrcOptimal, buffer,
                             plan->sources);
    const vk::BufferMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = 0,
        .size = plan->byte_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &barrier,
    });
    cmdbuf.copyBufferToImage(buffer, destination.GetImage(), vk::ImageLayout::eTransferDstOptimal,
                             plan->destinations);
    destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, {});
    // The copy buffer is GPU-owned until the command buffer's submission has completed.
    // Capture the allocator directly: image eviction must not invalidate a deferred `this`.
    scheduler.DeferOperation(
        [allocator, buffer, allocation] { vmaDestroyBuffer(allocator, buffer, allocation); });
    return true;
}

void Image::CopyImage(Image& src_image) {
    if (!IsValid() || !src_image.IsValid()) {
        LOG_WARNING(Render_Vulkan, "Skipping image copy for invalid image backing");
        return;
    }
    if (GetHostScale() != src_image.GetHostScale()) {
        ASSERT(SetHostScale(100));
        ASSERT(src_image.SetHostScale(100));
    }
    const auto& src_info = src_image.info;

    const u32 num_mips = std::min(src_info.resources.levels, info.resources.levels);

    // Format mismatch warning (safe but useful)
    if (src_info.pixel_format != info.pixel_format) {
        LOG_DEBUG(Render_Vulkan,
                  "Copy between different formats: src={}, dst={}. "
                  "Result may be undefined.",
                  vk::to_string(src_info.pixel_format), vk::to_string(info.pixel_format));
    }

    const auto src_host_extent = src_image.GetHostExtent();
    const u32 base_width = src_host_extent.width;
    const u32 base_height = src_host_extent.height;
    const u32 base_depth =
        info.type == AmdGpu::ImageType::Color3D ? info.size.depth : src_info.size.depth;

    // Match sample count before copying
    SetBackingSamples(info.num_samples, false);
    src_image.SetBackingSamples(src_info.num_samples);

    boost::container::small_vector<vk::ImageCopy, 8> regions;

    const vk::ImageAspectFlags src_aspect =
        src_image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil;

    const vk::ImageAspectFlags dst_aspect = aspect_mask & ~vk::ImageAspectFlagBits::eStencil;

    const bool src_is_2d = ConvertImageType(src_info.type) == vk::ImageType::e2D;
    const bool src_is_3d = ConvertImageType(src_info.type) == vk::ImageType::e3D;

    const bool dst_is_2d = ConvertImageType(info.type) == vk::ImageType::e2D;
    const bool dst_is_3d = ConvertImageType(info.type) == vk::ImageType::e3D;

    const bool is_2d_to_3d = src_is_2d && dst_is_3d;
    const bool is_3d_to_2d = src_is_3d && dst_is_2d;
    const bool is_same_type = !is_2d_to_3d && !is_3d_to_2d;

    for (u32 mip = 0; mip < num_mips; ++mip) {
        const u32 mip_w = std::max(base_width >> mip, 1u);
        const u32 mip_h = std::max(base_height >> mip, 1u);
        const u32 mip_d = std::max(base_depth >> mip, 1u);

        auto [src_layers, dst_layers] = SanitizeCopyLayers(src_info, info, mip_d);

        vk::ImageCopy region{};

        region.srcSubresource.aspectMask = src_aspect;
        region.srcSubresource.mipLevel = mip;
        region.srcSubresource.baseArrayLayer = 0;

        region.dstSubresource.aspectMask = dst_aspect;
        region.dstSubresource.mipLevel = mip;
        region.dstSubresource.baseArrayLayer = 0;

        if (is_same_type) {
            // 2D->2D OR 3D->3D
            if (src_is_3d) {
                // 3D images must use layerCount=1
                region.srcSubresource.layerCount = 1;
                region.dstSubresource.layerCount = 1;
                region.extent = vk::Extent3D(mip_w, mip_h, mip_d);
            } else {
                // Array images
                const u32 copy_layers = std::min(src_layers, dst_layers);
                region.srcSubresource.layerCount = copy_layers;
                region.dstSubresource.layerCount = copy_layers;
                region.extent = vk::Extent3D(mip_w, mip_h, 1);
            }
        } else if (is_2d_to_3d) {
            // 2D array -> 3D volume
            region.srcSubresource.layerCount = src_layers;
            region.dstSubresource.layerCount = 1;
            region.extent = vk::Extent3D(mip_w, mip_h, src_layers);
        } else if (is_3d_to_2d) {
            // 3D volume -> 2D array
            region.srcSubresource.layerCount = 1;
            region.dstSubresource.layerCount = dst_layers;
            region.extent = vk::Extent3D(mip_w, mip_h, dst_layers);
        }

        regions.push_back(region);
    }

    const auto depth_copy = SelectDepthCopyMode(
        src_image.backing->image.image_ci.format, backing->image.image_ci.format,
        src_image.backing->num_samples, backing->num_samples);
    if (depth_copy != DepthCopyMode::Direct) {
        if (depth_copy == DepthCopyMode::Buffer) {
            CopyDepthRegionsWithBuffer(*this, src_image, regions);
        } else {
            LOG_ERROR(Render_Vulkan,
                      "Unsupported depth image copy: {} ({} samples) to {} ({} samples)",
                      vk::to_string(src_image.backing->image.image_ci.format),
                      src_image.backing->num_samples, vk::to_string(backing->image.image_ci.format),
                      backing->num_samples);
        }
        return;
    }

    scheduler->EndRendering();

    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});

    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});

    auto cmdbuf = scheduler->CommandBuffer();

    if (!regions.empty()) {
        cmdbuf.copyImage(src_image.GetImage(), src_image.backing->state.layout, GetImage(),
                         backing->state.layout, regions);
    }

    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
}
void Image::CopyImageWithBuffer(Image& src_image, vk::Buffer buffer, u64 offset,
                                u64 buffer_size) {
    const auto& src_info = src_image.info;
    if (!SupportsImageBufferRoundTrip(src_info.num_samples, info.num_samples)) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping image buffer round trip for multisampled images ({} -> {} samples)",
                    src_info.num_samples, info.num_samples);
        return;
    }
    if (!IsValid() || !src_image.IsValid()) {
        LOG_WARNING(Render_Vulkan, "Skipping image buffer round trip for invalid image backing");
        return;
    }
    if ((backing->num_samples != info.num_samples && info.props.is_depth) ||
        (src_image.backing->num_samples != src_info.num_samples &&
         (src_info.props.is_depth || src_info.resources.levels != 1 ||
          src_info.resources.layers != 1))) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping image buffer round trip because a backing cannot be resolved for "
                    "buffer transfer");
        return;
    }
    ASSERT(SetHostScale(100));
    ASSERT(src_image.SetHostScale(100));
    const u32 num_mips = std::min(src_info.resources.levels, info.resources.levels);
    const u32 num_layers = std::min(src_info.resources.layers, info.resources.layers);
    ASSERT(src_info.resources.layers == info.resources.layers || num_mips == 1);

    SetBackingSamples(info.num_samples, false);
    src_image.SetBackingSamples(src_info.num_samples);
    if (!backing || !src_image.backing || backing->num_samples != 1 ||
        src_image.backing->num_samples != 1) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping image buffer round trip because a backing remains multisampled "
                    "({} -> {} samples)",
                    src_image.backing ? src_image.backing->num_samples : 0,
                    backing ? backing->num_samples : 0);
        return;
    }

    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < num_mips; ++mip) {
        const auto mip_w = std::max(src_info.size.width >> mip, 1u);
        const auto mip_h = std::max(src_info.size.height >> mip, 1u);
        const auto mip_d = std::max(src_info.size.depth >> mip, 1u);

        buffer_copies.emplace_back(vk::BufferImageCopy{
            .bufferOffset = offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = src_image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {mip_w, mip_h, mip_d},
        });
    }

    // The intermediate device-local stream buffer has a fixed capacity.  A malformed image can
    // otherwise produce a valid-looking vk::BufferImageCopy whose region extends beyond that
    // allocation; Vulkan reports this as pRegions-00171 and a later device-loss can obscure the
    // actual admission failure.  Validate each region before any transition or command recording.
    for (const auto& copy : buffer_copies) {
        u64 row_length = copy.bufferRowLength ? copy.bufferRowLength : copy.imageExtent.width;
        u64 image_height = copy.bufferImageHeight ? copy.bufferImageHeight : copy.imageExtent.height;
        if (src_info.props.is_block) {
            row_length = (row_length + 3) / 4;
            image_height = (image_height + 3) / 4;
        }
        const auto transfer_size = CheckedImageBufferTransferSize(
            row_length, image_height, copy.imageExtent.depth,
            copy.imageSubresource.layerCount, src_info.num_bits);
        if (!transfer_size || !IsImageBufferTransferRangeWithinBuffer(
                                  copy.bufferOffset, *transfer_size, buffer_size)) {
            LOG_WARNING(Render_Vulkan,
                        "Skipping image buffer copy whose region exceeds the utility buffer: "
                        "offset={} size={} capacity={} mip={} layers={}",
                        copy.bufferOffset, transfer_size.value_or(0), buffer_size,
                        copy.imageSubresource.mipLevel, copy.imageSubresource.layerCount);
            return;
        }
    }

    const vk::BufferMemoryBarrier2 pre_copy_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer,
        .offset = offset,
        .size = VK_WHOLE_SIZE,
    };

    const vk::BufferMemoryBarrier2 post_copy_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = buffer,
        .offset = offset,
        .size = VK_WHOLE_SIZE,
    };

    scheduler->EndRendering();
    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});

    auto cmdbuf = scheduler->CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_copy_barrier,
    });

    cmdbuf.copyImageToBuffer(src_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal, buffer,
                             buffer_copies);

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_copy_barrier,
    });

    for (auto& copy : buffer_copies) {
        copy.imageSubresource.aspectMask = aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    }

    cmdbuf.copyBufferToImage(buffer, GetImage(), vk::ImageLayout::eTransferDstOptimal,
                             buffer_copies);
    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
}

void Image::CopyMip(Image& src_image, u32 mip, u32 slice) {
    if (!IsValid() || !src_image.IsValid()) {
        LOG_WARNING(Render_Vulkan, "Skipping image mip copy for invalid image backing");
        return;
    }
    ASSERT(SetHostScale(100));
    ASSERT(src_image.SetHostScale(100));
    const auto& src_info = src_image.info;

    const auto dst_dim = info.props.is_block ? 2 : 0;
    const auto mip_block_w = std::max(info.size.width >> (mip + dst_dim), 1u);
    const auto mip_block_h = std::max(info.size.height >> (mip + dst_dim), 1u);
    const auto mip_block_p = std::max(info.mips_layout[mip].pitch >> dst_dim, 1u);

    const auto src_dim = src_info.props.is_block ? 2 : 0;
    ASSERT(mip_block_w == (src_info.size.width >> src_dim));
    ASSERT(mip_block_h == (src_info.size.height >> src_dim));
    ASSERT(mip_block_p == (src_info.pitch >> src_dim));

    auto [src_layers, dst_layers] = SanitizeCopyLayers(src_info, info, src_info.size.depth);

    // CopyMip writes the destination at a non-zero base slice when a smaller guest image is
    // merged into a larger backing image.  The layer counts from SanitizeCopyLayers describe the
    // complete resources, so bound them by the remaining destination range before publishing a
    // Vulkan copy region.  A malformed slice is not recoverable by coercing a layer count; skip
    // the copy before any transitions or command-buffer recording instead.
    const u32 destination_range_layers =
        slice < info.resources.layers ? info.resources.layers - slice : 0;
    const auto bounded_layers = ClampEqualImageCopyLayersToRanges(
        {src_layers, dst_layers}, src_info.resources.layers, destination_range_layers);
    src_layers = bounded_layers.source;
    dst_layers = bounded_layers.destination;
    if (dst_layers == 0) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping mip copy with out-of-range destination slice {} ({} layers)",
                    slice, info.resources.layers);
        return;
    }

    const vk::ImageCopy image_copy{
        .srcSubresource{
            .aspectMask = src_image.aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = src_layers,
        },
        .dstSubresource{
            .aspectMask = src_image.aspect_mask,
            .mipLevel = mip,
            .baseArrayLayer = slice,
            .layerCount = dst_layers,
        },
        .extent = {src_info.size.width, src_info.size.height, src_info.size.depth},
    };

    SetBackingSamples(info.num_samples);
    src_image.SetBackingSamples(src_info.num_samples);

    const auto depth_copy = SelectDepthCopyMode(
        src_image.backing->image.image_ci.format, backing->image.image_ci.format,
        src_image.backing->num_samples, backing->num_samples);
    if (depth_copy != DepthCopyMode::Direct) {
        if (depth_copy == DepthCopyMode::Buffer) {
            CopyDepthRegionsWithBuffer(*this, src_image, std::span{&image_copy, 1});
        } else {
            LOG_ERROR(Render_Vulkan,
                      "Unsupported depth mip copy: {} ({} samples) to {} ({} samples)",
                      vk::to_string(src_image.backing->image.image_ci.format),
                      src_image.backing->num_samples, vk::to_string(backing->image.image_ci.format),
                      backing->num_samples);
        }
        return;
    }

    scheduler->EndRendering();
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});
    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});

    const auto cmdbuf = scheduler->CommandBuffer();
    cmdbuf.copyImage(src_image.GetImage(), src_image.backing->state.layout, GetImage(),
                     backing->state.layout, image_copy);
    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
}

void Image::Resolve(Image& src_image, const VideoCore::SubresourceRange& mrt0_range,
                    const VideoCore::SubresourceRange& mrt1_range) {
    if (!IsValid() || !src_image.IsValid()) {
        LOG_WARNING(Render_Vulkan, "Skipping image resolve for invalid image backing");
        return;
    }
    if (!IsValidSubresourceRange(mrt0_range, src_image.info.resources) ||
        !IsValidSubresourceRange(mrt1_range, info.resources)) {
        LOG_WARNING(Render_Vulkan,
                    "Skipping resolve with out-of-range source or destination subresource");
        return;
    }
    if (GetHostScale() != src_image.GetHostScale()) {
        ASSERT(SetHostScale(100));
        ASSERT(src_image.SetHostScale(100));
    }
    SetBackingSamples(1, false);
    scheduler->EndRendering();

    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                      mrt0_range);
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, mrt1_range);

    auto [src_layers, dst_layers] = SanitizeCopyLayers(src_image.info, info, 1);
    if (ConvertImageType(src_image.info.type) == ConvertImageType(info.type)) {
        const auto bounded_layers = ClampEqualImageCopyLayersToRanges(
            {src_layers, dst_layers}, mrt0_range.extent.layers, mrt1_range.extent.layers);
        src_layers = bounded_layers.source;
        dst_layers = bounded_layers.destination;
    }
    if (src_image.backing->num_samples == 1) {
        const vk::ImageCopy region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = mrt0_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = mrt1_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = GetHostExtent(),
        };
        scheduler->CommandBuffer().copyImage(src_image.GetImage(),
                                             vk::ImageLayout::eTransferSrcOptimal, GetImage(),
                                             vk::ImageLayout::eTransferDstOptimal, region);
    } else {
        const vk::ImageResolve region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = mrt0_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = mrt1_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = GetHostExtent(),
        };
        scheduler->CommandBuffer().resolveImage(src_image.GetImage(),
                                                vk::ImageLayout::eTransferSrcOptimal, GetImage(),
                                                vk::ImageLayout::eTransferDstOptimal, region);
    }

    flags |= VideoCore::ImageFlagBits::GpuModified;
    flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Image::Clear(const vk::ClearValue& clear_value, const VideoCore::SubresourceRange& range) {
    if (!IsValid()) {
        LOG_WARNING(Render_Vulkan, "Skipping image clear for invalid image backing");
        return;
    }
    if (!IsValidSubresourceRange(range, info.resources)) {
        LOG_WARNING(Render_Vulkan, "Skipping clear with out-of-range subresource");
        return;
    }
    const vk::ImageSubresourceRange vk_range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = range.base.level,
        .levelCount = range.extent.levels,
        .baseArrayLayer = range.base.layer,
        .layerCount = range.extent.layers,
    };
    scheduler->EndRendering();
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});
    const auto cmdbuf = scheduler->CommandBuffer();
    cmdbuf.clearColorImage(GetImage(), vk::ImageLayout::eTransferDstOptimal, clear_value.color,
                           vk_range);
}

void Image::SetBackingSamples(u32 num_samples, bool copy_backing) {
    if (!IsValid() || backing->num_samples == num_samples) {
        return;
    }
    ASSERT_MSG(!info.props.is_depth, "Swapping samples is only valid for color images");
    BackingImage* new_backing;
    const auto current_extent = backing->image.image_ci.extent;
    auto it = std::ranges::find_if(backing_images, [&](const BackingImage& candidate) {
        return candidate.num_samples == num_samples &&
               candidate.image.image_ci.extent == current_extent;
    });
    if (it == backing_images.end()) {
        auto new_image_ci = backing->image.image_ci;
        new_image_ci.samples = LiverpoolToVK::NumSamples(num_samples, supported_samples);
        const auto selected_usage = SelectImageUsageForSamples(
            new_image_ci.usage, num_samples, info.storage_required,
            instance->IsShaderStorageImageMultisampleSupported());
        if (!selected_usage) {
            LOG_ERROR(Render_Vulkan,
                      "Cannot switch image backing to {} samples: required storage usage is "
                      "unsupported",
                      num_samples);
            return;
        }
        new_image_ci.usage = selected_usage->usage;

        new_backing = &backing_images.emplace_back();
        new_backing->num_samples = num_samples;
        new_backing->image = UniqueImage{instance->GetDevice(), instance->GetAllocator()};
        if (new_backing->image.TryCreate(new_image_ci, instance) != vk::Result::eSuccess) {
            backing_images.pop_back();
            return;
        }

        Vulkan::SetObjectName(instance->GetDevice(), new_backing->image.image,
                              "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{} (backing)",
                              info.size.width, info.size.height, info.size.depth,
                              AmdGpu::NameOf(info.tile_mode), vk::to_string(info.pixel_format),
                              info.guest_address, info.guest_size, info.resources.layers,
                              info.resources.levels, num_samples);
    } else {
        new_backing = std::addressof(*it);
    }

    if (copy_backing) {
        scheduler->EndRendering();
        ASSERT(info.resources.levels == 1 && info.resources.layers == 1);

        // Transition current backing to shader read layout
        auto barriers =
            GetBarriers(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
                        vk::PipelineStageFlagBits2::eFragmentShader, std::nullopt);

        // Transition dest backing to color attachment layout, not caring of previous contents
        constexpr auto dst_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        constexpr auto dst_access = vk::AccessFlagBits2::eColorAttachmentWrite;
        constexpr auto dst_layout = vk::ImageLayout::eColorAttachmentOptimal;
        barriers.push_back(vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = new_backing->image,
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = info.resources.layers,
            },
        });
        const auto cmdbuf = scheduler->CommandBuffer();
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });

        // Copy between ms and non ms backing images
        blit_helper->CopyBetweenMsImages(
            current_extent.width, current_extent.height, new_backing->num_samples,
            info.pixel_format, backing->num_samples > 1, backing->image, new_backing->image);

        // Update current layout in tracker to new backings layout
        new_backing->state.layout = dst_layout;
        new_backing->state.access_mask = dst_access;
        new_backing->state.pl_stage = dst_stage;
    }

    backing = new_backing;
}

} // namespace VideoCore
