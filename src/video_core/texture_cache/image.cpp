// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <ranges>
#include "common/assert.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

using namespace Vulkan;

Common::IncrementalIdProvider<u64> Image::global_image_uid{};
Common::IncrementalIdProvider<u64> Image::global_backing_generation{};

static vk::ImageUsageFlags ImageUsageFlags(const Vulkan::Instance* instance,
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
            // Always create images with storage flag to avoid needing re-creation in case of e.g
            // compute clears This sacrifices a bit of performance but is less work. ExtendedUsage
            // flag is also used.
            usage |= vk::ImageUsageFlagBits::eStorage;
        }
    } else {
        // Similarly to above, we specify storage usage. This is typically not supported by
        // compressed formats, but may be used for uncompressed views. In order to satisfy this,
        // we will also specify the extended usage bit.
        usage |= vk::ImageUsageFlagBits::eStorage;
    }

    return usage;
}

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return vk::ImageType::e1D;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DArray:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE();
    }
}

void Image::MarkDiagnosticTransfer(ImageTransferKind kind, vk::ImageAspectFlags aspect,
                                   SubresourceRange range, u32 old_samples, u32 new_samples,
                                   u32 source_samples, bool range_exact,
                                   bool request_matches_encoded, bool full_xy) noexcept {
    if (!IsPpSourceProducerTrackingEnabled()) {
        return;
    }
    diagnostic_depth_stencil = {};
    diagnostic_depth_writer_scope =
        InvalidateImageDepthWriterScope(diagnostic_depth_writer_scope);
    diagnostic_producer.Mark(ImageProducerClass::Transfer);
    const u64 mutation_epoch = diagnostic_producer.MutationEpoch();
    const u64 backing_generation = backing ? EnsureDiagnosticBackingGeneration() : 0;
    diagnostic_transfer = {
        .kind = kind,
        .mutation_epoch = mutation_epoch,
        .backing_generation = backing_generation,
        .aspect_mask = static_cast<u32>(static_cast<VkImageAspectFlags>(aspect)),
        .base_mip = range.base.level,
        .mip_count = range.extent.levels,
        .base_layer = range.base.layer,
        .layer_count = range.extent.layers,
        .source_samples = source_samples,
        .old_samples = old_samples,
        .new_samples = new_samples,
        .range_exact = range_exact,
        .request_matches_encoded = request_matches_encoded,
        .full_xy = full_xy,
        .valid = kind != ImageTransferKind::None && mutation_epoch != 0 &&
                 backing_generation != 0 && aspect && range.extent.levels != 0 &&
                 range.extent.layers != 0,
    };
    if (kind != ImageTransferKind::CopyImage) {
        diagnostic_transfer_source_uid = 0;
        diagnostic_transfer_source_backing_generation = 0;
        diagnostic_transfer_source_mutation_epoch = 0;
    }
    diagnostic_transfer_source_token =
        RetainDiagnosticTransferSourceToken(kind, diagnostic_transfer_source_token);
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
    // Note: StorageImage is intentionally ignored for now since it is always set, and can mess up
    // compatibility checks.
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

void UniqueImage::Create(const vk::ImageCreateInfo& image_ci) {
    this->image_ci = image_ci;
    ASSERT(!image);
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
    ASSERT_MSG(result == VK_SUCCESS, "Failed allocating image with error {}",
               vk::to_string(vk::Result{result}));
    image = vk::Image{unsafe_image};
}

Image::Image(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
             BlitHelper& blit_helper_, Common::SlotVector<ImageView>& slot_image_views_,
             const ImageInfo& info_)
    : instance{&instance_}, scheduler{&scheduler_}, blit_helper{&blit_helper_},
      slot_image_views{&slot_image_views_}, info{info_} {
    if (info.pixel_format == vk::Format::eUndefined) {
        return;
    }
    image_uid = global_image_uid.Next();
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

    usage_flags = ImageUsageFlags(instance, info);
    format_features = FormatFeatureFlags(usage_flags);
    if (info.props.is_depth) {
        aspect_mask = vk::ImageAspectFlagBits::eDepth;
        if (info.props.has_stencil) {
            aspect_mask |= vk::ImageAspectFlagBits::eStencil;
        }
    }

    constexpr auto tiling = vk::ImageTiling::eOptimal;
    const auto supported_format = instance->GetSupportedFormat(info.pixel_format, format_features);
    const vk::PhysicalDeviceImageFormatInfo2 format_info{
        .format = supported_format,
        .type = ConvertImageType(info.type),
        .tiling = tiling,
        .usage = usage_flags,
        .flags = flags,
    };
    const auto image_format_properties =
        instance->GetPhysicalDevice().getImageFormatProperties2(format_info);
    if (image_format_properties.result == vk::Result::eErrorFormatNotSupported) {
        LOG_ERROR(Render_Vulkan, "image format {} type {} is not supported (flags {}, usage {})",
                  vk::to_string(supported_format), vk::to_string(format_info.type),
                  vk::to_string(format_info.flags), vk::to_string(format_info.usage));
    }
    supported_samples = image_format_properties.result == vk::Result::eSuccess
                            ? image_format_properties.value.imageFormatProperties.sampleCounts
                            : vk::SampleCountFlagBits::e1;

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
    backing->image.Create(image_ci);

    Vulkan::SetObjectName(instance->GetDevice(), GetImage(),
                          "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{}", info.size.width,
                          info.size.height, info.size.depth, AmdGpu::NameOf(info.tile_mode),
                          vk::to_string(info.pixel_format), info.guest_address, info.guest_size,
                          info.resources.layers, info.resources.levels, info.num_samples);
}

Image::~Image() = default;

ImageView& Image::FindView(const ImageViewInfo& view_info, bool ensure_guest_samples,
                           bool* view_owner_exact) {
    if (ensure_guest_samples && backing->num_samples > 1 != info.num_samples > 1) {
        SetBackingSamples(info.num_samples);
    }
    const auto& view_infos = backing->image_view_infos;
    const auto it = std::ranges::find(view_infos, view_info);
    if (it != view_infos.end()) {
        const auto view_id = backing->image_view_ids[std::distance(view_infos.begin(), it)];
        auto& view = (*slot_image_views)[view_id];
        if (view_owner_exact != nullptr) {
            *view_owner_exact =
                view.info == view_info && view.CreationObservation().image_exact;
        }
        return view;
    }
    const auto view_id = slot_image_views->insert(*instance, view_info, *this);
    backing->image_view_infos.emplace_back(view_info);
    backing->image_view_ids.emplace_back(view_id);
    auto& view = (*slot_image_views)[view_id];
    if (view_owner_exact != nullptr) {
        *view_owner_exact = view.info == view_info && view.CreationObservation().image_exact;
    }
    return view;
}

Image::Barriers Image::GetBarriers(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                                   vk::PipelineStageFlags2 dst_stage,
                                   std::optional<SubresourceRange> subres_range) {
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

                const bool is_write = IsImageWriteAccess(state.access_mask);
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
        const bool is_write = IsImageWriteAccess(last_state.access_mask);
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

ImageReadStateObservation Image::ObserveReadState(
    vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
    std::optional<SubresourceRange> subres_range) const noexcept {
    const vk::PipelineStageFlags2 dst_stage =
        (dst_mask == vk::AccessFlagBits2::eTransferRead ||
         dst_mask == vk::AccessFlagBits2::eTransferWrite)
            ? vk::PipelineStageFlagBits2::eTransfer
            : ImageShaderReadStages;
    const SubresourceRange request =
        subres_range.value_or(SubresourceRange{.base = {}, .extent = info.resources});
    const bool uniform = backing && backing->subresource_states.empty();
    const ImageReadStateObservationSeed seed{
        .request_range = request,
        .resource_levels = info.resources.levels,
        .resource_layers = info.resources.layers,
        .request_aspect_mask = static_cast<u32>(static_cast<VkImageAspectFlags>(aspect_mask)),
        .destination_layout = dst_layout,
        .destination_access = dst_mask,
        .destination_stages = dst_stage,
        .uniform = uniform,
    };

    boost::container::static_vector<ImageReadStateSample,
                                    ImageTransitBarrierReceiptMaxSubresources>
        samples;
    if (!backing) {
        return BuildImageReadStateObservation(seed, samples);
    }
    if (uniform) {
        samples.emplace_back(ImageReadStateSample{
            .stages = backing->state.pl_stage,
            .access = backing->state.access_mask,
            .layout = backing->state.layout,
        });
        return BuildImageReadStateObservation(seed, samples);
    }

    const bool request_base_valid = request.base.level < seed.resource_levels &&
                                    request.base.layer < seed.resource_layers;
    const bool request_extent_valid = request.extent.levels != 0 &&
                                      request.extent.layers != 0 && request_base_valid &&
                                      request.extent.levels <=
                                          seed.resource_levels - request.base.level &&
                                      request.extent.layers <=
                                          seed.resource_layers - request.base.layer;
    const u64 selected_count = static_cast<u64>(request.extent.levels) *
                               static_cast<u64>(request.extent.layers);
    const u64 resource_count = static_cast<u64>(seed.resource_levels) *
                               static_cast<u64>(seed.resource_layers);
    if (!request_extent_valid ||
        selected_count > ImageTransitBarrierReceiptMaxSubresources ||
        backing->subresource_states.size() != resource_count) {
        return BuildImageReadStateObservation(seed, samples);
    }

    for (u32 mip_offset = 0; mip_offset < request.extent.levels; ++mip_offset) {
        const u32 mip = request.base.level + mip_offset;
        for (u32 layer_offset = 0; layer_offset < request.extent.layers; ++layer_offset) {
            const u32 layer = request.base.layer + layer_offset;
            const auto& state =
                backing->subresource_states[mip * seed.resource_layers + layer];
            samples.emplace_back(ImageReadStateSample{
                .stages = state.pl_stage,
                .access = state.access_mask,
                .layout = state.layout,
            });
        }
    }
    return BuildImageReadStateObservation(seed, samples);
}

void Image::Transit(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                    std::optional<SubresourceRange> range, vk::CommandBuffer cmdbuf /*= {}*/,
                    ImageTransitBarrierReceipt* receipt /*= nullptr*/) {
    // Adjust pipeline stage
    const vk::PipelineStageFlags2 dst_pl_stage =
        (dst_mask == vk::AccessFlagBits2::eTransferRead ||
         dst_mask == vk::AccessFlagBits2::eTransferWrite)
            ? vk::PipelineStageFlagBits2::eTransfer
            : vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;

    ImageTransitBarrierReceiptSeed receipt_seed{};
    if (receipt) {
        receipt_seed.request_range =
            range.value_or(SubresourceRange{.base = {}, .extent = info.resources});
        receipt_seed.resource_levels = info.resources.levels;
        receipt_seed.resource_layers = info.resources.layers;
        receipt_seed.request_aspect_mask =
            static_cast<u32>(static_cast<VkImageAspectFlags>(aspect_mask));
        receipt_seed.destination_layout = dst_layout;
        receipt_seed.destination_access = dst_mask;
        receipt_seed.destination_stages = dst_pl_stage;
        receipt_seed.transition_writer = diagnostic_depth_stencil;
        receipt_seed.uniform_before = backing->subresource_states.empty();
        receipt_seed.transit_invoked = true;
    }

    const auto capture_state_masks = [&](ImageTransitBarrierReceiptSeed& seed,
                                         u64* depth_stencil_write_mask,
                                         u64* readable_mask) noexcept {
        if (!receipt) {
            return;
        }
        if (depth_stencil_write_mask) {
            *depth_stencil_write_mask = 0;
        }
        if (readable_mask) {
            *readable_mask = 0;
        }

        const auto& request = seed.request_range;
        if (request.extent.levels == 0 || request.extent.layers == 0 ||
            request.base.level >= seed.resource_levels ||
            request.base.layer >= seed.resource_layers ||
            request.extent.levels > seed.resource_levels - request.base.level ||
            request.extent.layers > seed.resource_layers - request.base.layer) {
            seed.malformed = true;
            return;
        }
        const u64 selected_count = static_cast<u64>(request.extent.levels) *
                                   static_cast<u64>(request.extent.layers);
        if (selected_count > ImageTransitBarrierReceiptMaxSubresources) {
            seed.malformed = true;
            seed.overflowed = true;
            return;
        }

        const auto& subresource_states = backing->subresource_states;
        const u64 resource_count = static_cast<u64>(seed.resource_levels) *
                                   static_cast<u64>(seed.resource_layers);
        if (!subresource_states.empty() && subresource_states.size() != resource_count) {
            seed.malformed = true;
            return;
        }

        constexpr auto any_write = vk::AccessFlagBits2::eTransferWrite |
                                   vk::AccessFlagBits2::eShaderWrite |
                                   vk::AccessFlagBits2::eMemoryWrite |
                                   vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                   vk::AccessFlagBits2::eColorAttachmentWrite;
        for (u32 mip_offset = 0; mip_offset < request.extent.levels; ++mip_offset) {
            const u32 mip = request.base.level + mip_offset;
            for (u32 layer_offset = 0; layer_offset < request.extent.layers; ++layer_offset) {
                const u32 layer = request.base.layer + layer_offset;
                const u32 dense_index = mip_offset * request.extent.layers + layer_offset;
                const u64 bit = u64{1} << dense_index;
                const auto& state = subresource_states.empty()
                                        ? backing->state
                                        : subresource_states[mip * seed.resource_layers +
                                                             layer];
                if (depth_stencil_write_mask &&
                    static_cast<bool>(state.access_mask &
                                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite)) {
                    *depth_stencil_write_mask |= bit;
                }
                const bool access_covers =
                    (state.access_mask & dst_mask) == dst_mask &&
                    !static_cast<bool>(state.access_mask & any_write);
                const bool stages_cover =
                    (state.pl_stage & dst_pl_stage) == dst_pl_stage;
                if (readable_mask && dst_mask != vk::AccessFlagBits2::eNone &&
                    state.layout == dst_layout && access_covers && stages_cover) {
                    *readable_mask |= bit;
                }
            }
        }
    };

    Barriers barriers;
    if (receipt) {
        capture_state_masks(receipt_seed, &receipt_seed.prior_depth_stencil_write_mask,
                            &receipt_seed.already_readable_mask);
        auto result = QueryImageTransitBarriersAndBuildReceipt(
            receipt_seed,
            [&] { return GetBarriers(dst_layout, dst_mask, dst_pl_stage, range); },
            [&](ImageTransitBarrierReceiptSeed& observed_seed) {
                observed_seed.uniform_after = backing->subresource_states.empty();
                capture_state_masks(observed_seed, nullptr,
                                    &observed_seed.post_readable_mask);
            });
        barriers = std::move(result.first);
        *receipt = result.second;
    } else {
        barriers = GetBarriers(dst_layout, dst_mask, dst_pl_stage, range);
    }
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
    SetBackingSamples(info.num_samples, false);
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
    MarkDiagnosticProducer(ImageProducerClass::CpuUpload);
}

void Image::Download(std::span<const vk::BufferImageCopy> download_copies, vk::Buffer buffer,
                     u64 offset, u64 download_size) {
    SetBackingSamples(info.num_samples);
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

void Image::CopyImage(Image& src_image, ImageTransferOrigin origin, ImageId source_id) {
    const auto& src_info = src_image.info;
    const u32 old_samples = backing ? backing->num_samples : 0;

    const u32 num_mips = std::min(src_info.resources.levels, info.resources.levels);

    // Format mismatch warning (safe but useful)
    if (src_info.pixel_format != info.pixel_format) {
        LOG_DEBUG(Render_Vulkan,
                  "Copy between different formats: src={}, dst={}. "
                  "Result may be undefined.",
                  vk::to_string(src_info.pixel_format), vk::to_string(info.pixel_format));
    }

    const u32 base_width = src_info.size.width;
    const u32 base_height = src_info.size.height;
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
    u32 guaranteed_layers{};
    bool range_exact = true;
    bool full_xy = !regions.empty();
    for (u32 mip = 0; mip < regions.size(); ++mip) {
        const auto& region = regions[mip];
        const u32 layers = region.dstSubresource.layerCount;
        if (guaranteed_layers != 0 && guaranteed_layers != layers) {
            range_exact = false;
        }
        guaranteed_layers = guaranteed_layers == 0 ? layers : std::min(guaranteed_layers, layers);
        full_xy &= region.extent.width >= std::max(info.size.width >> mip, 1u) &&
                   region.extent.height >= std::max(info.size.height >> mip, 1u);
    }
    MarkDiagnosticTransfer(
        ImageTransferKind::CopyImage, dst_aspect,
        {.base = {.level = 0, .layer = 0},
         .extent = {.levels = static_cast<u32>(regions.size()), .layers = guaranteed_layers}},
        old_samples, backing ? backing->num_samples : 0,
        src_image.backing ? src_image.backing->num_samples : 0, range_exact, true, full_xy);
    if (diagnostic_transfer.valid) {
        diagnostic_transfer.origin = origin;
        const u64 source_generation = src_image.EnsureDiagnosticBackingGeneration();
        const u64 source_mutation_epoch = src_image.DiagnosticMutationEpoch();
        diagnostic_transfer_source_token = {
            .link = {.id = source_id, .uid = src_image.image_uid},
            .backing_generation = source_generation,
            .mutation_epoch = source_mutation_epoch,
        };
        if (!diagnostic_transfer_source_token.Valid()) {
            diagnostic_transfer_source_token = {};
        }
        const auto source_producer = src_image.PeekDiagnosticProducer();
        const auto source_transfer = src_image.PeekDiagnosticTransfer();
        const auto source_depth_stencil = src_image.PeekDiagnosticDepthStencilProducer();
        diagnostic_transfer.source_producer = source_producer.classification;
        diagnostic_transfer.source_is_depth = src_image.info.props.is_depth;
        diagnostic_transfer.source_depth_stencil_exact =
            source_depth_stencil.valid &&
            source_depth_stencil.mutation_epoch == source_mutation_epoch &&
            source_depth_stencil.backing_generation == source_generation;
        if (diagnostic_transfer.source_depth_stencil_exact) {
            diagnostic_transfer.source_depth_stencil_kind = source_depth_stencil.kind;
            diagnostic_transfer.source_depth_stencil_element_count =
                source_depth_stencil.element_count;
            diagnostic_transfer.source_depth_stencil_instance_count =
                source_depth_stencil.instance_count;
            diagnostic_transfer.source_depth_stencil_sampled_images =
                source_depth_stencil.sampled_images;
            diagnostic_transfer.source_depth_stencil_storage_writes =
                source_depth_stencil.storage_writes;
            diagnostic_transfer.source_depth_stencil_upstream_candidate_index =
                source_depth_stencil.upstream_candidate_index;
            diagnostic_transfer.source_depth_stencil_upstream_candidate_generation =
                source_depth_stencil.upstream_candidate_generation;
            diagnostic_transfer.source_depth_stencil_indexed = source_depth_stencil.indexed;
            diagnostic_transfer.source_depth_write = source_depth_stencil.depth_write;
            diagnostic_transfer.source_stencil_write = source_depth_stencil.stencil_write;
            diagnostic_transfer.source_depth_stencil_upstream_candidate_valid =
                source_depth_stencil.upstream_candidate_valid;
        }
        diagnostic_transfer.source_transfer_exact =
            source_producer.classification == ImageProducerClass::Transfer &&
            source_transfer.valid &&
            source_transfer.mutation_epoch == src_image.DiagnosticMutationEpoch() &&
            source_transfer.backing_generation == source_generation;
        if (diagnostic_transfer.source_transfer_exact) {
            diagnostic_transfer.source_transfer_kind = source_transfer.kind;
            diagnostic_transfer.source_transfer_origin = source_transfer.origin;
            diagnostic_transfer.nested_source_producer = source_transfer.source_producer;
            diagnostic_transfer.nested_source_transfer_kind =
                source_transfer.source_transfer_kind;
            diagnostic_transfer.nested_source_transfer_origin =
                source_transfer.source_transfer_origin;
            diagnostic_transfer.nested_source_transfer_exact =
                source_transfer.source_transfer_exact;
            diagnostic_transfer.nested_source_is_depth = source_transfer.source_is_depth;
        }
        diagnostic_transfer.source_identity_comparable = diagnostic_transfer_source_uid != 0;
        diagnostic_transfer.source_image_changed =
            diagnostic_transfer.source_identity_comparable &&
            diagnostic_transfer_source_uid != src_image.image_uid;
        diagnostic_transfer.source_backing_changed =
            diagnostic_transfer.source_identity_comparable &&
            (diagnostic_transfer.source_image_changed ||
             diagnostic_transfer_source_backing_generation != source_generation);
        diagnostic_transfer.source_mutation_comparable =
            diagnostic_transfer.source_identity_comparable &&
            !diagnostic_transfer.source_image_changed && diagnostic_transfer_source_mutation_epoch != 0 &&
            source_mutation_epoch != 0;
        diagnostic_transfer.source_mutation_changed =
            diagnostic_transfer.source_mutation_comparable &&
            diagnostic_transfer_source_mutation_epoch != source_mutation_epoch;
        diagnostic_transfer_source_uid = src_image.image_uid;
        diagnostic_transfer_source_backing_generation = source_generation;
        diagnostic_transfer_source_mutation_epoch = source_mutation_epoch;
    } else {
        diagnostic_transfer_source_token = {};
    }
}
void Image::CopyImageWithBuffer(Image& src_image, vk::Buffer buffer, u64 offset) {
    const auto& src_info = src_image.info;
    const u32 old_samples = backing ? backing->num_samples : 0;
    const u32 num_mips = std::min(src_info.resources.levels, info.resources.levels);
    const u32 num_layers = std::min(src_info.resources.layers, info.resources.layers);
    bool full_xy = num_mips != 0;
    for (u32 mip = 0; mip < num_mips; ++mip) {
        full_xy &= std::max(src_info.size.width >> mip, 1u) >=
                       std::max(info.size.width >> mip, 1u) &&
                   std::max(src_info.size.height >> mip, 1u) >=
                       std::max(info.size.height >> mip, 1u);
    }
    ASSERT(src_info.resources.layers == info.resources.layers || num_mips == 1);

    SetBackingSamples(info.num_samples, false);
    src_image.SetBackingSamples(src_info.num_samples);

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
    MarkDiagnosticTransfer(
        ImageTransferKind::CopyViaBuffer, aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
        {.base = {.level = 0, .layer = 0}, .extent = {.levels = num_mips, .layers = num_layers}},
        old_samples, backing ? backing->num_samples : 0,
        src_image.backing ? src_image.backing->num_samples : 0, true, true, full_xy);
}

void Image::CopyMip(Image& src_image, u32 mip, u32 slice) {
    const auto& src_info = src_image.info;
    const u32 old_samples = backing ? backing->num_samples : 0;

    const auto dst_dim = info.props.is_block ? 2 : 0;
    const auto mip_block_w = std::max(info.size.width >> (mip + dst_dim), 1u);
    const auto mip_block_h = std::max(info.size.height >> (mip + dst_dim), 1u);
    const auto mip_block_p = std::max(info.mips_layout[mip].pitch >> dst_dim, 1u);

    const auto src_dim = src_info.props.is_block ? 2 : 0;
    ASSERT(mip_block_w == (src_info.size.width >> src_dim));
    ASSERT(mip_block_h == (src_info.size.height >> src_dim));
    ASSERT(mip_block_p == (src_info.pitch >> src_dim));

    const auto [src_layers, dst_layers] = SanitizeCopyLayers(src_info, info, src_info.size.depth);
    const bool full_xy = src_info.size.width >= std::max(info.size.width >> mip, 1u) &&
                         src_info.size.height >= std::max(info.size.height >> mip, 1u);

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

    scheduler->EndRendering();
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});
    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});

    const auto cmdbuf = scheduler->CommandBuffer();
    cmdbuf.copyImage(src_image.GetImage(), src_image.backing->state.layout, GetImage(),
                     backing->state.layout, image_copy);
    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
    MarkDiagnosticTransfer(
        ImageTransferKind::CopyMip, image_copy.dstSubresource.aspectMask,
        {.base = {.level = mip, .layer = slice},
         .extent = {.levels = 1, .layers = image_copy.dstSubresource.layerCount}},
        old_samples, backing ? backing->num_samples : 0,
        src_image.backing ? src_image.backing->num_samples : 0, true, true, full_xy);
}

void Image::Resolve(Image& src_image, const VideoCore::SubresourceRange& mrt0_range,
                    const VideoCore::SubresourceRange& mrt1_range) {
    const u32 old_samples = backing ? backing->num_samples : 0;
    SetBackingSamples(1, false);
    scheduler->EndRendering();

    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                      mrt0_range);
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, mrt1_range);

    const auto [src_layers, dst_layers] = SanitizeCopyLayers(src_image.info, info, 1);
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
            .extent = {info.size.width, info.size.height, 1},
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
            .extent = {info.size.width, info.size.height, 1},
        };
        scheduler->CommandBuffer().resolveImage(src_image.GetImage(),
                                                vk::ImageLayout::eTransferSrcOptimal, GetImage(),
                                                vk::ImageLayout::eTransferDstOptimal, region);
    }

    flags |= VideoCore::ImageFlagBits::GpuModified;
    flags &= ~VideoCore::ImageFlagBits::Dirty;
    const bool request_matches_encoded =
        mrt0_range.base.level == 0 && mrt0_range.extent.levels == 1 &&
        mrt0_range.extent.layers == src_layers && mrt1_range.base.level == 0 &&
        mrt1_range.extent.levels == 1 && mrt1_range.extent.layers == dst_layers;
    MarkDiagnosticTransfer(ImageTransferKind::Resolve, vk::ImageAspectFlagBits::eColor,
                           {.base = {.level = 0, .layer = mrt1_range.base.layer},
                            .extent = {.levels = 1, .layers = dst_layers}},
                           old_samples, backing ? backing->num_samples : 0,
                           src_image.backing ? src_image.backing->num_samples : 0, true,
                           request_matches_encoded);
}

void Image::Clear(const vk::ClearValue& clear_value, const VideoCore::SubresourceRange& range) {
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
    const u32 samples = backing ? backing->num_samples : 0;
    MarkDiagnosticTransfer(ImageTransferKind::Clear, vk::ImageAspectFlagBits::eColor, range,
                           samples, samples, 0);
}

void Image::SetBackingSamples(u32 num_samples, bool copy_backing) {
    if (!backing || backing->num_samples == num_samples) {
        return;
    }
    const u32 old_samples = backing->num_samples;
    ResetDiagnosticProducer();
    ASSERT_MSG(!info.props.is_depth, "Swapping samples is only valid for color images");
    BackingImage* new_backing;
    auto it = std::ranges::find(backing_images, num_samples, &BackingImage::num_samples);
    if (it == backing_images.end()) {
        auto new_image_ci = backing->image.image_ci;
        new_image_ci.samples = LiverpoolToVK::NumSamples(num_samples, supported_samples);

        new_backing = &backing_images.emplace_back();
        new_backing->num_samples = num_samples;
        new_backing->image = UniqueImage{instance->GetDevice(), instance->GetAllocator()};
        new_backing->image.Create(new_image_ci);

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
            info.size.width, info.size.height, new_backing->num_samples, info.pixel_format,
            backing->num_samples > 1, backing->image, new_backing->image);

        // Update current layout in tracker to new backings layout
        new_backing->state.layout = dst_layout;
        new_backing->state.access_mask = dst_access;
        new_backing->state.pl_stage = dst_stage;
    }

    backing = new_backing;
    if (copy_backing) {
        MarkDiagnosticTransfer(ImageTransferKind::BackingSampleCopy, aspect_mask,
                               {.base = {.level = 0, .layer = 0},
                                .extent = {.levels = 1, .layers = info.resources.layers}},
                               old_samples, backing->num_samples, old_samples);
    }
}

} // namespace VideoCore
