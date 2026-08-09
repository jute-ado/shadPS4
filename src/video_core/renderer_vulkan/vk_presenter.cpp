// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/system/systemservice.h"
#include "imgui/friends_layer.h"
#include "imgui/invitation_prompt_layer.h"
#include "imgui/notifications_layer.h"
#include "imgui/renderer/imgui_core.h"
#include "imgui/renderer/imgui_impl_vulkan.h"
#include "imgui/shadnet_notifications_layer.h"
#include "sdl_window.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/completed_readback.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/final_guest_surface_content.h"
#include "video_core/renderer_vulkan/present_frame_ownership.h"
#include "video_core/renderer_vulkan/present_frame_transition.h"
#include "video_core/renderer_vulkan/presented_frame_timing_trace.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/image.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csetjmp>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string_view>
#include <system_error>
#include <vector>
#include <imgui.h>
#include <png.h>
#include <vk_mem_alloc.h>

namespace Vulkan {

void RecordPresentedFrameTiming(const u32 presented_frame) {
    static auto trace = PresentedFrameTimingTrace::CreateFromEnvironment();
    if (trace) {
        trace->Record(presented_frame, PresentedFrameTimingTrace::MonotonicNanoseconds());
    }
}

bool CanBlitToSwapchain(const vk::PhysicalDevice physical_device, vk::Format format) {
    const vk::FormatProperties props{physical_device.getFormatProperties(format)};
    return static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);
}

[[nodiscard]] vk::ImageSubresourceLayers MakeImageSubresourceLayers() {
    return vk::ImageSubresourceLayers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlit(s32 frame_width, s32 frame_height, s32 dst_width,
                                          s32 dst_height, s32 offset_x, s32 offset_y) {
    return vk::ImageBlit{
        .srcSubresource = MakeImageSubresourceLayers(),
        .srcOffsets =
            std::array{
                vk::Offset3D{
                    .x = 0,
                    .y = 0,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = frame_width,
                    .y = frame_height,
                    .z = 1,
                },
            },
        .dstSubresource = MakeImageSubresourceLayers(),
        .dstOffsets =
            std::array{
                vk::Offset3D{
                    .x = offset_x,
                    .y = offset_y,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = offset_x + dst_width,
                    .y = offset_y + dst_height,
                    .z = 1,
                },
            },
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlitStretch(s32 frame_width, s32 frame_height,
                                                 s32 swapchain_width, s32 swapchain_height) {
    return MakeImageBlit(frame_width, frame_height, swapchain_width, swapchain_height, 0, 0);
}

static vk::Rect2D FitImage(s32 frame_width, s32 frame_height, s32 swapchain_width,
                           s32 swapchain_height) {
    float frame_aspect = static_cast<float>(frame_width) / frame_height;
    float swapchain_aspect = static_cast<float>(swapchain_width) / swapchain_height;

    u32 dst_width = swapchain_width;
    u32 dst_height = swapchain_height;

    if (frame_aspect > swapchain_aspect) {
        dst_height = static_cast<s32>(swapchain_width / frame_aspect);
    } else {
        dst_width = static_cast<s32>(swapchain_height * frame_aspect);
    }

    const s32 offset_x = (swapchain_width - dst_width) / 2;
    const s32 offset_y = (swapchain_height - dst_height) / 2;

    return vk::Rect2D{{offset_x, offset_y}, {dst_width, dst_height}};
}

[[nodiscard]] vk::ImageBlit MakeImageBlitFit(s32 frame_width, s32 frame_height, s32 swapchain_width,
                                             s32 swapchain_height) {
    const auto& dst_rect = FitImage(frame_width, frame_height, swapchain_width, swapchain_height);

    return MakeImageBlit(frame_width, frame_height, dst_rect.extent.width, dst_rect.extent.height,
                         dst_rect.offset.x, dst_rect.offset.y);
}

enum class ScreenshotKind : u8 {
    GameOnly,
    WithOverlays,
};

struct ScreenshotReadback {
    ScreenshotKind kind{};
    bool notify{};
    std::vector<std::filesystem::path> paths{};
    VideoCore::Buffer buffer;
    u32 width{};
    u32 height{};
    vk::Format format{};
    bool hdr_encoded{};

    ScreenshotReadback(const Instance& instance, Scheduler& scheduler, ScreenshotKind kind_,
                       const bool notify_, std::vector<std::filesystem::path> paths_,
                       const u32 width_, const u32 height_, const vk::Format format_,
                       const bool hdr_encoded_)
        : kind{kind_}, notify{notify_}, paths{std::move(paths_)},
          buffer{instance,
                 scheduler,
                 VideoCore::MemoryUsage::Download,
                 0,
                 vk::BufferUsageFlagBits::eTransferDst,
                 static_cast<u64>(width_) * static_cast<u64>(height_) * 4},
          width{width_}, height{height_}, format{format_}, hdr_encoded{hdr_encoded_} {}
};

namespace {

[[nodiscard]] std::optional<FinalGuestSurfaceContentConfig> ReadFinalGuestSurfaceContentConfig() {
    return ResolveFinalGuestSurfaceContentConfig([](const char* name) {
        const char* value = std::getenv(name);
        if (value == nullptr) {
            return std::optional<std::string_view>{};
        }
        return std::optional<std::string_view>{value};
    });
}

[[nodiscard]] FinalGuestSurfaceFormat ToFinalGuestSurfaceFormat(vk::Format format) noexcept {
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        return FinalGuestSurfaceFormat::Rgba8;
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
        return FinalGuestSurfaceFormat::Bgra8;
    case vk::Format::eA2R10G10B10UnormPack32:
        return FinalGuestSurfaceFormat::A2R10G10B10;
    case vk::Format::eA2B10G10R10UnormPack32:
        return FinalGuestSurfaceFormat::A2B10G10R10;
    case vk::Format::eR16G16B16A16Sfloat:
        return FinalGuestSurfaceFormat::Rgba16Float;
    case vk::Format::eBc1RgbUnormBlock:
    case vk::Format::eBc1RgbSrgbBlock:
    case vk::Format::eBc1RgbaUnormBlock:
    case vk::Format::eBc1RgbaSrgbBlock:
    case vk::Format::eBc4UnormBlock:
    case vk::Format::eBc4SnormBlock:
        return FinalGuestSurfaceFormat::Block8;
    case vk::Format::eBc2UnormBlock:
    case vk::Format::eBc2SrgbBlock:
    case vk::Format::eBc3UnormBlock:
    case vk::Format::eBc3SrgbBlock:
    case vk::Format::eBc5UnormBlock:
    case vk::Format::eBc5SnormBlock:
    case vk::Format::eBc6HUfloatBlock:
    case vk::Format::eBc6HSfloatBlock:
    case vk::Format::eBc7UnormBlock:
    case vk::Format::eBc7SrgbBlock:
        return FinalGuestSurfaceFormat::Block16;
    default:
        return FinalGuestSurfaceFormat::Unsupported;
    }
}

[[nodiscard]] FinalGuestSurfaceImageType ToFinalGuestSurfaceImageType(
    AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
        return FinalGuestSurfaceImageType::Color1D;
    case AmdGpu::ImageType::Color2D:
        return FinalGuestSurfaceImageType::Color2D;
    case AmdGpu::ImageType::Color3D:
        return FinalGuestSurfaceImageType::Color3D;
    default:
        return FinalGuestSurfaceImageType::Other;
    }
}

void DestroyPpSourceReconstructionResources(const Instance& instance, Frame& frame) {
    const auto device = instance.GetDevice();
    if (frame.pp_source_reconstruction_snapshot_view) {
        device.destroyImageView(frame.pp_source_reconstruction_snapshot_view);
        frame.pp_source_reconstruction_snapshot_view = nullptr;
    }
    if (frame.pp_source_reconstruction_snapshot_image) {
        vmaDestroyImage(instance.GetAllocator(), frame.pp_source_reconstruction_snapshot_image,
                        frame.pp_source_reconstruction_snapshot_allocation);
        frame.pp_source_reconstruction_snapshot_image = nullptr;
        frame.pp_source_reconstruction_snapshot_allocation = {};
    }
    if (frame.pp_source_reconstruction_output_view) {
        device.destroyImageView(frame.pp_source_reconstruction_output_view);
        frame.pp_source_reconstruction_output_view = nullptr;
    }
    if (frame.pp_source_reconstruction_output_image) {
        vmaDestroyImage(instance.GetAllocator(), frame.pp_source_reconstruction_output_image,
                        frame.pp_source_reconstruction_output_allocation);
        frame.pp_source_reconstruction_output_image = nullptr;
        frame.pp_source_reconstruction_output_allocation = {};
    }
    frame.pp_source_reconstruction_image_format = {};
    frame.pp_source_reconstruction_view_format = {};
    frame.pp_source_reconstruction_mapping = {};
    frame.pp_source_reconstruction_source_extent = {};
    frame.pp_source_reconstruction_output_format = {};
    frame.pp_source_reconstruction_output_extent = {};
}

[[nodiscard]] bool EnsurePpSourceReconstructionResources(
    const Instance& instance, Frame& frame, vk::Format source_image_format,
    vk::Format source_view_format, vk::ComponentMapping source_mapping, vk::Extent2D source_extent,
    vk::Format output_format, vk::Extent2D output_extent) {
    const bool matches = frame.pp_source_reconstruction_snapshot_image &&
                         frame.pp_source_reconstruction_snapshot_view &&
                         frame.pp_source_reconstruction_output_image &&
                         frame.pp_source_reconstruction_output_view &&
                         frame.pp_source_reconstruction_image_format == source_image_format &&
                         frame.pp_source_reconstruction_view_format == source_view_format &&
                         frame.pp_source_reconstruction_mapping == source_mapping &&
                         frame.pp_source_reconstruction_source_extent == source_extent &&
                         frame.pp_source_reconstruction_output_format == output_format &&
                         frame.pp_source_reconstruction_output_extent == output_extent;
    if (matches) {
        return true;
    }
    DestroyPpSourceReconstructionResources(instance, frame);
    if (source_extent.width == 0 || source_extent.height == 0 || output_extent.width == 0 ||
        output_extent.height == 0) {
        return false;
    }

    const VmaAllocationCreateInfo allocation_info{
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    const auto create_image = [&](const vk::ImageCreateInfo& image_info, vk::Image& image,
                                  VmaAllocation& allocation) {
        VkImage unsafe_image{};
        auto unsafe_info = static_cast<VkImageCreateInfo>(image_info);
        const auto result = vmaCreateImage(instance.GetAllocator(), &unsafe_info, &allocation_info,
                                           &unsafe_image, &allocation, nullptr);
        if (result != VK_SUCCESS) {
            return false;
        }
        image = vk::Image{unsafe_image};
        return true;
    };

    const vk::ImageCreateInfo source_info{
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = source_image_format,
        .extent = {source_extent.width, source_extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    if (!create_image(source_info, frame.pp_source_reconstruction_snapshot_image,
                      frame.pp_source_reconstruction_snapshot_allocation)) {
        DestroyPpSourceReconstructionResources(instance, frame);
        return false;
    }
    const vk::ImageViewCreateInfo source_view_info{
        .image = frame.pp_source_reconstruction_snapshot_image,
        .viewType = vk::ImageViewType::e2D,
        .format = source_view_format,
        .components = source_mapping,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    const auto [source_view_result, source_view] =
        instance.GetDevice().createImageView(source_view_info);
    if (source_view_result != vk::Result::eSuccess) {
        DestroyPpSourceReconstructionResources(instance, frame);
        return false;
    }
    frame.pp_source_reconstruction_snapshot_view = source_view;

    const vk::ImageCreateInfo output_info{
        .imageType = vk::ImageType::e2D,
        .format = output_format,
        .extent = {output_extent.width, output_extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    if (!create_image(output_info, frame.pp_source_reconstruction_output_image,
                      frame.pp_source_reconstruction_output_allocation)) {
        DestroyPpSourceReconstructionResources(instance, frame);
        return false;
    }
    const vk::ImageViewCreateInfo output_view_info{
        .image = frame.pp_source_reconstruction_output_image,
        .viewType = vk::ImageViewType::e2D,
        .format = output_format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    const auto [output_view_result, output_view] =
        instance.GetDevice().createImageView(output_view_info);
    if (output_view_result != vk::Result::eSuccess) {
        DestroyPpSourceReconstructionResources(instance, frame);
        return false;
    }
    frame.pp_source_reconstruction_output_view = output_view;
    frame.pp_source_reconstruction_image_format = source_image_format;
    frame.pp_source_reconstruction_view_format = source_view_format;
    frame.pp_source_reconstruction_mapping = source_mapping;
    frame.pp_source_reconstruction_source_extent = source_extent;
    frame.pp_source_reconstruction_output_format = output_format;
    frame.pp_source_reconstruction_output_extent = output_extent;
    return true;
}

} // namespace

class FinalGuestSurfaceContentState {
public:
    struct PendingCapture {
        FinalGuestSurfaceFrameStamp stamp{};
        FinalGuestSurfaceTransport transport{};
        FinalGuestSurfaceTilePlan plan{};
        FinalGuestSurfaceReadbackSlotPool::Token slot{};
        u64 slot_offset{};

        [[nodiscard]] bool HasCopy() const noexcept {
            return static_cast<bool>(slot) && plan.status == FinalGuestSurfaceStatus::Complete;
        }
    };

    FinalGuestSurfaceContentState(const Instance& instance_, Scheduler& scheduler_,
                                  FinalGuestSurfaceContentConfig config_)
        : scheduler{scheduler_}, config{config_}, reducer{config.lag, config.watch_ordinals},
          calibrated_triplets{config.calibrated_triplets, config.watch_ordinals, config.window,
                              config.expected_calibrations},
          screenshot_calibration{true},
          slot_stride{Common::AlignUp<u64>(FinalGuestSurfaceTileLimits{}.max_bytes,
                                           std::max<u64>(1, instance_.NonCoherentAtomSize()))},
          download{instance_,
                   scheduler,
                   VideoCore::MemoryUsage::Download,
                   0,
                   vk::BufferUsageFlagBits::eTransferDst,
                   slot_stride * FinalGuestSurfaceReadbackSlotPool::MaxSlots} {
        LOG_INFO(Render,
                 "FinalGuestSurfaceContentConfig enabled=1 stage={} frame_start={} frame_count={} "
                 "logical_window=32 logical_stride=16 max_windows={} copy_regions={} max_bytes={} "
                 "slots={} lag_cadence_us={} lag_tolerance_us={} selector_count={} "
                 "selector_status={} selector_loss={} calibrated_triplets={} "
                 "expected_calibrations={}",
                 static_cast<u32>(config.stage), config.window.frame_start,
                 config.window.frame_count, FinalGuestSurfaceTilePlan::MaxTiles,
                 config.stage == FinalGuestSurfaceStage::PpSourceReconstruction ? 4
                 : config.stage == FinalGuestSurfaceStage::PpSampledInput       ? 3
                                                                                : 1,
                 FinalGuestSurfaceTileLimits{}.max_bytes,
                 FinalGuestSurfaceReadbackSlotPool::MaxSlots, config.lag.cadence_us,
                 config.lag.tolerance_us, config.watch_ordinals.count,
                 static_cast<u32>(config.watch_ordinals.status), config.watch_ordinals.loss,
                 config.calibrated_triplets, config.expected_calibrations);
    }

    [[nodiscard]] bool ShouldCapture(u64 sequence) const noexcept {
        return config.window.Contains(sequence);
    }

    [[nodiscard]] const FinalGuestSurfaceWatchOrdinals& WatchOrdinals() const noexcept {
        return config.watch_ordinals;
    }

    [[nodiscard]] bool IsPostPpStage() const noexcept {
        return config.stage == FinalGuestSurfaceStage::PostPp;
    }

    [[nodiscard]] bool IsPpInputShadowStage() const noexcept {
        return config.stage == FinalGuestSurfaceStage::PpInputShadow;
    }

    [[nodiscard]] bool IsPpSampledInputStage() const noexcept {
        return config.stage == FinalGuestSurfaceStage::PpSampledInput ||
               config.stage == FinalGuestSurfaceStage::PpSourceReconstruction;
    }

    [[nodiscard]] bool IsPpSourceReconstructionStage() const noexcept {
        return config.stage == FinalGuestSurfaceStage::PpSourceReconstruction;
    }

    [[nodiscard]] bool IsPresentStage() const noexcept {
        return IsPresentFinalGuestSurfaceStage(config.stage);
    }

    [[nodiscard]] FinalGuestSurfaceStage Stage() const noexcept {
        return config.stage;
    }

    [[nodiscard]] PendingCapture PrepareGuest(FinalGuestSurfaceFrameStamp stamp,
                                              VideoCore::Image& image, u32 frame_output_width,
                                              u32 frame_output_height) {
        ++selected_frames;
        PendingCapture pending{
            .stamp = stamp,
            .transport =
                {
                    .surface_identity = image.image_uid,
                    .backing_generation = image.EnsureDiagnosticBackingGeneration(),
                    .format = ToFinalGuestSurfaceFormat(image.backing->image.image_ci.format),
                    .width = image.info.size.width,
                    .height = image.info.size.height,
                },
        };
        pending.plan = PlanFinalGuestSurfaceTiles({
            .width = image.info.size.width,
            .height = image.info.size.height,
            .depth = image.info.size.depth,
            .mip_level = 0,
            .mip_levels = image.info.resources.levels,
            .base_array_layer = 0,
            .array_layers = image.info.resources.layers,
            .samples = image.backing->num_samples,
            .type = ToFinalGuestSurfaceImageType(image.info.type),
            .aspect = image.aspect_mask == vk::ImageAspectFlagBits::eColor
                          ? FinalGuestSurfaceAspect::Color
                          : FinalGuestSurfaceAspect::Other,
            .format = pending.transport.format,
            .stage = FinalGuestSurfaceStage::GuestPreFsr,
            .logical_width = frame_output_width,
            .logical_height = frame_output_height,
            .logical_full_fit = true,
            .logical_top_left = true,
            .logical_no_y_flip = true,
        });
        if (pending.plan.status != FinalGuestSurfaceStatus::Complete) {
            return pending;
        }
        const auto slot = slots.TryAcquire();
        if (!slot) {
            pending.plan.status = FinalGuestSurfaceStatus::BusyLoss;
            pending.plan.loss.busy = 1;
            return pending;
        }
        pending.slot = *slot;
        pending.slot_offset = static_cast<u64>(slot->slot) * slot_stride;
        return pending;
    }

    [[nodiscard]] PendingCapture PreparePostPp(FinalGuestSurfaceFrameStamp stamp, u32 width,
                                               u32 height, vk::Format format, bool hdr) {
        ++selected_frames;
        PendingCapture pending{
            .stamp = stamp,
            .transport =
                post_pp_transport.Observe(ToFinalGuestSurfaceFormat(format), width, height, hdr),
        };
        pending.plan = PlanFinalGuestSurfaceTiles({
            .width = width,
            .height = height,
            .depth = 1,
            .mip_level = 0,
            .mip_levels = 1,
            .base_array_layer = 0,
            .array_layers = 1,
            .samples = 1,
            .type = FinalGuestSurfaceImageType::Color2D,
            .aspect = FinalGuestSurfaceAspect::Color,
            .format = pending.transport.format,
            .comparison = FinalGuestSurfaceComparison::LocalizedVisualReturn,
            .stage = FinalGuestSurfaceStage::PostPp,
            .logical_width = width,
            .logical_height = height,
            .logical_full_fit = true,
            .logical_top_left = true,
            .logical_no_y_flip = true,
        });
        if (pending.plan.status != FinalGuestSurfaceStatus::Complete) {
            return pending;
        }
        const auto slot = slots.TryAcquire();
        if (!slot) {
            pending.plan.status = FinalGuestSurfaceStatus::BusyLoss;
            pending.plan.loss.busy = 1;
            return pending;
        }
        pending.slot = *slot;
        pending.slot_offset = static_cast<u64>(slot->slot) * slot_stride;
        return pending;
    }

    [[nodiscard]] PendingCapture PreparePpInputShadow(
        FinalGuestSurfaceFrameStamp stamp, const FinalGuestSurfacePpInputMetadata& metadata) {
        ++selected_frames;
        PendingCapture pending{
            .stamp = stamp,
            .transport =
                {
                    .surface_identity = 1,
                    .backing_generation = metadata.config_generation,
                    .format = metadata.output_format,
                    .width = metadata.output_width,
                    .height = metadata.output_height,
                },
        };
        pending.plan = PlanFinalGuestSurfaceTiles({
            .width = metadata.output_width,
            .height = metadata.output_height,
            .depth = 1,
            .mip_level = 0,
            .mip_levels = 1,
            .base_array_layer = 0,
            .array_layers = 1,
            .samples = 1,
            .type = FinalGuestSurfaceImageType::Color2D,
            .aspect = FinalGuestSurfaceAspect::Color,
            .format = metadata.output_format,
            .comparison = FinalGuestSurfaceComparison::LocalizedVisualReturn,
            .stage = FinalGuestSurfaceStage::PpInputShadow,
            .logical_width = metadata.output_width,
            .logical_height = metadata.output_height,
            .logical_full_fit = true,
            .logical_top_left = true,
            .logical_no_y_flip = true,
        });
        if (!metadata.valid) {
            pending.plan.status = FinalGuestSurfaceStatus::InvalidationLoss;
            pending.plan.loss.invalidation = 1;
            return pending;
        }
        if (metadata.config_generation != last_metadata_generation) {
            last_metadata_generation = metadata.config_generation;
            LOG_INFO(Render, "FGSCM g={} fb={} sf={} sw={} sh={} of={} ow={} oh={} gm={} hdr={}",
                     metadata.config_generation, metadata.fsr_bypassed,
                     static_cast<u32>(metadata.source_format), metadata.source_width,
                     metadata.source_height, static_cast<u32>(metadata.output_format),
                     metadata.output_width, metadata.output_height, metadata.gamma_bits,
                     metadata.hdr);
        }
        if (pending.plan.status != FinalGuestSurfaceStatus::Complete) {
            return pending;
        }
        const auto slot = slots.TryAcquire();
        if (!slot) {
            pending.plan.status = FinalGuestSurfaceStatus::BusyLoss;
            pending.plan.loss.busy = 1;
            return pending;
        }
        pending.slot = *slot;
        pending.slot_offset = static_cast<u64>(slot->slot) * slot_stride;
        return pending;
    }

    [[nodiscard]] PendingCapture PreparePpSampledInput(
        FinalGuestSurfaceFrameStamp stamp, const FinalGuestSurfaceSampledInputMetadata& metadata,
        const PpSourceBackingFootprintPlan& source_backing,
        const HostPasses::PpSourceReconstructionPlan& source_reconstruction,
        FinalGuestSurfaceStatus observation_status) {
        ++selected_frames;
        PendingCapture pending{
            .stamp = stamp,
            .transport =
                {
                    .surface_identity = 1,
                    .backing_generation = metadata.config_generation,
                    .format = metadata.output_format,
                    .width = metadata.output_width,
                    .height = metadata.output_height,
                },
        };
        const auto output_plan = PlanFinalGuestSurfaceTiles({
            .width = metadata.output_width,
            .height = metadata.output_height,
            .depth = 1,
            .mip_level = 0,
            .mip_levels = 1,
            .base_array_layer = 0,
            .array_layers = 1,
            .samples = 1,
            .type = FinalGuestSurfaceImageType::Color2D,
            .aspect = FinalGuestSurfaceAspect::Color,
            .format = metadata.output_format,
            .comparison = FinalGuestSurfaceComparison::LocalizedVisualReturn,
            .stage = config.stage,
            .logical_width = metadata.output_width,
            .logical_height = metadata.output_height,
            .logical_full_fit = true,
            .logical_top_left = true,
            .logical_no_y_flip = true,
        });
        const auto pair = PlanPpSampledInputPairedCapture({
            .enabled = true,
            .width = metadata.output_width,
            .height = metadata.output_height,
            .output_format = metadata.output_format,
            .sampled_format = FinalGuestSurfaceFormat::Rgba16Float,
            .slot_bytes = slot_stride,
            .alignment = 8,
        });
        pending.plan = MakePpSampledInputSourceBackingTilePlan(
            MakePpSampledInputPairedTilePlan(output_plan, pair), source_backing, slot_stride, 16);
        if (IsPpSourceReconstructionStage()) {
            pending.plan =
                HostPasses::AttachPpSourceReconstructionPlane(pending.plan, source_reconstruction);
        }
        pending.plan = ApplyPpSampledInputObservationStatus(pending.plan, observation_status);
        if (!metadata.valid && observation_status == FinalGuestSurfaceStatus::Complete) {
            pending.plan = ApplyPpSampledInputObservationStatus(
                pending.plan, FinalGuestSurfaceStatus::InvalidationLoss);
        }
        if (metadata.config_generation != last_metadata_generation) {
            last_metadata_generation = metadata.config_generation;
            LOG_INFO(Render,
                     "FGSCM g={} fb={} sf={} sw={} sh={} of={} ow={} oh={} gm={} mip={}/{} "
                     "layer={}/{} bound_mip={}/{} bound_layer={}/{} mismatch={} srgb={} snap={}",
                     metadata.config_generation, metadata.fsr_bypassed,
                     static_cast<u32>(metadata.source_format), metadata.source_width,
                     metadata.source_height, static_cast<u32>(metadata.output_format),
                     metadata.output_width, metadata.output_height, metadata.gamma_bits,
                     metadata.resolved_base_mip, metadata.resolved_mip_count,
                     metadata.resolved_base_layer, metadata.resolved_layer_count,
                     metadata.bound_base_mip, metadata.bound_mip_count, metadata.bound_base_layer,
                     metadata.bound_layer_count, metadata.resolved_range_mismatch,
                     metadata.source_view_srgb, metadata.settings_snapshot_matches_push);
        }
        if (pending.plan.status != FinalGuestSurfaceStatus::Complete) {
            return pending;
        }
        const auto slot = slots.TryAcquire();
        if (!slot) {
            pending.plan.status = FinalGuestSurfaceStatus::BusyLoss;
            pending.plan.loss.busy = 1;
            return pending;
        }
        pending.slot = *slot;
        pending.slot_offset = static_cast<u64>(slot->slot) * slot_stride;
        return pending;
    }

    void ReportPpSampledInputLoss(const FinalGuestSurfaceSampledInputTakeResult& frame,
                                  FinalGuestSurfaceStatus status) {
        if (!frame.emit || !ShouldCapture(frame.payload.sequence)) {
            return;
        }
        auto metadata = frame.payload.metadata;
        metadata.valid = false;
        auto pending = PreparePpSampledInput(
            {frame.payload.sequence, frame.payload.process_time_us}, metadata,
            frame.payload.source_backing, frame.payload.source_reconstruction, status);
        ASSERT(!pending.HasCopy());
        DeferReport(std::move(pending));
    }

    void Record(PendingCapture pending, vk::Image image, vk::CommandBuffer cmdbuf) {
        if (!pending.HasCopy()) {
            DeferReport(std::move(pending));
            return;
        }
        const vk::BufferMemoryBarrier2 pre_barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eHost,
            .srcAccessMask = vk::AccessFlagBits2::eHostRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = download.Handle(),
            .offset = pending.slot_offset,
            .size = slot_stride,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });

        const vk::BufferImageCopy copy{
            .bufferOffset = pending.slot_offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent = {pending.plan.surface_width, pending.plan.surface_height, 1},
        };
        cmdbuf.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, download.Handle(), 1,
                                 &copy);

        const vk::BufferMemoryBarrier2 post_barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
            .buffer = download.Handle(),
            .offset = pending.slot_offset,
            .size = slot_stride,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &post_barrier,
        });
        DeferReport(std::move(pending));
    }

    void RecordPpSampledInput(PendingCapture pending, vk::Image output_image,
                              vk::Image sampled_image, vk::Buffer source_backing_snapshot,
                              vk::Image source_reconstruction_image, vk::CommandBuffer cmdbuf) {
        if (!pending.HasCopy()) {
            DeferReport(std::move(pending));
            return;
        }
        const bool has_reconstruction =
            pending.plan.paired_reconstruction_format != FinalGuestSurfaceFormat::Unsupported;
        ASSERT(pending.plan.paired_sampled_format == FinalGuestSurfaceFormat::Rgba16Float &&
               pending.plan.paired_backing_format != FinalGuestSurfaceFormat::Unsupported &&
               pending.plan.copy_region_count == (has_reconstruction ? 4u : 3u) &&
               pending.plan.paired_sampled_offset != 0 && pending.plan.paired_backing_offset != 0 &&
               source_backing_snapshot && (!has_reconstruction || source_reconstruction_image));
        const vk::BufferMemoryBarrier2 pre_barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eHost,
            .srcAccessMask = vk::AccessFlagBits2::eHostRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = download.Handle(),
            .offset = pending.slot_offset,
            .size = slot_stride,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });

        const auto make_copy = [&](u64 offset) {
            return vk::BufferImageCopy{
                .bufferOffset = offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = {0, 0, 0},
                .imageExtent = {pending.plan.surface_width, pending.plan.surface_height, 1},
            };
        };
        const auto output_copy = make_copy(pending.slot_offset);
        cmdbuf.copyImageToBuffer(output_image, vk::ImageLayout::eTransferSrcOptimal,
                                 download.Handle(), 1, &output_copy);
        const auto sampled_copy =
            make_copy(pending.slot_offset + pending.plan.paired_sampled_offset);
        cmdbuf.copyImageToBuffer(sampled_image, vk::ImageLayout::eTransferSrcOptimal,
                                 download.Handle(), 1, &sampled_copy);
        const vk::BufferCopy backing_copy{
            .srcOffset = 0,
            .dstOffset = pending.slot_offset + pending.plan.paired_backing_offset,
            .size = pending.plan.paired_backing_bytes,
        };
        cmdbuf.copyBuffer(source_backing_snapshot, download.Handle(), 1, &backing_copy);
        if (has_reconstruction) {
            const auto reconstruction_copy =
                make_copy(pending.slot_offset + pending.plan.paired_reconstruction_offset);
            cmdbuf.copyImageToBuffer(source_reconstruction_image,
                                     vk::ImageLayout::eTransferSrcOptimal, download.Handle(), 1,
                                     &reconstruction_copy);
        }

        const vk::BufferMemoryBarrier2 post_barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
            .buffer = download.Handle(),
            .offset = pending.slot_offset,
            .size = slot_stride,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &post_barrier,
        });
        DeferReport(std::move(pending));
    }

    void CalibrateScreenshots(const FinalGuestSurfaceFrameDiagnosticStamp& stamp,
                              FinalGuestSurfacePresentationMapping mapping, u32 count,
                              u64 fallback_process_time_us) {
        for (u32 request = 0; request < count; ++request) {
            const auto report =
                screenshot_calibration.Observe(stamp, mapping, fallback_process_time_us);
            if (!report.emit) {
                continue;
            }
            if (report.emit_mapping) {
                LOG_INFO(Render, "{}", FormatFinalGuestSurfaceCompactMapping(report));
            }
            LOG_INFO(Render, "{}", FormatFinalGuestSurfaceCompactCalibration(report));
            if (config.calibrated_triplets) {
                const FinalGuestSurfaceCalibratedStamp calibrated{
                    .request_ordinal = report.request_ordinal,
                    .sequence = report.surface_sequence,
                    .process_time_us = report.surface_process_time_us,
                    .valid = stamp.valid && report.surface_sequence != 0 &&
                             report.surface_process_time_us != 0 && !report.fallback_time &&
                             report.exact_scaled_mapping && report.mapping_loss == 0 &&
                             report.overflow_loss == 0 && !report.overflow_marker,
                };
                scheduler.DeferOperation([this, calibrated] { ConsumeCalibration(calibrated); });
            }
        }
    }

private:
    void DeferReport(PendingCapture pending) {
        scheduler.DeferOperation(
            [this, pending = std::move(pending)]() mutable { Consume(std::move(pending)); });
    }

    void LogCalibratedReports() {
        for (const auto& report : calibrated_triplets.TakeReports()) {
            LOG_INFO(Render, "{}", FormatFinalGuestSurfaceCalibratedReport(report));
        }
    }

    void ConsumeCalibration(FinalGuestSurfaceCalibratedStamp stamp) {
        calibrated_triplets.ObserveCalibration(stamp, reducer);
        if (stamp.request_ordinal == config.expected_calibrations) {
            calibrated_triplets.Finish(reducer);
        }
        LogCalibratedReports();
        if (calibrated_triplets.CoverageReady() && !calibrated_coverage_logged) {
            LOG_INFO(Render, "{}",
                     FormatFinalGuestSurfaceCalibratedCoverage(calibrated_triplets.GetCoverage()));
            calibrated_coverage_logged = true;
        }
    }

    void Consume(PendingCapture pending) {
        std::span<const std::byte> bytes{};
        if (pending.HasCopy()) {
            FinalGuestSurfaceReadbackCompletion completion;
            const auto status = completion.TryConsume(true, download.is_coherent, [&] {
                return download.InvalidateMappedRange(pending.slot_offset, slot_stride);
            });
            if (status == FinalGuestSurfaceStatus::InvalidationLoss) {
                pending.plan.status = status;
                pending.plan.loss.invalidation = 1;
            } else {
                bytes = {reinterpret_cast<const std::byte*>(download.mapped_data.data() +
                                                            pending.slot_offset),
                         pending.plan.sample_bytes};
            }
        }
        const auto report = reducer.Observe(pending.stamp.sequence, pending.stamp.process_time_us,
                                            pending.transport, pending.plan, bytes);
        calibrated_triplets.Reconcile(reducer);
        LogCalibratedReports();
        ++emitted_frames;
        complete_frames += report.status == FinalGuestSurfaceStatus::Complete && !report.loss.Any();
        loss_frames += report.loss.Any() || report.selector_loss != 0;
        if (FinalGuestSurfaceLogPolicy(config.stage).verbose_frame_reports) {
            LOG_INFO(Render, "{}", FormatFinalGuestSurfaceCompactReport(report));
        }

        if (pending.slot && !slots.ReleaseAfterCpuConsume(pending.slot)) {
            LOG_ERROR(Render,
                      "FinalGuestSurfaceContent sequence={} process_time_us={} surface_ordinal={} "
                      "tiles={} status={} slot_release_loss=1",
                      report.sequence, report.process_time_us, report.surface_ordinal,
                      report.tile_count,
                      static_cast<u32>(FinalGuestSurfaceStatus::AlreadyConsumed));
            ++loss_frames;
        }
        if (config.window.IsFinal(pending.stamp.sequence)) {
            LOG_INFO(
                Render,
                "FinalGuestSurfaceContentCoverage sequence={} process_time_us={} "
                "stage={} surface_ordinal={} tiles={} status={} selected={} emitted={} complete={} "
                "loss={} selector_count={} selector_status={} selector_loss={}",
                report.sequence, report.process_time_us, static_cast<u32>(report.stage),
                report.surface_ordinal, report.tile_count, static_cast<u32>(report.status),
                selected_frames, emitted_frames, complete_frames, loss_frames,
                report.selector_count, static_cast<u32>(report.selector_status),
                report.selector_loss);
        }
    }

    Scheduler& scheduler;
    FinalGuestSurfaceContentConfig config{};
    FinalGuestSurfaceReducer reducer;
    FinalGuestSurfaceCalibratedTriplets calibrated_triplets;
    FinalGuestSurfaceScreenshotCalibration screenshot_calibration;
    FinalGuestSurfacePostPpTransportTracker post_pp_transport;
    FinalGuestSurfaceReadbackSlotPool slots{};
    u64 slot_stride{};
    VideoCore::Buffer download;
    u32 selected_frames{};
    u32 emitted_frames{};
    u32 complete_frames{};
    u32 loss_frames{};
    u64 last_metadata_generation{};
    bool calibrated_coverage_logged{};
};

static std::string SanitizeFilenameComponent(std::string value) {
    for (char& c : value) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_' && c != '-') {
            c = '_';
        }
    }
    if (value.empty()) {
        return "UNKNOWN";
    }
    return value;
}

static std::vector<std::filesystem::path> BuildScreenshotPaths(const ScreenshotKind kind,
                                                               const u32 count) {
    static std::atomic<u64> screenshot_sequence{0};
    std::vector<std::filesystem::path> paths{};
    if (count == 0) {
        return paths;
    }

    const auto& screenshots_dir = Common::FS::GetUserPath(Common::FS::PathType::ScreenshotsDir);
    std::filesystem::create_directories(screenshots_dir);

    const auto game_id =
        SanitizeFilenameComponent(std::string(Common::ElfInfo::Instance().GameSerial()));
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
        1000;

    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    std::ostringstream stamp;
    stamp << std::put_time(&local_tm, "%Y%m%d_%H%M%S") << '_' << std::setw(3) << std::setfill('0')
          << ms;

    const char* suffix = kind == ScreenshotKind::GameOnly ? "game" : "hud";
    const auto first_sequence = screenshot_sequence.fetch_add(count, std::memory_order_relaxed);

    paths.reserve(count);
    const auto stamp_str = stamp.str();
    for (u32 i = 0; i < count; ++i) {
        paths.emplace_back(screenshots_dir / fmt::format("{}_{}_{}_{:06}.png", game_id, stamp_str,
                                                         suffix, first_sequence + i));
    }

    return paths;
}

static float PqToNits(const float encoded) {
    // ST.2084 inverse EOTF
    constexpr float m1 = 2610.0f / 16384.0f;
    constexpr float m2 = 2523.0f / 32.0f;
    constexpr float c1 = 3424.0f / 4096.0f;
    constexpr float c2 = 2413.0f / 128.0f;
    constexpr float c3 = 2392.0f / 128.0f;

    const float v = std::clamp(encoded, 0.0f, 1.0f);
    const float vp = std::pow(v, 1.0f / m2);
    const float num = std::max(vp - c1, 0.0f);
    const float den = std::max(c2 - c3 * vp, 1e-6f);
    return 10000.0f * std::pow(num / den, 1.0f / m1);
}

static float ToneMapToSdrLinear(const float nits) {
    // Map absolute HDR luminance into SDR [0,1], preserving 100-nit white.
    constexpr float sdr_white_nits = 100.0f;
    const float x = std::max(nits, 0.0f) / sdr_white_nits;
    const float mapped = (2.0f * x) / (1.0f + x);
    return std::clamp(mapped, 0.0f, 1.0f);
}

static float LinearToSrgb(const float linear) {
    const float x = std::clamp(linear, 0.0f, 1.0f);
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

static const std::array<float, 1024>& GetPqDecodeNitsLut() {
    static const std::array<float, 1024> lut = [] {
        std::array<float, 1024> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = PqToNits(static_cast<float>(i) / 1023.0f);
        }
        return values;
    }();
    return lut;
}

static const std::array<u8, 1024>& GetUnorm10ToU8Lut() {
    static const std::array<u8, 1024> lut = [] {
        std::array<u8, 1024> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<u8>((i * 255u + 511u) / 1023u);
        }
        return values;
    }();
    return lut;
}

static void CopyImageToReadback(const vk::CommandBuffer& cmdbuf, const vk::Image image,
                                const vk::ImageLayout layout, ScreenshotReadback& readback) {
    const vk::BufferImageCopy copy_region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {readback.width, readback.height, 1},
    };
    cmdbuf.copyImageToBuffer(image, layout, readback.buffer.Handle(), copy_region);
}

static bool ConvertReadbackToRgba8(const ScreenshotReadback& readback, std::vector<u8>& out_rgba) {
    const u64 pixel_count = static_cast<u64>(readback.width) * static_cast<u64>(readback.height);
    const u64 byte_size = pixel_count * 4;
    if (readback.buffer.mapped_data.size() < byte_size) {
        LOG_ERROR(Render_Vulkan, "Screenshot readback buffer size mismatch (have {}, need {})",
                  readback.buffer.mapped_data.size(), byte_size);
        return false;
    }

    const auto src =
        std::span<const u8>{readback.buffer.mapped_data.data(), static_cast<size_t>(byte_size)};
    out_rgba.resize(static_cast<size_t>(byte_size));

    switch (readback.format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        std::memcpy(out_rgba.data(), src.data(), out_rgba.size());
        for (u64 i = 0; i < pixel_count; ++i) {
            out_rgba[static_cast<size_t>(i) * 4 + 3] = 255;
        }
        return true;
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            out_rgba[o + 0] = src[o + 2];
            out_rgba[o + 1] = src[o + 1];
            out_rgba[o + 2] = src[o + 0];
            out_rgba[o + 3] = 255;
        }
        return true;
    case vk::Format::eA2R10G10B10UnormPack32: {
        const auto& pq_decode_lut = GetPqDecodeNitsLut();
        const auto& unorm10_to_u8 = GetUnorm10ToU8Lut();

        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            const u32 packed = static_cast<u32>(src[o + 0]) | (static_cast<u32>(src[o + 1]) << 8) |
                               (static_cast<u32>(src[o + 2]) << 16) |
                               (static_cast<u32>(src[o + 3]) << 24);
            const u32 b = (packed >> 0) & 0x3FF;
            const u32 g = (packed >> 10) & 0x3FF;
            const u32 r = (packed >> 20) & 0x3FF;

            if (readback.hdr_encoded) {
                // Rec.2020 + PQ. Convert to SDR Rec.709 for PNG output.
                const float r2020 = pq_decode_lut[r];
                const float g2020 = pq_decode_lut[g];
                const float b2020 = pq_decode_lut[b];

                const float r709_nits = 1.6605f * r2020 - 0.5876f * g2020 - 0.0728f * b2020;
                const float g709_nits = -0.1246f * r2020 + 1.1329f * g2020 - 0.0083f * b2020;
                const float b709_nits = -0.0182f * r2020 - 0.1006f * g2020 + 1.1187f * b2020;

                const float r_srgb = LinearToSrgb(ToneMapToSdrLinear(r709_nits));
                const float g_srgb = LinearToSrgb(ToneMapToSdrLinear(g709_nits));
                const float b_srgb = LinearToSrgb(ToneMapToSdrLinear(b709_nits));

                out_rgba[o + 0] = static_cast<u8>(std::clamp(r_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 1] = static_cast<u8>(std::clamp(g_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 2] = static_cast<u8>(std::clamp(b_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
            } else {
                out_rgba[o + 0] = unorm10_to_u8[r];
                out_rgba[o + 1] = unorm10_to_u8[g];
                out_rgba[o + 2] = unorm10_to_u8[b];
            }
            out_rgba[o + 3] = 255;
        }
        return true;
    }
    case vk::Format::eA2B10G10R10UnormPack32: {
        const auto& pq_decode_lut = GetPqDecodeNitsLut();
        const auto& unorm10_to_u8 = GetUnorm10ToU8Lut();

        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            const u32 packed = static_cast<u32>(src[o + 0]) | (static_cast<u32>(src[o + 1]) << 8) |
                               (static_cast<u32>(src[o + 2]) << 16) |
                               (static_cast<u32>(src[o + 3]) << 24);
            const u32 r = (packed >> 0) & 0x3FF;
            const u32 g = (packed >> 10) & 0x3FF;
            const u32 b = (packed >> 20) & 0x3FF;

            if (readback.hdr_encoded) {
                // HDR swapchain path is Rec.2020 + PQ. Convert to SDR Rec.709 for PNG output.
                const float r2020 = pq_decode_lut[r];
                const float g2020 = pq_decode_lut[g];
                const float b2020 = pq_decode_lut[b];

                const float r709_nits = 1.6605f * r2020 - 0.5876f * g2020 - 0.0728f * b2020;
                const float g709_nits = -0.1246f * r2020 + 1.1329f * g2020 - 0.0083f * b2020;
                const float b709_nits = -0.0182f * r2020 - 0.1006f * g2020 + 1.1187f * b2020;

                const float r_srgb = LinearToSrgb(ToneMapToSdrLinear(r709_nits));
                const float g_srgb = LinearToSrgb(ToneMapToSdrLinear(g709_nits));
                const float b_srgb = LinearToSrgb(ToneMapToSdrLinear(b709_nits));

                out_rgba[o + 0] = static_cast<u8>(std::clamp(r_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 1] = static_cast<u8>(std::clamp(g_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 2] = static_cast<u8>(std::clamp(b_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
            } else {
                out_rgba[o + 0] = unorm10_to_u8[r];
                out_rgba[o + 1] = unorm10_to_u8[g];
                out_rgba[o + 2] = unorm10_to_u8[b];
            }
            out_rgba[o + 3] = 255;
        }
        return true;
    }
    default:
        LOG_WARNING(Render_Vulkan, "Unsupported screenshot format: {}",
                    vk::to_string(readback.format));
        return false;
    }
}

static bool WritePng(const std::filesystem::path& path, const std::span<const u8> rgba,
                     const u32 width, const u32 height) {
    Common::FS::IOFile file(path, Common::FS::FileAccessMode::Create);
    if (!file.IsOpen()) {
        return false;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png_ptr == nullptr) {
        return false;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == nullptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr)) != 0) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    png_init_io(png_ptr, file.file);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    thread_local std::vector<png_bytep> rows;
    rows.resize(height);
    for (u32 y = 0; y < height; ++y) {
        rows[y] = const_cast<png_bytep>(rgba.data() + static_cast<size_t>(y) * width * 4);
    }

    png_write_image(png_ptr, rows.data());
    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;
}

static void SaveCompletedScreenshot(const ScreenshotReadback& readback) {
    std::vector<u8> rgba;
    if (!ConvertReadbackToRgba8(readback, rgba)) {
        return;
    }

    const auto& primary_path = readback.paths.front();
    if (!WritePng(primary_path, rgba, readback.width, readback.height)) {
        LOG_ERROR(Render_Vulkan, "Failed saving screenshot to {}", primary_path.string());
        return;
    }

    LOG_INFO(Render_Vulkan, "Saved screenshot: {}", primary_path.string());

    if (readback.notify) {
        std::ifstream file(primary_path, std::ios::binary);
        std::vector<u8> imgdata;
        if (file) {
            imgdata = std::vector<u8>(std::istreambuf_iterator<char>(file),
                                      std::istreambuf_iterator<char>());
        }
        shadNotifications::QueueNotification("Saved screenshot:\n" + primary_path.string(), 3.0f,
                                             shadNotifications::position::BottomRight, imgdata);
    }

    for (size_t i = 1; i < readback.paths.size(); ++i) {
        const auto& path = readback.paths[i];
        std::error_code ec{};
        std::filesystem::copy_file(primary_path, path, std::filesystem::copy_options::none, ec);
        if (ec) {
            // Fallback for platforms/filesystems where copy_file can fail for transient reasons.
            if (!WritePng(path, rgba, readback.width, readback.height)) {
                LOG_ERROR(Render_Vulkan, "Failed saving screenshot to {}", path.string());
                continue;
            }
        }

        LOG_INFO(Render_Vulkan, "Saved screenshot: {}", path.string());
        if (readback.notify) {
            std::ifstream file(path, std::ios::binary);
            std::vector<u8> imgdata;
            if (file) {
                imgdata = std::vector<u8>(std::istreambuf_iterator<char>(file),
                                          std::istreambuf_iterator<char>());
            }
            shadNotifications::QueueNotification("Saved screenshot:\n" + path.string(), 3.0f,
                                                 shadNotifications::position::BottomRight, imgdata);
        }
    }
}

static void SavePendingScreenshots(std::vector<ScreenshotReadback>& readbacks) {
    for (auto& readback : readbacks) {
        if (readback.paths.empty()) {
            continue;
        }

        const auto result = VideoCore::ConsumeCompletedReadback(
            readback.buffer.is_coherent,
            [&] { return readback.buffer.InvalidateMappedRange(0, readback.buffer.SizeBytes()); },
            [&] { SaveCompletedScreenshot(readback); });
        if (result == VideoCore::CompletedReadbackResult::InvalidationFailed) {
            LOG_ERROR(Render_Vulkan,
                      "Screenshot readback invalidation failed; screenshot output suppressed");
        }
    }
}

Presenter::Presenter(Frontend::WindowSDL& window_, AmdGpu::Liverpool* liverpool_)
    : window{window_}, liverpool{liverpool_},
      instance{window, EmulatorSettings.GetGpuId(), EmulatorSettings.IsVkValidationEnabled(),
               EmulatorSettings.IsVkCrashDiagnosticEnabled()},
      draw_scheduler{instance}, present_scheduler{instance}, swapchain{instance, window},
      rasterizer{std::make_unique<Rasterizer>(instance, draw_scheduler, liverpool)},
      texture_cache{rasterizer->GetTextureCache()} {
    if (const auto config = ReadFinalGuestSurfaceContentConfig()) {
        auto& content_scheduler =
            IsPresentFinalGuestSurfaceStage(config->stage) ? present_scheduler : draw_scheduler;
        final_guest_surface_content =
            std::make_unique<FinalGuestSurfaceContentState>(instance, content_scheduler, *config);
    }
    const u32 num_images = swapchain.GetImageCount();
    const vk::Device device = instance.GetDevice();

    // Create presentation frames.
    present_frames.resize(num_images);
    for (u32 i = 0; i < num_images; i++) {
        Frame& frame = present_frames[i];
        frame.id = i;
        auto fence = Check<"create present done fence">(
            device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        frame.present_done = fence;
        free_queue.push(&frame);
    }

    fsr_settings.enable = EmulatorSettings.IsFsrEnabled();
    fsr_settings.use_rcas = EmulatorSettings.IsRcasEnabled();
    fsr_settings.rcas_attenuation =
        static_cast<float>(EmulatorSettings.GetRcasAttenuation() / 1000.f);

    fsr_pass.Create(device, instance.GetAllocator(), num_images);
    pp_pass.Create(device, swapchain.GetSurfaceFormat().format,
                   final_guest_surface_content
                       ? PpDiagnosticModeForStage(final_guest_surface_content->Stage())
                       : HostPasses::PpDiagnosticMode::None);

    ImGui::Layer::AddLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
    ImGui::Friends::Register();
    ImGui::ShadNetNotify::Register();
    ImGui::InvitationPrompt::Register();
}

Presenter::~Presenter() {
    ImGui::InvitationPrompt::Unregister();
    ImGui::ShadNetNotify::Unregister();
    ImGui::Friends::Unregister();
    ImGui::Layer::RemoveLayer(Common::Singleton<Core::Devtools::Layer>::Instance());

    const bool present_surface =
        final_guest_surface_content && final_guest_surface_content->IsPresentStage();
    draw_scheduler.Finish();
    if (final_guest_surface_content && !present_surface) {
        draw_scheduler.PopPendingOperations();
    }
    present_scheduler.Finish();
    if (present_surface) {
        present_scheduler.PopPendingOperations();
    }
    Check(draw_scheduler.CommandBuffer().reset());
    Check(present_scheduler.CommandBuffer().reset());

    const vk::Device device = instance.GetDevice();
    for (auto& frame : present_frames) {
        DestroyPpSourceReconstructionResources(instance, frame);
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        device.destroyImageView(frame.image_view);
        if (frame.pp_input_shadow_image) {
            device.destroyImageView(frame.pp_input_shadow_view);
            vmaDestroyImage(instance.GetAllocator(), frame.pp_input_shadow_image,
                            frame.pp_input_shadow_allocation);
        }
        device.destroyFence(frame.present_done);
    }
}

bool Presenter::IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const {
    return std::ranges::find(vo_buffers_addr, color_buffer.Address()) != vo_buffers_addr.cend();
}

void Presenter::RecreateFrame(Frame* frame, u32 width, u32 height) {
    const vk::Device device = instance.GetDevice();
    frame->pp_input_shadow_state.Clear();
    frame->pp_sampled_input_state.Clear();
    DestroyPpSourceReconstructionResources(instance, *frame);
    if (frame->pp_input_shadow_view) {
        device.destroyImageView(frame->pp_input_shadow_view);
        frame->pp_input_shadow_view = nullptr;
    }
    if (frame->pp_input_shadow_image) {
        vmaDestroyImage(instance.GetAllocator(), frame->pp_input_shadow_image,
                        frame->pp_input_shadow_allocation);
        frame->pp_input_shadow_image = nullptr;
        frame->pp_input_shadow_allocation = {};
    }
    if (frame->imgui_texture) {
        ImGui::Vulkan::RemoveTexture(frame->imgui_texture);
    }
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);
    }
    if (frame->image) {
        vmaDestroyImage(instance.GetAllocator(), frame->image, frame->allocation);
    }

    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::ImageCreateInfo image_info = {
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &frame->allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}",
                     vk::to_string(vk::Result{result}));
        UNREACHABLE();
    }
    frame->image = vk::Image{unsafe_image};
    SetObjectName(device, frame->image, "Frame image #{}", frame->id);

    const vk::ImageViewCreateInfo view_info = {
        .image = frame->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    auto view = Check<"create frame image view">(device.createImageView(view_info));
    frame->image_view = view;

    const bool sampled_input_stage =
        final_guest_surface_content && final_guest_surface_content->IsPpSampledInputStage();
    if (final_guest_surface_content &&
        (final_guest_surface_content->IsPpInputShadowStage() || sampled_input_stage)) {
        auto shadow_image_info = image_info;
        shadow_image_info.flags = {};
        shadow_image_info.format = sampled_input_stage ? vk::Format::eR16G16B16A16Sfloat : format;
        shadow_image_info.usage =
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
        VkImage unsafe_shadow_image{};
        VkImageCreateInfo unsafe_shadow_image_info =
            static_cast<VkImageCreateInfo>(shadow_image_info);
        result = vmaCreateImage(instance.GetAllocator(), &unsafe_shadow_image_info, &alloc_info,
                                &unsafe_shadow_image, &frame->pp_input_shadow_allocation, nullptr);
        if (result != VK_SUCCESS) [[unlikely]] {
            LOG_CRITICAL(Render_Vulkan, "Failed allocating PP input shadow with error {}",
                         vk::to_string(vk::Result{result}));
            UNREACHABLE();
        }
        frame->pp_input_shadow_image = vk::Image{unsafe_shadow_image};
        SetObjectName(device, frame->pp_input_shadow_image, "PP input shadow #{}", frame->id);
        auto shadow_view_info = view_info;
        shadow_view_info.image = frame->pp_input_shadow_image;
        shadow_view_info.format = shadow_image_info.format;
        frame->pp_input_shadow_view =
            Check<"create PP input shadow image view">(device.createImageView(shadow_view_info));
    }
    frame->width = width;
    frame->height = height;

    frame->imgui_texture = ImGui::Vulkan::AddTexture(view, vk::ImageLayout::eShaderReadOnlyOptimal);
    frame->is_hdr = swapchain.GetHDR();
}

Frame* Presenter::PrepareLastFrame() {
    if (last_submit_frame == nullptr) {
        return nullptr;
    }

    Frame* frame = last_submit_frame;

    while (true) {
        vk::Result result = instance.GetDevice().waitForFences(frame->present_done, false,
                                                               std::numeric_limits<u64>::max());
        if (result == vk::Result::eSuccess) {
            break;
        }
        if (result == vk::Result::eTimeout) {
            continue;
        }
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
    }

    return frame;
}

static vk::Format GetFrameViewFormat(const Libraries::VideoOut::PixelFormat format) {
    switch (format) {
    case Libraries::VideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A2R10G10B10:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        break;
    }
    UNREACHABLE_MSG("Unknown format={}", static_cast<u32>(format));
    return {};
}

[[nodiscard]] static bool IsSrgbViewFormat(vk::Format format) noexcept {
    return format == vk::Format::eR8G8B8A8Srgb || format == vk::Format::eB8G8R8A8Srgb;
}

Frame* Presenter::PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address, FinalGuestSurfaceFrameStamp stamp) {
    auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
    const auto image_id = texture_cache.FindImage(desc);
    texture_cache.UpdateImage(image_id);

    Frame* frame = GetRenderFrame();

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    std::array<vk::ImageMemoryBarrier2, 2> pre_barriers{};
    u32 pre_barrier_count = 1;
    pre_barriers[0] = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange{frame_subresources},
    };
    const bool pp_input_shadow_stage =
        final_guest_surface_content && final_guest_surface_content->IsPpInputShadowStage();
    const bool pp_sampled_input_stage =
        final_guest_surface_content && final_guest_surface_content->IsPpSampledInputStage();
    const bool pp_source_reconstruction_stage =
        final_guest_surface_content && final_guest_surface_content->IsPpSourceReconstructionStage();
    const bool pp_diagnostic_stage = pp_input_shadow_stage || pp_sampled_input_stage;
    if (pp_diagnostic_stage && frame->pp_input_shadow_image) {
        pre_barriers[pre_barrier_count++] = vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eNone,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = frame->pp_input_shadow_image,
            .subresourceRange{frame_subresources},
        };
    }

    draw_scheduler.EndRendering();
    const auto cmdbuf = draw_scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = pre_barrier_count,
        .pImageMemoryBarriers = pre_barriers.data(),
    });

    VideoCore::ImageViewInfo view_info{};
    view_info.format = GetFrameViewFormat(attribute.attrib.pixel_format);
    // Exclude alpha from output frame to avoid blending with UI.
    view_info.mapping.a = vk::ComponentSwizzle::eOne;

    auto& image = texture_cache.GetImage(image_id);
    auto& source_view = image.FindView(view_info);
    auto image_view = *source_view.image_view;
    const vk::Extent2D image_size = {image.info.size.width, image.info.size.height};
    const bool present_surface =
        final_guest_surface_content && final_guest_surface_content->IsPresentStage();
    frame->final_surface_diagnostic.Assign(final_guest_surface_content != nullptr, stamp.sequence,
                                           stamp.process_time_us,
                                           present_surface ? frame->width : image_size.width,
                                           present_surface ? frame->height : image_size.height);
    expected_ratio = static_cast<float>(image_size.width) / static_cast<float>(image_size.height);

    std::optional<FinalGuestSurfaceContentState::PendingCapture> final_surface_capture;
    if (final_guest_surface_content && !present_surface &&
        final_guest_surface_content->ShouldCapture(stamp.sequence)) {
        final_surface_capture =
            final_guest_surface_content->PrepareGuest(stamp, image, frame->width, frame->height);
    }

    const auto capture_game_only = VideoCore::ConsumeGameOnlyScreenshotRequests();
    std::vector<ScreenshotReadback> pending_screenshots;
    if (capture_game_only.Total() > 0 ||
        (final_surface_capture && final_surface_capture->HasCopy())) {
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
                      cmdbuf);
    }
    if (capture_game_only.Total() > 0) {
        pending_screenshots.reserve(2);
        const bool hdr_encoded =
            attribute.attrib.pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq;

        // Capture the guest output before any host-side scaling (FSR/PP) is applied.
        const auto append_readback = [&](const u32 count, const bool notify) {
            if (count == 0) {
                return;
            }
            pending_screenshots.emplace_back(
                instance, draw_scheduler, ScreenshotKind::GameOnly, notify,
                BuildScreenshotPaths(ScreenshotKind::GameOnly, count), image_size.width,
                image_size.height, view_info.format, hdr_encoded);
            CopyImageToReadback(cmdbuf, image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                pending_screenshots.back());
        };
        append_readback(capture_game_only.notifying_count, true);
        append_readback(capture_game_only.silent_count, false);
    }
    if (final_surface_capture) {
        final_guest_surface_content->Record(std::move(*final_surface_capture), image.GetImage(),
                                            cmdbuf);
    }

    // Continue with host-side passes that draw the displayed (scaled) frame.
    image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {},
                  cmdbuf);

    image_view = fsr_pass.Render(cmdbuf, image_view, image_size, {frame->width, frame->height},
                                 fsr_settings, frame->is_hdr);
    const auto frame_pp_settings = pp_settings;
    FinalGuestSurfacePpInputMetadata pp_input_metadata{};
    if (pp_input_shadow_stage) {
        pp_input_metadata = pp_input_shadow_config.Observe({
            .fsr_enabled = fsr_settings.enable,
            .input_width = image_size.width,
            .input_height = image_size.height,
            .output_width = frame->width,
            .output_height = frame->height,
            .source_format = ToFinalGuestSurfaceFormat(view_info.format),
            .output_format = ToFinalGuestSurfaceFormat(swapchain.GetSurfaceFormat().format),
            .gamma_bits = std::bit_cast<u32>(frame_pp_settings.gamma),
            .pp_hdr = frame_pp_settings.hdr != 0,
            .frame_hdr = frame->is_hdr,
        });
    }
    FinalGuestSurfaceSampledInputMetadata sampled_input_metadata{};
    if (pp_sampled_input_stage) {
        const auto view_assessment = AssessPpSampledInputSourceView({
            .resolved_base_mip = desc.view_info.range.base.level,
            .resolved_mip_count = desc.view_info.range.extent.levels,
            .resolved_base_layer = desc.view_info.range.base.layer,
            .resolved_layer_count = desc.view_info.range.extent.layers,
            .bound_base_mip = source_view.info.range.base.level,
            .bound_mip_count = source_view.info.range.extent.levels,
            .bound_base_layer = source_view.info.range.base.layer,
            .bound_layer_count = source_view.info.range.extent.layers,
        });
        if (view_assessment.resolved_range_mismatch && !pp_sampled_input_view_mismatch_logged) {
            pp_sampled_input_view_mismatch_logged = true;
            LOG_INFO(Render,
                     "PPSampledInputSourceView mismatch=1 resolved_mip={}/{} "
                     "resolved_layer={}/{} bound_mip={}/{} bound_layer={}/{}",
                     desc.view_info.range.base.level, desc.view_info.range.extent.levels,
                     desc.view_info.range.base.layer, desc.view_info.range.extent.layers,
                     source_view.info.range.base.level, source_view.info.range.extent.levels,
                     source_view.info.range.base.layer, source_view.info.range.extent.layers);
        }
        sampled_input_metadata = pp_sampled_input_config.Observe({
            .fsr_enabled = fsr_settings.enable,
            .input_width = image_size.width,
            .input_height = image_size.height,
            .output_width = frame->width,
            .output_height = frame->height,
            .source_format = ToFinalGuestSurfaceFormat(source_view.info.format),
            .output_format = ToFinalGuestSurfaceFormat(swapchain.GetSurfaceFormat().format),
            .resolved_base_mip = desc.view_info.range.base.level,
            .resolved_mip_count = desc.view_info.range.extent.levels,
            .resolved_base_layer = desc.view_info.range.base.layer,
            .resolved_layer_count = desc.view_info.range.extent.layers,
            .bound_base_mip = source_view.info.range.base.level,
            .bound_mip_count = source_view.info.range.extent.levels,
            .bound_base_layer = source_view.info.range.base.layer,
            .bound_layer_count = source_view.info.range.extent.layers,
            .source_image_uid = image.image_uid,
            .source_backing_generation = image.EnsureDiagnosticBackingGeneration(),
            .gamma_bits = std::bit_cast<u32>(frame_pp_settings.gamma),
            .source_view_srgb = IsSrgbViewFormat(source_view.info.format),
            .bound_view_observed = true,
            .settings_snapshot_matches_push = true,
            .pp_hdr = frame_pp_settings.hdr != 0,
            .frame_hdr = frame->is_hdr,
        });
    }
    const bool diagnostic_metadata_valid = pp_input_metadata.valid || sampled_input_metadata.valid;
    const bool shadow_written =
        pp_pass.Render(cmdbuf, image_view, image_size, *frame, frame_pp_settings,
                       diagnostic_metadata_valid ? frame->pp_input_shadow_view : vk::ImageView{});
    PpSourceBackingFootprintPlan source_backing_plan{};
    bool source_backing_captured{};
    if (pp_sampled_input_stage && final_guest_surface_content->ShouldCapture(stamp.sequence)) {
        source_backing_plan = PlanPpSourceBackingFootprints({
            .enabled = true,
            .in_window = true,
            .pp_draw_encoded = shadow_written,
            .fsr_bypassed = sampled_input_metadata.fsr_bypassed,
            .source_width = image_size.width,
            .source_height = image_size.height,
            .logical_width = frame->width,
            .logical_height = frame->height,
            .source_format = ToFinalGuestSurfaceFormat(image.backing->image.image_ci.format),
            .samples = image.backing->num_samples,
            .resolved_base_mip = sampled_input_metadata.resolved_base_mip,
            .resolved_mip_count = sampled_input_metadata.resolved_mip_count,
            .resolved_base_layer = sampled_input_metadata.resolved_base_layer,
            .resolved_layer_count = sampled_input_metadata.resolved_layer_count,
            .bound_base_mip = sampled_input_metadata.bound_base_mip,
            .bound_mip_count = sampled_input_metadata.bound_mip_count,
            .bound_base_layer = sampled_input_metadata.bound_base_layer,
            .bound_layer_count = sampled_input_metadata.bound_layer_count,
            .logical_full_fit = true,
            .logical_top_left = true,
            .logical_no_y_flip = true,
            .buffer_alignment = 16,
            .max_regions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals,
            .max_bytes = PpSourceBackingSnapshotBytes,
            .selector = final_guest_surface_content->WatchOrdinals(),
        });
        if (source_backing_plan.status == FinalGuestSurfaceStatus::Complete) {
            if (!frame->pp_source_backing_snapshot) {
                frame->pp_source_backing_snapshot = std::make_unique<VideoCore::Buffer>(
                    instance, draw_scheduler, VideoCore::MemoryUsage::DeviceLocal, 0,
                    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                    PpSourceBackingSnapshotBytes);
            }
            const auto handoff = PlanPpSourceBackingHandoff({
                .enabled = true,
                .frame_is_new = true,
                .metadata_valid = sampled_input_metadata.valid,
                .snapshot_buffer_available = frame->pp_source_backing_snapshot != nullptr,
                .backing = source_backing_plan,
            });
            if (handoff.copy) {
                if (const auto barrier = frame->pp_source_backing_snapshot->GetBarrier(
                        vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eTransfer)) {
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .bufferMemoryBarrierCount = 1,
                        .pBufferMemoryBarriers = &*barrier,
                    });
                }
                image.Transit(vk::ImageLayout::eTransferSrcOptimal,
                              vk::AccessFlagBits2::eTransferRead, source_view.info.range, cmdbuf);
                std::array<vk::BufferImageCopy, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>
                    copies{};
                for (u32 index = 0; index < source_backing_plan.region_count; ++index) {
                    const auto& region = source_backing_plan.regions[index];
                    copies[index] = {
                        .bufferOffset = region.buffer_offset,
                        .bufferRowLength = 0,
                        .bufferImageHeight = 0,
                        .imageSubresource =
                            {
                                .aspectMask = vk::ImageAspectFlagBits::eColor,
                                .mipLevel = sampled_input_metadata.bound_base_mip,
                                .baseArrayLayer = sampled_input_metadata.bound_base_layer,
                                .layerCount = 1,
                            },
                        .imageOffset = {static_cast<s32>(region.x), static_cast<s32>(region.y), 0},
                        .imageExtent = {region.width, region.height, 1},
                    };
                }
                cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                         frame->pp_source_backing_snapshot->Handle(),
                                         source_backing_plan.region_count, copies.data());
                image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::AccessFlagBits2::eShaderRead, source_view.info.range, cmdbuf);
                if (const auto barrier = frame->pp_source_backing_snapshot->GetBarrier(
                        vk::AccessFlagBits2::eTransferRead,
                        vk::PipelineStageFlagBits2::eTransfer)) {
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .bufferMemoryBarrierCount = 1,
                        .pBufferMemoryBarriers = &*barrier,
                    });
                }
                source_backing_captured = true;
            }
        }
    }
    HostPasses::PpSourceReconstructionPlan source_reconstruction_plan{};
    bool source_reconstruction_captured{};
    if (pp_source_reconstruction_stage && source_backing_captured &&
        final_guest_surface_content->ShouldCapture(stamp.sequence)) {
        const auto pair = PlanPpSampledInputPairedCapture({
            .enabled = true,
            .width = frame->width,
            .height = frame->height,
            .output_format = sampled_input_metadata.output_format,
            .sampled_format = FinalGuestSurfaceFormat::Rgba16Float,
            .slot_bytes = FinalGuestSurfaceTileLimits{}.max_bytes,
            .alignment = 8,
        });
        const u64 existing_readback_bytes =
            HostPasses::AlignPpSourceReconstructionOffset(pair.total_bytes, 16) +
            PpSourceBackingSnapshotBytes;
        auto reconstruction_descriptor = HostPasses::PpSourceReconstructionDescriptor{
            .enabled = true,
            .in_window = true,
            .frame_is_new = true,
            .visible_pp_draw_encoded = shadow_written,
            .sampled_metadata_valid = sampled_input_metadata.valid,
            .fsr_bypassed = sampled_input_metadata.fsr_bypassed,
            .source_view_matches_baseline = !sampled_input_metadata.resolved_range_mismatch,
            .source_view_srgb = sampled_input_metadata.source_view_srgb,
            .source_snapshot_available = true,
            .source_snapshot_view_available = true,
            .reconstruction_output_available = true,
            .source_width = image_size.width,
            .source_height = image_size.height,
            .output_width = frame->width,
            .output_height = frame->height,
            .bound_base_mip = sampled_input_metadata.bound_base_mip,
            .bound_mip_count = sampled_input_metadata.bound_mip_count,
            .bound_base_layer = sampled_input_metadata.bound_base_layer,
            .bound_layer_count = sampled_input_metadata.bound_layer_count,
            .source_format = ToFinalGuestSurfaceFormat(source_view.info.format),
            .output_format = sampled_input_metadata.output_format,
            .existing_readback_bytes = existing_readback_bytes,
            .slot_bytes = FinalGuestSurfaceTileLimits{}.max_bytes,
            .alignment = 16,
        };
        source_reconstruction_plan =
            HostPasses::PlanPpSourceReconstruction(reconstruction_descriptor);
        if (source_reconstruction_plan.status == FinalGuestSurfaceStatus::Complete) {
            const bool resources_available = EnsurePpSourceReconstructionResources(
                instance, *frame, image.backing->image.image_ci.format, source_view.info.format,
                source_view.info.mapping, image_size, swapchain.GetSurfaceFormat().format,
                {frame->width, frame->height});
            if (!resources_available) {
                reconstruction_descriptor.source_snapshot_available = false;
                reconstruction_descriptor.source_snapshot_view_available = false;
                reconstruction_descriptor.reconstruction_output_available = false;
                source_reconstruction_plan =
                    HostPasses::PlanPpSourceReconstruction(reconstruction_descriptor);
            }
        }
        if (source_reconstruction_plan.status == FinalGuestSurfaceStatus::Complete) {
            const vk::ImageSubresourceRange simple_range{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                          source_view.info.range, cmdbuf);
            const vk::ImageMemoryBarrier2 snapshot_to_transfer{
                .srcStageMask = vk::PipelineStageFlagBits2::eNone,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .image = frame->pp_source_reconstruction_snapshot_image,
                .subresourceRange = simple_range,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &snapshot_to_transfer,
            });
            const vk::ImageCopy source_copy{
                .srcSubresource = MakeImageSubresourceLayers(),
                .srcOffset = {0, 0, 0},
                .dstSubresource = MakeImageSubresourceLayers(),
                .dstOffset = {0, 0, 0},
                .extent = {image_size.width, image_size.height, 1},
            };
            cmdbuf.copyImage(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             frame->pp_source_reconstruction_snapshot_image,
                             vk::ImageLayout::eTransferDstOptimal, 1, &source_copy);
            image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
                          source_view.info.range, cmdbuf);
            const std::array reconstruction_barriers{
                vk::ImageMemoryBarrier2{
                    .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .image = frame->pp_source_reconstruction_snapshot_image,
                    .subresourceRange = simple_range,
                },
                vk::ImageMemoryBarrier2{
                    .srcStageMask = vk::PipelineStageFlagBits2::eNone,
                    .srcAccessMask = vk::AccessFlagBits2::eNone,
                    .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .image = frame->pp_source_reconstruction_output_image,
                    .subresourceRange = simple_range,
                },
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount = reconstruction_barriers.size(),
                .pImageMemoryBarriers = reconstruction_barriers.data(),
            });
            Frame reconstruction_target{};
            reconstruction_target.width = frame->width;
            reconstruction_target.height = frame->height;
            reconstruction_target.image = frame->pp_source_reconstruction_output_image;
            reconstruction_target.image_view = frame->pp_source_reconstruction_output_view;
            (void)pp_pass.Render(cmdbuf, frame->pp_source_reconstruction_snapshot_view, image_size,
                                 reconstruction_target, frame_pp_settings);
            source_reconstruction_captured = true;
        }
    }
    if (pp_input_shadow_stage) {
        if (!shadow_written) {
            pp_input_metadata.valid = false;
        }
        const u64 token = next_pp_input_shadow_token++;
        const auto status = frame->pp_input_shadow_state.Assign(
            {stamp.sequence, stamp.process_time_us, token, pp_input_metadata});
        if (status != FinalGuestSurfaceStatus::Complete) {
            LOG_INFO(Render, "PPInputShadowFrameState seq={} st={}", stamp.sequence,
                     static_cast<u32>(status));
        }
    }
    if (pp_sampled_input_stage) {
        const bool in_window = final_guest_surface_content->ShouldCapture(stamp.sequence);
        const auto observation = PlanPpSampledInputObservation({
            .in_window = in_window,
            .stamp_valid = stamp.sequence != 0 && stamp.process_time_us != 0,
            .metadata_valid = shadow_written && sampled_input_metadata.valid &&
                              source_backing_captured &&
                              (!pp_source_reconstruction_stage || source_reconstruction_captured),
        });
        if (observation.emit) {
            const u64 token = next_pp_input_shadow_token++;
            const auto status = frame->pp_sampled_input_state.Assign(
                observation, {stamp.sequence, stamp.process_time_us, token, sampled_input_metadata,
                              source_backing_plan, source_reconstruction_plan,
                              source_backing_captured, source_reconstruction_captured});
            if (status != FinalGuestSurfaceStatus::Complete) {
                LOG_INFO(Render, "PPSampledInputFrameState seq={} st={}", stamp.sequence,
                         static_cast<u32>(status));
            }
        }
    }

    DebugState.game_resolution = {image_size.width, image_size.height};
    DebugState.output_resolution = {frame->width, frame->height};

    std::shared_ptr<std::vector<ScreenshotReadback>> deferred_screenshots{};
    if (!pending_screenshots.empty()) {
        deferred_screenshots =
            std::make_shared<std::vector<ScreenshotReadback>>(std::move(pending_screenshots));
        draw_scheduler.DeferPriorityOperation(
            [deferred_screenshots]() { SavePendingScreenshots(*deferred_screenshots); });
    }

    // Flush frame creation commands.
    frame->ready_semaphore = draw_scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    SubmitInfo info{};
    draw_scheduler.Flush(info);
    return frame;
}

Frame* Presenter::PrepareBlankFrame(bool present_thread) {
    // Request a free presentation frame.
    Frame* frame = GetRenderFrame();
    frame->final_surface_diagnostic.Clear();
    const auto discarded_shadow = frame->pp_input_shadow_state.TakeForPresent(true);
    if (discarded_shadow.status == FinalGuestSurfaceStatus::GapLoss) {
        LOG_INFO(Render, "PPInputShadowFrameState blank_discard=1 st={}",
                 static_cast<u32>(discarded_shadow.status));
    }
    const auto discarded_sample =
        frame->pp_sampled_input_state.MarkPendingLoss(FinalGuestSurfaceStatus::GapLoss);
    if (discarded_sample == FinalGuestSurfaceStatus::GapLoss) {
        LOG_INFO(Render, "PPSampledInputFrameState blank_discard=1 st={}",
                 static_cast<u32>(discarded_sample));
    }

    auto& scheduler = present_thread ? present_scheduler : draw_scheduler;
    scheduler.EndRendering();

    const auto cmdbuf = scheduler.CommandBuffer();

    constexpr vk::ImageSubresourceRange simple_subresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const auto post_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const vk::RenderingAttachmentInfo attachment = {
        .imageView = frame->image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .extent = {frame->width, frame->height},
            },
        .layerCount = 1,
        .colorAttachmentCount = 1u,
        .pColorAttachments = &attachment,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    cmdbuf.beginRendering(rendering_info);
    cmdbuf.endRendering();

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &post_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return frame;
}

void Presenter::Present(Frame* frame, bool is_reusing_frame) {
    const auto complete_frame = [&](const bool was_presented) {
        bool released_frame{};
        {
            std::scoped_lock lock{free_mutex};
            released_frame = CompletePresentFrameOwnership(free_queue, last_submit_frame, frame,
                                                           is_reusing_frame, was_presented);
        }
        if (released_frame) {
            free_cv.notify_one();
        }
    };

    const bool pp_input_shadow_stage =
        final_guest_surface_content && final_guest_surface_content->IsPpInputShadowStage();
    const bool pp_sampled_input_stage =
        final_guest_surface_content && final_guest_surface_content->IsPpSampledInputStage();
    const bool pp_source_reconstruction_stage =
        final_guest_surface_content && final_guest_surface_content->IsPpSourceReconstructionStage();
    FinalGuestSurfacePpInputTakeResult pp_input_shadow_frame{};
    if (pp_input_shadow_stage) {
        pp_input_shadow_frame = frame->pp_input_shadow_state.TakeForPresent(is_reusing_frame);
        if (pp_input_shadow_frame.status != FinalGuestSurfaceStatus::Complete &&
            !(is_reusing_frame &&
              pp_input_shadow_frame.status == FinalGuestSurfaceStatus::AlreadyConsumed)) {
            LOG_INFO(Render, "PPInputShadowFrameState present_take=1 reused={} st={}",
                     is_reusing_frame, static_cast<u32>(pp_input_shadow_frame.status));
        }
    }
    FinalGuestSurfaceSampledInputTakeResult pp_sampled_input_frame{};
    if (pp_sampled_input_stage) {
        pp_sampled_input_frame = frame->pp_sampled_input_state.TakeForPresent(is_reusing_frame);
        if (pp_sampled_input_frame.status != FinalGuestSurfaceStatus::Complete &&
            !(is_reusing_frame &&
              pp_sampled_input_frame.status == FinalGuestSurfaceStatus::AlreadyConsumed)) {
            LOG_INFO(Render, "PPSampledInputFrameState present_take=1 reused={} st={}",
                     is_reusing_frame, static_cast<u32>(pp_sampled_input_frame.status));
        }
    }

    // Recreate the swapchain if the window was resized.
    if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }

    if (!swapchain.AcquireNextImage()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
        if (!swapchain.AcquireNextImage()) {
            // User resizes the window too fast and GPU can't keep up. Skip this frame.
            LOG_WARNING(Render_Vulkan, "Skipping frame!");
            if (pp_sampled_input_stage) {
                final_guest_surface_content->ReportPpSampledInputLoss(
                    pp_sampled_input_frame, FinalGuestSurfaceStatus::GapLoss);
            }
            complete_frame(false);
            return;
        }
    }

    // Reset fence for queue submission. Do it here instead of GetRenderFrame() because we may
    // skip frame because of slow swapchain recreation. If a frame skip occurs, we skip signal
    // the frame's present fence and future GetRenderFrame() call will hang waiting for this frame.
    const auto reset_result = instance.GetDevice().resetFences(frame->present_done);
    ASSERT_MSG(reset_result == vk::Result::eSuccess,
               "Unexpected error resetting present done fence: {}", vk::to_string(reset_result));

    ImGuiID dockId = ImGui::Core::NewFrame(is_reusing_frame);

    const vk::Image swapchain_image = swapchain.Image();
    const vk::ImageView swapchain_image_view = swapchain.ImageView();

    auto& scheduler = present_scheduler;
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto capture_with_overlays = VideoCore::ConsumeWithOverlaysScreenshotRequests();
    FinalGuestSurfacePresentationMapping final_surface_mapping{};
    std::vector<ScreenshotReadback> pending_screenshots;
    if (capture_with_overlays.Total() > 0) {
        pending_screenshots.reserve(2);
    }

    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
            .pLabelName = "Present",
        });
    }

    {
        auto* profiler_ctx = instance.GetProfilerContext();
        TracyVkNamedZoneC(profiler_ctx, renderer_gpu_zone, cmdbuf, "Host frame",
                          MarkersPalette::GpuMarkerColor, profiler_ctx != nullptr);

        const vk::Extent2D extent = swapchain.GetExtent();
        std::optional<FinalGuestSurfaceContentState::PendingCapture> post_pp_surface_capture;
        std::optional<FinalGuestSurfaceContentState::PendingCapture> pp_input_shadow_capture;
        std::optional<FinalGuestSurfaceContentState::PendingCapture> pp_sampled_input_capture;
        if (final_guest_surface_content && final_guest_surface_content->IsPostPpStage()) {
            const auto& diagnostic = frame->final_surface_diagnostic;
            const bool in_window = diagnostic.valid && final_guest_surface_content->ShouldCapture(
                                                           diagnostic.surface_sequence);
            if (ShouldObserveFinalGuestSurfaceAtPresent(final_guest_surface_content->Stage(),
                                                        is_reusing_frame, diagnostic.valid,
                                                        in_window)) {
                post_pp_surface_capture = final_guest_surface_content->PreparePostPp(
                    {diagnostic.surface_sequence, diagnostic.surface_process_time_us}, frame->width,
                    frame->height, swapchain.GetSurfaceFormat().format, frame->is_hdr);
            }
        }
        if (pp_input_shadow_stage && pp_input_shadow_frame.emit &&
            final_guest_surface_content->ShouldCapture(pp_input_shadow_frame.payload.sequence)) {
            pp_input_shadow_capture = final_guest_surface_content->PreparePpInputShadow(
                {pp_input_shadow_frame.payload.sequence,
                 pp_input_shadow_frame.payload.process_time_us},
                pp_input_shadow_frame.payload.metadata);
        }
        if (pp_sampled_input_stage && pp_sampled_input_frame.emit &&
            final_guest_surface_content->ShouldCapture(pp_sampled_input_frame.payload.sequence)) {
            auto metadata = pp_sampled_input_frame.payload.metadata;
            if (pp_sampled_input_frame.status != FinalGuestSurfaceStatus::Complete) {
                metadata.valid = false;
            }
            pp_sampled_input_capture = final_guest_surface_content->PreparePpSampledInput(
                {pp_sampled_input_frame.payload.sequence,
                 pp_sampled_input_frame.payload.process_time_us},
                metadata, pp_sampled_input_frame.payload.source_backing,
                pp_sampled_input_frame.payload.source_reconstruction,
                pp_sampled_input_frame.status);
        }
        const bool copy_frame_before_sampling =
            (post_pp_surface_capture && post_pp_surface_capture->HasCopy()) ||
            (pp_sampled_input_capture && pp_sampled_input_capture->HasCopy());
        const auto frame_transitions =
            GetPresentFrameTransitions(is_reusing_frame, copy_frame_before_sampling);
        std::array<vk::ImageMemoryBarrier2, 2> pre_barriers;
        u32 num_pre_barriers = 1;
        pre_barriers[0] = vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eNone,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        const auto& frame_transition = frame_transitions.before;
        if (frame_transition.required) {
            pre_barriers[num_pre_barriers++] = vk::ImageMemoryBarrier2{
                .srcStageMask = frame_transition.src_stage,
                .srcAccessMask = frame_transition.src_access,
                .dstStageMask = frame_transition.dst_stage,
                .dstAccessMask = frame_transition.dst_access,
                .oldLayout = frame_transition.old_layout,
                .newLayout = frame_transition.new_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = frame->image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
        }

        bool swapchain_copied_for_screenshot = false;

        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .imageMemoryBarrierCount = num_pre_barriers,
            .pImageMemoryBarriers = pre_barriers.data(),
        });

        if (pp_input_shadow_capture) {
            const auto handoff =
                PlanPpInputShadowPresentHandoff(true, is_reusing_frame, pp_input_shadow_frame.emit,
                                                pp_input_shadow_capture->HasCopy());
            ASSERT(handoff.scheduler == FinalGuestSurfaceDeferredScheduler::Present);
            ASSERT(!handoff.defer_on_draw_scheduler && handoff.callback_payload_is_scalar_only);
            const auto transition = GetPpInputShadowCaptureTransition(handoff.copy);
            if (transition.required) {
                const vk::ImageMemoryBarrier2 shadow_to_transfer{
                    .srcStageMask = transition.src_stage,
                    .srcAccessMask = transition.src_access,
                    .dstStageMask = transition.dst_stage,
                    .dstAccessMask = transition.dst_access,
                    .oldLayout = transition.old_layout,
                    .newLayout = transition.new_layout,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = frame->pp_input_shadow_image,
                    .subresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &shadow_to_transfer,
                });
            }
            final_guest_surface_content->Record(std::move(*pp_input_shadow_capture),
                                                frame->pp_input_shadow_image, cmdbuf);
        }
        if (pp_sampled_input_capture) {
            const auto transfer = PlanPpSampledInputTransfer(
                true, is_reusing_frame, pp_sampled_input_frame.emit,
                pp_sampled_input_capture->HasCopy(), pp_source_reconstruction_stage);
            ASSERT(IsPpSampledInputTransferContractValid(transfer));
            const auto transition = GetPpInputShadowCaptureTransition(transfer.copy);
            if (transition.required) {
                const vk::ImageMemoryBarrier2 sampled_to_transfer{
                    .srcStageMask = transition.src_stage,
                    .srcAccessMask = transition.src_access,
                    .dstStageMask = transition.dst_stage,
                    .dstAccessMask = transition.dst_access,
                    .oldLayout = transition.old_layout,
                    .newLayout = transition.new_layout,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = frame->pp_input_shadow_image,
                    .subresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &sampled_to_transfer,
                });
            }
            const auto reconstruction_transition = GetPpSourceReconstructionCaptureTransition(
                transfer.copy && transfer.paired_source_reconstruction);
            if (reconstruction_transition.required) {
                const vk::ImageMemoryBarrier2 reconstruction_to_transfer{
                    .srcStageMask = reconstruction_transition.src_stage,
                    .srcAccessMask = reconstruction_transition.src_access,
                    .dstStageMask = reconstruction_transition.dst_stage,
                    .dstAccessMask = reconstruction_transition.dst_access,
                    .oldLayout = reconstruction_transition.old_layout,
                    .newLayout = reconstruction_transition.new_layout,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = frame->pp_source_reconstruction_output_image,
                    .subresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &reconstruction_to_transfer,
                });
            }
            final_guest_surface_content->RecordPpSampledInput(
                std::move(*pp_sampled_input_capture), frame->image, frame->pp_input_shadow_image,
                frame->pp_source_backing_snapshot ? frame->pp_source_backing_snapshot->Handle()
                                                  : vk::Buffer{},
                frame->pp_source_reconstruction_output_image, cmdbuf);
        }
        if (post_pp_surface_capture) {
            final_guest_surface_content->Record(std::move(*post_pp_surface_capture), frame->image,
                                                cmdbuf);
        }
        if (frame_transitions.after.required) {
            const auto& transition = frame_transitions.after;
            const vk::ImageMemoryBarrier2 restore_frame{
                .srcStageMask = transition.src_stage,
                .srcAccessMask = transition.src_access,
                .dstStageMask = transition.dst_stage,
                .dstAccessMask = transition.dst_access,
                .oldLayout = transition.old_layout,
                .newLayout = transition.new_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = frame->image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &restore_frame,
            });
        }

        { // Draw the game
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
            ImGui::SetNextWindowDockID(dockId, ImGuiCond_Once);
            if (ImGui::Begin("Display##game_display", nullptr, ImGuiWindowFlags_NoNav)) {
                auto game_texture = frame->imgui_texture;
                auto game_width = frame->width;
                auto game_height = frame->height;
                bool displaying_guest_frame = true;

                if (Libraries::SystemService::IsSplashVisible()) { // draw splash
                    if (!splash_img.has_value()) {
                        splash_img.emplace();
                        auto splash_path = Common::ElfInfo::Instance().GetSplashPath();
                        if (!splash_path.empty()) {
                            splash_img = ImGui::RefCountedTexture::DecodePngFile(splash_path);
                        }
                    }
                    if (auto& splash_image = this->splash_img.value()) {
                        auto [im_id, width, height] = splash_image.GetTexture();
                        game_texture = im_id;
                        game_width = width;
                        game_height = height;
                        displaying_guest_frame = false;
                    }
                }

                ImVec2 contentArea = ImGui::GetContentRegionAvail();
                SetExpectedGameSize((s32)contentArea.x, (s32)contentArea.y);

                const auto imgRect =
                    FitImage(game_width, game_height, (s32)contentArea.x, (s32)contentArea.y);
                ImVec2 offset{
                    static_cast<float>(imgRect.offset.x),
                    static_cast<float>(imgRect.offset.y),
                };
                ImVec2 size{
                    static_cast<float>(imgRect.extent.width),
                    static_cast<float>(imgRect.extent.height),
                };

                ImGui::SetCursorPos(ImGui::GetCursorStartPos() + offset);
                const ImVec2 output_position = ImGui::GetCursorScreenPos();
                ImGui::Image(game_texture, size);
                final_surface_mapping = {
                    .guest_width = frame->final_surface_diagnostic.guest_width,
                    .guest_height = frame->final_surface_diagnostic.guest_height,
                    .swapchain_width = extent.width,
                    .swapchain_height = extent.height,
                    .output_x = static_cast<s32>(output_position.x),
                    .output_y = static_cast<s32>(output_position.y),
                    .output_width = imgRect.extent.width,
                    .output_height = imgRect.extent.height,
                    .top_left = displaying_guest_frame,
                    .no_y_flip = displaying_guest_frame,
                };

                if (EmulatorSettings.IsNullGPU()) {
                    Core::Devtools::Layer::DrawNullGpuNotice();
                }
            }
            ImGui::End();
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor();
        }
        ImGui::Core::Render(cmdbuf, swapchain_image_view, swapchain.GetExtent());

        if (capture_with_overlays.Total() > 0) {
            const u32 calibration_count = FinalGuestSurfaceAutomationCalibrationCount(
                capture_with_overlays.notifying_count, capture_with_overlays.silent_count);
            if (final_guest_surface_content && calibration_count != 0) {
                final_guest_surface_content->CalibrateScreenshots(
                    frame->final_surface_diagnostic, final_surface_mapping, calibration_count,
                    Libraries::Kernel::sceKernelGetProcessTime());
            }
            const vk::ImageMemoryBarrier to_transfer{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, to_transfer);
            const auto append_readback = [&](const u32 count, const bool notify) {
                if (count == 0) {
                    return;
                }
                pending_screenshots.emplace_back(
                    instance, scheduler, ScreenshotKind::WithOverlays, notify,
                    BuildScreenshotPaths(ScreenshotKind::WithOverlays, count), extent.width,
                    extent.height,
                    swapchain.GetHDR() ? vk::Format::eA2B10G10R10UnormPack32
                                       : swapchain.GetSurfaceFormat().format,
                    swapchain.GetHDR());
                CopyImageToReadback(cmdbuf, swapchain_image, vk::ImageLayout::eTransferSrcOptimal,
                                    pending_screenshots.back());
            };
            append_readback(capture_with_overlays.notifying_count, true);
            append_readback(capture_with_overlays.silent_count, false);
            swapchain_copied_for_screenshot = true;
        }

        const vk::AccessFlags post_src_access_mask =
            swapchain_copied_for_screenshot ? vk::AccessFlagBits::eTransferRead
                                            : vk::AccessFlagBits::eColorAttachmentWrite;
        const vk::ImageLayout post_old_layout = swapchain_copied_for_screenshot
                                                    ? vk::ImageLayout::eTransferSrcOptimal
                                                    : vk::ImageLayout::eColorAttachmentOptimal;
        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = post_src_access_mask,
            .dstAccessMask = vk::AccessFlagBits::eNone,
            .oldLayout = post_old_layout,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);

        if (profiler_ctx) {
            TracyVkCollect(profiler_ctx, cmdbuf);
        }
    }
    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.endDebugUtilsLabelEXT();
    }

    // Flush vulkan commands.
    std::shared_ptr<std::vector<ScreenshotReadback>> deferred_screenshots{};
    if (!pending_screenshots.empty()) {
        deferred_screenshots =
            std::make_shared<std::vector<ScreenshotReadback>>(std::move(pending_screenshots));
        scheduler.DeferPriorityOperation(
            [deferred_screenshots]() { SavePendingScreenshots(*deferred_screenshots); });
    }

    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore());
    info.AddWait(frame->ready_semaphore, frame->ready_tick,
                 FrameReadyWaitStage(pp_input_shadow_stage || pp_sampled_input_stage));
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    info.AddSignal(frame->present_done);
    void* const vulkan_instance =
        reinterpret_cast<void*>(static_cast<VkInstance>(instance.GetInstance()));
    void* const window_handle = window.GetWindowInfo().render_surface;
    const bool is_renderdoc_capture =
        VideoCore::BeginNextPresentedFrameCapture(vulkan_instance, window_handle);
    scheduler.Flush(info);

    // Present to swapchain.
    {
        std::scoped_lock submit_lock{Scheduler::submit_mutex};
        if (!swapchain.Present()) {
            swapchain.Recreate(window.GetWidth(), window.GetHeight());
        }
    }
    if (is_renderdoc_capture) {
        VideoCore::EndPresentedFrameCapture(vulkan_instance, window_handle);
    }

    complete_frame(true);
    if (!is_reusing_frame) {
        const auto presented_frame = DebugState.IncFlipFrameNum();
        RecordPresentedFrameTiming(presented_frame);
    }
}

Frame* Presenter::GetRenderFrame() {
    // Wait for free presentation frames
    Frame* frame;
    {
        std::unique_lock lock{free_mutex};
        free_cv.wait(lock, [this] { return !free_queue.empty(); });
        LOG_DEBUG(Render_Vulkan, "Got render frame, remaining {}", free_queue.size() - 1);

        // Take the frame from the queue
        frame = free_queue.front();
        free_queue.pop();
    }

    const vk::Device device = instance.GetDevice();
    vk::Result result{};

    const auto wait = [&]() {
        result = device.waitForFences(frame->present_done, false, std::numeric_limits<u64>::max());
        return result;
    };

    // Wait for the presentation to be finished so all frame resources are free
    while (wait() != vk::Result::eSuccess) {
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
        // Retry if the waiting times out
        if (result == vk::Result::eTimeout) {
            continue;
        }
    }

    if (frame->width != expected_frame_width || frame->height != expected_frame_height ||
        frame->is_hdr != swapchain.GetHDR()) {
        RecreateFrame(frame, expected_frame_width, expected_frame_height);
    }

    return frame;
}

void Presenter::SetExpectedGameSize(s32 width, s32 height) {
    const float ratio = (float)width / (float)height;

    expected_frame_height = height;
    expected_frame_width = width;
    if (ratio > expected_ratio) {
        expected_frame_width = static_cast<s32>(height * expected_ratio);
    } else {
        expected_frame_height = static_cast<s32>(width / expected_ratio);
    }
}

} // namespace Vulkan
