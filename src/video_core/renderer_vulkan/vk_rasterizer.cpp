// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/debug.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/quad_list_indices.h"
#include "video_core/renderer_vulkan/attachment_feedback_loop_access.h"
#include "video_core/renderer_vulkan/stencil_reference.h"
#include "video_core/renderer_vulkan/depth_attachment_clear.h"
#include "video_core/renderer_vulkan/host_passes/pp_source_producer_scope.h"
#include "video_core/renderer_vulkan/host_passes/pp_terminal_scope_content.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_feedback_pre_post.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_input_content.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_sample_tee.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_sample_tee_vulkan.h"
#include "video_core/renderer_vulkan/host_passes/pp_upstream_target_color_reuse.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/depth_association.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vulkan/vulkan_format_traits.hpp>


#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Vulkan {

namespace {

[[nodiscard]] std::optional<PpTerminalScopeRuntimeConfig> ReadPpTerminalScopeRuntimeConfig() {
    return ResolvePpTerminalScopeRuntimeConfig([](const char* name) {
        const char* value = std::getenv(name);
        return value ? std::optional<std::string_view>{value} : std::optional<std::string_view>{};
    });
}

[[nodiscard]] FinalGuestSurfaceFormat TerminalScopeFormat(vk::Format format) noexcept {
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        return FinalGuestSurfaceFormat::Rgba8;
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
        return FinalGuestSurfaceFormat::Bgra8;
    default:
        return FinalGuestSurfaceFormat::Unsupported;
    }
}

[[nodiscard]] u32 RawTerminalScopeBytesPerTexel(vk::Format format) noexcept {
    if (format == vk::Format::eUndefined) {
        return 0;
    }
    const auto extent = vk::blockExtent(format);
    if (extent[0] != 1 || extent[1] != 1 || extent[2] != 1) {
        return 0;
    }
    const u32 bytes = vk::blockSize(format);
    return bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16 ? bytes : 0;
}

[[nodiscard]] u32 RawTerminalScopeDepthBytesPerTexel(vk::Format format) noexcept {
    switch (format) {
    case vk::Format::eD16Unorm:
    case vk::Format::eD16UnormS8Uint:
        return 2;
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32Sfloat:
    case vk::Format::eD32SfloatS8Uint:
        return 4;
    default:
        return 0;
    }
}

[[nodiscard]] bool IsEffectiveDepthWrite(const RenderState& state,
                                         const DynamicState& dynamic_state) noexcept {
    return !dynamic_state.rasterizer_discard_enable &&
           state.depth_stencil_attachment.has_depth && dynamic_state.depth_test_enabled &&
           dynamic_state.depth_write_enabled;
}

struct PpUpstreamTargetFixedFunctionState {
    u32 depth_compare{};
    u32 z_order{};
    u32 num_samples{};
    u32 mrt_mask{};
    u32 logic_op{};
    u32 color_src_factor{};
    u32 color_dst_factor{};
    u32 color_blend_op{};
    u32 alpha_src_factor{};
    u32 alpha_dst_factor{};
    u32 alpha_blend_op{};
    u32 write_mask{};
    u32 shader_mask{};
    u32 round_mode{};
    bool valid{};
    bool depth_enable{};
    bool depth_write_enable{};
    bool color_on_depth_fail{};
    bool no_color_on_depth_pass{};
    bool z_export_enable{};
    bool kill_enable{};
    bool alpha_to_mask_disable{};
    bool coverage_to_mask_enable{};
    bool mask_export_enable{};
    bool blend_enable{};

    auto operator<=>(const PpUpstreamTargetFixedFunctionState&) const = default;
};

[[nodiscard]] PpUpstreamTargetFixedFunctionState MakePpUpstreamTargetFixedFunctionState(
    const AmdGpu::Regs& regs, const GraphicsPipeline* pipeline) noexcept {
    if (!pipeline) {
        return {};
    }
    const auto& key = pipeline->GetGraphicsKey();
    const auto& blend = key.blend_controls[0];
    return {
        .depth_compare = static_cast<u32>(regs.depth_control.depth_func),
        .z_order = static_cast<u32>(regs.depth_shader_control.z_order),
        .num_samples = key.num_samples,
        .mrt_mask = key.mrt_mask,
        .logic_op = static_cast<u32>(key.logic_op),
        .color_src_factor = static_cast<u32>(blend.color_src_factor),
        .color_dst_factor = static_cast<u32>(blend.color_dst_factor),
        .color_blend_op = static_cast<u32>(blend.color_func),
        .alpha_src_factor = static_cast<u32>(blend.alpha_src_factor),
        .alpha_dst_factor = static_cast<u32>(blend.alpha_dst_factor),
        .alpha_blend_op = static_cast<u32>(blend.alpha_func),
        .write_mask = static_cast<u32>(key.write_masks[0]),
        .shader_mask = key.cb_shader_mask.GetMask(0),
        .round_mode = static_cast<u32>(regs.color_buffers[0].info.round_mode),
        .valid = true,
        .depth_enable = bool(regs.depth_control.depth_enable),
        .depth_write_enable = bool(regs.depth_control.depth_write_enable),
        .color_on_depth_fail = bool(regs.depth_control.enable_color_writes_on_depth_fail),
        .no_color_on_depth_pass = bool(regs.depth_control.disable_color_writes_on_depth_pass),
        .z_export_enable = bool(regs.depth_shader_control.z_export_enable),
        .kill_enable = bool(regs.depth_shader_control.kill_enable),
        .alpha_to_mask_disable = bool(regs.depth_shader_control.alpha_to_mask_disable),
        .coverage_to_mask_enable = bool(regs.depth_shader_control.coverage_to_mask_enable),
        .mask_export_enable = bool(regs.depth_shader_control.mask_export_enable),
        .blend_enable = bool(blend.enable),
    };
}

[[nodiscard]] VideoCore::ImageColorScopeTrackedView MakeImageColorScopeTrackedView(
    VideoCore::ImageColorScopePrivateLink link, VideoCore::Image& image,
    const VideoCore::ImageViewInfo& view, bool view_conflict) noexcept {
    const auto& range = view.range;
    const auto& resources = image.info.resources;
    const bool range_valid = range.extent.levels != 0 && range.extent.layers != 0 &&
                             range.base.level < resources.levels &&
                             range.extent.levels <= resources.levels - range.base.level &&
                             range.base.layer < resources.layers &&
                             range.extent.layers <= resources.layers - range.base.layer;
    const u64 mutation_epoch = image.DiagnosticMutationEpoch();
    if (!link.Valid() || link.uid != image.image_uid || !image.backing || !range_valid ||
        image.info.size.width == 0 || image.info.size.height == 0 || mutation_epoch == 0) {
        return {};
    }
    const auto mip_dimension = [](u32 dimension, u32 level) {
        return level < 32 ? std::max(1u, dimension >> level) : 0u;
    };
    const bool type_2d =
        view.type == AmdGpu::ImageType::Color2D || view.type == AmdGpu::ImageType::Color2DArray;
    return {
        .link = link,
        .backing_generation = image.EnsureDiagnosticBackingGeneration(),
        .mutation_epoch = mutation_epoch,
        .width = mip_dimension(image.info.size.width, range.base.level),
        .height = mip_dimension(image.info.size.height, range.base.level),
        .bytes_per_texel = !image.info.props.is_block && image.info.num_bits % 8 == 0
                               ? image.info.num_bits / 8
                               : 0,
        .type = static_cast<u32>(view.type),
        .format = static_cast<u32>(view.format),
        .base_mip = range.base.level,
        .mip_count = range.extent.levels,
        .base_layer = range.base.layer,
        .layer_count = range.extent.layers,
        .mapping = {static_cast<u32>(view.mapping.r), static_cast<u32>(view.mapping.g),
                    static_cast<u32>(view.mapping.b), static_cast<u32>(view.mapping.a)},
        .min_lod = view.min_lod,
        .samples = image.backing->num_samples,
        .color = image.aspect_mask == vk::ImageAspectFlagBits::eColor,
        .type_2d = type_2d,
        .uniform_state = image.backing->subresource_states.empty(),
        .view_conflict = view_conflict,
        .storage = view.is_storage,
        .valid = true,
    };
}

struct PpUpstreamSampleTeeTransferObservation {
    VideoCore::ImageTransferKind kind{VideoCore::ImageTransferKind::None};
    VideoCore::ImageTransferOrigin origin{VideoCore::ImageTransferOrigin::Unknown};
    VideoCore::ImageProducerClass source_producer{VideoCore::ImageProducerClass::Unknown};
    VideoCore::ImageTransferKind source_transfer_kind{VideoCore::ImageTransferKind::None};
    VideoCore::ImageTransferOrigin source_transfer_origin{VideoCore::ImageTransferOrigin::Unknown};
    VideoCore::ImageProducerClass nested_source_producer{VideoCore::ImageProducerClass::Unknown};
    VideoCore::ImageTransferKind nested_source_transfer_kind{VideoCore::ImageTransferKind::None};
    VideoCore::ImageTransferOrigin nested_source_transfer_origin{
        VideoCore::ImageTransferOrigin::Unknown};
    u32 base_mip{};
    u32 mip_count{};
    u32 base_layer{};
    u32 layer_count{};
    u32 source_samples{};
    u32 old_samples{};
    u32 new_samples{};
    u32 requested_width{};
    u32 requested_height{};
    u32 backing_width{};
    u32 backing_height{};
    bool observed{};
    bool range_exact{};
    bool request_matches_encoded{};
    bool full_xy{};
    bool exact{};
    bool aspect_overlap{};
    bool range_overlap{};
    bool range_covers{};
    bool extent_valid{};
    bool extent_conflict{};
    bool extent_matches{};
    bool source_transfer_exact{};
    bool source_identity_comparable{};
    bool source_image_changed{};
    bool source_backing_changed{};
    bool source_mutation_comparable{};
    bool source_mutation_changed{};
    bool source_is_depth{};
    bool nested_source_transfer_exact{};
    bool nested_source_is_depth{};
    VideoCore::ImageColorScopeDrawKind source_depth_stencil_kind{
        VideoCore::ImageColorScopeDrawKind::Unknown};
    u32 source_depth_stencil_element_count{};
    u32 source_depth_stencil_instance_count{};
    u32 source_depth_stencil_sampled_images{};
    u32 source_depth_stencil_storage_writes{};
    u32 source_depth_stencil_scope_first_element_count{};
    u32 source_depth_stencil_scope_first_instance_count{};
    u32 source_depth_stencil_scope_first_sampled_images{};
    u32 source_depth_stencil_scope_first_storage_writes{};
    u32 selected_aspect{};
    u64 source_depth_stencil_fragment_pgm_hash{};
    u64 source_depth_stencil_scope_first_fragment_pgm_hash{};
    u32 source_depth_stencil_user_clip_plane_mask{};
    VideoCore::ImageColorScopeDrawKind source_depth_stencil_scope_first_kind{
        VideoCore::ImageColorScopeDrawKind::Unknown};
    u8 source_depth_stencil_fragment_mrt_mask{};
    u8 source_depth_stencil_guest_primitive_type{};
    u8 source_depth_stencil_effective_primitive_type{};
    u8 source_depth_stencil_cull_mode{};
    u8 source_depth_stencil_front_face{};
    u8 source_depth_stencil_scope_first_guest_primitive_type{};
    u8 source_depth_stencil_scope_first_effective_primitive_type{};
    u8 source_depth_stencil_scope_first_cull_mode{};
    u8 source_depth_stencil_scope_first_front_face{};
    bool source_depth_stencil_exact{};
    bool source_depth_stencil_direct{};
    bool source_depth_stencil_indexed{};
    bool source_depth_write{};
    bool source_stencil_write{};
    bool selected_is_depth{};
    bool source_depth_stencil_fragment_writes_depth{};
    bool source_depth_stencil_fragment_has_discard{};
    bool source_depth_stencil_quad_list_converted{};
    bool source_depth_stencil_scope_first_valid{};
    bool source_depth_stencil_scope_first_indexed{};
    bool source_depth_stencil_scope_first_quad_list_converted{};
    bool source_depth_stencil_sampled_id_alias{};
    bool source_depth_stencil_sampled_backing_alias{};
    bool source_depth_stencil_sampled_aspect_alias{};
    bool source_depth_stencil_sampled_range_alias{};
    bool source_depth_stencil_sampled_feedback_alias{};
};

struct PpUpstreamSampledExtentObservation {
    u32 requested_width{};
    u32 requested_height{};
    u32 backing_width{};
    u32 backing_height{};
    bool conflict{};
    bool valid{};
};

[[nodiscard]] PpUpstreamSampleTeeTransferObservation MakePpUpstreamSampleTeeTransferObservation(
    VideoCore::Image& image, const VideoCore::ImageColorScopeTrackedView& view,
    vk::ImageAspectFlags resolved_aspect,
    const PpUpstreamSampledExtentObservation& extent) noexcept {
    const auto transfer = image.PeekDiagnosticTransfer();
    const auto direct_depth_stencil = image.PeekDiagnosticDepthStencilProducer();
    PpUpstreamSampleTeeTransferObservation result{
        .kind = transfer.kind,
        .origin = transfer.origin,
        .source_producer = transfer.source_producer,
        .source_transfer_kind = transfer.source_transfer_kind,
        .source_transfer_origin = transfer.source_transfer_origin,
        .nested_source_producer = transfer.nested_source_producer,
        .nested_source_transfer_kind = transfer.nested_source_transfer_kind,
        .nested_source_transfer_origin = transfer.nested_source_transfer_origin,
        .base_mip = transfer.base_mip,
        .mip_count = transfer.mip_count,
        .base_layer = transfer.base_layer,
        .layer_count = transfer.layer_count,
        .source_samples = transfer.source_samples,
        .old_samples = transfer.old_samples,
        .new_samples = transfer.new_samples,
        .requested_width = extent.requested_width,
        .requested_height = extent.requested_height,
        .backing_width = extent.backing_width,
        .backing_height = extent.backing_height,
        .observed = transfer.kind != VideoCore::ImageTransferKind::None,
        .range_exact = transfer.range_exact,
        .request_matches_encoded = transfer.request_matches_encoded,
        .full_xy = transfer.full_xy,
        .extent_valid = extent.valid,
        .extent_conflict = extent.conflict,
        .extent_matches = extent.valid && !extent.conflict &&
                          extent.requested_width == extent.backing_width &&
                          extent.requested_height == extent.backing_height,
        .source_transfer_exact = transfer.source_transfer_exact,
        .source_identity_comparable = transfer.source_identity_comparable,
        .source_image_changed = transfer.source_image_changed,
        .source_backing_changed = transfer.source_backing_changed,
        .source_mutation_comparable = transfer.source_mutation_comparable,
        .source_mutation_changed = transfer.source_mutation_changed,
        .source_is_depth = transfer.source_is_depth,
        .nested_source_transfer_exact = transfer.nested_source_transfer_exact,
        .nested_source_is_depth = transfer.nested_source_is_depth,
        .source_depth_stencil_kind = transfer.source_depth_stencil_kind,
        .source_depth_stencil_element_count = transfer.source_depth_stencil_element_count,
        .source_depth_stencil_instance_count = transfer.source_depth_stencil_instance_count,
        .source_depth_stencil_sampled_images = transfer.source_depth_stencil_sampled_images,
        .source_depth_stencil_storage_writes = transfer.source_depth_stencil_storage_writes,
        .selected_aspect =
            static_cast<u32>(static_cast<VkImageAspectFlags>(resolved_aspect)),
        .source_depth_stencil_exact = transfer.source_depth_stencil_exact,
        .source_depth_stencil_indexed = transfer.source_depth_stencil_indexed,
        .source_depth_write = transfer.source_depth_write,
        .source_stencil_write = transfer.source_stencil_write,
        .selected_is_depth = image.info.props.is_depth,
    };
    if (direct_depth_stencil.valid && view.valid &&
        direct_depth_stencil.mutation_epoch == view.mutation_epoch &&
        direct_depth_stencil.backing_generation == view.backing_generation &&
        direct_depth_stencil.mutation_epoch == image.DiagnosticMutationEpoch() &&
        direct_depth_stencil.backing_generation == image.EnsureDiagnosticBackingGeneration()) {
        result.source_depth_stencil_kind = direct_depth_stencil.kind;
        result.source_depth_stencil_element_count = direct_depth_stencil.element_count;
        result.source_depth_stencil_instance_count = direct_depth_stencil.instance_count;
        result.source_depth_stencil_sampled_images = direct_depth_stencil.sampled_images;
        result.source_depth_stencil_storage_writes = direct_depth_stencil.storage_writes;
        result.source_depth_stencil_fragment_pgm_hash = direct_depth_stencil.fragment_pgm_hash;
        result.source_depth_stencil_scope_first_kind =
            direct_depth_stencil.scope_first_depth_kind;
        result.source_depth_stencil_scope_first_fragment_pgm_hash =
            direct_depth_stencil.scope_first_depth_fragment_pgm_hash;
        result.source_depth_stencil_scope_first_element_count =
            direct_depth_stencil.scope_first_depth_element_count;
        result.source_depth_stencil_scope_first_instance_count =
            direct_depth_stencil.scope_first_depth_instance_count;
        result.source_depth_stencil_scope_first_sampled_images =
            direct_depth_stencil.scope_first_depth_sampled_images;
        result.source_depth_stencil_scope_first_storage_writes =
            direct_depth_stencil.scope_first_depth_storage_writes;
        result.source_depth_stencil_user_clip_plane_mask =
            direct_depth_stencil.user_clip_plane_mask;
        result.source_depth_stencil_fragment_mrt_mask = direct_depth_stencil.fragment_mrt_mask;
        result.source_depth_stencil_guest_primitive_type =
            direct_depth_stencil.guest_primitive_type;
        result.source_depth_stencil_effective_primitive_type =
            direct_depth_stencil.effective_primitive_type;
        result.source_depth_stencil_cull_mode = direct_depth_stencil.cull_mode;
        result.source_depth_stencil_front_face = direct_depth_stencil.front_face;
        result.source_depth_stencil_scope_first_guest_primitive_type =
            direct_depth_stencil.scope_first_depth_guest_primitive_type;
        result.source_depth_stencil_scope_first_effective_primitive_type =
            direct_depth_stencil.scope_first_depth_effective_primitive_type;
        result.source_depth_stencil_scope_first_cull_mode =
            direct_depth_stencil.scope_first_depth_cull_mode;
        result.source_depth_stencil_scope_first_front_face =
            direct_depth_stencil.scope_first_depth_front_face;
        result.source_depth_stencil_exact = true;
        result.source_depth_stencil_direct = true;
        result.source_depth_stencil_indexed = direct_depth_stencil.indexed;
        result.source_depth_write = direct_depth_stencil.depth_write;
        result.source_stencil_write = direct_depth_stencil.stencil_write;
        result.source_depth_stencil_fragment_writes_depth =
            direct_depth_stencil.fragment_writes_depth;
        result.source_depth_stencil_fragment_has_discard =
            direct_depth_stencil.fragment_has_discard;
        result.source_depth_stencil_quad_list_converted =
            direct_depth_stencil.quad_list_converted;
        result.source_depth_stencil_scope_first_valid =
            direct_depth_stencil.scope_first_depth_valid;
        result.source_depth_stencil_scope_first_indexed =
            direct_depth_stencil.scope_first_depth_indexed;
        result.source_depth_stencil_scope_first_quad_list_converted =
            direct_depth_stencil.scope_first_depth_quad_list_converted;
        result.source_depth_stencil_sampled_id_alias =
            direct_depth_stencil.sampled_id_alias;
        result.source_depth_stencil_sampled_backing_alias =
            direct_depth_stencil.sampled_backing_alias;
        result.source_depth_stencil_sampled_aspect_alias =
            direct_depth_stencil.sampled_aspect_alias;
        result.source_depth_stencil_sampled_range_alias =
            direct_depth_stencil.sampled_range_alias;
        result.source_depth_stencil_sampled_feedback_alias =
            direct_depth_stencil.sampled_feedback_alias;
    }
    if (!transfer.valid || !view.valid || !result.observed ||
        transfer.mutation_epoch != view.mutation_epoch ||
        transfer.backing_generation != view.backing_generation ||
        transfer.mutation_epoch != image.DiagnosticMutationEpoch() ||
        transfer.backing_generation != image.EnsureDiagnosticBackingGeneration() ||
        image.PeekDiagnosticProducer().classification != VideoCore::ImageProducerClass::Transfer) {
        return result;
    }
    result.exact = true;
    result.aspect_overlap =
        (transfer.aspect_mask &
         static_cast<u32>(static_cast<VkImageAspectFlags>(resolved_aspect))) != 0;
    const auto overlaps = [](u32 first_base, u32 first_count, u32 second_base, u32 second_count) {
        return first_count != 0 && second_count != 0 &&
               static_cast<u64>(first_base) < static_cast<u64>(second_base) + second_count &&
               static_cast<u64>(second_base) < static_cast<u64>(first_base) + first_count;
    };
    const auto covers = [](u32 outer_base, u32 outer_count, u32 inner_base, u32 inner_count) {
        return outer_count != 0 && inner_count != 0 && outer_base <= inner_base &&
               static_cast<u64>(inner_base) + inner_count <=
                   static_cast<u64>(outer_base) + outer_count;
    };
    result.range_overlap =
        overlaps(transfer.base_mip, transfer.mip_count, view.base_mip, view.mip_count) &&
        overlaps(transfer.base_layer, transfer.layer_count, view.base_layer, view.layer_count);
    result.range_covers =
        covers(transfer.base_mip, transfer.mip_count, view.base_mip, view.mip_count) &&
        covers(transfer.base_layer, transfer.layer_count, view.base_layer, view.layer_count);
    return result;
}

} // namespace

class PpTerminalScopeContentRuntime {
public:
    static constexpr u32 MaxTargets = 8;

    struct UpstreamSampleTeePreparation {
        HostPasses::PpUpstreamSampleTeeCapturePlan capture{};
        HostPasses::PpUpstreamSampleTeeCommandPlan commands{};
        HostPasses::PpUpstreamSampleTeeVulkanBarrier reuse_barrier{};
        HostPasses::PpUpstreamSampleTeeVulkanBarrier fragment_barrier{};
        HostPasses::PpUpstreamSampleTeeVulkanBarrier host_barrier{};
        HostPasses::PpUpstreamSampleTeeVulkanPushStages push_stages{};
        HostPasses::PpUpstreamSampleTeeTransferredDepthSourcePlan transferred_depth_source{};
        HostPasses::PpUpstreamSampleTeeSelectedDepthPlan selected_depth{};
    };

    struct DepthWriteBoundaryRuntimeFacts {
        VideoCore::ImageColorScopePrivateLink current_link{};
        VideoCore::SubresourceRange current_range{};
        u64 current_backing_generation{};
        u64 current_mutation_epoch{};
        HostPasses::PpDepthPredecessorCrossScopeRuntimeBoundaryCheckpoints
            boundary_checkpoints{};
        u64 boundary_after_guest_rendering_serial{};
        u64 boundary_after_host_rendering_instance_id{};
        u64 live_target_guest_rendering_serial{};
        u64 live_target_host_rendering_instance_id{};
        bool raw_boundary_active{};
        bool raw_boundary_tracked{};
        bool boundary_after_rendering_active{};
        bool boundary_after_diagnostic_origin{};
        bool live_target_rendering_active{};
        bool live_target_diagnostic_origin{};
    };

    PpTerminalScopeContentRuntime(const Instance& instance_, Scheduler& scheduler_,
                                  VideoCore::TextureCache& texture_cache_,
                                  PpTerminalScopeRuntimeConfig config_)
        : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_}, config{config_},
          reducer{config.window, 32, config.join_final_backing}, calibration{true},
          upstream_registry{config.content.upstream},
          upstream_feedback_frame{config.content.upstream,
                                  HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates},
          depth_predecessor_hint{
              HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates},
          depth_predecessor_two_hop{
              HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates, 0,
              HostPasses::IsPpDepthPredecessorCrossScopePairHintMode(
                  config_.content.capture_upstream_depth_predecessor_hint,
                  config_.content.upstream_depth_predecessor_hint_hops)
                  ? HostPasses::ResolvePpDepthPredecessorCrossScopeBootstrapLifecycleHops(
                        config_.content.capture_upstream_depth_predecessor_hint,
                        config_.content.upstream_depth_predecessor_hint_hops)
                  : HostPasses::ResolvePpDepthPredecessorMultiHopLifecycleHops(
                        config_.content.capture_upstream_depth_predecessor_hint,
                        config_.content.upstream_depth_predecessor_hint_hops)},
          depth_predecessor_cross_scope{
              HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates},
          upstream_input_gate{config.content.upstream_candidate_index},
          upstream_feedback_reducer{config.window, 32, false},
          slot_stride{Common::AlignUp<u64>(config.content.capture_upstream_input_content
                                               ? HostPasses::PpUpstreamInputSnapshotBytes
                                               : PpTerminalScopeSnapshotBytes,
                                           std::max<u64>(1, instance.NonCoherentAtomSize()))},
          download{instance,
                   scheduler,
                   VideoCore::MemoryUsage::Download,
                   0,
                   vk::BufferUsageFlagBits::eTransferDst,
                   slot_stride * FinalGuestSurfaceReadbackSlotPool::MaxSlots} {
        if (config.content.capture_upstream_sample_tee) {
            upstream_sample_tee_buffer = std::make_unique<VideoCore::Buffer>(
                instance, scheduler, VideoCore::MemoryUsage::Download, 0, VideoCore::AllFlags,
                Shader::PpUpstreamSampleTeeBufferBytes);
            upstream_sample_tee_reducer =
                std::make_unique<HostPasses::PpUpstreamSampleTeeCalibratedReducer>(
                    config.window, HostPasses::PpUpstreamSampleTeeCalibratedReducer::MaxHistory,
                    config.content.upstream_candidate_index, config.expected_calibrations);
        }
        LOG_INFO(
            Render,
            "PpTerminalScopeContentConfig enabled=1 frame_start={} frame_count={} "
            "selector_count={} expected_calibrations={} first={}/{}/{}/{}/{}/{} "
            "second={}/{}/{}/{}/{}/{} consumer={}/{}/{}/{}/{}/{} "
            "predecessor={}/{}/{}/{}/{}/{} targets={} slots={} max_bytes={} pre_first={} "
             "capture_predecessor={} input_content={} upstream_inputs={} upstream_pre_post={} "
             "upstream_input_content={} upstream_sample_routes={} upstream_sample_tee={} "
             "upstream_depth_pre_post={} upstream_depth_predecessor_hint={} "
            "upstream_depth_predecessor_hint_hops={} "
            "upstream_index={} "
            "upstream_candidate={} upstream_sample_route_input={} "
            "upstream={}/{}/{}/{}/{}/{} final_backing_join={}",
            config.window.frame_start, config.window.frame_count, config.watch_ordinals.count,
            config.expected_calibrations, static_cast<u32>(config.content.first.kind),
            config.content.first.indexed, config.content.first.element_count,
            config.content.first.instance_count, config.content.first.sampled_images,
            config.content.first.storage_writes, static_cast<u32>(config.content.second.kind),
            config.content.second.indexed, config.content.second.element_count,
            config.content.second.instance_count, config.content.second.sampled_images,
            config.content.second.storage_writes, static_cast<u32>(config.content.consumer.kind),
            config.content.consumer.indexed, config.content.consumer.element_count,
            config.content.consumer.instance_count, config.content.consumer.sampled_images,
            config.content.consumer.storage_writes,
            static_cast<u32>(config.content.predecessor.kind), config.content.predecessor.indexed,
            config.content.predecessor.element_count, config.content.predecessor.instance_count,
            config.content.predecessor.sampled_images, config.content.predecessor.storage_writes,
            MaxTargets, FinalGuestSurfaceReadbackSlotPool::MaxSlots, slot_stride,
            config.content.capture_pre_first, config.content.capture_predecessor,
            config.content.capture_sampled_input_content, config.content.capture_upstream_inputs,
            config.content.capture_upstream_pre_post, config.content.capture_upstream_input_content,
             config.content.capture_upstream_sample_routes,
             config.content.capture_upstream_sample_tee,
             config.content.capture_upstream_depth_pre_post,
             config.content.capture_upstream_depth_predecessor_hint,
             config.content.upstream_depth_predecessor_hint_hops,
             config.content.upstream_input_index,
            config.content.upstream_candidate_index,
            config.content.upstream_sample_route_input_index,
            static_cast<u32>(config.content.upstream.kind), config.content.upstream.indexed,
            config.content.upstream.element_count, config.content.upstream.instance_count,
            config.content.upstream.sampled_images, config.content.upstream.storage_writes,
            config.join_final_backing);
    }

    [[nodiscard]] VideoCore::Buffer* UpstreamSampleTeeBuffer() noexcept {
        return upstream_sample_tee_buffer.get();
    }

    [[nodiscard]] bool CapturesUpstreamSampleTee() const noexcept {
        return config.content.capture_upstream_sample_tee;
    }

    [[nodiscard]] bool CapturesUpstreamDepthPrePost() const noexcept {
        return config.content.capture_upstream_depth_pre_post;
    }

    void ObserveUpstreamSampleTeeProducerPreDepthScopeEntryBegin(
        VideoCore::ImageId image_id, VideoCore::Image& image, const RenderState& state,
        const VideoCore::SubresourceRange& range, u64 rendering_serial,
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash,
        bool effective_depth_write) {
        if (!config.content.capture_upstream_sample_tee || !scheduler.IsRendering()) {
            return;
        }
        const auto rendering_origin = scheduler.HostRenderingOrigin();
        const bool regular_origin =
            rendering_origin == HostRenderingInstanceOrigin::Regular;
        const bool diagnostic_origin =
            rendering_origin == HostRenderingInstanceOrigin::DiagnosticResume;
        if (!regular_origin && !diagnostic_origin) {
            return;
        }
        const VideoCore::ImageColorScopePrivateLink link{.id = image_id, .uid = image.image_uid};
        const u64 current_backing_generation =
            image.backing ? image.EnsureDiagnosticBackingGeneration() : 0;
        const u64 current_mutation_epoch = image.DiagnosticMutationEpoch();
        const bool current = image.backing && current_backing_generation != 0 &&
                             current_mutation_epoch != 0;
        const auto predecessor_endpoint =
            BuildCurrentDepthPredecessorEndpoint(image_id, image, range);
        const auto predecessor_shape = HostPasses::
            MakePpUpstreamSampleTeeProducerPreDepthScopeLoadWriterShape({
                .producer = image.PeekDiagnosticDepthStencilProducer(),
                .endpoint = predecessor_endpoint,
            });
        const u64 host_instance = scheduler.HostRenderingInstanceId();
        const HostPasses::
            PpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBeginDescriptor
                resource_descriptor{
                    .enabled = true,
                    .sample_tee_enabled = config.content.capture_upstream_sample_tee,
                    .upstream_feedback_armed = upstream_feedback_armed,
                    .target_hint = upstream_sample_tee_producer_pre_hint,
                    .resolved_attachment_link = link,
                    .resolved_attachment_range = range,
                    .resolved_backing_generation = current_backing_generation,
                    .resolved_mutation_epoch = current_mutation_epoch,
                    .current_predecessor_endpoint = predecessor_endpoint,
                    .predecessor_shape = predecessor_shape.shape,
                    .predecessor_shape_exact = predecessor_shape.exact,
                    .source_still_live = current,
                    .guest_rendering_serial = rendering_serial,
                    .host_rendering_instance_id = host_instance,
                    .attachment_has_depth = state.depth_stencil_attachment.has_depth,
                    .depth_clear = state.depth_stencil_attachment.depth_clear,
                    .stencil_clear = state.depth_stencil_attachment.stencil_clear,
                    .rendering_active = true,
                    .regular_origin = regular_origin,
                    .diagnostic_origin = diagnostic_origin,
                };
        const auto resource_observed = HostPasses::
            ObservePpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBegin(
                upstream_sample_tee_producer_pre_scope_entry, resource_descriptor);
        const auto evidence_terminal = diagnostic_origin
                                           ? resource_observed
                                           : HostPasses::
                                                 ObservePpUpstreamSampleTeeProducerPreDepthScopeEntryResourceBegin(
                                                     {}, resource_descriptor);
        upstream_sample_tee_scope_load_candidate_evidence.ObserveRegularLoad(
            HostPasses::AdaptPpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidence(
                resource_descriptor, evidence_terminal));
        if (diagnostic_origin) {
            return;
        }

        if (!state.depth_stencil_attachment.depth_clear) {
            if (resource_observed.collision) {
                upstream_sample_tee_producer_pre_scope_entry_collision = true;
            } else if (resource_observed.valid) {
                upstream_sample_tee_producer_pre_scope_entry = resource_observed;
            }
            return;
        }

        // Scope clears retain the existing exact target-writer gate. Only a Load entry may be
        // latched from the prior private resource hint before current-frame target arming.
        if (!upstream_feedback_armed) {
            return;
        }
        const auto intent = BuildDepthPredecessorIntent(
            link, image, range, draw, fragment_pgm_hash, effective_depth_write);
        if (!HostPasses::
                IsPpUpstreamSampleTeeProducerPreDepthScopeEntryRuntimeEligible({
                    .armed = upstream_feedback_armed,
                    .learned_hint = upstream_sample_tee_producer_pre_hint,
                    .observed_writer = intent,
                    .current_backing_generation = current_backing_generation,
                    .current_mutation_epoch = current_mutation_epoch,
                    .source_still_live = current,
                })) {
            return;
        }
        if (upstream_sample_tee_producer_pre_scope_entry.valid &&
            upstream_sample_tee_producer_pre_scope_entry.guest_rendering_serial ==
                rendering_serial &&
            upstream_sample_tee_producer_pre_scope_entry.host_rendering_instance_id ==
                host_instance) {
            return;
        }
        const auto format = image.info.pixel_format;
        const bool exact_clear_encoding = format == vk::Format::eD32Sfloat ||
                                          format == vk::Format::eD32SfloatS8Uint;
        const auto observed =
            HostPasses::ObservePpUpstreamSampleTeeProducerPreDepthScopeEntryBegin(
                upstream_sample_tee_producer_pre_scope_entry,
                {.enabled = true,
                 .predecessor_endpoint = predecessor_endpoint,
                 .guest_rendering_serial = rendering_serial,
                 .host_rendering_instance_id = host_instance,
                 .attachment_has_depth = state.depth_stencil_attachment.has_depth,
                 .depth_clear = state.depth_stencil_attachment.depth_clear,
                 .stencil_clear = state.depth_stencil_attachment.stencil_clear,
                 .private_depth_clear_bits = state.depth_stencil_attachment.clear_value[0],
                 .raw_depth_clear_encoding_exact = exact_clear_encoding,
                 .rendering_active = true,
                 .regular_origin = true});
        if (observed.collision) {
            upstream_sample_tee_producer_pre_scope_entry_collision = true;
        } else if (observed.valid) {
            upstream_sample_tee_producer_pre_scope_entry = observed;
        }
    }

    void ResetUpstreamSampleTeeBinding() noexcept {
        upstream_sample_tee_binding = {};
        upstream_sample_tee_binding_count = 0;
    }

    [[nodiscard]] bool RetainUpstreamSampleTeeBinding(
        const HostPasses::PpUpstreamSampleTeeBindingPlan& binding) noexcept {
        if (upstream_sample_tee_binding_count != 0 || !binding.bind ||
            binding.status != FinalGuestSurfaceStatus::Complete || binding.loss.Any()) {
            return false;
        }
        upstream_sample_tee_binding = binding;
        upstream_sample_tee_binding_count = 1;
        return true;
    }

    [[nodiscard]] bool IsPredecessorCandidate(
        const VideoCore::ImageColorScopeDrawDescriptor& draw) const noexcept {
        return config.content.capture_predecessor &&
               MatchesPpTerminalScopeDraw(config.content.predecessor,
                                          {
                                              .kind = draw.kind,
                                              .indexed = draw.indexed,
                                              .element_count = draw.element_count,
                                              .instance_count = draw.instance_count,
                                              .sampled_images = draw.sampled_images,
                                              .storage_writes = draw.storage_writes,
                                          });
    }

    [[nodiscard]] bool ShouldClassifyInputs(
        const VideoCore::ImageColorScopeDrawDescriptor& draw) const noexcept {
        const PpTerminalScopeDrawSelector observed{
            .kind = draw.kind,
            .indexed = draw.indexed,
            .element_count = draw.element_count,
            .instance_count = draw.instance_count,
            .sampled_images = draw.sampled_images,
            .storage_writes = draw.storage_writes,
        };
        return IsPredecessorCandidate(draw) ||
               (config.content.capture_upstream_inputs &&
                !config.content.capture_upstream_depth_predecessor_hint &&
                MatchesPpTerminalScopeUpstreamTarget(config.content, observed,
                                                     draw.fragment_pgm_hash));
    }

    void ObserveConsumer(const VideoCore::ImageColorScopeDrawDescriptor& draw,
                         VideoCore::Image* sampled_input,
                         VideoCore::ImageColorScopePrivateLink output_link,
                         VideoCore::Image* output_image, u32 output_attachment_index,
                         bool single_output, const RenderState& state, u64 rendering_serial) {
        Entry* entry = FindLineageTail(draw.sampled_input_image);
        if (!entry) {
            entry = Find(draw.sampled_input_image);
        }
        if (!entry) {
            return;
        }
        const auto gate_link = entry->discovered ? entry->lineage.Root() : draw.sampled_input_image;
        const auto action =
            entry->gate.ObserveConsumer(gate_link.uid, {
                                                           .kind = draw.kind,
                                                           .indexed = draw.indexed,
                                                           .element_count = draw.element_count,
                                                           .instance_count = draw.instance_count,
                                                           .sampled_images = draw.sampled_images,
                                                           .storage_writes = draw.storage_writes,
                                                       });
        if (action != PpTerminalScopeConsumerAction::CaptureConsumer) {
            return;
        }
        if (entry->discovered) {
            progress_coverage.Observe({.consumer_action = true});
        }
        if (!sampled_input) {
            return;
        }
        if (entry->discovered) {
            auto handoff = PlanPpTerminalScopeLineageHandoff(
                true, entry->lineage.Status(), entry->lineage.Loss(), entry->recorded_plane_mask,
                single_output && output_link.Valid() && output_image);
            if (handoff.status == FinalGuestSurfaceStatus::Complete) {
                const auto extension = entry->lineage.Extend({
                    .input = draw.sampled_input_image,
                    .input_generation = sampled_input->EnsureDiagnosticBackingGeneration(),
                    .output = output_link,
                    .output_generation = output_image->EnsureDiagnosticBackingGeneration(),
                    .single_input = draw.sampled_images == 1,
                    .single_output = single_output,
                });
                handoff =
                    PlanPpTerminalScopeLineageHandoff(true, extension.status, extension.loss,
                                                      entry->recorded_plane_mask, single_output);
            }
            if (!handoff.capture_consumer) {
                entry->status = handoff.status;
                entry->loss = handoff.loss;
                return;
            }
            RecordPlane(*entry, *sampled_input, state, 0, rendering_serial, 2);
            if (entry->status == FinalGuestSurfaceStatus::Complete && !entry->loss.Any() &&
                handoff.capture_output) {
                RecordPlane(*entry, *output_image, state, output_attachment_index, rendering_serial,
                            3, true);
            }
            const u32 complete_mask = config.content.capture_predecessor ? 0x7fu : 0x1fu;
            if (entry->status == FinalGuestSurfaceStatus::Complete && !entry->loss.Any() &&
                entry->recorded_plane_mask == complete_mask && handoff.publish_flip_alias) {
                entry->flip_alias = output_link;
                entry->flip_alias_ready = true;
            }
            progress_coverage.Observe({
                .consumer_captured = (entry->recorded_plane_mask & (1u << 2)) != 0,
                .output_captured = (entry->recorded_plane_mask & (1u << 3)) != 0,
                .alias_ready = entry->flip_alias_ready,
            });
            return;
        }
        entry->status = FinalGuestSurfaceStatus::Complete;
        entry->loss = {};
        RecordPlane(*entry, *sampled_input, state, 0, rendering_serial, 2, true);
    }

    void ObserveTargetSampledDepthWriter(
        std::span<const HostPasses::PpUpstreamFeedbackProducerDepthConsumerInput>
            sampled_depth_consumers) noexcept {
        if (sampled_depth_consumers.size() != 1) {
            target_sampled_depth_writer_collision = true;
            return;
        }
        const auto& writer = sampled_depth_consumers.front().current_writer;
        const bool writer_valid = writer.valid && writer.fragment_pgm_hash != 0;
        if (!writer_valid) {
            target_sampled_depth_writer_collision = true;
            return;
        }
        if (!target_sampled_depth_writer_valid) {
            target_sampled_depth_writer_hash = writer.fragment_pgm_hash;
            target_sampled_depth_writer_valid = true;
            return;
        }
        if (target_sampled_depth_writer_hash != writer.fragment_pgm_hash) {
            target_sampled_depth_writer_collision = true;
        }
    }

    void ObserveTargetFixedFunctionState(
        const PpUpstreamTargetFixedFunctionState& observed) noexcept {
        if (!observed.valid) {
            target_fixed_function_state_collision = true;
            return;
        }
        if (!target_fixed_function_state_valid) {
            target_fixed_function_state = observed;
            target_fixed_function_state_valid = true;
            return;
        }
        if (target_fixed_function_state != observed) {
            target_fixed_function_state_collision = true;
        }
    }

    void ObservePreDraw(
        VideoCore::ImageId image_id, VideoCore::Image& image, const RenderState& state,
        u32 attachment_index, u64 rendering_serial,
        const VideoCore::ImageColorScopeDrawDescriptor& draw,
        const PpTerminalScopeSampledInputs& sampled_inputs,
        std::span<VideoCore::Image* const> sampled_images,
        std::span<const VideoCore::ImageColorScopePrivateLink> sampled_links,
        std::span<const VideoCore::ImageColorScopeTrackedView> sampled_views,
        std::span<const VideoCore::ImageColorScopeProducerObservation> sampled_scopes,
        std::span<const HostPasses::PpUpstreamFeedbackProducerDepthConsumerInput>
            sampled_depth_consumers,
        const HostPasses::PpUpstreamFeedbackSampledViewBindingObservation& sampled_view_binding,
        std::span<const HostPasses::PpUpstreamInputContentView> upstream_views,
        std::span<const PpUpstreamSampledExtentObservation> sampled_extents,
        std::span<const HostPasses::PpUpstreamSampleRouteObservation> upstream_sample_routes,
        std::span<const HostPasses::PpUpstreamFeedbackCubeLiveObservation>
            diagnostic_sampled_cube_bindings,
        const PpUpstreamTargetFixedFunctionState& target_fixed_function_state,
        const Shader::Info* fragment_info,
        const HostPasses::PpUpstreamSampleTeeOutputPlan& tee_outputs,
        vk::PipelineLayout pipeline_layout) {
        const PpTerminalScopeDrawSelector observed{
            .kind = draw.kind,
            .indexed = draw.indexed,
            .element_count = draw.element_count,
            .instance_count = draw.instance_count,
            .sampled_images = draw.sampled_images,
            .storage_writes = draw.storage_writes,
        };
        const VideoCore::ImageColorScopePrivateLink link{.id = image_id, .uid = image.image_uid};
        const u64 fragment_pgm_hash = fragment_info ? fragment_info->pgm_hash : 0;
        if (config.content.capture_upstream_inputs &&
            !config.content.capture_upstream_depth_pre_post &&
            MatchesPpTerminalScopeUpstreamTarget(config.content, observed, fragment_pgm_hash) &&
            (!config.content.capture_upstream_sample_tee ||
             draw.fragment_pgm_hash == fragment_pgm_hash)) {
            ObserveTargetSampledDepthWriter(sampled_depth_consumers);
            ObserveTargetFixedFunctionState(target_fixed_function_state);
            ObserveUpstreamFeedbackPreDraw(link, image, state, attachment_index, rendering_serial,
                                           observed, upstream_views, sampled_images, sampled_views,
                                           sampled_scopes, sampled_extents, upstream_sample_routes,
                                           diagnostic_sampled_cube_bindings,
                                           fragment_info, tee_outputs, pipeline_layout);
            static_cast<void>(HostPasses::ObservePpUpstreamFeedbackSelectionProducer(
                upstream_feedback_selection_authority, upstream_registry, link, observed,
                sampled_inputs,
                sampled_depth_consumers.size() == 1 ? sampled_depth_consumers.front()
                                                    : HostPasses::PpUpstreamFeedbackProducerDepthConsumerInput{},
                sampled_view_binding));
        }
        Entry* entry = Find(link);
        const bool initially_tracked = entry != nullptr;
        const auto& allocation_selector =
            config.content.capture_predecessor ? config.content.predecessor : config.content.first;
        const bool exact_candidate = MatchesPpTerminalScopeDraw(allocation_selector, observed);
        const bool mapping_valid = mapping.Valid();
        const bool target_valid = IsValidTarget(link, &image);
        const bool target_capacity = HasTargetCapacity();
        const auto discovery = PlanPpTerminalScopeDiscoveryDecision(
            config.content.enabled, initially_tracked, exact_candidate, mapping_valid, target_valid,
            target_capacity);
        bool allocated = false;
        if (!entry && discovery.allocate) {
            entry = FindOrAllocate(link);
            if (entry) {
                allocated = true;
                ConfigureEntry(*entry, link, image, true);
            }
        }
        if (!entry) {
            discovery_coverage.Observe({
                .exact_candidate = exact_candidate,
                .tracked = initially_tracked,
                .mapping_valid = mapping_valid,
                .target_valid = target_valid,
                .capacity = target_capacity,
                .allocated = allocated,
            });
            return;
        }
        const bool restart =
            mapping_valid && target_valid &&
            entry->gate.CanRestartAtFirst(image.image_uid, rendering_serial, observed);
        discovery_coverage.Observe({
            .exact_candidate = exact_candidate,
            .tracked = initially_tracked,
            .mapping_valid = mapping_valid,
            .target_valid = target_valid,
            .capacity = target_capacity,
            .allocated = allocated,
            .restarted = restart,
        });
        if (restart) {
            RestartEntry(*entry, link, image);
            progress_coverage.Observe({
                .restart = true,
                .restart_plan_complete = entry->plan.status == FinalGuestSurfaceStatus::Complete &&
                                         !entry->plan.loss.Any() && entry->plan.copy,
            });
        }
        const auto action = entry->gate.PreviewDraw(image.image_uid, rendering_serial, observed);
        if (action == PpTerminalScopePreDrawAction::ShapeLoss) {
            entry->status = FinalGuestSurfaceStatus::GapLoss;
            entry->loss = {.gap = 1};
            return;
        }
        if (action == PpTerminalScopePreDrawAction::CaptureBeforePredecessor) {
            entry->sampled_inputs = sampled_inputs;
            if (config.content.capture_upstream_inputs &&
                !config.content.capture_upstream_depth_predecessor_hint) {
                const u32 selected = config.content.upstream_input_index;
                PpTerminalScopeUpstreamInputRegistryResolution resolution{};
                if (config.content.capture_upstream_depth_pre_post &&
                    selected < sampled_images.size() && sampled_images[selected] &&
                    selected < sampled_views.size()) {
                    auto& selected_image = *sampled_images[selected];
                    const auto& tracked = sampled_views[selected];
                    const u64 current_generation = selected_image.EnsureDiagnosticBackingGeneration();
                    const u64 current_epoch = selected_image.DiagnosticMutationEpoch();
                    const bool tracked_exact =
                        tracked.valid && !tracked.view_conflict && !tracked.storage &&
                        tracked.uniform_state && tracked.samples == 1 &&
                        tracked.aspect_mask == static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT) &&
                        static_cast<bool>(selected_image.aspect_mask &
                                          vk::ImageAspectFlagBits::eDepth);
                    const auto range_covers = [&tracked](u32 base_mip, u32 mip_count,
                                                         u32 base_layer, u32 layer_count) {
                        return mip_count != 0 && layer_count != 0 && tracked.mip_count != 0 &&
                               tracked.layer_count != 0 && base_mip <= tracked.base_mip &&
                               tracked.base_mip - base_mip < mip_count &&
                               tracked.mip_count <=
                                   mip_count - (tracked.base_mip - base_mip) &&
                               base_layer <= tracked.base_layer &&
                               tracked.base_layer - base_layer < layer_count &&
                               tracked.layer_count <=
                                   layer_count - (tracked.base_layer - base_layer);
                    };
                    u32 candidate = HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates;
                    const auto direct = selected_image.PeekDiagnosticDepthStencilProducer();
                    const bool direct_match =
                        tracked_exact && tracked.base_mip == 0 && tracked.mip_count == 1 &&
                        tracked.base_layer == 0 && tracked.layer_count == 1 && direct.valid &&
                        direct.upstream_candidate_valid &&
                        direct.upstream_candidate_generation == depth_candidate_generation &&
                        direct.mutation_epoch == tracked.mutation_epoch &&
                        direct.backing_generation == tracked.backing_generation &&
                        tracked.mutation_epoch == current_epoch &&
                        tracked.backing_generation == current_generation;
                    const auto transfer = selected_image.PeekDiagnosticTransfer();
                    const u32 depth_aspect = static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT);
                    const bool transfer_match =
                        tracked_exact && transfer.valid && transfer.source_depth_stencil_exact &&
                        transfer.source_depth_stencil_upstream_candidate_valid &&
                        transfer.source_depth_stencil_upstream_candidate_generation ==
                            depth_candidate_generation &&
                        transfer.mutation_epoch == tracked.mutation_epoch &&
                        transfer.backing_generation == tracked.backing_generation &&
                        tracked.mutation_epoch == current_epoch &&
                        tracked.backing_generation == current_generation && transfer.full_xy &&
                        (transfer.aspect_mask & depth_aspect) != 0 &&
                        range_covers(transfer.base_mip, transfer.mip_count, transfer.base_layer,
                                     transfer.layer_count);
                    if (direct_match) {
                        candidate = direct.upstream_candidate_index;
                    } else if (transfer_match) {
                        candidate = transfer.source_depth_stencil_upstream_candidate_index;
                    }
                    const bool candidate_valid = candidate < depth_candidate_inputs.size();
                    const bool inputs_valid =
                        candidate_valid && depth_candidate_inputs_valid[candidate];
                    const bool selected_candidate =
                        inputs_valid && (!config.content.capture_upstream_pre_post ||
                                         upstream_feedback_frame.SelectCandidate(candidate));
                    if (selected_candidate) {
                        resolution = {.inputs = depth_candidate_inputs[candidate],
                                      .status = FinalGuestSurfaceStatus::Complete,
                                      .matched = true};
                    }
                } else if (!config.content.capture_upstream_pre_post) {
                    if (selected < sampled_links.size()) {
                        resolution = upstream_registry.Resolve(sampled_links[selected]);
                    }
                } else {
                    const auto selection_attempt =
                        HostPasses::ObservePpUpstreamFeedbackSelectionPredecessorAttempt(
                            upstream_feedback_selection_authority, upstream_registry,
                            upstream_feedback_frame, sampled_links, selected,
                            true);
                    resolution = selection_attempt.resolution;
                }
                entry->upstream_inputs = resolution.inputs;
                if (!resolution.matched) {
                    entry->status = resolution.status == FinalGuestSurfaceStatus::AlreadyConsumed
                                        ? FinalGuestSurfaceStatus::GapLoss
                                        : resolution.status;
                    entry->loss = resolution.loss;
                    if (!entry->loss.Any()) {
                        entry->loss.gap = 1;
                    }
                }
            }
            std::array<PpTerminalScopeSampledInputView, PpTerminalScopeSampledInputs::MaxInputs>
                sampled_views{};
            for (u32 input = 0; input < sampled_inputs.count; ++input) {
                sampled_views[input] = sampled_inputs.inputs[input].view;
            }
            entry->input_content_plan = PlanPpTerminalScopeSampledInputContent({
                .enabled = config.content.capture_sampled_input_content,
                .logical_width = mapping.logical_width,
                .logical_height = mapping.logical_height,
                .base_offset = entry->plan.total_bytes,
                .selector = config.watch_ordinals,
                .views = std::span{sampled_views.data(), sampled_inputs.count},
                .buffer_alignment = 16,
                .max_regions = PpTerminalScopeSampledInputContentPlan::MaxInputs *
                               FinalGuestSurfaceWatchOrdinals::MaxOrdinals,
                .max_bytes = PpTerminalScopeSnapshotBytes,
            });
            if (config.content.capture_sampled_input_content &&
                (entry->input_content_plan.status != FinalGuestSurfaceStatus::Complete ||
                 entry->input_content_plan.loss.Any() || !entry->input_content_plan.copy)) {
                entry->status = entry->input_content_plan.status;
                entry->loss = entry->input_content_plan.loss;
            }
            RecordPlane(*entry, image, state, attachment_index, rendering_serial, 5, false,
                        sampled_images);
            if (entry->discovered) {
                progress_coverage.Observe({
                    .predecessor_before_action = true,
                    .predecessor_before_captured = (entry->recorded_plane_mask & (1u << 5)) != 0,
                });
            }
        } else if (action == PpTerminalScopePreDrawAction::CaptureBeforeFirst) {
            entry->predecessor = ClassifyPpTerminalScopePredecessor(
                image.PeekDiagnosticProducer(), image.ObserveDiagnosticColorScope());
            RecordPlane(*entry, image, state, attachment_index, rendering_serial, 4);
            if (entry->discovered) {
                progress_coverage.Observe({
                    .first_before_action = true,
                    .first_before_captured = (entry->recorded_plane_mask & (1u << 4)) != 0,
                });
            }
        }
    }

    void ObserveDepthUpstreamPreDraw(
        VideoCore::ImageId image_id, VideoCore::Image& image, const RenderState& state,
        u64 rendering_serial, const VideoCore::ImageColorScopeDrawDescriptor& draw,
        const PpTerminalScopeSampledInputs& sampled_inputs,
        const PpTerminalScopeDepthWriteBoundary& depth_write_boundary,
        const DepthWriteBoundaryRuntimeFacts& boundary_facts,
        const VideoCore::SubresourceRange& current_range, bool effective_depth_write,
        const Shader::Info* fragment_info) {
        if (!config.content.capture_upstream_inputs ||
            !config.content.capture_upstream_depth_pre_post) {
            return;
        }
        const PpTerminalScopeDrawSelector observed{
            .kind = draw.kind,
            .indexed = draw.indexed,
            .element_count = draw.element_count,
            .instance_count = draw.instance_count,
            .sampled_images = draw.sampled_images,
            .storage_writes = draw.storage_writes,
        };
        const u64 fragment_pgm_hash = fragment_info ? fragment_info->pgm_hash : 0;
        const bool hint_mode = config.content.capture_upstream_depth_predecessor_hint;
        const bool cross_scope_hint_mode = IsCrossScopeDepthPredecessorHintMode();
        const bool two_hop_hint_mode =
            IsDepthPredecessorStructuralMultiHopRuntime();
        const bool one_hop_hint_mode = IsOneHopDepthPredecessorHintMode();
        const bool configured_anchor = MatchesPpTerminalScopeUpstreamTarget(
            config.content, observed, fragment_pgm_hash);
        const VideoCore::ImageColorScopePrivateLink link{.id = image_id, .uid = image.image_uid};
        const bool raw_prefix_restart =
            two_hop_hint_mode && depth_predecessor_two_hop.MatchesRawPrefixRestart(
                                     observed, fragment_pgm_hash);

        if (cross_scope_hint_mode &&
            !depth_predecessor_runtime_plan.begin_cross_scope &&
            !depth_predecessor_runtime_plan.begin_multi_hop) {
            return;
        }

        if (cross_scope_hint_mode &&
            depth_predecessor_runtime_plan.path ==
                HostPasses::PpDepthPredecessorRuntimePath::CrossScopeArmed) {
            if (!depth_predecessor_cross_scope_frame_open ||
                depth_predecessor_cross_scope_capture_closed) {
                return;
            }
            const bool matches_source =
                depth_predecessor_cross_scope.MatchesArmedSource(observed,
                                                                 fragment_pgm_hash);
            const bool matches_target =
                depth_predecessor_cross_scope.MatchesTargetPre(observed,
                                                               fragment_pgm_hash);
            const bool matches_target_repeat =
                depth_predecessor_cross_scope.MatchesTargetRepeat(
                    observed, fragment_pgm_hash);
            const bool matches_tail =
                depth_predecessor_cross_scope.MatchesTail(observed,
                                                          fragment_pgm_hash);
            const u32 open_count =
                static_cast<u32>(open_depth_predecessor_cross_scope_source_draw.valid) +
                static_cast<u32>(open_depth_predecessor_cross_scope_target_draw.valid) +
                static_cast<u32>(
                    open_depth_predecessor_cross_scope_p0_repeat_draw.valid) +
                static_cast<u32>(open_depth_predecessor_cross_scope_tail_draw.valid);
            const auto dispatch =
                HostPasses::ClassifyPpDepthPredecessorCrossScopeRuntimeDispatch({
                    .phase = depth_predecessor_cross_scope_outcome.result.phase,
                    .role = depth_predecessor_cross_scope_outcome.result.role,
                    .pre_draw = true,
                    .configured_anchor = configured_anchor,
                    .matches_source_post = matches_source,
                    .matches_target_pre = matches_target,
                    .matches_target_repeat = matches_target_repeat,
                    .matches_tail = matches_tail,
                    .source_reservation_open =
                        open_depth_predecessor_cross_scope_source_draw.valid,
                    .target_post_open =
                        open_depth_predecessor_cross_scope_target_draw.valid,
                    .target_repeat_post_open =
                        open_depth_predecessor_cross_scope_p0_repeat_draw.valid,
                    .tail_post_open =
                        open_depth_predecessor_cross_scope_tail_draw.valid,
                    .cross_open_collision = open_count > 1,
                    .source_restart_allowed =
                        depth_predecessor_cross_scope.CanBeginSourcePostRestart(),
                });
            using CrossDispatch =
                HostPasses::PpDepthPredecessorCrossScopeRuntimeDispatch;
            switch (dispatch) {
            case CrossDispatch::Anchor: {
                const auto current = BuildCurrentDepthPredecessorObservation(
                    image_id, image, current_range, depth_candidate_generation);
                const HostPasses::PpDepthPredecessorHint anchor_hint{
                    .draw = observed,
                    .fragment_pgm_hash = fragment_pgm_hash,
                    .valid = true,
                };
                const auto anchor = depth_predecessor_cross_scope.ObserveAnchor(
                    anchor_hint, current.writer);
                AccumulateCrossScopeOutcome(anchor);
                bool selected{};
                if (!IsCrossScopeFailure(anchor.decision) && anchor.select &&
                    upstream_feedback_armed) {
                    selected = upstream_feedback_frame.SelectCandidate(
                        anchor.candidate_index);
                }
                depth_predecessor_cross_scope_outcome.actual_selected |= selected;
                depth_predecessor_cross_scope_capture_closed = true;
                open_depth_predecessor_cross_scope_source_draw = {};
                open_depth_predecessor_cross_scope_target_draw = {};
                open_depth_predecessor_cross_scope_tail_draw = {};
                open_depth_predecessor_cross_scope_p0_repeat_draw = {};
                if (!selected || IsCrossScopeFailure(anchor.decision)) {
                    RejectCrossScopeCapture();
                }
                return;
            }
            case CrossDispatch::SourcePreReserve:
                if (!effective_depth_write ||
                    !HostPasses::IsCanonicalPpDepthPredecessorRange(current_range) ||
                    !ReserveDepthPredecessorCrossScopeSource(
                        link, image, observed, current_range, fragment_pgm_hash,
                        rendering_serial)) {
                    RejectCrossScopeCapture();
                }
                return;
            case CrossDispatch::SourcePreRestart:
                if (!effective_depth_write ||
                    !HostPasses::IsCanonicalPpDepthPredecessorRange(
                        current_range) ||
                    !RestartDepthPredecessorCrossScopeSource(
                        image_id, link, image, observed, current_range,
                        fragment_pgm_hash, rendering_serial)) {
                    RejectCrossScopeCapture();
                }
                return;
            case CrossDispatch::TargetPrePlane1: {
                const auto boundary_before = HostPasses::
                    SelectPpDepthPredecessorCrossScopeArmedResumeCheckpoint(
                        boundary_facts.boundary_checkpoints);
                const auto endpoint = BuildCurrentDepthPredecessorEndpoint(
                    image_id, image, current_range);
                const auto intent = BuildDepthPredecessorIntent(
                    link, image, current_range, observed, fragment_pgm_hash,
                    effective_depth_write);
                const auto first_writer_pre =
                    BuildDepthFirstWriterPreObservation(image, rendering_serial);
                auto candidate =
                    depth_predecessor_cross_scope_resume.source_candidate;
                candidate.recorded_plane_mask =
                    upstream_feedback_frame.RecordedPlaneMask(
                        candidate.candidate_index);
                const HostPasses::
                    PpDepthPredecessorCrossScopeTargetPreRuntimeObservation
                        raw_target{
                        .source_resume = depth_predecessor_cross_scope_resume,
                        .target_pre_current_writer = endpoint.writer,
                        .target_intent = intent,
                        .candidate = candidate,
                        .raw_current_link = boundary_facts.current_link,
                        .raw_current_range = boundary_facts.current_range,
                        .raw_current_backing_generation =
                            boundary_facts.current_backing_generation,
                        .raw_current_mutation_epoch =
                            boundary_facts.current_mutation_epoch,
                        .raw_boundary_active =
                            boundary_facts.raw_boundary_active,
                        .raw_boundary_tracked =
                            boundary_facts.raw_boundary_tracked,
                        .boundary_before_guest_rendering_serial =
                            boundary_before.guest_rendering_serial,
                        .boundary_before_host_rendering_instance_id =
                            boundary_before.host_rendering_instance_id,
                        .boundary_after_guest_rendering_serial =
                            boundary_facts.boundary_after_guest_rendering_serial,
                        .boundary_after_host_rendering_instance_id =
                            boundary_facts.boundary_after_host_rendering_instance_id,
                        .boundary_before_rendering_active =
                            boundary_before.rendering_active,
                        .boundary_before_diagnostic_origin =
                            boundary_before.diagnostic_origin,
                        .boundary_after_rendering_active =
                            boundary_facts.boundary_after_rendering_active,
                        .boundary_after_regular_origin =
                            !boundary_facts.boundary_after_diagnostic_origin,
                        .boundary_after_diagnostic_origin =
                            boundary_facts.boundary_after_diagnostic_origin,
                        .live_target_guest_rendering_serial =
                            boundary_facts.live_target_guest_rendering_serial,
                        .live_target_host_rendering_instance_id =
                            boundary_facts.live_target_host_rendering_instance_id,
                        .live_target_rendering_active =
                            boundary_facts.live_target_rendering_active,
                        .live_target_regular_origin =
                            !boundary_facts.live_target_diagnostic_origin,
                        .live_target_diagnostic_origin =
                            boundary_facts.live_target_diagnostic_origin,
                        .source_outside_target_guest_scope =
                            endpoint.guest_rendering_serial != rendering_serial,
                        .after_begin_rendering =
                            boundary_facts.boundary_after_rendering_active,
                        .target_depth_load =
                            state.depth_stencil_attachment.has_depth &&
                            !state.depth_stencil_attachment.depth_clear,
                        .target_depth_clear =
                            state.depth_stencil_attachment.depth_clear,
                        .target_stencil_clear =
                            state.depth_stencil_attachment.stencil_clear,
                        .depth_aspect =
                            state.depth_stencil_attachment.has_depth &&
                            static_cast<bool>(image.aspect_mask &
                                              vk::ImageAspectFlagBits::eDepth),
                        .canonical_attachment =
                            HostPasses::IsCanonicalPpDepthPredecessorRange(
                                current_range),
                    };
                const bool stored_source_exact =
                    depth_predecessor_cross_scope.MatchesStoredSourcePost(
                        raw_target.source_resume.source_post);
                const auto attempt = HostPasses::
                    ClassifyPpDepthPredecessorCrossScopeTargetPreAttempt(
                        raw_target, stored_source_exact);
                using TargetPreDisposition = HostPasses::
                    PpDepthPredecessorCrossScopeTargetPreAttemptDisposition;
                if (attempt.disposition ==
                    TargetPreDisposition::IgnoreNoTransition) {
                    return;
                }
                const auto& built = attempt.boundary;
                const auto armed_target_pre_evidence = HostPasses::
                    SnapshotPpDepthPredecessorCrossScopeArmedTargetPreEvidence(
                        raw_target, stored_source_exact);
                depth_predecessor_cross_scope_outcome
                    .armed_target_pre_evidence = HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeArmedTargetPreEvidence(
                        depth_predecessor_cross_scope_outcome
                            .armed_target_pre_evidence,
                        {
                            .event = HostPasses::
                                PpDepthPredecessorCrossScopeArmedTargetPreEvidenceEvent::
                                    Attempt,
                            .evidence = armed_target_pre_evidence,
                        });
                const auto target_pre =
                    depth_predecessor_cross_scope.ObserveTargetPre(
                        built.observation);
                AccumulateCrossScopeOutcome(target_pre);
                if (!built.exact || IsCrossScopeFailure(target_pre.decision) ||
                    !target_pre.boundary_exact ||
                    !RecordDepthPredecessorCrossScopeTargetPre(
                        image, state, rendering_serial, intent, observed,
                        fragment_pgm_hash, first_writer_pre)) {
                    RejectCrossScopeCapture();
                }
                return;
            }
            case CrossDispatch::TargetRepeatPreMetadata: {
                const auto endpoint = BuildCurrentDepthPredecessorEndpoint(
                    image_id, image, current_range);
                const auto intent = BuildDepthPredecessorIntent(
                    link, image, current_range, observed, fragment_pgm_hash,
                    effective_depth_write);
                auto candidate =
                    depth_predecessor_cross_scope_resume.source_candidate;
                candidate.recorded_plane_mask =
                    upstream_feedback_frame.RecordedPlaneMask(
                        candidate.candidate_index);
                const HostPasses::
                    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp
                        slot_stamp{
                            .generation =
                                depth_predecessor_cross_scope_outcome.result
                                    .generation,
                            .candidate_index = candidate.candidate_index,
                            .present = static_cast<bool>(upstream_feedback_slot),
                        };
                const auto repeat_pre =
                    depth_predecessor_cross_scope
                        .ObserveTargetRepeatPreMetadata({
                            .stored_latest =
                                depth_predecessor_cross_scope_outcome
                                    .target_post_endpoint,
                            .current_endpoint = endpoint,
                            .intent = intent,
                            .candidate = candidate,
                            .slot = slot_stamp,
                            .retained_slot_token = upstream_feedback_slot,
                            .live_guest_rendering_serial =
                                scheduler.RenderingSerial(),
                            .live_host_rendering_instance_id =
                                scheduler.HostRenderingInstanceId(),
                            .live_rendering_active = scheduler.IsRendering(),
                            .live_regular_origin =
                                scheduler.HostRenderingOrigin() ==
                                HostRenderingInstanceOrigin::Regular,
                            .live_diagnostic_origin =
                                scheduler.HostRenderingOrigin() ==
                                HostRenderingInstanceOrigin::DiagnosticResume,
                            .configured_effective_depth_write =
                                effective_depth_write,
                            .canonical_draw = HostPasses::
                                IsCanonicalPpDepthPredecessorRange(current_range),
                            .cross_open_collision = open_count != 0,
                        });
                AccumulateCrossScopeOutcome(repeat_pre);
                static_cast<void>(
                    depth_predecessor_cross_scope_p0_repeat_probe.ObservePre({
                        .lifecycle = repeat_pre,
                        .stored_probe_latest =
                            depth_predecessor_cross_scope_p0_repeat_probe
                                .Result()
                                .latest_endpoint,
                        .current_endpoint = endpoint,
                        .intent = intent,
                        .candidate = candidate,
                        .slot = slot_stamp,
                        .live_guest_rendering_serial =
                            scheduler.RenderingSerial(),
                        .live_host_rendering_instance_id =
                            scheduler.HostRenderingInstanceId(),
                        .live_rendering_active = scheduler.IsRendering(),
                        .live_regular_origin =
                            scheduler.HostRenderingOrigin() ==
                            HostRenderingInstanceOrigin::Regular,
                        .live_diagnostic_origin =
                            scheduler.HostRenderingOrigin() ==
                            HostRenderingInstanceOrigin::DiagnosticResume,
                        .configured_effective_depth_write = effective_depth_write,
                        .canonical_draw = HostPasses::
                            IsCanonicalPpDepthPredecessorRange(current_range),
                        .tail_pre_open =
                            open_depth_predecessor_cross_scope_tail_draw.valid,
                        .anchor_seen = configured_anchor,
                        .repeat_pre_open =
                            open_depth_predecessor_cross_scope_p0_repeat_draw.valid,
                        .collision = open_count != 0,
                    }));
                if (IsCrossScopeFailure(repeat_pre.decision) ||
                    !repeat_pre.target_repeat_pre_open) {
                    open_depth_predecessor_cross_scope_p0_repeat_draw = {};
                    RejectCrossScopeCapture();
                } else {
                    open_depth_predecessor_cross_scope_p0_repeat_draw = {
                        .link = link,
                        .range = current_range,
                        .draw = observed,
                        .first_writer_pre =
                            BuildDepthFirstWriterPreObservation(image,
                                                                rendering_serial),
                        .fragment_pgm_hash = fragment_pgm_hash,
                        .valid = true,
                    };
                }
                return;
            }
            case CrossDispatch::TailPreMetadata: {
                const auto intent = BuildDepthPredecessorIntent(
                    link, image, current_range, observed, fragment_pgm_hash,
                    effective_depth_write);
                const auto endpoint = BuildCurrentDepthPredecessorEndpoint(
                    image_id, image, current_range);
                auto repeat_candidate =
                    depth_predecessor_cross_scope_resume.source_candidate;
                repeat_candidate.recorded_plane_mask =
                    upstream_feedback_frame.RecordedPlaneMask(
                        repeat_candidate.candidate_index);
                const auto* repeat_tail_hint =
                    depth_predecessor_cross_scope.PeekArmedTailHint();
                const auto repeat_evidence =
                    depth_predecessor_cross_scope_p0_repeat_probe
                        .SnapshotTailPre({
                            .lifecycle =
                                depth_predecessor_cross_scope_outcome.result,
                            .current_endpoint = endpoint,
                            .tail_intent = intent,
                            .expected_p1_hint = repeat_tail_hint
                                                    ? *repeat_tail_hint
                                                    : HostPasses::
                                                          PpDepthPredecessorHint{},
                            .candidate = repeat_candidate,
                            .slot = {
                                .generation =
                                    depth_predecessor_cross_scope_outcome.result
                                        .generation,
                                .candidate_index =
                                    repeat_candidate.candidate_index,
                                .present = static_cast<bool>(
                                    upstream_feedback_slot),
                            },
                            .first_tail_pre =
                                depth_predecessor_cross_scope_outcome.result
                                        .observed_tail_occurrences == 0,
                            .anchor_seen = false,
                        });
                using RepeatEvent = HostPasses::
                    PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceEvent;
                depth_predecessor_cross_scope_outcome
                    .armed_p0_repeat_probe_evidence = HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidence(
                        depth_predecessor_cross_scope_outcome
                            .armed_p0_repeat_probe_evidence,
                        {
                            .event = repeat_evidence.attempted
                                         ? RepeatEvent::Attempt
                                         : RepeatEvent::NonAttempt,
                            .evidence = repeat_evidence,
                        });
                const auto drift_evidence =
                    depth_predecessor_cross_scope.SnapshotFirstTailPreP0Drift(
                        depth_predecessor_cross_scope_outcome
                            .target_post_endpoint,
                        endpoint, intent, scheduler.RenderingSerial(),
                        scheduler.HostRenderingInstanceId(),
                        scheduler.IsRendering(),
                        scheduler.HostRenderingOrigin() ==
                            HostRenderingInstanceOrigin::Regular,
                        scheduler.HostRenderingOrigin() ==
                            HostRenderingInstanceOrigin::DiagnosticResume);
                const auto tail = depth_predecessor_cross_scope.ObserveTailPre(
                    intent, endpoint.writer, upstream_feedback_slot);
                using DriftEvent = HostPasses::
                    PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceEvent;
                depth_predecessor_cross_scope_outcome
                    .first_tail_pre_drift_evidence = HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence(
                        depth_predecessor_cross_scope_outcome
                            .first_tail_pre_drift_evidence,
                        {
                            .event = drift_evidence.attempted
                                         ? DriftEvent::Attempt
                                         : DriftEvent::NonAttempt,
                            .evidence = drift_evidence,
                        });
                AccumulateCrossScopeOutcome(tail);
                if (IsCrossScopeFailure(tail.decision) || !tail.tail_exact) {
                    RejectCrossScopeCapture();
                } else {
                    open_depth_predecessor_cross_scope_tail_draw = {
                        .link = link,
                        .range = current_range,
                        .draw = observed,
                        .fragment_pgm_hash = fragment_pgm_hash,
                        .valid = true,
                    };
                }
                return;
            }
            case CrossDispatch::Ignore:
                return;
            default:
                RejectCrossScopeCapture();
                return;
            }
        }

        if (cross_scope_hint_mode &&
            depth_predecessor_runtime_plan.path ==
                HostPasses::PpDepthPredecessorRuntimePath::CrossScopeBootstrap &&
            depth_predecessor_cross_scope_frame_open &&
            !depth_predecessor_cross_scope_capture_closed &&
            depth_predecessor_cross_scope_bootstrap.valid &&
            depth_predecessor_cross_scope_outcome.result.role ==
                HostPasses::PpDepthPredecessorCrossScopePairRole::TargetPre &&
            MatchesPpTerminalScopeDraw(
                depth_predecessor_cross_scope_bootstrap.p0_raw_prefix.draw,
                observed) &&
            depth_predecessor_cross_scope_bootstrap.p0_raw_prefix.fragment_pgm_hash ==
                fragment_pgm_hash) {
            const auto boundary_before = HostPasses::
                SelectPpDepthPredecessorCrossScopeLearningSourceCheckpoint(
                    boundary_facts.boundary_checkpoints);
            const auto endpoint = BuildCurrentDepthPredecessorEndpoint(
                image_id, image, current_range);
            const auto intent = BuildDepthPredecessorIntent(
                link, image, current_range, observed, fragment_pgm_hash,
                effective_depth_write);
            const auto first_writer_pre =
                BuildDepthFirstWriterPreObservation(image, rendering_serial);
            const auto built =
                HostPasses::BuildPpDepthPredecessorCrossScopeLearningTargetPreBoundary({
                    .source_current = endpoint,
                    .target_intent = intent,
                    .predecessor = first_writer_pre,
                    .raw_current_link = boundary_facts.current_link,
                    .raw_current_range = boundary_facts.current_range,
                    .raw_current_backing_generation =
                        boundary_facts.current_backing_generation,
                    .raw_current_mutation_epoch =
                        boundary_facts.current_mutation_epoch,
                    .raw_boundary_active = boundary_facts.raw_boundary_active,
                    .raw_boundary_tracked = boundary_facts.raw_boundary_tracked,
                    .boundary_before_guest_rendering_serial =
                        boundary_before.guest_rendering_serial,
                    .boundary_before_host_rendering_instance_id =
                        boundary_before.host_rendering_instance_id,
                    .boundary_after_guest_rendering_serial =
                        boundary_facts.boundary_after_guest_rendering_serial,
                    .boundary_after_host_rendering_instance_id =
                        boundary_facts.boundary_after_host_rendering_instance_id,
                    .boundary_before_rendering_active =
                        boundary_before.rendering_active,
                    .boundary_before_regular_origin =
                        boundary_before.regular_origin,
                    .boundary_before_diagnostic_origin =
                        boundary_before.diagnostic_origin,
                    .boundary_after_rendering_active =
                        boundary_facts.boundary_after_rendering_active,
                    .boundary_after_regular_origin =
                        !boundary_facts.boundary_after_diagnostic_origin,
                    .boundary_after_diagnostic_origin =
                        boundary_facts.boundary_after_diagnostic_origin,
                    .live_target_guest_rendering_serial =
                        boundary_facts.live_target_guest_rendering_serial,
                    .live_target_host_rendering_instance_id =
                        boundary_facts.live_target_host_rendering_instance_id,
                    .live_target_rendering_active =
                        boundary_facts.live_target_rendering_active,
                    .live_target_regular_origin =
                        !boundary_facts.live_target_diagnostic_origin,
                    .live_target_diagnostic_origin =
                        boundary_facts.live_target_diagnostic_origin,
                    .source_outside_target_guest_scope =
                        endpoint.guest_rendering_serial != rendering_serial,
                    .after_begin_rendering =
                        boundary_facts.boundary_after_rendering_active,
                    .target_depth_load =
                        state.depth_stencil_attachment.has_depth &&
                        !state.depth_stencil_attachment.depth_clear,
                    .target_depth_clear =
                        state.depth_stencil_attachment.depth_clear,
                    .target_stencil_clear =
                        state.depth_stencil_attachment.stencil_clear,
                    .depth_aspect = state.depth_stencil_attachment.has_depth &&
                                    static_cast<bool>(
                                        image.aspect_mask &
                                        vk::ImageAspectFlagBits::eDepth),
                    .canonical_attachment =
                        HostPasses::IsCanonicalPpDepthPredecessorRange(current_range),
                });
            depth_predecessor_cross_scope_outcome.learning =
                HostPasses::SnapshotPpDepthPredecessorCrossScopeLearningEvidence(
                    built);
            depth_predecessor_cross_scope_outcome.learning_logical_candidates =
                HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeLearningLogicalCandidate(
                        depth_predecessor_cross_scope_outcome
                            .learning_logical_candidates,
                        depth_predecessor_cross_scope_outcome.learning);
            depth_predecessor_cross_scope_outcome.learning_transition_candidates =
                HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeLearningTransitionCandidate(
                        depth_predecessor_cross_scope_outcome
                            .learning_transition_candidates,
                        depth_predecessor_cross_scope_outcome.learning);
            const auto learning_attempt = HostPasses::
                PlanPpDepthPredecessorCrossScopeLearningTargetPreAttempt(built);
            if (learning_attempt.reject_shared_feedback) {
                RejectCrossScopeCapture();
            } else if (learning_attempt.observe_cross_scope) {
                const auto target_pre =
                    depth_predecessor_cross_scope.ObserveLearningTargetPre(
                        built.observation);
                AccumulateCrossScopeOutcome(target_pre);
                if (IsCrossScopeFailure(target_pre.decision) ||
                    !target_pre.boundary_exact) {
                    RejectCrossScopeCapture();
                } else {
                    open_depth_predecessor_cross_scope_target_draw = {
                        .link = link,
                        .range = current_range,
                        .draw = observed,
                        .intent = intent,
                        .first_writer_pre = first_writer_pre,
                        .fragment_pgm_hash = fragment_pgm_hash,
                        .valid = true,
                    };
                }
            }
            if (!learning_attempt.continue_nested_multi_hop) {
                return;
            }
        }

        if (two_hop_hint_mode && configured_anchor) {
            if (depth_predecessor_two_hop_outcome.anchor_count !=
                std::numeric_limits<u32>::max()) {
                ++depth_predecessor_two_hop_outcome.anchor_count;
            }
            const bool collision = open_depth_predecessor_hint_draw.valid ||
                                   open_depth_predecessor_two_hop_p1_draw.valid;
            const auto built = BuildCurrentDepthPredecessorObservation(
                image_id, image, current_range, depth_candidate_generation);
            const HostPasses::PpDepthPredecessorHint anchor_hint{
                .draw = observed,
                .fragment_pgm_hash = fragment_pgm_hash,
                .valid = true,
            };
            const auto anchor = depth_predecessor_two_hop.ObserveAnchor(
                anchor_hint,
                effective_depth_write ? built.writer
                                      : HostPasses::PpDepthPredecessorWriterStamp{});
            AccumulateTwoHopOutcome(anchor);
            const bool cross_scope_bootstrap_selection =
                cross_scope_hint_mode &&
                depth_predecessor_runtime_plan.path ==
                    HostPasses::PpDepthPredecessorRuntimePath::CrossScopeBootstrap &&
                depth_predecessor_cross_scope_frame_open;
            const auto selected_latest =
                cross_scope_bootstrap_selection
                    ? depth_predecessor_two_hop
                          .SnapshotCrossScopeSelectedLatestPrivateWitness(anchor)
                    : HostPasses::
                          PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness{};
            depth_predecessor_hint_capture_closed = true;
            open_depth_predecessor_hint_draw = {};
            open_depth_predecessor_two_hop_p1_draw = {};

            bool selected{};
            const bool selection_attempted =
                !collision && anchor.select && upstream_feedback_armed;
            if (selection_attempted) {
                selected = upstream_feedback_frame.SelectCandidate(anchor.candidate_index);
            }
            if (cross_scope_bootstrap_selection && selection_attempted) {
                const u32 live_recorded_plane_mask =
                    upstream_feedback_frame.RecordedPlaneMask(anchor.candidate_index);
                depth_predecessor_cross_scope_outcome.selected_latest_private =
                    selected_latest;
                depth_predecessor_cross_scope_outcome.mode3_runtime_selection =
                    HostPasses::
                        AccumulatePpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot(
                            depth_predecessor_cross_scope_outcome
                                .mode3_runtime_selection,
                            {
                                .event = HostPasses::
                                    PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshotEvent::
                                        Selection,
                                .selected_candidate =
                                    selected_latest.latest_candidate,
                                .frame_generation = anchor.generation,
                                .expected_generation = depth_candidate_generation,
                                .frame_candidate_index = anchor.candidate_index,
                                .live_recorded_plane_mask =
                                    live_recorded_plane_mask,
                                .restart_count =
                                    selected_latest.restart_completion_count,
                                .chain_kind = anchor.chain_kind,
                                .select_attempted = true,
                                .select_succeeded = selected,
                                .slot_present =
                                    static_cast<bool>(upstream_feedback_slot),
                                .frame_complete =
                                    selected && live_recorded_plane_mask == 0x3u,
                                .frame_loss_free = selected,
                                .final_binding_point = true,
                            });
            }
            depth_predecessor_two_hop_outcome.actual_selected |= selected;
            const bool failed = IsTwoHopFailure(anchor.decision);
            if (collision || failed ||
                (upstream_feedback_armed && (!anchor.select || !selected))) {
                RejectTwoHopCapture(collision);
            }
            return;
        }

        if (one_hop_hint_mode && configured_anchor) {
            if (depth_predecessor_hint_outcome.anchor_count !=
                std::numeric_limits<u32>::max()) {
                ++depth_predecessor_hint_outcome.anchor_count;
            }
            const bool collision = open_depth_predecessor_hint_draw.valid;
            const auto latest = image.PeekDiagnosticDepthStencilProducer();
            const u64 current_mutation_epoch = image.DiagnosticMutationEpoch();
            const u64 current_backing_generation =
                image.backing ? image.EnsureDiagnosticBackingGeneration() : 0;
            const u32 recorded_plane_mask = upstream_feedback_frame.RecordedPlaneMask(
                latest.upstream_candidate_index);
            const auto built = HostPasses::BuildPpDepthPredecessorObservation({
                .observed =
                    {
                        .kind = latest.kind,
                        .link = latest.link,
                        .range = latest.range,
                        .mutation_epoch = latest.mutation_epoch,
                        .backing_generation = latest.backing_generation,
                        .fragment_pgm_hash = latest.fragment_pgm_hash,
                        .upstream_candidate_generation =
                            latest.upstream_candidate_generation,
                        .element_count = latest.element_count,
                        .instance_count = latest.instance_count,
                        .sampled_images = latest.sampled_images,
                        .storage_writes = latest.storage_writes,
                        .upstream_candidate_index = latest.upstream_candidate_index,
                        .depth_aspect =
                            (latest.attachment_aspect_mask &
                             static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT)) != 0,
                        .indexed = latest.indexed,
                        .depth_write = latest.depth_write,
                        .upstream_candidate_valid = latest.upstream_candidate_valid,
                        .valid = latest.valid,
                    },
                .current_link = link,
                .current_range = current_range,
                .current_backing_generation = current_backing_generation,
                .current_mutation_epoch = current_mutation_epoch,
                .expected_candidate_generation = depth_candidate_generation,
                .recorded_plane_mask = recorded_plane_mask,
            });
            const bool exact_latest = built.hint.Valid();
            const bool hint_match =
                exact_latest && depth_predecessor_hint.MatchesArmed(
                                    built.hint.draw, built.hint.fragment_pgm_hash);
            const bool current_fresh_tag =
                exact_latest && latest.upstream_candidate_valid &&
                depth_candidate_generation != 0 &&
                latest.upstream_candidate_generation == depth_candidate_generation &&
                latest.upstream_candidate_index <
                    HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates;
            const auto anchor = depth_predecessor_hint.ObserveAnchor(
                built.hint.draw, built.hint.fragment_pgm_hash,
                effective_depth_write ? built.candidate
                                      : HostPasses::PpDepthPredecessorCandidateStamp{});
            depth_predecessor_hint_outcome.exact_latest |= exact_latest;
            depth_predecessor_hint_outcome.hint_match |= hint_match;
            depth_predecessor_hint_outcome.current_fresh_tag |= current_fresh_tag;
            depth_predecessor_hint_outcome.exact_candidate_join |=
                effective_depth_write && built.candidate.valid;
            depth_predecessor_hint_outcome.next_updated =
                built.hint.Valid() &&
                anchor.decision != HostPasses::PpDepthPredecessorHintDecision::DuplicateAnchor;
            depth_predecessor_hint_outcome.decision = anchor.decision;
            depth_predecessor_hint_capture_closed = true;
            open_depth_predecessor_hint_draw = {};

            bool selected{};
            if (!collision && anchor.select && upstream_feedback_armed) {
                selected = upstream_feedback_frame.SelectCandidate(anchor.candidate_index);
            }
            depth_predecessor_hint_outcome.actual_selected |= selected;
            const bool duplicate =
                anchor.decision == HostPasses::PpDepthPredecessorHintDecision::DuplicateAnchor;
            if (collision || duplicate || (upstream_feedback_armed && !selected)) {
                depth_predecessor_hint_outcome.duplicate_collision |= collision || duplicate;
                if (upstream_feedback_armed) {
                    upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
                }
            }
            return;
        }

        if (two_hop_hint_mode &&
            depth_predecessor_two_hop.MatchesP1(observed, fragment_pgm_hash)) {
            if (depth_predecessor_two_hop_outcome.p1_pre_count !=
                std::numeric_limits<u32>::max()) {
                ++depth_predecessor_two_hop_outcome.p1_pre_count;
            }
            const bool collision = open_depth_predecessor_hint_draw.valid ||
                                   open_depth_predecessor_two_hop_p1_draw.valid;
            const u64 current_mutation_epoch = image.DiagnosticMutationEpoch();
            const u64 current_backing_generation =
                image.backing ? image.EnsureDiagnosticBackingGeneration() : 0;
            const HostPasses::PpDepthPredecessorWriterIntent intent{
                .hint = {.draw = observed,
                         .fragment_pgm_hash = fragment_pgm_hash,
                         .valid = true},
                .link = link,
                .range = current_range,
                .backing_generation = current_backing_generation,
                .mutation_epoch = current_mutation_epoch,
                .valid = effective_depth_write &&
                         HostPasses::IsCanonicalPpDepthPredecessorRange(current_range),
            };
            const auto built = BuildCurrentDepthPredecessorObservation(
                image_id, image, current_range, depth_candidate_generation);
            const auto p1_pre = depth_predecessor_two_hop.ObserveP1Pre(intent, built);
            AccumulateTwoHopOutcome(p1_pre);
            depth_predecessor_two_hop_outcome.current_fresh_tag |= built.candidate.valid;
            depth_predecessor_hint_capture_closed = true;
            open_depth_predecessor_hint_draw = {};
            const bool accepted =
                !collision && !IsTwoHopFailure(p1_pre.decision) && p1_pre.p1_intent &&
                p1_pre.p2_exact && (p1_pre.metadata_only || p1_pre.candidate_saved);
            if (accepted) {
                open_depth_predecessor_two_hop_p1_draw = {
                    .link = link,
                    .range = current_range,
                    .draw = observed,
                    .fragment_pgm_hash = fragment_pgm_hash,
                    .valid = true,
                };
            } else {
                RejectTwoHopCapture(collision);
            }
            return;
        }

        if ((!hint_mode &&
             (!configured_anchor ||
              !HostPasses::IsCanonicalPpDepthPredecessorRange(current_range))) ||
            (hint_mode &&
             (depth_predecessor_hint_capture_closed || !upstream_feedback_armed ||
              !effective_depth_write ||
               !HostPasses::IsCanonicalPpDepthPredecessorRange(current_range) ||
               !(one_hop_hint_mode
                     ? depth_predecessor_hint.MatchesArmed(observed, fragment_pgm_hash)
                    : depth_predecessor_two_hop.MatchesRawP2(observed,
                                                             fragment_pgm_hash) ||
                          raw_prefix_restart)))) {
            return;
        }
        if (hint_mode &&
            (open_depth_predecessor_hint_draw.valid ||
             (two_hop_hint_mode && open_depth_predecessor_two_hop_p1_draw.valid))) {
            if (two_hop_hint_mode) {
                RejectTwoHopCapture(true);
            } else {
                depth_predecessor_hint_outcome.duplicate_collision = true;
                depth_predecessor_hint_capture_closed = true;
                open_depth_predecessor_hint_draw = {};
                upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
            }
            return;
        }
        if (raw_prefix_restart) {
            const auto restart = depth_predecessor_two_hop.BeginRawPrefixRestart();
            AccumulateTwoHopOutcome(restart);
            if (IsTwoHopFailure(restart.decision) ||
                restart.occurrence_role !=
                    HostPasses::PpDepthPredecessorTwoHopOccurrenceRole::RawPrefix) {
                RejectTwoHopCapture();
                return;
            }
        }
        const auto producer = image.PeekDiagnosticProducer();
        const auto transfer = image.PeekDiagnosticTransfer();
        const auto depth_stencil = image.PeekDiagnosticDepthStencilProducer();
        const u64 mutation_epoch = image.DiagnosticMutationEpoch();
        const u64 backing_generation =
            image.backing ? image.EnsureDiagnosticBackingGeneration() : 0;
        const bool transfer_exact =
            producer.classification == VideoCore::ImageProducerClass::Transfer && transfer.valid &&
            transfer.mutation_epoch == mutation_epoch &&
            transfer.backing_generation == backing_generation;
        const bool depth_stencil_exact =
            depth_stencil.valid && depth_stencil.mutation_epoch == mutation_epoch &&
            depth_stencil.backing_generation == backing_generation;
        const PendingUpstreamFeedback::DepthPredecessorObservation predecessor{
            .producer = producer.classification,
            .transfer_kind = transfer_exact ? transfer.kind : VideoCore::ImageTransferKind::None,
            .transfer_origin =
                transfer_exact ? transfer.origin : VideoCore::ImageTransferOrigin::Unknown,
            .depth_stencil_kind = depth_stencil_exact
                                      ? depth_stencil.kind
                                      : VideoCore::ImageColorScopeDrawKind::Unknown,
            .element_count = depth_stencil_exact ? depth_stencil.element_count : 0,
            .instance_count = depth_stencil_exact ? depth_stencil.instance_count : 0,
            .sampled_images = depth_stencil_exact ? depth_stencil.sampled_images : 0,
            .storage_writes = depth_stencil_exact ? depth_stencil.storage_writes : 0,
            .recorded = true,
            .transfer_exact = transfer_exact,
            .depth_stencil_exact = depth_stencil_exact,
            .depth_stencil_same_scope =
                depth_stencil_exact && depth_stencil.rendering_serial == rendering_serial,
            .scope_first_depth_kind =
                depth_stencil_exact ? depth_stencil.scope_first_depth_kind
                                    : VideoCore::ImageColorScopeDrawKind::Unknown,
            .scope_first_depth_fragment_pgm_hash =
                depth_stencil_exact ? depth_stencil.scope_first_depth_fragment_pgm_hash : 0,
            .scope_first_depth_element_count =
                depth_stencil_exact ? depth_stencil.scope_first_depth_element_count : 0,
            .scope_first_depth_instance_count =
                depth_stencil_exact ? depth_stencil.scope_first_depth_instance_count : 0,
            .scope_first_depth_sampled_images =
                depth_stencil_exact ? depth_stencil.scope_first_depth_sampled_images : 0,
            .scope_first_depth_storage_writes =
                depth_stencil_exact ? depth_stencil.scope_first_depth_storage_writes : 0,
            .scope_depth_writer_count =
                depth_stencil_exact ? depth_stencil.scope_depth_writer_count : 0,
            .indexed = depth_stencil_exact && depth_stencil.indexed,
            .scope_first_depth_indexed =
                depth_stencil_exact && depth_stencil.scope_first_depth_indexed,
            .scope_first_depth_valid =
                depth_stencil_exact && depth_stencil.scope_first_depth_valid,
            .scope_depth_writer_count_overflow =
                depth_stencil_exact && depth_stencil.scope_depth_writer_count_overflow,
            .scope_depth_history_valid =
                depth_stencil_exact && depth_stencil.scope_depth_history_valid,
            .scope_depth_clear = depth_stencil_exact && depth_stencil.scope_depth_clear,
            .scope_stencil_clear = depth_stencil_exact && depth_stencil.scope_stencil_clear,
            .fragment_writes_depth = depth_stencil_exact && depth_stencil.fragment_writes_depth,
            .fragment_has_discard = depth_stencil_exact && depth_stencil.fragment_has_discard,
            .depth_write = depth_stencil_exact && depth_stencil.depth_write,
            .stencil_write = depth_stencil_exact && depth_stencil.stencil_write,
            .feedback_alias = depth_stencil_exact && depth_stencil.sampled_feedback_alias,
            .depth_clear = state.depth_stencil_attachment.depth_clear,
            .stencil_clear = state.depth_stencil_attachment.stencil_clear,
            .depth_boundary_valid = depth_write_boundary.valid,
            .depth_boundary_join_exact = depth_write_boundary.join_exact,
            .depth_boundary_writer_exact = depth_write_boundary.writer_exact,
            .depth_boundary_pre_rendering = depth_write_boundary.pre_rendering,
            .depth_boundary_post_transit_rendering =
                depth_write_boundary.post_transit_rendering,
            .depth_boundary_pre_external_rendering =
                depth_write_boundary.pre_external_begin_rendering,
            .depth_boundary_after_rendering = depth_write_boundary.after_rendering,
            .depth_boundary_split_interval = depth_write_boundary.split_interval_observed,
            .depth_boundary_target_load = depth_write_boundary.target_depth_load,
            .depth_boundary_state_exact = depth_write_boundary.state_exact,
            .depth_boundary_skipped_write =
                depth_write_boundary.skipped_same_state_depth_stencil_write,
            .depth_boundary_same_host = depth_write_boundary.same_host_rendering_instance,
            .depth_boundary_cross_host = depth_write_boundary.cross_host_rendering_instance,
            .depth_boundary_instrumented_cross_host =
                depth_write_boundary.instrumented_cross_host_rendering_instance,
            .depth_boundary_before_diagnostic = depth_write_boundary.before_diagnostic_resume,
            .depth_boundary_diagnostic_excluded =
                depth_write_boundary.diagnostic_resume_excluded,
            .depth_boundary_barrier_recognized_write =
                depth_write_boundary.barrier_recognized_write,
        };
        u32 copied_candidate = HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates;
        ObserveUpstreamFeedbackPreDraw(link, image, state, 0, rendering_serial, observed, {}, {},
                                       {}, {}, {}, {}, {}, fragment_info, {}, {},
                                       vk::ImageAspectFlagBits::eDepth, &copied_candidate,
                                       hint_mode, raw_prefix_restart);
        if (copied_candidate < depth_candidate_predecessors.size()) {
            depth_candidate_predecessors[copied_candidate] = predecessor;
        }
        if (hint_mode) {
            if (copied_candidate < depth_candidate_predecessors.size() &&
                !depth_predecessor_hint_capture_closed) {
                open_depth_predecessor_hint_draw = {
                    .link = link,
                    .range = current_range,
                    .draw = observed,
                    .fragment_pgm_hash = fragment_pgm_hash,
                    .candidate_index = copied_candidate,
                    .valid = true,
                };
            } else if (!depth_predecessor_hint_capture_closed) {
                if (two_hop_hint_mode) {
                    RejectTwoHopCapture();
                } else {
                    depth_predecessor_hint_outcome.duplicate_collision = true;
                    depth_predecessor_hint_capture_closed = true;
                    upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
                }
            }
            return;
        }
        if (copied_candidate < depth_candidate_inputs.size() &&
            sampled_inputs.status == FinalGuestSurfaceStatus::Complete &&
            !sampled_inputs.loss.Any() &&
            sampled_inputs.count == config.content.upstream.sampled_images) {
            depth_candidate_inputs[copied_candidate] = sampled_inputs;
            depth_candidate_inputs_valid[copied_candidate] = true;
        }
        static_cast<void>(upstream_registry.Observe(link, observed, sampled_inputs));
    }

    [[nodiscard]] bool ObserveEncodedUpstreamPostDraw(
        VideoCore::ImageId image_id, VideoCore::Image& image, const RenderState& state,
        u32 attachment_index, u64 rendering_serial,
        const VideoCore::ImageColorScopeDrawDescriptor& draw) {
        const PpTerminalScopeDrawSelector observed{
            .kind = draw.kind,
            .indexed = draw.indexed,
            .element_count = draw.element_count,
            .instance_count = draw.instance_count,
            .sampled_images = draw.sampled_images,
            .storage_writes = draw.storage_writes,
        };
        const VideoCore::ImageColorScopePrivateLink link{.id = image_id, .uid = image.image_uid};
        if (config.content.capture_upstream_depth_pre_post ||
            !MatchesPpTerminalScopeUpstreamTarget(config.content, observed,
                                                  draw.fragment_pgm_hash)) {
            return true;
        }
        return ObserveUpstreamFeedbackPostDraw(link, image, state, attachment_index,
                                               rendering_serial, observed,
                                               draw.fragment_pgm_hash);
    }

    [[nodiscard]] bool ObserveEncodedDepthUpstreamPostDraw(
        VideoCore::ImageId image_id, VideoCore::Image& image, const RenderState& state,
        u64 rendering_serial, const VideoCore::ImageColorScopeDrawDescriptor& draw,
        const VideoCore::SubresourceRange& current_range, bool effective_depth_write) {
        if (!config.content.capture_upstream_depth_pre_post) {
            return true;
        }
        const PpTerminalScopeDrawSelector observed{
            .kind = draw.kind,
            .indexed = draw.indexed,
            .element_count = draw.element_count,
            .instance_count = draw.instance_count,
            .sampled_images = draw.sampled_images,
            .storage_writes = draw.storage_writes,
        };
        const VideoCore::ImageColorScopePrivateLink link{.id = image_id, .uid = image.image_uid};
        const bool hint_mode = config.content.capture_upstream_depth_predecessor_hint;
        const bool cross_scope_hint_mode = IsCrossScopeDepthPredecessorHintMode();
        const bool two_hop_hint_mode =
            IsDepthPredecessorStructuralMultiHopRuntime();
        const bool one_hop_hint_mode = IsOneHopDepthPredecessorHintMode();

        if (cross_scope_hint_mode &&
            depth_predecessor_runtime_plan.path ==
                HostPasses::PpDepthPredecessorRuntimePath::CrossScopeArmed) {
            if (!depth_predecessor_cross_scope_frame_open ||
                depth_predecessor_cross_scope_capture_closed) {
                return true;
            }
            const auto source_exact = [&] {
                const auto& open =
                    open_depth_predecessor_cross_scope_source_draw;
                return open.valid && open.link == link && open.range == current_range &&
                       MatchesPpTerminalScopeDraw(open.draw, observed) &&
                       open.fragment_pgm_hash == draw.fragment_pgm_hash;
            }();
            const auto target_exact = [&] {
                const auto& open =
                    open_depth_predecessor_cross_scope_target_draw;
                return open.valid && open.link == link && open.range == current_range &&
                       MatchesPpTerminalScopeDraw(open.draw, observed) &&
                       open.fragment_pgm_hash == draw.fragment_pgm_hash;
            }();
            const auto tail_exact = [&] {
                const auto& open = open_depth_predecessor_cross_scope_tail_draw;
                return open.valid && open.link == link && open.range == current_range &&
                       MatchesPpTerminalScopeDraw(open.draw, observed) &&
                       open.fragment_pgm_hash == draw.fragment_pgm_hash;
            }();
            const auto target_repeat_exact = [&] {
                const auto& open =
                    open_depth_predecessor_cross_scope_p0_repeat_draw;
                return open.valid && open.link == link &&
                       open.range == current_range &&
                       MatchesPpTerminalScopeDraw(open.draw, observed) &&
                       open.fragment_pgm_hash == draw.fragment_pgm_hash;
            }();
            const u32 open_count =
                static_cast<u32>(open_depth_predecessor_cross_scope_source_draw.valid) +
                static_cast<u32>(open_depth_predecessor_cross_scope_target_draw.valid) +
                static_cast<u32>(
                    open_depth_predecessor_cross_scope_p0_repeat_draw.valid) +
                static_cast<u32>(open_depth_predecessor_cross_scope_tail_draw.valid);
            const auto dispatch =
                HostPasses::ClassifyPpDepthPredecessorCrossScopeRuntimeDispatch({
                    .phase = depth_predecessor_cross_scope_outcome.result.phase,
                    .role = depth_predecessor_cross_scope_outcome.result.role,
                    .pre_draw = false,
                    .configured_anchor = false,
                    .matches_source_post = source_exact,
                    .matches_target_pre = target_exact,
                    .matches_target_repeat = target_repeat_exact,
                    .matches_tail = tail_exact,
                    .source_reservation_open =
                        open_depth_predecessor_cross_scope_source_draw.valid,
                    .target_post_open =
                        open_depth_predecessor_cross_scope_target_draw.valid,
                    .target_repeat_post_open =
                        open_depth_predecessor_cross_scope_p0_repeat_draw.valid,
                    .tail_post_open =
                        open_depth_predecessor_cross_scope_tail_draw.valid,
                    .cross_open_collision = open_count > 1,
                    .source_restart_open =
                        open_depth_predecessor_cross_scope_source_draw.restart,
                });
            using CrossDispatch =
                HostPasses::PpDepthPredecessorCrossScopeRuntimeDispatch;
            switch (dispatch) {
            case CrossDispatch::SourcePostPlane0:
            case CrossDispatch::SourcePostRestartPlane0:
                if (!RecordDepthPredecessorCrossScopeSourcePost(
                        image_id, image, state, rendering_serial, current_range,
                        observed, draw.fragment_pgm_hash, effective_depth_write)) {
                    RejectCrossScopeCapture();
                }
                return true;
            case CrossDispatch::TargetRepeatPostMetadata: {
                const auto open =
                    open_depth_predecessor_cross_scope_p0_repeat_draw;
                const auto endpoint = BuildCurrentDepthPredecessorEndpoint(
                    image_id, image, current_range);
                const auto classification = ClassifyCurrentDepthFirstWriter(
                    open.first_writer_pre, image, draw, rendering_serial);
                auto candidate =
                    depth_predecessor_cross_scope_resume.source_candidate;
                candidate.recorded_plane_mask =
                    upstream_feedback_frame.RecordedPlaneMask(
                        candidate.candidate_index);
                const HostPasses::
                    PpDepthPredecessorCrossScopeArmedP0RepeatProbeSlotStamp
                        slot_stamp{
                            .generation =
                                depth_predecessor_cross_scope_outcome.result
                                    .generation,
                            .candidate_index = candidate.candidate_index,
                            .present = static_cast<bool>(upstream_feedback_slot),
                        };
                const u32 completion_before =
                    depth_predecessor_cross_scope_outcome.result
                        .target_repeat_completion_count;
                const auto repeat_post =
                    depth_predecessor_cross_scope
                        .ObserveTargetRepeatPostMetadata({
                            .post_endpoint = endpoint,
                            .candidate = candidate,
                            .slot = slot_stamp,
                            .retained_slot_token = upstream_feedback_slot,
                            .configured_effective_depth_write =
                                effective_depth_write,
                            .regular_origin =
                                scheduler.HostRenderingOrigin() ==
                                HostRenderingInstanceOrigin::Regular,
                            .diagnostic_origin =
                                scheduler.HostRenderingOrigin() ==
                                HostRenderingInstanceOrigin::DiagnosticResume,
                            .first_writer = classification,
                        });
                AccumulateCrossScopeOutcome(repeat_post);
                static_cast<void>(
                    depth_predecessor_cross_scope_p0_repeat_probe.ObservePost({
                        .lifecycle = repeat_post,
                        .post_endpoint = endpoint,
                        .candidate = candidate,
                        .slot = slot_stamp,
                        .configured_effective_depth_write = effective_depth_write,
                        .regular_origin =
                            scheduler.HostRenderingOrigin() ==
                            HostRenderingInstanceOrigin::Regular,
                        .diagnostic_origin =
                            scheduler.HostRenderingOrigin() ==
                            HostRenderingInstanceOrigin::DiagnosticResume,
                        .first_writer = classification,
                    }));
                open_depth_predecessor_cross_scope_p0_repeat_draw = {};
                if (!effective_depth_write ||
                    IsCrossScopeFailure(repeat_post.decision) ||
                    repeat_post.target_repeat_pre_open ||
                    repeat_post.target_repeat_completion_count !=
                        completion_before + 1) {
                    RejectCrossScopeCapture();
                } else {
                    depth_predecessor_cross_scope_outcome
                        .target_post_endpoint = endpoint;
                }
                return true;
            }
            case CrossDispatch::TargetPostFirstWriter: {
                const auto open = open_depth_predecessor_cross_scope_target_draw;
                const auto endpoint = BuildCurrentDepthPredecessorEndpoint(
                    image_id, image, current_range);
                const auto classification = ClassifyCurrentDepthFirstWriter(
                    open.first_writer_pre, image, draw, rendering_serial);
                const auto target_post =
                    depth_predecessor_cross_scope.ObserveTargetPost(
                        endpoint, classification, upstream_feedback_slot);
                AccumulateCrossScopeOutcome(target_post);
                open_depth_predecessor_cross_scope_target_draw = {};
                if (!effective_depth_write ||
                    IsCrossScopeFailure(target_post.decision) ||
                    !target_post.target_first_writer_exact) {
                    RejectCrossScopeCapture();
                } else {
                    depth_predecessor_cross_scope_outcome.target_post_endpoint =
                        endpoint;
                    auto candidate =
                        depth_predecessor_cross_scope_resume.source_candidate;
                    candidate.recorded_plane_mask =
                        upstream_feedback_frame.RecordedPlaneMask(
                            candidate.candidate_index);
                    static_cast<void>(
                        depth_predecessor_cross_scope_p0_repeat_probe
                            .ObserveTargetPost({
                                .lifecycle = target_post,
                                .target_post = endpoint,
                                .candidate = candidate,
                                .slot = {
                                    .generation = target_post.generation,
                                    .candidate_index = candidate.candidate_index,
                                    .present = static_cast<bool>(
                                        upstream_feedback_slot),
                                },
                                .tail_pre_open =
                                    open_depth_predecessor_cross_scope_tail_draw
                                        .valid,
                                .anchor_seen = false,
                                .repeat_pre_open =
                                    open_depth_predecessor_cross_scope_p0_repeat_draw
                                        .valid,
                                .collision = false,
                            }));
                }
                return true;
            }
            case CrossDispatch::TailPostMetadata: {
                const auto current = BuildCurrentDepthPredecessorObservation(
                    image_id, image, current_range, depth_candidate_generation);
                const auto tail_post =
                    depth_predecessor_cross_scope.ObserveTailPost(current.writer);
                AccumulateCrossScopeOutcome(tail_post);
                open_depth_predecessor_cross_scope_tail_draw = {};
                if (!effective_depth_write ||
                    IsCrossScopeFailure(tail_post.decision) ||
                    !tail_post.tail_exact) {
                    RejectCrossScopeCapture();
                }
                return true;
            }
            case CrossDispatch::Ignore:
                return true;
            default:
                RejectCrossScopeCapture();
                return true;
            }
        }

        if (cross_scope_hint_mode &&
            depth_predecessor_runtime_plan.path ==
                HostPasses::PpDepthPredecessorRuntimePath::CrossScopeBootstrap &&
            depth_predecessor_cross_scope_frame_open &&
            !depth_predecessor_cross_scope_capture_closed &&
            open_depth_predecessor_cross_scope_target_draw.valid) {
            const auto open = open_depth_predecessor_cross_scope_target_draw;
            const bool exact = effective_depth_write && open.link == link &&
                               open.range == current_range &&
                               MatchesPpTerminalScopeDraw(open.draw, observed) &&
                               open.fragment_pgm_hash == draw.fragment_pgm_hash;
            const auto current = BuildCurrentDepthPredecessorObservation(
                image_id, image, current_range, depth_candidate_generation);
            const auto classification = ClassifyCurrentDepthFirstWriter(
                open.first_writer_pre, image, draw, rendering_serial);
            const auto target_post =
                depth_predecessor_cross_scope.ObserveTargetPost(current.writer,
                                                                classification);
            AccumulateCrossScopeOutcome(target_post);
            open_depth_predecessor_cross_scope_target_draw = {};
            if (!exact || IsCrossScopeFailure(target_post.decision) ||
                !target_post.target_first_writer_exact) {
                RejectCrossScopeCapture();
            }
        }
        HostPasses::PpDepthPredecessorTwoHopPostDispatch two_hop_dispatch =
            HostPasses::PpDepthPredecessorTwoHopPostDispatch::Ignore;
        if (two_hop_hint_mode) {
            const bool matches_raw =
                depth_predecessor_two_hop.MatchesRawP2(observed, draw.fragment_pgm_hash) ||
                depth_predecessor_two_hop.MatchesRawPrefixRestart(
                    observed, draw.fragment_pgm_hash);
            const bool matches_bridge =
                depth_predecessor_two_hop.MatchesP1(observed, draw.fragment_pgm_hash);
            const bool raw_open_exact =
                !depth_predecessor_hint_capture_closed &&
                open_depth_predecessor_hint_draw.valid && effective_depth_write &&
                open_depth_predecessor_hint_draw.link == link &&
                open_depth_predecessor_hint_draw.range == current_range &&
                MatchesPpTerminalScopeDraw(open_depth_predecessor_hint_draw.draw, observed) &&
                open_depth_predecessor_hint_draw.fragment_pgm_hash == draw.fragment_pgm_hash;
            const bool bridge_open_exact =
                depth_predecessor_hint_capture_closed &&
                open_depth_predecessor_two_hop_p1_draw.valid && effective_depth_write &&
                open_depth_predecessor_two_hop_p1_draw.link == link &&
                open_depth_predecessor_two_hop_p1_draw.range == current_range &&
                MatchesPpTerminalScopeDraw(open_depth_predecessor_two_hop_p1_draw.draw,
                                           observed) &&
                open_depth_predecessor_two_hop_p1_draw.fragment_pgm_hash ==
                    draw.fragment_pgm_hash;
            two_hop_dispatch = HostPasses::ClassifyPpDepthPredecessorTwoHopPostDispatch({
                .occurrence_role = depth_predecessor_two_hop_outcome.occurrence_role,
                .matches_raw = matches_raw,
                .matches_bridge = matches_bridge,
                .raw_open = open_depth_predecessor_hint_draw.valid,
                .raw_open_exact = raw_open_exact,
                .bridge_open = open_depth_predecessor_two_hop_p1_draw.valid,
                .bridge_open_exact = bridge_open_exact,
            });
            if (two_hop_dispatch ==
                HostPasses::PpDepthPredecessorTwoHopPostDispatch::Reject) {
                RejectTwoHopCapture(open_depth_predecessor_hint_draw.valid ||
                                    open_depth_predecessor_two_hop_p1_draw.valid);
                return true;
            }
            if (two_hop_dispatch ==
                HostPasses::PpDepthPredecessorTwoHopPostDispatch::Ignore) {
                return true;
            }
        }
        if (two_hop_dispatch == HostPasses::PpDepthPredecessorTwoHopPostDispatch::P1Bridge ||
            two_hop_dispatch == HostPasses::PpDepthPredecessorTwoHopPostDispatch::P1Tail) {
            if (depth_predecessor_two_hop_outcome.p1_post_count !=
                std::numeric_limits<u32>::max()) {
                ++depth_predecessor_two_hop_outcome.p1_post_count;
            }
            const auto built = BuildCurrentDepthPredecessorObservation(
                image_id, image, current_range, depth_candidate_generation);
            const auto p1_post = depth_predecessor_two_hop.ObserveP1Post(built.writer);
            AccumulateTwoHopOutcome(p1_post);
            open_depth_predecessor_two_hop_p1_draw = {};
            if (IsTwoHopFailure(p1_post.decision) || !p1_post.p1_post_exact) {
                RejectTwoHopCapture();
            } else if (two_hop_dispatch ==
                       HostPasses::PpDepthPredecessorTwoHopPostDispatch::P1Bridge) {
                depth_predecessor_hint_capture_closed = false;
            } else {
                depth_predecessor_hint_capture_closed = true;
            }
            return true;
        }
        u32 expected_hint_candidate =
            HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates;
        if (hint_mode) {
            if (depth_predecessor_hint_capture_closed ||
                !open_depth_predecessor_hint_draw.valid) {
                if (two_hop_hint_mode) {
                    RejectTwoHopCapture();
                }
                return true;
            }
            const bool exact_open =
                effective_depth_write && open_depth_predecessor_hint_draw.link == link &&
                open_depth_predecessor_hint_draw.range == current_range &&
                MatchesPpTerminalScopeDraw(open_depth_predecessor_hint_draw.draw, observed) &&
                open_depth_predecessor_hint_draw.fragment_pgm_hash == draw.fragment_pgm_hash &&
                (one_hop_hint_mode
                     ? depth_predecessor_hint.MatchesArmed(observed, draw.fragment_pgm_hash)
                     : two_hop_dispatch ==
                               HostPasses::PpDepthPredecessorTwoHopPostDispatch::RawP2 ||
                           two_hop_dispatch ==
                               HostPasses::PpDepthPredecessorTwoHopPostDispatch::RawPrefix);
            if (!exact_open) {
                if (two_hop_hint_mode) {
                    RejectTwoHopCapture(true);
                } else {
                    depth_predecessor_hint_outcome.duplicate_collision = true;
                    depth_predecessor_hint_capture_closed = true;
                    open_depth_predecessor_hint_draw = {};
                    if (upstream_feedback_armed) {
                        upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss,
                                                       {.gap = 1});
                    }
                }
                return true;
            }
            expected_hint_candidate = open_depth_predecessor_hint_draw.candidate_index;
        } else if (!MatchesPpTerminalScopeUpstreamTarget(config.content, observed,
                                                         draw.fragment_pgm_hash)) {
            return true;
        }
        u32 copied_candidate = HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates;
        const bool continue_draw = ObserveUpstreamFeedbackPostDraw(
            link, image, state, 0, rendering_serial, observed, draw.fragment_pgm_hash,
            vk::ImageAspectFlagBits::eDepth, &copied_candidate, hint_mode);
        if (hint_mode) {
            open_depth_predecessor_hint_draw = {};
        }
        const bool exact_hint_candidate =
            copied_candidate == expected_hint_candidate &&
            copied_candidate < depth_candidate_predecessors.size() &&
            (!two_hop_hint_mode ||
             upstream_feedback_frame.RecordedPlaneMask(copied_candidate) == 0x3u);
        if (hint_mode && exact_hint_candidate) {
            image.TagDiagnosticDepthStencilUpstreamCandidate(depth_candidate_generation,
                                                             copied_candidate);
            if (two_hop_hint_mode &&
                (depth_predecessor_two_hop_outcome.chain_kind ==
                     HostPasses::PpDepthPredecessorTwoHopChainKind::SameWriterImmediatePrior ||
                 depth_predecessor_two_hop_outcome.chain_kind ==
                     HostPasses::PpDepthPredecessorTwoHopChainKind::
                         DistinctPrefixBeforeSameWriter)) {
                const u32 prior_observed =
                    depth_predecessor_two_hop_outcome.observed_occurrences;
                const u32 prior_raw = depth_predecessor_two_hop_outcome.raw_occurrences;
                const bool prefix_chain =
                    depth_predecessor_two_hop_outcome.chain_kind ==
                    HostPasses::PpDepthPredecessorTwoHopChainKind::
                        DistinctPrefixBeforeSameWriter;
                const auto built = BuildCurrentDepthPredecessorObservation(
                    image_id, image, current_range, depth_candidate_generation);
                depth_predecessor_two_hop_outcome.current_fresh_tag |= built.candidate.valid;
                const auto raw_complete =
                    depth_predecessor_two_hop.ObserveRawP2Complete(built);
                AccumulateTwoHopOutcome(raw_complete);
                const bool role_transition =
                    prior_observed != std::numeric_limits<u32>::max() &&
                    prior_raw != std::numeric_limits<u32>::max() &&
                    raw_complete.observed_occurrences ==
                        prior_observed + (prefix_chain ? 0u : 1u) &&
                    raw_complete.raw_occurrences == prior_raw + 1 &&
                    raw_complete.occurrence_role ==
                        (prefix_chain
                             ? HostPasses::PpDepthPredecessorTwoHopOccurrenceRole::P1Tail
                             : HostPasses::PpDepthPredecessorTwoHopOccurrenceRole::P1Bridge);
                if (IsTwoHopFailure(raw_complete.decision) || !raw_complete.p2_exact ||
                    !role_transition) {
                    RejectTwoHopCapture();
                } else if (prefix_chain) {
                    depth_predecessor_hint_capture_closed = false;
                }
            }
        } else if (hint_mode) {
            if (two_hop_hint_mode) {
                RejectTwoHopCapture();
            } else {
                depth_predecessor_hint_outcome.duplicate_collision = true;
                depth_predecessor_hint_capture_closed = true;
                if (upstream_feedback_armed) {
                    upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
                }
            }
        }
        if (copied_candidate < depth_candidate_predecessors.size()) {
            auto& candidate = depth_candidate_predecessors[copied_candidate];
            const auto target = image.PeekDiagnosticDepthStencilProducer();
            const u64 current_mutation_epoch = image.DiagnosticMutationEpoch();
            const u64 current_backing_generation =
                image.backing ? image.EnsureDiagnosticBackingGeneration() : 0;
            const auto classification = HostPasses::ClassifyPpUpstreamDepthFirstWriter(
                {
                    .recorded = candidate.recorded,
                    .depth_stencil_exact = candidate.depth_stencil_exact,
                    .same_scope = candidate.depth_stencil_same_scope,
                    .scope_depth_history_valid = candidate.scope_depth_history_valid,
                    .first_depth_writer_valid = candidate.scope_first_depth_valid,
                    .depth_writer_count = candidate.scope_depth_writer_count,
                    .depth_writer_count_overflow =
                        candidate.scope_depth_writer_count_overflow,
                },
                {
                    .recorded = true,
                    .current_mutation_epoch = current_mutation_epoch,
                    .current_backing_generation = current_backing_generation,
                    .current_rendering_serial = rendering_serial,
                    .target_shape =
                        {
                            .kind = draw.kind,
                            .fragment_pgm_hash = draw.fragment_pgm_hash,
                            .element_count = draw.element_count,
                            .instance_count = draw.instance_count,
                            .sampled_images = draw.sampled_images,
                            .storage_writes = draw.storage_writes,
                            .indexed = draw.indexed,
                        },
                    .target_depth_write = true,
                    .observed_valid = target.valid,
                    .observed_mutation_epoch = target.mutation_epoch,
                    .observed_backing_generation = target.backing_generation,
                    .observed_rendering_serial = target.rendering_serial,
                    .observed_shape =
                        {
                            .kind = target.kind,
                            .fragment_pgm_hash = target.fragment_pgm_hash,
                            .element_count = target.element_count,
                            .instance_count = target.instance_count,
                            .sampled_images = target.sampled_images,
                            .storage_writes = target.storage_writes,
                            .indexed = target.indexed,
                        },
                    .observed_depth_write = target.depth_write,
                    .scope =
                        {
                            .rendering_serial = target.rendering_serial,
                            .backing_generation = target.backing_generation,
                            .latest_mutation_epoch = target.mutation_epoch,
                            .first_shape =
                                {
                                    .kind = target.scope_first_depth_kind,
                                    .fragment_pgm_hash =
                                        target.scope_first_depth_fragment_pgm_hash,
                                    .element_count = target.scope_first_depth_element_count,
                                    .instance_count = target.scope_first_depth_instance_count,
                                    .sampled_images = target.scope_first_depth_sampled_images,
                                    .storage_writes = target.scope_first_depth_storage_writes,
                                    .indexed = target.scope_first_depth_indexed,
                                },
                            .writer_count = target.scope_depth_writer_count,
                            .first_valid = target.scope_first_depth_valid,
                            .writer_count_overflow =
                                target.scope_depth_writer_count_overflow,
                            .depth_clear = target.scope_depth_clear,
                            .stencil_clear = target.scope_stencil_clear,
                            .valid = target.scope_depth_history_valid,
                        },
                });
            candidate.target_post_exact = classification.post_exact;
            candidate.target_first_depth_writer = classification.target_first_depth_writer;
            candidate.target_scope_load = classification.target_scope_load;
            candidate.target_scope_clear = classification.target_scope_clear;
        }
        if (!hint_mode && copied_candidate < depth_candidate_inputs.size() &&
            depth_candidate_inputs_valid[copied_candidate]) {
            image.TagDiagnosticDepthStencilUpstreamCandidate(depth_candidate_generation,
                                                             copied_candidate);
        }
        return continue_draw;
    }

    void ObserveDraw(VideoCore::ImageId image_id, VideoCore::Image& image, const RenderState& state,
                     u32 attachment_index, u64 rendering_serial,
                     const VideoCore::ImageColorScopeDrawDescriptor& draw) {
        const PpTerminalScopeDrawSelector observed{
            .kind = draw.kind,
            .indexed = draw.indexed,
            .element_count = draw.element_count,
            .instance_count = draw.instance_count,
            .sampled_images = draw.sampled_images,
            .storage_writes = draw.storage_writes,
        };
        const VideoCore::ImageColorScopePrivateLink link{.id = image_id, .uid = image.image_uid};
        Entry* entry = Find(link);
        if (!entry) {
            return;
        }
        const auto action = entry->gate.ObserveDraw(image.image_uid, rendering_serial, observed);
        const auto action_result = ApplyPpTerminalScopeContentAction(
            entry->status, entry->loss, action, config.content.capture_pre_first);
        entry->status = action_result.status;
        entry->loss = action_result.loss;
        if (action != PpTerminalScopeContentAction::CapturePredecessor &&
            action != PpTerminalScopeContentAction::CaptureFirst &&
            action != PpTerminalScopeContentAction::CaptureSecond) {
            return;
        }
        const u32 plane = action == PpTerminalScopeContentAction::CapturePredecessor ? 6
                          : action == PpTerminalScopeContentAction::CaptureFirst     ? 0
                                                                                     : 1;
        RecordPlane(*entry, image, state, attachment_index, rendering_serial, plane);
        if (entry->discovered) {
            progress_coverage.Observe({
                .predecessor_after_action =
                    action == PpTerminalScopeContentAction::CapturePredecessor,
                .predecessor_after_captured =
                    action == PpTerminalScopeContentAction::CapturePredecessor &&
                    (entry->recorded_plane_mask & (1u << 6)) != 0,
                .first_after_action = action == PpTerminalScopeContentAction::CaptureFirst,
                .first_after_captured = action == PpTerminalScopeContentAction::CaptureFirst &&
                                        (entry->recorded_plane_mask & (1u << 0)) != 0,
                .second_action = action == PpTerminalScopeContentAction::CaptureSecond,
                .second_captured = action == PpTerminalScopeContentAction::CaptureSecond &&
                                   (entry->recorded_plane_mask & (1u << 1)) != 0,
            });
        }
    }

    void ObserveFlip(u64 sequence, u64 process_time_us, VideoCore::ImageColorScopePrivateLink link,
                     VideoCore::Image* target, u32 final_source_width, u32 final_source_height,
                     u32 logical_width, u32 logical_height) {
        const bool hint_mode = config.content.capture_upstream_depth_predecessor_hint;
        const bool cross_scope_hint_mode = IsCrossScopeDepthPredecessorHintMode();
        const bool two_hop_hint_mode = IsTwoHopDepthPredecessorHintMode();
        const bool one_hop_hint_mode = IsOneHopDepthPredecessorHintMode();
        if (cross_scope_hint_mode &&
            depth_predecessor_runtime_plan.path ==
                HostPasses::PpDepthPredecessorRuntimePath::CrossScopeBootstrap) {
            const auto mode3_end = depth_predecessor_two_hop.EndFrame();
            AccumulateTwoHopOutcome(mode3_end);
            const u32 mode3_recorded_plane_mask =
                upstream_feedback_frame.RecordedPlaneMask(
                    mode3_end.candidate_index);
            const bool invalidate_runtime_selection =
                depth_predecessor_cross_scope_outcome.frame_failed ||
                IsTwoHopFailure(mode3_end.decision) ||
                !depth_predecessor_two_hop_outcome.actual_selected ||
                !upstream_feedback_slot || mode3_recorded_plane_mask != 0x3u;
            if (invalidate_runtime_selection) {
                depth_predecessor_cross_scope_outcome.mode3_runtime_selection =
                    HostPasses::
                        AccumulatePpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot(
                            depth_predecessor_cross_scope_outcome
                                .mode3_runtime_selection,
                            {.event = HostPasses::
                                 PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshotEvent::
                                     FrameInvalidated});
            }
            const auto mode3_selected_proof =
                depth_predecessor_cross_scope_frame_open
                    ? depth_predecessor_cross_scope
                          .ClassifyMode3SelectedProof(mode3_end)
                    : HostPasses::
                          PpDepthPredecessorCrossScopeMode3SelectedProof{};
            depth_predecessor_cross_scope_outcome.bootstrap = HostPasses::
                AccumulatePpDepthPredecessorCrossScopeBootstrapRuntimeEvidence(
                    depth_predecessor_cross_scope_outcome.bootstrap,
                    {
                        .event = HostPasses::
                            PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceEvent::End,
                        .end_decision = mode3_end.decision,
                        .selected_proof = mode3_selected_proof,
                    });
            if (depth_predecessor_cross_scope_frame_open) {
                depth_predecessor_cross_scope_outcome.selected_latest_evidence =
                    depth_predecessor_cross_scope
                        .ClassifyMode3SelectedLatestBinding(
                            mode3_end,
                            depth_predecessor_cross_scope_outcome
                                .selected_latest_private,
                            depth_predecessor_cross_scope_outcome
                                .mode3_runtime_selection,
                            depth_predecessor_cross_scope_outcome
                                .learning_logical_candidates,
                            depth_predecessor_cross_scope_outcome
                                .learning_transition_candidates);
                depth_predecessor_cross_scope_outcome.root_binding_evidence =
                    depth_predecessor_cross_scope.ClassifyMode3RootBinding(
                        depth_predecessor_cross_scope_outcome
                            .selected_latest_private,
                        depth_predecessor_cross_scope_outcome
                            .selected_latest_evidence);
                if (depth_predecessor_cross_scope_outcome
                        .root_binding_evidence.exact) {
                    depth_predecessor_cross_scope_p0_repeat_probe_configuration =
                        HostPasses::
                            MakePpDepthPredecessorCrossScopeArmedP0RepeatProbeConfiguration(
                                depth_predecessor_cross_scope_outcome
                                    .selected_latest_private,
                                depth_predecessor_cross_scope_outcome
                                    .selected_latest_evidence,
                                depth_predecessor_cross_scope_outcome
                                    .root_binding_evidence);
                    depth_predecessor_cross_scope_p0_repeat_probe_configured =
                        depth_predecessor_cross_scope_p0_repeat_probe
                            .Configure(
                                depth_predecessor_cross_scope_p0_repeat_probe_configuration)
                            .root_exact;
                }
                const auto chain =
                    depth_predecessor_cross_scope_outcome.root_binding_evidence.exact
                        ? depth_predecessor_cross_scope
                              .ObserveLearningRootBoundChainComplete(
                                  mode3_end,
                                  depth_predecessor_cross_scope_outcome
                                      .selected_latest_private,
                                  depth_predecessor_cross_scope_outcome
                                      .mode3_runtime_selection,
                                  depth_predecessor_cross_scope_outcome
                                      .learning_logical_candidates,
                                  depth_predecessor_cross_scope_outcome
                                      .learning_transition_candidates)
                        : depth_predecessor_cross_scope
                              .ObserveLearningChainComplete(mode3_end);
                AccumulateCrossScopeOutcome(chain);
                const auto cross_end = depth_predecessor_cross_scope.EndFrame();
                AccumulateCrossScopeOutcome(cross_end);
            }
            const auto disposition =
                HostPasses::ClassifyPpDepthPredecessorRuntimeFrameDisposition({
                    .path = depth_predecessor_runtime_plan.path,
                    .feedback_armed = upstream_feedback_armed,
                    .slot_present = static_cast<bool>(upstream_feedback_slot),
                    .recorded_plane_mask =
                        upstream_feedback_frame.RecordedPlaneMask(
                            mode3_end.candidate_index),
                    .candidate_saved = mode3_end.candidate_saved,
                    .lifecycle_selected =
                        mode3_end.select &&
                        depth_predecessor_two_hop_outcome.actual_selected,
                    .frame_failed =
                        depth_predecessor_cross_scope_outcome.frame_failed ||
                        IsTwoHopFailure(mode3_end.decision),
                });
            if (disposition.reject_before_take && upstream_feedback_armed) {
                upstream_feedback_frame.Reject(
                    FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
            }
            const auto outcome =
                HostPasses::MakePpDepthPredecessorCrossScopeRuntimeOutcome(
                    depth_predecessor_cross_scope_outcome.result,
                    disposition.disposition,
                    depth_predecessor_cross_scope_outcome.source_plane_recorded,
                    depth_predecessor_cross_scope_outcome.target_plane_recorded,
                    depth_predecessor_cross_scope_outcome.learning,
                    depth_predecessor_cross_scope_outcome
                        .learning_logical_candidates,
                    depth_predecessor_cross_scope_outcome
                        .learning_transition_candidates,
                    depth_predecessor_cross_scope_outcome.bootstrap,
                    depth_predecessor_cross_scope_outcome
                        .armed_target_pre_evidence,
                    depth_predecessor_cross_scope_outcome
                        .source_restart_evidence,
                    depth_predecessor_cross_scope_outcome
                        .selected_latest_evidence,
                    depth_predecessor_cross_scope_outcome
                        .root_binding_evidence,
                    depth_predecessor_cross_scope_outcome
                        .first_tail_pre_drift_evidence,
                    depth_predecessor_cross_scope_outcome
                        .armed_p0_repeat_probe_evidence);
            LOG_INFO(
                Render, "{}",
                HostPasses::FormatPpDepthPredecessorCrossScopeRuntimeOutcome(
                    outcome));
        } else if (cross_scope_hint_mode &&
                   depth_predecessor_runtime_plan.path ==
                       HostPasses::PpDepthPredecessorRuntimePath::CrossScopeArmed) {
            HostPasses::PpDepthPredecessorCrossScopePairResult cross_end =
                depth_predecessor_cross_scope_outcome.result;
            if (depth_predecessor_cross_scope_frame_open) {
                cross_end = depth_predecessor_cross_scope.EndFrame();
                AccumulateCrossScopeOutcome(cross_end);
            }
            const u32 candidate_index =
                depth_predecessor_cross_scope_resume.valid
                    ? depth_predecessor_cross_scope_resume.source_candidate
                          .candidate_index
                    : cross_end.candidate_index;
            const auto disposition =
                HostPasses::ClassifyPpDepthPredecessorRuntimeFrameDisposition({
                    .path = depth_predecessor_runtime_plan.path,
                    .feedback_armed = upstream_feedback_armed,
                    .slot_present = static_cast<bool>(upstream_feedback_slot),
                    .recorded_plane_mask =
                        upstream_feedback_frame.RecordedPlaneMask(candidate_index),
                    .candidate_saved = cross_end.candidate_saved,
                    .lifecycle_selected =
                        cross_end.select &&
                        depth_predecessor_cross_scope_outcome.actual_selected,
                    .frame_failed =
                        depth_predecessor_cross_scope_outcome.frame_failed ||
                        IsCrossScopeFailure(cross_end.decision),
                });
            if (disposition.reject_before_take && upstream_feedback_armed) {
                upstream_feedback_frame.Reject(
                    FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
            }
            const auto outcome =
                HostPasses::MakePpDepthPredecessorCrossScopeRuntimeOutcome(
                    cross_end, disposition.disposition,
                    depth_predecessor_cross_scope_outcome.source_plane_recorded,
                    depth_predecessor_cross_scope_outcome.target_plane_recorded,
                    depth_predecessor_cross_scope_outcome.learning,
                    depth_predecessor_cross_scope_outcome
                        .learning_logical_candidates,
                    depth_predecessor_cross_scope_outcome
                        .learning_transition_candidates,
                    depth_predecessor_cross_scope_outcome.bootstrap,
                    depth_predecessor_cross_scope_outcome
                        .armed_target_pre_evidence,
                    depth_predecessor_cross_scope_outcome
                        .source_restart_evidence,
                    {}, {},
                    depth_predecessor_cross_scope_outcome
                        .first_tail_pre_drift_evidence,
                    depth_predecessor_cross_scope_outcome
                        .armed_p0_repeat_probe_evidence);
            LOG_INFO(
                Render, "{}",
                HostPasses::FormatPpDepthPredecessorCrossScopeRuntimeOutcome(
                    outcome));
        } else if (one_hop_hint_mode) {
            const auto end = depth_predecessor_hint.EndFrame();
            depth_predecessor_hint_outcome.decision = end.decision;
            if (upstream_feedback_armed &&
                (!end.select || !depth_predecessor_hint_outcome.actual_selected)) {
                upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
            }
            LOG_INFO(
                Render,
                "FGSCTUH s={} ah={} ac={} ex={} hm={} ft={} cj={} sel={} nu={} rr={} dc={} d={}",
                sequence, depth_predecessor_hint_outcome.active_hint,
                depth_predecessor_hint_outcome.anchor_count,
                depth_predecessor_hint_outcome.exact_latest,
                depth_predecessor_hint_outcome.hint_match,
                depth_predecessor_hint_outcome.current_fresh_tag,
                depth_predecessor_hint_outcome.exact_candidate_join,
                depth_predecessor_hint_outcome.actual_selected,
                depth_predecessor_hint_outcome.next_updated,
                depth_predecessor_hint_outcome.restart_count,
                depth_predecessor_hint_outcome.duplicate_collision,
                static_cast<u32>(depth_predecessor_hint_outcome.decision));
        } else if (two_hop_hint_mode) {
            const auto end = depth_predecessor_two_hop.EndFrame();
            AccumulateTwoHopOutcome(end);
            depth_predecessor_two_hop_outcome.committed = end.committed;
            if (upstream_feedback_armed &&
                (!end.select || !depth_predecessor_two_hop_outcome.actual_selected)) {
                upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
            }
            LOG_INFO(
                Render,
                "FGSCTUH2 s={} ph={} md={} ra={} ck={} ex={} oc={} rc={} bc={} p1c={} p1x={} "
                "ic={} p1={} p2={} pj={} ft={} ppc={} pp={} cs={} ac={} ax={} cm={} sel={} rr={} "
                "dc={} d={} pjd={}",
                sequence, static_cast<u32>(depth_predecessor_two_hop_outcome.phase),
                depth_predecessor_two_hop_outcome.metadata_only,
                depth_predecessor_two_hop_outcome.raw_armed,
                static_cast<u32>(depth_predecessor_two_hop_outcome.chain_kind),
                depth_predecessor_two_hop_outcome.expected_occurrences,
                depth_predecessor_two_hop_outcome.observed_occurrences,
                depth_predecessor_two_hop_outcome.raw_occurrences,
                depth_predecessor_two_hop_outcome.bridge_occurrences,
                depth_predecessor_two_hop_outcome.p1_pre_count,
                depth_predecessor_two_hop_outcome.collision_p1_ordinal,
                static_cast<u32>(depth_predecessor_two_hop_outcome.identity_collision),
                depth_predecessor_two_hop_outcome.p1_intent,
                depth_predecessor_two_hop_outcome.p2_exact,
                depth_predecessor_two_hop_outcome.prefix_join_exact,
                depth_predecessor_two_hop_outcome.current_fresh_tag,
                depth_predecessor_two_hop_outcome.p1_post_count,
                depth_predecessor_two_hop_outcome.p1_post_exact,
                depth_predecessor_two_hop_outcome.candidate_saved,
                depth_predecessor_two_hop_outcome.anchor_count,
                depth_predecessor_two_hop_outcome.anchor_exact,
                depth_predecessor_two_hop_outcome.committed,
                depth_predecessor_two_hop_outcome.actual_selected,
                depth_predecessor_two_hop_outcome.restart_count,
                depth_predecessor_two_hop_outcome.duplicate_collision,
                static_cast<u32>(depth_predecessor_two_hop_outcome.decision),
                static_cast<u32>(
                    depth_predecessor_two_hop_outcome.prefix_join_discriminator));
        }
        TakeUpstreamFeedbackAtFlip(sequence, process_time_us);
        upstream_registry.Reset();
        mapping = {
            .final_source_width = final_source_width,
            .final_source_height = final_source_height,
            .logical_width = logical_width,
            .logical_height = logical_height,
        };
        if (!mapping.Valid()) {
            mapping = {};
        }
        if (config.content.capture_upstream_pre_post) {
            upstream_input_gate.Reset();
            upstream_input_plan = {};
            upstream_input_ready = false;
            upstream_sample_routes = {};
            upstream_sample_tee_producer_trace = {};
            upstream_sample_tee_source = {};
            upstream_sample_tee_transfer = {};
            upstream_sample_tee_transferred_depth_source = {};
            upstream_sample_tee_selected_depth = {};
            upstream_sample_tee_candidate = {};
            upstream_sample_tee_capture = {};
            upstream_sample_tee_commands = {};
            upstream_sample_tee_controller = {};
            upstream_sample_tee_attempted = false;
            upstream_sample_tee_status = FinalGuestSurfaceStatus::AlreadyConsumed;
            upstream_sample_tee_loss = {};
            upstream_sample_tee_copy = false;
            depth_candidate_inputs = {};
            depth_candidate_inputs_valid = {};
            depth_candidate_predecessors = {};
            if (hint_mode) {
                open_depth_predecessor_hint_draw = {};
                open_depth_predecessor_two_hop_p1_draw = {};
                open_depth_predecessor_cross_scope_source_draw = {};
                open_depth_predecessor_cross_scope_target_draw = {};
                open_depth_predecessor_cross_scope_tail_draw = {};
                const auto p0_repeat_frame_reset = HostPasses::
                    ResetPpDepthPredecessorCrossScopeArmedP0RepeatProbeRuntimeFrame({
                        .correlation_open =
                            open_depth_predecessor_cross_scope_p0_repeat_draw
                                .valid,
                        .evidence = depth_predecessor_cross_scope_outcome
                                        .armed_p0_repeat_probe_evidence,
                    });
                if (!p0_repeat_frame_reset.correlation_open) {
                    open_depth_predecessor_cross_scope_p0_repeat_draw = {};
                }
                depth_predecessor_hint_capture_closed = false;
                depth_predecessor_cross_scope_capture_closed = false;
                depth_predecessor_cross_scope_frame_open = false;
                depth_predecessor_cross_scope_bootstrap = {};
                depth_predecessor_cross_scope_resume = {};
                depth_predecessor_cross_scope_outcome = {};
                depth_predecessor_cross_scope_outcome.bootstrap = HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeBootstrapRuntimeEvidence(
                        depth_predecessor_cross_scope_outcome.bootstrap,
                        {.event = HostPasses::
                             PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceEvent::Reset});
                depth_predecessor_cross_scope_outcome.source_restart_evidence =
                    HostPasses::
                        AccumulatePpDepthPredecessorCrossScopeSourceRestartEvidence(
                            depth_predecessor_cross_scope_outcome
                                .source_restart_evidence,
                            {.event = HostPasses::
                                 PpDepthPredecessorCrossScopeSourceRestartEvidenceEvent::
                                     Reset});
                depth_predecessor_cross_scope_outcome
                    .first_tail_pre_drift_evidence = HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidence(
                        depth_predecessor_cross_scope_outcome
                            .first_tail_pre_drift_evidence,
                        {.event = HostPasses::
                             PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceEvent::
                                 Reset});
                depth_predecessor_cross_scope_outcome
                    .armed_p0_repeat_probe_evidence =
                    p0_repeat_frame_reset.evidence;
                const bool prospective_capture =
                    sequence != std::numeric_limits<u64>::max() &&
                    config.window.Contains(sequence + 1);
                depth_predecessor_runtime_plan =
                    HostPasses::PlanPpDepthPredecessorRuntimeFrame({
                        .hint_mode = hint_mode,
                        .configured_hops =
                            config.content.upstream_depth_predecessor_hint_hops,
                        .cross_scope_pair_ready =
                            depth_predecessor_cross_scope.PeekArmedSourceHint() !=
                            nullptr,
                        .prospective_capture = prospective_capture,
                    });
                upstream_feedback_armed = false;
                if (cross_scope_hint_mode &&
                    depth_predecessor_runtime_plan.begin_multi_hop) {
                    depth_predecessor_cross_scope_p0_repeat_probe.Reset();
                    depth_predecessor_cross_scope_p0_repeat_probe_configuration = {};
                    depth_predecessor_cross_scope_p0_repeat_probe_configured = false;
                    depth_predecessor_two_hop_outcome = {};
                    const auto begin = depth_predecessor_two_hop.BeginFrame();
                    AccumulateTwoHopOutcome(begin);
                    depth_candidate_generation = begin.generation;
                    const auto bootstrap =
                        depth_predecessor_two_hop
                            .ClassifyCrossScopeBootstrapSnapshot(begin);
                    depth_predecessor_cross_scope_bootstrap = bootstrap.snapshot;
                    depth_predecessor_cross_scope_outcome.bootstrap = HostPasses::
                        AccumulatePpDepthPredecessorCrossScopeBootstrapRuntimeEvidence(
                            depth_predecessor_cross_scope_outcome.bootstrap,
                            {
                                .event = HostPasses::
                                    PpDepthPredecessorCrossScopeBootstrapRuntimeEvidenceEvent::
                                        Begin,
                                .begin = begin,
                                .readiness = bootstrap.readiness,
                            });
                    if (depth_predecessor_cross_scope_bootstrap.valid) {
                        const auto cross_begin =
                            depth_predecessor_cross_scope.BeginLearningFrame(
                                begin,
                                depth_predecessor_cross_scope_bootstrap
                                    .p0_raw_prefix,
                                depth_predecessor_cross_scope_bootstrap.p1_tail,
                                depth_predecessor_cross_scope_bootstrap
                                    .configured_anchor);
                        AccumulateCrossScopeOutcome(cross_begin);
                        depth_predecessor_cross_scope_frame_open =
                            !IsCrossScopeFailure(cross_begin.decision);
                    }
                    const auto* armed_hint =
                        depth_predecessor_two_hop.PeekArmedRawP2Hint();
                    upstream_feedback_armed =
                        depth_predecessor_runtime_plan.arm_feedback &&
                        begin.raw_armed && armed_hint &&
                        HostPasses::ArmPpUpstreamFeedbackSelectionAuthority(
                            upstream_feedback_selection_authority, upstream_feedback_frame,
                            armed_hint->draw,
                            HostPasses::PpUpstreamFeedbackSelectionAuthorityMode::
                                PredecessorHint);
                } else if (cross_scope_hint_mode &&
                           depth_predecessor_runtime_plan.begin_cross_scope) {
                    const auto cross_begin =
                        depth_predecessor_cross_scope.BeginFrame();
                    AccumulateCrossScopeOutcome(cross_begin);
                    depth_predecessor_cross_scope_frame_open =
                        !IsCrossScopeFailure(cross_begin.decision) &&
                        cross_begin.phase ==
                            HostPasses::PpDepthPredecessorCrossScopePairPhase::Armed;
                    depth_candidate_generation = cross_begin.generation;
                    if (depth_predecessor_cross_scope_p0_repeat_probe_configured) {
                        static_cast<void>(
                            depth_predecessor_cross_scope_p0_repeat_probe.Configure(
                                depth_predecessor_cross_scope_p0_repeat_probe_configuration));
                    } else {
                        depth_predecessor_cross_scope_p0_repeat_probe.Reset();
                    }
                    const auto* source_hint =
                        depth_predecessor_cross_scope.PeekArmedSourceHint();
                    upstream_feedback_armed =
                        depth_predecessor_cross_scope_frame_open &&
                        depth_predecessor_runtime_plan.arm_feedback && source_hint &&
                        HostPasses::ArmPpUpstreamFeedbackSelectionAuthority(
                            upstream_feedback_selection_authority, upstream_feedback_frame,
                            source_hint->draw,
                            HostPasses::PpUpstreamFeedbackSelectionAuthorityMode::
                                PredecessorHint);
                } else if (two_hop_hint_mode) {
                    depth_predecessor_two_hop_outcome = {};
                    const auto begin = depth_predecessor_two_hop.BeginFrame();
                    AccumulateTwoHopOutcome(begin);
                    depth_candidate_generation = begin.generation;
                    const auto* armed_hint =
                        depth_predecessor_two_hop.PeekArmedRawP2Hint();
                    upstream_feedback_armed =
                        prospective_capture && begin.raw_armed && armed_hint &&
                        HostPasses::ArmPpUpstreamFeedbackSelectionAuthority(
                            upstream_feedback_selection_authority, upstream_feedback_frame,
                            armed_hint->draw,
                            HostPasses::PpUpstreamFeedbackSelectionAuthorityMode::
                                PredecessorHint);
                } else if (one_hop_hint_mode) {
                    depth_predecessor_hint_outcome = {};
                    const auto begin = depth_predecessor_hint.BeginFrame();
                    depth_predecessor_hint_outcome.active_hint = begin.armed;
                    depth_predecessor_hint_outcome.decision = begin.decision;
                    depth_candidate_generation = begin.generation;
                    const auto* armed_hint = depth_predecessor_hint.PeekArmedHint();
                    upstream_feedback_armed =
                        prospective_capture && armed_hint &&
                        HostPasses::ArmPpUpstreamFeedbackSelectionAuthority(
                            upstream_feedback_selection_authority, upstream_feedback_frame,
                            armed_hint->draw,
                            HostPasses::PpUpstreamFeedbackSelectionAuthorityMode::
                                PredecessorHint);
                }
            } else {
                const bool arm_upstream_feedback =
                    !config.content.capture_upstream_depth_pre_post ||
                    (sequence != std::numeric_limits<u64>::max() &&
                     config.window.Contains(sequence + 1));
                upstream_feedback_armed =
                    arm_upstream_feedback &&
                    HostPasses::ArmPpUpstreamFeedbackSelectionAuthority(
                        upstream_feedback_selection_authority, upstream_feedback_frame,
                        config.content.capture_upstream_depth_pre_post
                            ? HostPasses::PpUpstreamFeedbackSelectionAuthorityMode::DepthPrePost
                            : HostPasses::PpUpstreamFeedbackSelectionAuthorityMode::Exact);
                if (upstream_feedback_armed) {
                    ++depth_candidate_generation;
                    if (depth_candidate_generation == 0) {
                        ++depth_candidate_generation;
                    }
                }
            }
        }
        const bool target_valid = mapping.Valid() && IsValidTarget(link, target);
        const u64 target_generation =
            target_valid ? target->EnsureDiagnosticBackingGeneration() : 0;
        Entry* entry = target_valid ? FindFlipAlias(link, target_generation) : nullptr;
        const bool lineage_entry = entry != nullptr;
        if (!entry) {
            entry = Find(link);
        }
        const bool had_entry = entry != nullptr;
        if (entry) {
            const auto gate_link = lineage_entry ? entry->lineage.Root() : link;
            const u64 gate_generation =
                lineage_entry ? entry->lineage.RootGeneration() : target_generation;
            const auto take = entry->gate.Take(gate_link.uid, gate_generation);
            Pending pending{
                .sequence = sequence,
                .process_time_us = process_time_us,
                .layout = MakeHistoryLayout(entry->plan, entry->input_content_plan,
                                            entry->recorded_plane_mask),
                .status = take.status,
                .loss = take.loss,
                .draw_count = take.draw_count,
                .consumer_observations = take.consumer_observations,
                .consumer_phase_mask = take.consumer_phase_mask,
                .consumer_shape_matches = take.consumer_shape_matches,
                .consumer_frozen = take.consumer_frozen,
                .lineage_hops = lineage_entry ? entry->lineage.Hops() : 0,
                .lineage_status = lineage_entry ? entry->lineage.Status()
                                                : FinalGuestSurfaceStatus::AlreadyConsumed,
                .lineage_loss = lineage_entry ? entry->lineage.Loss() : FinalGuestSurfaceLoss{},
                .predecessor = entry->predecessor,
                .sampled_inputs = entry->sampled_inputs,
                .upstream_inputs = entry->upstream_inputs,
                .slot = entry->slot,
                .slot_offset = entry->slot ? entry->slot.slot * slot_stride : 0,
            };
            if (entry->status != FinalGuestSurfaceStatus::Complete || entry->loss.Any()) {
                pending.status = entry->status;
                pending.loss = entry->loss;
            }
            entry->slot = {};
            Defer(std::move(pending));
            if (lineage_entry) {
                ReleaseEntry(entry);
            }
        }

        if (!ShouldArmPpTerminalScopeFallbackAfterFlip(lineage_entry)) {
            return;
        }

        entry = target_valid ? FindOrAllocate(link) : nullptr;
        const auto decision = PlanPpTerminalScopeFlipDecision(
            had_entry, target_valid, entry != nullptr, config.window.Contains(sequence));
        if (decision.synthesize_loss) {
            Defer({
                .sequence = sequence,
                .process_time_us = process_time_us,
                .status = decision.status,
                .loss = decision.loss,
            });
        }
        if (!decision.arm_next || !entry) {
            return;
        }
        ConfigureEntry(*entry, link, *target, false);
    }

    void Calibrate(const FinalGuestSurfaceFrameDiagnosticStamp& stamp,
                   FinalGuestSurfacePresentationMapping mapping, u32 count,
                   u64 fallback_process_time_us) {
        std::scoped_lock lock{reducer_mutex};
        for (u32 index = 0; index < count; ++index) {
            const auto report = calibration.Observe(stamp, mapping, fallback_process_time_us);
            if (!report.emit) {
                continue;
            }
            const FinalGuestSurfaceCalibratedStamp calibrated{
                .request_ordinal = report.request_ordinal,
                .sequence = report.surface_sequence,
                .process_time_us = report.surface_process_time_us,
                .valid = stamp.valid && report.surface_sequence != 0 &&
                         report.surface_process_time_us != 0 && !report.fallback_time &&
                         report.exact_scaled_mapping && report.mapping_loss == 0 &&
                         report.overflow_loss == 0 && !report.overflow_marker,
            };
            reducer.ObserveCalibration(calibrated);
            if (config.content.capture_upstream_pre_post) {
                upstream_feedback_reducer.ObserveCalibration(calibrated);
            }
            if (upstream_sample_tee_reducer) {
                upstream_sample_tee_reducer->ObserveCalibration(calibrated);
            }
            LogReports();
            LogUpstreamFeedbackReports();
            LogCalibratedCoverage();
            LogUpstreamFeedbackCalibratedCoverage();
        }
        LogUpstreamSampleTeeCalibratedOutput();
    }

    [[nodiscard]] bool HasFinalBackingJoin() const noexcept {
        return config.join_final_backing;
    }

    void ObserveFinalBacking(u64 sequence, FinalGuestSurfaceStage stage,
                             const FinalGuestSurfaceTilePlan& plan,
                             std::span<const std::byte> bytes, FinalGuestSurfaceStatus status,
                             FinalGuestSurfaceLoss loss) {
        if (!config.join_final_backing || !config.window.Contains(sequence)) {
            return;
        }
        std::scoped_lock lock{reducer_mutex};
        if (status != FinalGuestSurfaceStatus::Complete || loss.Any()) {
            reducer.ObserveFinalBacking(sequence, {}, {}, status, loss);
            LogReports();
            LogCalibratedCoverage();
            return;
        }
        const auto decision = PlanPpTerminalScopeFinalBackingObservation(
            true, stage, plan,
            static_cast<u64>(plan.paired_backing_offset) + plan.paired_backing_bytes <=
                bytes.size());
        if (!decision.observe) {
            reducer.ObserveFinalBacking(sequence, {}, {}, decision.status, decision.loss);
        } else {
            reducer.ObserveFinalBacking(
                sequence, decision.layout,
                bytes.subspan(decision.source_offset, decision.source_bytes), decision.status,
                decision.loss);
        }
        LogReports();
        LogCalibratedCoverage();
    }

    void Finalize() {
        scheduler.PopPendingOperations();
        if (discovery_coverage_emission.Finalize()) {
            LOG_INFO(Render, "{}", FormatPpTerminalScopeDiscoveryCoverage(discovery_coverage));
            LOG_INFO(Render, "{}", FormatPpTerminalScopeProgressCoverage(progress_coverage));
        }
        std::scoped_lock lock{reducer_mutex};
        LogReports();
        LogUpstreamFeedbackReports();
        LogCalibratedCoverage(true);
        LogUpstreamFeedbackCalibratedCoverage(true);
        LogUpstreamSampleTeeCalibratedOutput(true);
    }

private:
    struct MappingTemplate {
        u32 final_source_width{};
        u32 final_source_height{};
        u32 logical_width{};
        u32 logical_height{};

        [[nodiscard]] bool Valid() const noexcept {
            return final_source_width != 0 && final_source_height != 0 && logical_width != 0 &&
                   logical_height != 0;
        }
    };

    struct DepthPredecessorHintOutcome {
        u32 anchor_count{};
        u32 restart_count{};
        bool active_hint{};
        bool exact_latest{};
        bool hint_match{};
        bool current_fresh_tag{};
        bool exact_candidate_join{};
        bool actual_selected{};
        bool next_updated{};
        bool duplicate_collision{};
        HostPasses::PpDepthPredecessorHintDecision decision{
            HostPasses::PpDepthPredecessorHintDecision::Idle};
    };

    struct DepthPredecessorTwoHopOutcome {
        u32 p1_pre_count{};
        u32 collision_p1_ordinal{};
        u32 p1_post_count{};
        u32 anchor_count{};
        u32 restart_count{};
        u32 expected_occurrences{};
        u32 observed_occurrences{};
        u32 raw_occurrences{};
        u32 bridge_occurrences{};
        HostPasses::PpDepthPredecessorTwoHopOccurrenceRole occurrence_role{
            HostPasses::PpDepthPredecessorTwoHopOccurrenceRole::None};
        bool metadata_only{};
        bool raw_armed{};
        bool p1_intent{};
        bool p2_exact{};
        bool prefix_join_exact{};
        bool p1_post_exact{};
        bool current_fresh_tag{};
        bool candidate_saved{};
        bool anchor_exact{};
        bool committed{};
        bool actual_selected{};
        bool duplicate_collision{};
        HostPasses::PpDepthPredecessorTwoHopIdentityCollision identity_collision{
            HostPasses::PpDepthPredecessorTwoHopIdentityCollision::None};
        HostPasses::PpDepthPredecessorPrefixJoinDiscriminator prefix_join_discriminator{
            HostPasses::PpDepthPredecessorPrefixJoinDiscriminator::NotAttempted};
        HostPasses::PpDepthPredecessorTwoHopChainKind chain_kind{
            HostPasses::PpDepthPredecessorTwoHopChainKind::Distinct};
        HostPasses::PpDepthPredecessorTwoHopPhase phase{
            HostPasses::PpDepthPredecessorTwoHopPhase::LearnP1};
        HostPasses::PpDepthPredecessorTwoHopDecision decision{
            HostPasses::PpDepthPredecessorTwoHopDecision::Idle};
    };

    struct OpenDepthPredecessorHintDraw {
        VideoCore::ImageColorScopePrivateLink link{};
        VideoCore::SubresourceRange range{};
        PpTerminalScopeDrawSelector draw{};
        u64 fragment_pgm_hash{};
        u32 candidate_index{};
        bool valid{};
    };

    struct OpenDepthPredecessorTwoHopP1Draw {
        VideoCore::ImageColorScopePrivateLink link{};
        VideoCore::SubresourceRange range{};
        PpTerminalScopeDrawSelector draw{};
        u64 fragment_pgm_hash{};
        bool valid{};
    };

    struct DepthPredecessorCrossScopeOutcome {
        HostPasses::PpDepthPredecessorCrossScopePairResult result{};
        HostPasses::PpDepthPredecessorCrossScopeBootstrapRuntimeEvidence bootstrap{};
        HostPasses::PpDepthPredecessorCrossScopeLearningEvidence learning{};
        HostPasses::PpDepthPredecessorCrossScopeLearningLogicalCandidateSummary
            learning_logical_candidates{};
        HostPasses::PpDepthPredecessorCrossScopeLearningTransitionCandidateSummary
            learning_transition_candidates{};
        HostPasses::PpDepthPredecessorCrossScopeArmedTargetPreEvidenceSummary
            armed_target_pre_evidence{};
        HostPasses::PpDepthPredecessorCrossScopeSourceRestartEvidence
            source_restart_evidence{};
        HostPasses::PpDepthPredecessorCrossScopeMode3SelectedLatestPrivateWitness
            selected_latest_private{};
        HostPasses::PpDepthPredecessorCrossScopeMode3RuntimeSelectionSnapshot
            mode3_runtime_selection{};
        HostPasses::PpDepthPredecessorCrossScopeMode3SelectedLatestBindingEvidence
            selected_latest_evidence{};
        HostPasses::PpDepthPredecessorCrossScopeMode3RootBindingEvidence
            root_binding_evidence{};
        HostPasses::PpDepthPredecessorCrossScopeFirstTailPreP0DriftEvidenceSummary
            first_tail_pre_drift_evidence{};
        HostPasses::PpDepthPredecessorCrossScopeArmedP0RepeatProbeEvidenceSummary
            armed_p0_repeat_probe_evidence{};
        HostPasses::PpDepthPredecessorEndpointStamp target_post_endpoint{};
        bool actual_selected{};
        bool source_plane_recorded{};
        bool target_plane_recorded{};
        bool frame_failed{};
    };

    struct OpenDepthPredecessorCrossScopeSourceDraw {
        VideoCore::ImageColorScopePrivateLink link{};
        VideoCore::SubresourceRange range{};
        PpTerminalScopeDrawSelector draw{};
        HostPasses::PpUpstreamFeedbackPrePostPlan plan{};
        HostPasses::PpDepthPredecessorCrossScopeResumeWitness
            restart_prior_resume{};
        u64 fragment_pgm_hash{};
        u64 rendering_serial{};
        u32 candidate_index{};
        bool restart{};
        bool valid{};
    };

    struct OpenDepthPredecessorCrossScopeTargetDraw {
        VideoCore::ImageColorScopePrivateLink link{};
        VideoCore::SubresourceRange range{};
        PpTerminalScopeDrawSelector draw{};
        HostPasses::PpDepthPredecessorWriterIntent intent{};
        HostPasses::PpUpstreamDepthFirstWriterPreObservation first_writer_pre{};
        u64 fragment_pgm_hash{};
        bool valid{};
    };

    struct OpenDepthPredecessorCrossScopeTailDraw {
        VideoCore::ImageColorScopePrivateLink link{};
        VideoCore::SubresourceRange range{};
        PpTerminalScopeDrawSelector draw{};
        u64 fragment_pgm_hash{};
        bool valid{};
    };

    struct OpenDepthPredecessorCrossScopeArmedP0RepeatDraw {
        VideoCore::ImageColorScopePrivateLink link{};
        VideoCore::SubresourceRange range{};
        PpTerminalScopeDrawSelector draw{};
        HostPasses::PpUpstreamDepthFirstWriterPreObservation first_writer_pre{};
        u64 fragment_pgm_hash{};
        bool valid{};
    };

    [[nodiscard]] bool IsCrossScopeDepthPredecessorHintMode() const noexcept {
        return HostPasses::IsPpDepthPredecessorCrossScopePairHintMode(
            config.content.capture_upstream_depth_predecessor_hint,
            config.content.upstream_depth_predecessor_hint_hops);
    }

    [[nodiscard]] bool IsTwoHopDepthPredecessorHintMode() const noexcept {
        return HostPasses::IsPpDepthPredecessorMultiHopHintMode(
            config.content.capture_upstream_depth_predecessor_hint,
            config.content.upstream_depth_predecessor_hint_hops);
    }

    [[nodiscard]] bool IsDepthPredecessorStructuralMultiHopRuntime() const noexcept {
        return IsTwoHopDepthPredecessorHintMode() ||
               (depth_predecessor_runtime_plan.path ==
                    HostPasses::PpDepthPredecessorRuntimePath::CrossScopeBootstrap &&
                depth_predecessor_runtime_plan.begin_multi_hop);
    }

    [[nodiscard]] bool IsOneHopDepthPredecessorHintMode() const noexcept {
        return config.content.capture_upstream_depth_predecessor_hint &&
               config.content.upstream_depth_predecessor_hint_hops == 1;
    }

    [[nodiscard]] static constexpr bool IsCrossScopeFailure(
        HostPasses::PpDepthPredecessorCrossScopePairDecision decision) noexcept {
        switch (decision) {
        case HostPasses::PpDepthPredecessorCrossScopePairDecision::Idle:
        case HostPasses::PpDepthPredecessorCrossScopePairDecision::Committed:
        case HostPasses::PpDepthPredecessorCrossScopePairDecision::Selected:
            return false;
        default:
            return true;
        }
    }

    void AccumulateCrossScopeOutcome(
        const HostPasses::PpDepthPredecessorCrossScopePairResult& result) noexcept {
        depth_predecessor_cross_scope_outcome.result = result;
        depth_predecessor_cross_scope_outcome.frame_failed |=
            IsCrossScopeFailure(result.decision);
    }

    void RejectCrossScopeCapture() noexcept {
        depth_predecessor_cross_scope_outcome.frame_failed = true;
        depth_predecessor_cross_scope_capture_closed = true;
        open_depth_predecessor_cross_scope_source_draw = {};
        open_depth_predecessor_cross_scope_target_draw = {};
        open_depth_predecessor_cross_scope_tail_draw = {};
        open_depth_predecessor_cross_scope_p0_repeat_draw = {};
        depth_predecessor_cross_scope_resume = {};
        if (upstream_feedback_armed) {
            upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
        }
    }

    [[nodiscard]] static constexpr bool IsTwoHopFailure(
        HostPasses::PpDepthPredecessorTwoHopDecision decision) noexcept {
        switch (decision) {
        case HostPasses::PpDepthPredecessorTwoHopDecision::Idle:
        case HostPasses::PpDepthPredecessorTwoHopDecision::LearnedP1:
        case HostPasses::PpDepthPredecessorTwoHopDecision::Committed:
        case HostPasses::PpDepthPredecessorTwoHopDecision::Selected:
            return false;
        default:
            return true;
        }
    }

    void AccumulateTwoHopOutcome(
        const HostPasses::PpDepthPredecessorTwoHopResult& result) noexcept {
        depth_predecessor_two_hop_outcome.phase = result.phase;
        depth_predecessor_two_hop_outcome.decision = result.decision;
        depth_predecessor_two_hop_outcome.chain_kind = result.chain_kind;
        depth_predecessor_two_hop_outcome.expected_occurrences = result.expected_occurrences;
        depth_predecessor_two_hop_outcome.observed_occurrences = result.observed_occurrences;
        depth_predecessor_two_hop_outcome.raw_occurrences = result.raw_occurrences;
        depth_predecessor_two_hop_outcome.bridge_occurrences = result.bridge_occurrences;
        depth_predecessor_two_hop_outcome.occurrence_role = result.occurrence_role;
        depth_predecessor_two_hop_outcome.prefix_join_discriminator =
            result.prefix_join_discriminator;
        depth_predecessor_two_hop_outcome.metadata_only |= result.metadata_only;
        depth_predecessor_two_hop_outcome.raw_armed |= result.raw_armed;
        depth_predecessor_two_hop_outcome.p1_intent |= result.p1_intent;
        depth_predecessor_two_hop_outcome.p2_exact |= result.p2_exact;
        depth_predecessor_two_hop_outcome.prefix_join_exact |= result.prefix_join_exact;
        depth_predecessor_two_hop_outcome.p1_post_exact |= result.p1_post_exact;
        if (result.identity_collision !=
                HostPasses::PpDepthPredecessorTwoHopIdentityCollision::None &&
            depth_predecessor_two_hop_outcome.identity_collision ==
                HostPasses::PpDepthPredecessorTwoHopIdentityCollision::None) {
            depth_predecessor_two_hop_outcome.identity_collision = result.identity_collision;
            depth_predecessor_two_hop_outcome.collision_p1_ordinal =
                result.collision_p1_ordinal;
        }
        depth_predecessor_two_hop_outcome.candidate_saved |= result.candidate_saved;
        depth_predecessor_two_hop_outcome.anchor_exact |= result.anchor_exact;
        depth_predecessor_two_hop_outcome.committed |= result.committed;
        switch (result.decision) {
        case HostPasses::PpDepthPredecessorTwoHopDecision::DuplicateP1:
        case HostPasses::PpDepthPredecessorTwoHopDecision::DuplicateP1Post:
        case HostPasses::PpDepthPredecessorTwoHopDecision::DuplicateAnchor:
        case HostPasses::PpDepthPredecessorTwoHopDecision::IdentityCollision:
            depth_predecessor_two_hop_outcome.duplicate_collision = true;
            break;
        default:
            break;
        }
    }

    void RejectTwoHopCapture(bool collision = false) noexcept {
        depth_predecessor_two_hop_outcome.duplicate_collision |= collision;
        depth_predecessor_hint_capture_closed = true;
        open_depth_predecessor_hint_draw = {};
        open_depth_predecessor_two_hop_p1_draw = {};
        if (upstream_feedback_armed) {
            upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
        }
    }

    [[nodiscard]] HostPasses::PpDepthPredecessorObservationResult
    BuildCurrentDepthPredecessorObservation(
        VideoCore::ImageId image_id, VideoCore::Image& image,
        const VideoCore::SubresourceRange& current_range,
        u64 expected_candidate_generation) const noexcept {
        const auto latest = image.PeekDiagnosticDepthStencilProducer();
        const VideoCore::ImageColorScopePrivateLink link{.id = image_id,
                                                         .uid = image.image_uid};
        const u64 current_mutation_epoch = image.DiagnosticMutationEpoch();
        const u64 current_backing_generation =
            image.backing ? image.EnsureDiagnosticBackingGeneration() : 0;
        return HostPasses::BuildPpDepthPredecessorObservation({
            .observed =
                {
                    .kind = latest.kind,
                    .link = latest.link,
                    .range = latest.range,
                    .mutation_epoch = latest.mutation_epoch,
                    .backing_generation = latest.backing_generation,
                    .fragment_pgm_hash = latest.fragment_pgm_hash,
                    .upstream_candidate_generation = latest.upstream_candidate_generation,
                    .element_count = latest.element_count,
                    .instance_count = latest.instance_count,
                    .sampled_images = latest.sampled_images,
                    .storage_writes = latest.storage_writes,
                    .upstream_candidate_index = latest.upstream_candidate_index,
                    .depth_aspect =
                        (latest.attachment_aspect_mask &
                         static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT)) != 0,
                    .indexed = latest.indexed,
                    .depth_write = latest.depth_write,
                    .upstream_candidate_valid = latest.upstream_candidate_valid,
                    .valid = latest.valid,
                },
            .current_link = link,
            .current_range = current_range,
            .current_backing_generation = current_backing_generation,
            .current_mutation_epoch = current_mutation_epoch,
            .expected_candidate_generation = expected_candidate_generation,
            .recorded_plane_mask =
                upstream_feedback_frame.RecordedPlaneMask(latest.upstream_candidate_index),
        });
    }

    [[nodiscard]] HostPasses::PpDepthPredecessorEndpointStamp
    BuildCurrentDepthPredecessorEndpoint(
        VideoCore::ImageId image_id, VideoCore::Image& image,
        const VideoCore::SubresourceRange& current_range) const noexcept {
        const auto latest = image.PeekDiagnosticDepthStencilProducer();
        const auto built = BuildCurrentDepthPredecessorObservation(
            image_id, image, current_range, depth_candidate_generation);
        const bool producer_marked =
            latest.valid && latest.depth_write &&
            (latest.attachment_aspect_mask &
             static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT)) != 0;
        return {
            .writer = built.writer,
            .guest_rendering_serial = latest.rendering_serial,
            .host_rendering_instance_id = latest.host_rendering_instance_id,
            .producer_marked = producer_marked,
            .after_diagnostic_resume = latest.after_diagnostic_resume,
            .valid = producer_marked && built.writer.Valid(),
        };
    }

    [[nodiscard]] HostPasses::PpDepthPredecessorWriterIntent
    BuildDepthPredecessorIntent(
        VideoCore::ImageColorScopePrivateLink link, VideoCore::Image& image,
        const VideoCore::SubresourceRange& range,
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash,
        bool effective_depth_write) const noexcept {
        return {
            .hint = {.draw = draw,
                     .fragment_pgm_hash = fragment_pgm_hash,
                     .valid = true},
            .link = link,
            .range = range,
            .backing_generation =
                image.backing ? image.EnsureDiagnosticBackingGeneration() : 0,
            .mutation_epoch = image.DiagnosticMutationEpoch(),
            .valid = effective_depth_write &&
                     HostPasses::IsCanonicalPpDepthPredecessorRange(range),
        };
    }

    [[nodiscard]] static HostPasses::PpUpstreamDepthFirstWriterPreObservation
    BuildDepthFirstWriterPreObservation(VideoCore::Image& image,
                                        u64 rendering_serial) noexcept {
        const auto latest = image.PeekDiagnosticDepthStencilProducer();
        const u64 mutation_epoch = image.DiagnosticMutationEpoch();
        const u64 backing_generation =
            image.backing ? image.EnsureDiagnosticBackingGeneration() : 0;
        const bool exact = latest.valid && latest.mutation_epoch == mutation_epoch &&
                           latest.backing_generation == backing_generation;
        return {
            .recorded = true,
            .depth_stencil_exact = exact,
            .same_scope = exact && latest.rendering_serial == rendering_serial,
            .scope_depth_history_valid = exact && latest.scope_depth_history_valid,
            .first_depth_writer_valid = exact && latest.scope_first_depth_valid,
            .depth_writer_count = exact ? latest.scope_depth_writer_count : 0,
            .depth_writer_count_overflow =
                exact && latest.scope_depth_writer_count_overflow,
        };
    }

    [[nodiscard]] static HostPasses::PpUpstreamDepthFirstWriterPostObservation
    BuildDepthFirstWriterPostObservation(
        const HostPasses::PpUpstreamDepthFirstWriterPreObservation& pre,
        VideoCore::Image& image,
        const VideoCore::ImageColorScopeDrawDescriptor& draw,
        u64 rendering_serial) noexcept {
        const auto target = image.PeekDiagnosticDepthStencilProducer();
        const u64 mutation_epoch = image.DiagnosticMutationEpoch();
        const u64 backing_generation =
            image.backing ? image.EnsureDiagnosticBackingGeneration() : 0;
        return {
                .recorded = true,
                .current_mutation_epoch = mutation_epoch,
                .current_backing_generation = backing_generation,
                .current_rendering_serial = rendering_serial,
                .target_shape =
                    {
                        .kind = draw.kind,
                        .fragment_pgm_hash = draw.fragment_pgm_hash,
                        .element_count = draw.element_count,
                        .instance_count = draw.instance_count,
                        .sampled_images = draw.sampled_images,
                        .storage_writes = draw.storage_writes,
                        .indexed = draw.indexed,
                    },
                .target_depth_write = true,
                .observed_valid = target.valid,
                .observed_mutation_epoch = target.mutation_epoch,
                .observed_backing_generation = target.backing_generation,
                .observed_rendering_serial = target.rendering_serial,
                .observed_shape =
                    {
                        .kind = target.kind,
                        .fragment_pgm_hash = target.fragment_pgm_hash,
                        .element_count = target.element_count,
                        .instance_count = target.instance_count,
                        .sampled_images = target.sampled_images,
                        .storage_writes = target.storage_writes,
                        .indexed = target.indexed,
                    },
                .observed_depth_write = target.depth_write,
                .scope =
                    {
                        .rendering_serial = target.rendering_serial,
                        .backing_generation = target.backing_generation,
                        .latest_mutation_epoch = target.mutation_epoch,
                        .first_shape =
                            {
                                .kind = target.scope_first_depth_kind,
                                .fragment_pgm_hash =
                                    target.scope_first_depth_fragment_pgm_hash,
                                .element_count =
                                    target.scope_first_depth_element_count,
                                .instance_count =
                                    target.scope_first_depth_instance_count,
                                .sampled_images =
                                    target.scope_first_depth_sampled_images,
                                .storage_writes =
                                    target.scope_first_depth_storage_writes,
                                .indexed = target.scope_first_depth_indexed,
                            },
                        .writer_count = target.scope_depth_writer_count,
                        .first_valid = target.scope_first_depth_valid,
                        .writer_count_overflow =
                            target.scope_depth_writer_count_overflow,
                        .depth_clear = target.scope_depth_clear,
                        .stencil_clear = target.scope_stencil_clear,
                        .valid = target.scope_depth_history_valid,
                    },
            };
    }

    [[nodiscard]] static HostPasses::PpUpstreamDepthFirstWriterClassification
    ClassifyCurrentDepthFirstWriter(
        const HostPasses::PpUpstreamDepthFirstWriterPreObservation& pre,
        VideoCore::Image& image,
        const VideoCore::ImageColorScopeDrawDescriptor& draw,
        u64 rendering_serial) noexcept {
        return HostPasses::ClassifyPpUpstreamDepthFirstWriter(
            pre, BuildDepthFirstWriterPostObservation(pre, image, draw, rendering_serial));
    }

    struct Entry {
        explicit Entry(PpTerminalScopeContentConfig config_) : gate{config_} {}

        VideoCore::ImageColorScopePrivateLink link{};
        PpTerminalScopeContentGate gate;
        PpTerminalScopeContentPlan plan{};
        PpTerminalScopeSampledInputContentPlan input_content_plan{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
        FinalGuestSurfaceReadbackSlotPool::Token slot{};
        u32 recorded_plane_mask{};
        PpTerminalScopePrivateLineage lineage{};
        PpTerminalScopePredecessor predecessor{};
        PpTerminalScopeSampledInputs sampled_inputs{};
        PpTerminalScopeSampledInputs upstream_inputs{};
        VideoCore::ImageColorScopePrivateLink flip_alias{};
        bool discovered{};
        bool flip_alias_ready{};
    };

    struct Pending {
        u64 sequence{};
        u64 process_time_us{};
        PpTerminalScopeContentHistoryLayout layout{};
        FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss loss{};
        u32 draw_count{};
        u32 consumer_observations{};
        u32 consumer_phase_mask{};
        u32 consumer_shape_matches{};
        bool consumer_frozen{};
        u32 lineage_hops{};
        FinalGuestSurfaceStatus lineage_status{FinalGuestSurfaceStatus::AlreadyConsumed};
        FinalGuestSurfaceLoss lineage_loss{};
        PpTerminalScopePredecessor predecessor{};
        PpTerminalScopeSampledInputs sampled_inputs{};
        PpTerminalScopeSampledInputs upstream_inputs{};
        FinalGuestSurfaceReadbackSlotPool::Token slot{};
        u64 slot_offset{};
    };

    struct PendingUpstreamFeedback {
        struct DepthPredecessorObservation {
            VideoCore::ImageProducerClass producer{VideoCore::ImageProducerClass::Unknown};
            VideoCore::ImageTransferKind transfer_kind{VideoCore::ImageTransferKind::None};
            VideoCore::ImageTransferOrigin transfer_origin{VideoCore::ImageTransferOrigin::Unknown};
            VideoCore::ImageColorScopeDrawKind depth_stencil_kind{
                VideoCore::ImageColorScopeDrawKind::Unknown};
            u32 element_count{};
            u32 instance_count{};
            u32 sampled_images{};
            u32 storage_writes{};
            bool recorded{};
            bool transfer_exact{};
            bool depth_stencil_exact{};
            bool depth_stencil_same_scope{};
            VideoCore::ImageColorScopeDrawKind scope_first_depth_kind{
                VideoCore::ImageColorScopeDrawKind::Unknown};
            u64 scope_first_depth_fragment_pgm_hash{};
            u32 scope_first_depth_element_count{};
            u32 scope_first_depth_instance_count{};
            u32 scope_first_depth_sampled_images{};
            u32 scope_first_depth_storage_writes{};
            u32 scope_depth_writer_count{};
            bool indexed{};
            bool scope_first_depth_indexed{};
            bool scope_first_depth_valid{};
            bool scope_depth_writer_count_overflow{};
            bool scope_depth_history_valid{};
            bool scope_depth_clear{};
            bool scope_stencil_clear{};
            bool fragment_writes_depth{};
            bool fragment_has_discard{};
            bool depth_write{};
            bool stencil_write{};
            bool feedback_alias{};
            bool depth_clear{};
            bool stencil_clear{};
            bool target_post_exact{};
            bool target_first_depth_writer{};
            bool target_scope_load{};
            bool target_scope_clear{};
            bool depth_boundary_valid{};
            bool depth_boundary_join_exact{};
            bool depth_boundary_writer_exact{};
            bool depth_boundary_pre_rendering{};
            bool depth_boundary_post_transit_rendering{};
            bool depth_boundary_pre_external_rendering{};
            bool depth_boundary_after_rendering{};
            bool depth_boundary_split_interval{};
            bool depth_boundary_target_load{};
            bool depth_boundary_state_exact{};
            bool depth_boundary_skipped_write{};
            bool depth_boundary_same_host{};
            bool depth_boundary_cross_host{};
            bool depth_boundary_instrumented_cross_host{};
            bool depth_boundary_before_diagnostic{};
            bool depth_boundary_diagnostic_excluded{};
            bool depth_boundary_barrier_recognized_write{};
        };

        u64 sequence{};
        u64 process_time_us{};
        HostPasses::PpUpstreamFeedbackPrePostTakeResult take{};
        HostPasses::PpUpstreamFeedbackPrePostTakeEvaluation take_evaluation{};
        HostPasses::PpUpstreamInputContentPlan input_plan{};
        HostPasses::PpUpstreamSampleRouteSet sample_routes{};
        PpUpstreamSampleTeeProducerTrace producer_trace{};
        HostPasses::PpUpstreamSampleTeeSourceObservation tee_source{};
        PpUpstreamSampleTeeTransferObservation tee_transfer{};
        HostPasses::PpUpstreamSampleTeeTransferredDepthSourcePlan transferred_depth_source{};
        HostPasses::PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned
            producer_pre_scope_entry{};
        DepthPredecessorObservation depth_predecessor{};
        bool input_ready{};
        HostPasses::PpUpstreamSampleTeeDeferredReadbackDescriptor tee_attempt{};
        HostPasses::PpUpstreamFeedbackTeeAuthorizedTargetPrePostReceipt
            tee_target_pre_post_receipt{};
        HostPasses::PpUpstreamSampleTeeScopeWriterTraceFramePlan scope_writer_trace_frame{};
        HostPasses::PpUpstreamSampleTeeScopeWriterTraceTailPlan scope_writer_trace_tail{};
        u64 slot_offset{};
    };

    [[nodiscard]] Entry* Find(VideoCore::ImageColorScopePrivateLink link) noexcept {
        for (auto& entry : entries) {
            if (entry && entry->link == link) {
                return entry.get();
            }
        }
        return nullptr;
    }

    [[nodiscard]] Entry* FindOrAllocate(VideoCore::ImageColorScopePrivateLink link) {
        if (auto* entry = Find(link)) {
            return entry;
        }
        for (auto& entry : entries) {
            if (!entry) {
                entry = std::make_unique<Entry>(config.content);
                entry->link = link;
                return entry.get();
            }
        }
        return nullptr;
    }

    [[nodiscard]] Entry* FindLineageTail(VideoCore::ImageColorScopePrivateLink link) noexcept {
        Entry* match{};
        for (auto& entry : entries) {
            if (!entry || !entry->discovered || entry->lineage.Tail() != link) {
                continue;
            }
            if (match) {
                return nullptr;
            }
            match = entry.get();
        }
        return match;
    }

    [[nodiscard]] Entry* FindFlipAlias(VideoCore::ImageColorScopePrivateLink link,
                                       u64 generation) noexcept {
        Entry* match{};
        for (auto& entry : entries) {
            if (!entry || !entry->flip_alias_ready || entry->flip_alias != link ||
                !entry->lineage.Resolve(link, generation).matched) {
                continue;
            }
            if (match) {
                return nullptr;
            }
            match = entry.get();
        }
        return match;
    }

    void ReleaseEntry(Entry* target) noexcept {
        for (auto& entry : entries) {
            if (entry.get() == target) {
                entry.reset();
                return;
            }
        }
    }

    [[nodiscard]] bool HasTargetCapacity() const noexcept {
        return std::ranges::any_of(entries, [](const auto& entry) { return !entry; });
    }

    [[nodiscard]] static bool IsValidTarget(VideoCore::ImageColorScopePrivateLink link,
                                            const VideoCore::Image* target) noexcept {
        return target && link.Valid() && target->image_uid == link.uid &&
               target->aspect_mask == vk::ImageAspectFlagBits::eColor &&
               target->info.resources.levels == 1 && target->info.resources.layers == 1 &&
               target->info.size.width != 0 && target->info.size.height != 0 && target->backing &&
               target->backing->subresource_states.empty();
    }

    void ConfigureEntry(Entry& entry, VideoCore::ImageColorScopePrivateLink link,
                        VideoCore::Image& target, bool discovered) {
        entry.plan = PlanPpTerminalScopeContent({
            .enabled = true,
            .armed = true,
            .capture_predecessor = config.content.capture_predecessor,
            .target_width = target.info.size.width,
            .target_height = target.info.size.height,
            .final_source_width = mapping.final_source_width,
            .final_source_height = mapping.final_source_height,
            .logical_width = mapping.logical_width,
            .logical_height = mapping.logical_height,
            .format = TerminalScopeFormat(target.backing->image.image_ci.format),
            .samples = target.backing->num_samples,
            .selector = config.watch_ordinals,
            .buffer_alignment = 16,
            .max_regions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals *
                           (config.content.capture_predecessor ? 7 : 5),
            .max_bytes = PpTerminalScopeSnapshotBytes,
        });
        entry.status = entry.plan.status;
        entry.loss = entry.plan.loss;
        entry.slot = {};
        entry.recorded_plane_mask = 0;
        entry.discovered = discovered;
        entry.flip_alias = {};
        entry.flip_alias_ready = false;
        entry.lineage.Reset();
        entry.predecessor = {};
        entry.sampled_inputs = {};
        entry.upstream_inputs = {};
        entry.input_content_plan = {};
        const u64 generation = target.EnsureDiagnosticBackingGeneration();
        if (discovered && !entry.lineage.Start(link, generation)) {
            entry.status = entry.lineage.Status();
            entry.loss = entry.lineage.Loss();
        }
        if (!entry.gate.Arm(link.uid, generation)) {
            entry.status = FinalGuestSurfaceStatus::InvalidationLoss;
            entry.loss = {.invalidation = 1};
        }
    }

    void RestartEntry(Entry& entry, VideoCore::ImageColorScopePrivateLink link,
                      VideoCore::Image& target) {
        const auto slot = entry.slot;
        ConfigureEntry(entry, link, target, true);
        entry.slot = slot;
    }

    [[nodiscard]] static PpTerminalScopeContentHistoryLayout MakeHistoryLayout(
        const PpTerminalScopeContentPlan& plan,
        const PpTerminalScopeSampledInputContentPlan& input_plan, u32 plane_mask = 0) noexcept {
        PpTerminalScopeContentHistoryLayout layout{
            .region_count = plan.region_count,
            .plane_bytes = plan.plane_bytes,
            .second_plane_offset = plan.second_plane_offset,
            .consumer_plane_offset = plan.consumer_plane_offset,
            .output_plane_offset = plan.output_plane_offset,
            .pre_first_plane_offset = plan.pre_first_plane_offset,
            .predecessor_pre_plane_offset = plan.predecessor_pre_plane_offset,
            .predecessor_post_plane_offset = plan.predecessor_post_plane_offset,
            .total_bytes = input_plan.copy ? input_plan.total_bytes : plan.total_bytes,
            .input_count = input_plan.input_count,
            .input_capture_mask = input_plan.capture_mask,
            .input_unavailable_mask = input_plan.unavailable_mask,
            .plane_mask = plane_mask,
            .format = plan.format,
        };
        for (u32 index = 0; index < plan.region_count; ++index) {
            layout.regions[index] = {
                .logical_ordinal = plan.regions[index].logical_ordinal,
                .buffer_offset = plan.regions[index].buffer_offset,
                .byte_size = plan.regions[index].byte_size,
            };
        }
        for (u32 input_index = 0; input_index < input_plan.input_count; ++input_index) {
            const auto& input = input_plan.inputs[input_index];
            auto& history_input = layout.input_planes[input_index];
            history_input = {
                .region_count = input.region_count,
                .plane_offset = input.plane_offset,
                .plane_bytes = input.plane_bytes,
                .format = input.format,
                .status = input.status,
                .loss = input.loss,
            };
            for (u32 region_index = 0; region_index < input.region_count; ++region_index) {
                history_input.regions[region_index] = {
                    .logical_ordinal = input.regions[region_index].logical_ordinal,
                    .buffer_offset = input.regions[region_index].buffer_offset,
                    .byte_size = input.regions[region_index].byte_size,
                };
            }
        }
        return layout;
    }

    [[nodiscard]] HostPasses::PpUpstreamFeedbackPrePostPlan MakeUpstreamFeedbackPlan(
        const VideoCore::Image& image,
        vk::ImageAspectFlagBits copy_aspect = vk::ImageAspectFlagBits::eColor) const noexcept {
        const bool depth = copy_aspect == vk::ImageAspectFlagBits::eDepth;
        const bool color = copy_aspect == vk::ImageAspectFlagBits::eColor;
        const auto format = image.backing ? TerminalScopeFormat(image.backing->image.image_ci.format)
                                          : FinalGuestSurfaceFormat::Unsupported;
        const u32 guest_bytes_per_texel =
            !image.info.props.is_block && image.info.num_bits % 8 == 0 ? image.info.num_bits / 8 : 0;
        const u32 backing_bytes_per_texel = image.backing
                                                ? depth
                                                      ? RawTerminalScopeDepthBytesPerTexel(
                                                            image.backing->image.image_ci.format)
                                                      : RawTerminalScopeBytesPerTexel(
                                                            image.backing->image.image_ci.format)
                                                : 0;
        const u32 raw_bytes_per_texel =
            format == FinalGuestSurfaceFormat::Unsupported && backing_bytes_per_texel != 0 &&
                    guest_bytes_per_texel != 0 &&
                    guest_bytes_per_texel == backing_bytes_per_texel
                ? backing_bytes_per_texel
                : 0;
        return HostPasses::PlanPpUpstreamFeedbackPrePost({
            .enabled = config.content.capture_upstream_pre_post,
            .logical_width = mapping.logical_width,
            .logical_height = mapping.logical_height,
            .source_width = image.info.size.width,
            .source_height = image.info.size.height,
            .format = format,
            .storage_format = image.backing ? image.backing->image.image_ci.format
                                            : vk::Format::eUndefined,
            .raw_bytes_per_texel = raw_bytes_per_texel,
            .samples = image.backing ? image.backing->num_samples : 0,
            .base_mip = 0,
            .mip_count = image.info.resources.levels,
            .base_layer = 0,
            .layer_count = image.info.resources.layers,
            .color = color && image.aspect_mask == vk::ImageAspectFlagBits::eColor,
            .depth = depth && static_cast<bool>(image.aspect_mask & vk::ImageAspectFlagBits::eDepth),
            .type_2d = image.info.type == AmdGpu::ImageType::Color2D ||
                       image.info.type == AmdGpu::ImageType::Color2DArray,
            .uniform_state = image.backing && image.backing->subresource_states.empty(),
            .selector = config.watch_ordinals,
            .candidate_capacity = HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates,
            .buffer_alignment = 16,
            .max_regions = HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates * 2 *
                           FinalGuestSurfaceWatchOrdinals::MaxOrdinals,
            .max_bytes = config.content.capture_upstream_sample_tee
                             ? HostPasses::PpUpstreamSampleTeeDownloadPrefixBytes
                             : PpTerminalScopeSnapshotBytes,
        });
    }

    [[nodiscard]] bool CaptureUpstreamSampleTeeScopeWriterTracePlane(
        VideoCore::Image& image, const RenderState& state, u64 rendering_serial,
        const VideoCore::SubresourceRange& range,
        const auto& intent,
        const HostPasses::PpUpstreamInputContentPlane& plane) {
        if (!upstream_feedback_slot || !scheduler.IsRendering() || !image.backing ||
            plane.status != FinalGuestSurfaceStatus::Complete || plane.loss.Any() ||
            plane.region_count == 0 || plane.region_count > plane.regions.size() ||
            plane.plane_bytes == 0) {
            return false;
        }
        const auto aspect_restore = HostPasses::PlanPpUpstreamSampleTeeDepthCopyAspectRestore(
            static_cast<u32>(static_cast<VkImageAspectFlags>(image.aspect_mask)),
            static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT));
        const u64 absolute_slot = static_cast<u64>(upstream_feedback_slot.slot) * slot_stride;
        if (!aspect_restore.exact ||
            absolute_slot + plane.plane_offset + plane.plane_bytes >
                std::numeric_limits<u32>::max()) {
            return false;
        }
        const HostPasses::PpUpstreamSampleTeeBarrierPlan pre_barrier{
            .src_stages = HostPasses::PpUpstreamSampleTeeStage::AllCommands |
                          HostPasses::PpUpstreamSampleTeeStage::Host,
            .src_access = HostPasses::PpUpstreamSampleTeeAccess::MemoryWrite |
                          HostPasses::PpUpstreamSampleTeeAccess::HostRead,
            .dst_stages = HostPasses::PpUpstreamSampleTeeStage::Transfer,
            .dst_access = HostPasses::PpUpstreamSampleTeeAccess::TransferWrite,
            .offset = static_cast<u32>(absolute_slot + plane.plane_offset),
            .bytes = plane.plane_bytes,
        };
        const HostPasses::PpUpstreamSampleTeeBarrierPlan host_read{
            .src_stages = HostPasses::PpUpstreamSampleTeeStage::Transfer,
            .src_access = HostPasses::PpUpstreamSampleTeeAccess::TransferWrite,
            .dst_stages = HostPasses::PpUpstreamSampleTeeStage::Host,
            .dst_access = HostPasses::PpUpstreamSampleTeeAccess::HostRead,
            .offset = static_cast<u32>(absolute_slot + plane.plane_offset),
            .bytes = plane.plane_bytes,
        };
        const auto mapped_pre = HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(pre_barrier);
        const auto mapped_host = HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(host_read);
        if (!mapped_pre.valid || !mapped_host.valid ||
            image.backing->subresource_states.size() != 0 ||
            image.backing->state.layout == vk::ImageLayout::eUndefined) {
            return false;
        }
        const auto old_state = image.backing->state;
        const auto cmdbuf = scheduler.CommandBuffer();
        scheduler.EndRendering();
        const vk::BufferMemoryBarrier2 buffer_pre{
            .srcStageMask = mapped_pre.src_stages,
            .srcAccessMask = mapped_pre.src_access,
            .dstStageMask = mapped_pre.dst_stages,
            .dstAccessMask = mapped_pre.dst_access,
            .buffer = download.Handle(),
            .offset = mapped_pre.offset,
            .size = mapped_pre.bytes,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &buffer_pre,
        });
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                      range, cmdbuf);
        auto* resolved = texture_cache.ResolveDiagnosticColorScopeImage(intent.link);
        const bool live =
            resolved == &image && resolved->backing &&
            resolved->backing->subresource_states.empty() && resolved->backing->num_samples == 1 &&
            bool{resolved->aspect_mask & vk::ImageAspectFlagBits::eDepth} &&
            resolved->EnsureDiagnosticBackingGeneration() == intent.backing_generation &&
            resolved->DiagnosticMutationEpoch() == intent.mutation_epoch &&
            resolved->image_uid == intent.link.uid;
        if (live) {
            std::array<vk::BufferImageCopy, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> copies{};
            for (u32 index = 0; index < plane.region_count; ++index) {
                const auto& region = plane.regions[index];
                copies[index] = {
                    .bufferOffset = absolute_slot + plane.plane_offset + region.buffer_offset,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                         .mipLevel = region.mip_level,
                                         .baseArrayLayer = plane.base_layer,
                                         .layerCount = 1},
                    .imageOffset = {static_cast<s32>(region.x), static_cast<s32>(region.y), 0},
                    .imageExtent = {region.width, region.height, 1},
                };
            }
            cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     download.Handle(), plane.region_count, copies.data());
        }
        const vk::ImageMemoryBarrier2 restore{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = old_state.pl_stage,
            .dstAccessMask = old_state.access_mask,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = old_state.layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.GetImage(),
            .subresourceRange = {.aspectMask = vk::ImageAspectFlags{static_cast<VkImageAspectFlags>(
                                     aspect_restore.restore_aspect)},
                                 .baseMipLevel = range.base.level,
                                 .levelCount = range.extent.levels,
                                 .baseArrayLayer = range.base.layer,
                                 .layerCount = range.extent.layers},
        };
        const vk::BufferMemoryBarrier2 buffer_host{
            .srcStageMask = mapped_host.src_stages,
            .srcAccessMask = mapped_host.src_access,
            .dstStageMask = mapped_host.dst_stages,
            .dstAccessMask = mapped_host.dst_access,
            .buffer = download.Handle(),
            .offset = mapped_host.offset,
            .size = mapped_host.bytes,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &buffer_host,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &restore,
        });
        HostPasses::RestorePpUpstreamInputUniformTracker(
            image.backing->state, image.backing->subresource_states, old_state);
        return live && scheduler.ResumeRenderingForDiagnostic(state, rendering_serial);
    }

public:
    void ObserveUpstreamSampleTeePredecessorWriterPreDepth(
        VideoCore::ImageId image_id, VideoCore::Image& image, const RenderState& state,
        u64 rendering_serial, const PpTerminalScopeDrawSelector& draw,
        u64 fragment_pgm_hash, const VideoCore::SubresourceRange& range,
        bool effective_depth_write,
        const HostPasses::PpUpstreamSampleTeeProducerPreDepthHookDescriptor&
            generic_hook_descriptor,
        const HostPasses::PpUpstreamSampleTeeProducerPreDepthHookPlan& generic_hook_plan,
        const HostPasses::
            PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingDescriptor&
                occurrence_ordering_descriptor,
        const HostPasses::PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingPlan&
            occurrence_ordering_plan,
        const VideoCore::ImageDepthWriterShape& observed_shape,
        u32 observed_key_occurrence) {
        const VideoCore::ImageColorScopePrivateLink link{.id = image_id, .uid = image.image_uid};
        const auto intent = BuildDepthPredecessorIntent(
            link, image, range, draw, fragment_pgm_hash, effective_depth_write);
        const bool current = image.backing &&
                             image.EnsureDiagnosticBackingGeneration() == intent.backing_generation &&
                             image.DiagnosticMutationEpoch() == intent.mutation_epoch;
        if (upstream_sample_tee_scope_writer_trace_frame.exact) {
            const auto trace_capture =
                HostPasses::PlanPpUpstreamSampleTeeScopeWriterTraceCapture({
                    .frame = upstream_sample_tee_scope_writer_trace_frame,
                    .observed_hint = intent.hint,
                    .observed_shape = observed_shape,
                    .observed_key_occurrence = observed_key_occurrence,
                    .sample_tee_enabled = config.content.capture_upstream_sample_tee,
                    .upstream_feedback_armed = upstream_feedback_armed,
                    .pre_draw = true,
                    .effective_depth_write = effective_depth_write,
                    .actual_depth_aspect =
                        bool{image.aspect_mask & vk::ImageAspectFlagBits::eDepth},
                    .canonical_range = HostPasses::IsCanonicalPpDepthPredecessorRange(range),
                    .source_current = current,
                    .already_captured = upstream_sample_tee_scope_writer_trace_captured,
                });
            if (trace_capture.status != FinalGuestSurfaceStatus::AlreadyConsumed) {
                if (upstream_sample_tee_scope_writer_trace_capture_attempts !=
                    std::numeric_limits<u32>::max()) {
                    ++upstream_sample_tee_scope_writer_trace_capture_attempts;
                }
                upstream_sample_tee_scope_writer_trace_captured = true;
                if (!trace_capture.capture_pre || !trace_capture.capture_post ||
                    trace_capture.status != FinalGuestSurfaceStatus::Complete ||
                    trace_capture.loss.Any()) {
                    upstream_sample_tee_scope_writer_trace_tail = {
                        .status = trace_capture.status,
                        .loss = trace_capture.loss,
                    };
                    upstream_sample_tee_scope_writer_trace_open = false;
                    return;
                }
                if (!upstream_feedback_slot) {
                    const auto token = slots.TryAcquire();
                    if (!token) {
                        upstream_sample_tee_scope_writer_trace_tail = {
                            .status = FinalGuestSurfaceStatus::BusyLoss,
                            .loss = {.busy = 1},
                        };
                        return;
                    }
                    upstream_feedback_slot = *token;
                    upstream_feedback_base_prefix_initialized = false;
                }
                const bool source_2d = image.info.type == AmdGpu::ImageType::Color2D ||
                                       image.info.type == AmdGpu::ImageType::Color2DArray;
                const u32 bytes_per_texel =
                    RawTerminalScopeDepthBytesPerTexel(image.info.pixel_format);
                const std::array view{HostPasses::PpUpstreamInputContentView{
                    .width = image.info.size.width,
                    .height = image.info.size.height,
                    .bytes_per_texel = bytes_per_texel,
                    .base_mip = 0,
                    .mip_count = 1,
                    .base_layer = 0,
                    .layer_count = 1,
                    .color = true,
                    .type_2d = source_2d,
                    .uniform_state = image.backing && image.backing->subresource_states.empty(),
                }};
                const auto mapped = HostPasses::PlanPpUpstreamInputContent({
                    .enabled = true,
                    .logical_width = mapping.logical_width,
                    .logical_height = mapping.logical_height,
                    .base_offset = 0,
                    .selector = config.watch_ordinals,
                    .views = view,
                    .buffer_alignment = 16,
                    .max_regions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals,
                    .max_bytes =
                        HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthTailBytes,
                });
                auto tail = HostPasses::PlanPpUpstreamSampleTeeScopeWriterTraceTail({
                    .frame = upstream_sample_tee_scope_writer_trace_frame,
                    .plane = mapped.input_count == 1
                                 ? mapped.inputs[0]
                                 : HostPasses::PpUpstreamInputContentPlane{},
                    .tail_offset =
                        HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthTailOffset,
                    .tail_bytes =
                        HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthTailBytes,
                    .alignment = 16,
                });
                const u64 pre_host_instance = scheduler.HostRenderingInstanceId();
                if (tail.status == FinalGuestSurfaceStatus::Complete &&
                    !CaptureUpstreamSampleTeeScopeWriterTracePlane(
                        image, state, rendering_serial, range, intent, tail.pre_plane)) {
                    tail = {
                        .status = FinalGuestSurfaceStatus::InvalidationLoss,
                        .loss = {.invalidation = 1},
                    };
                }
                upstream_sample_tee_scope_writer_trace_tail = tail;
                if (tail.status == FinalGuestSurfaceStatus::Complete && !tail.loss.Any()) {
                    if (upstream_sample_tee_scope_writer_trace_pre_complete !=
                        std::numeric_limits<u32>::max()) {
                        ++upstream_sample_tee_scope_writer_trace_pre_complete;
                    }
                    upstream_sample_tee_scope_writer_trace_intent = intent;
                    upstream_sample_tee_scope_writer_trace_expected_serial = rendering_serial;
                    upstream_sample_tee_scope_writer_trace_host_instance = pre_host_instance;
                    upstream_sample_tee_scope_writer_trace_open = true;
                }
                return;
            }
        }
        if (upstream_sample_tee_scope_writer_trace_frame.exact) {
            return;
        }
        if (!config.content.capture_upstream_sample_tee ||
            !upstream_sample_tee_producer_pre_ordinal_arm.valid) {
            return;
        }
        const HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthHookDescriptor descriptor{
            .enabled = true,
            .sample_tee_enabled = true,
            .capture_upstream_depth_pre_post = config.content.capture_upstream_depth_pre_post,
            .predecessor_hint_armed = false,
            .pre_draw = true,
            .learned_hint = {},
            .observed_writer = intent,
            .effective_depth_write = effective_depth_write,
            .actual_depth_aspect = bool{image.aspect_mask & vk::ImageAspectFlagBits::eDepth},
            .canonical_range = HostPasses::IsCanonicalPpDepthPredecessorRange(range),
            .source_still_live = current,
            .source_identity_exact = current,
            .source_backing_exact = current,
            .source_mutation_exact = current,
            .current_target_armed = upstream_feedback_armed,
            .current_target_observed = upstream_sample_tee_selected_consumer_count != 0,
        };
        const auto hook =
            HostPasses::PlanPpUpstreamSampleTeePredecessorWriterPreDepthHook(descriptor);
        const HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCaptureDescriptor
            ordinal_capture_descriptor{
                .arm = upstream_sample_tee_producer_pre_ordinal_arm,
                .hook_descriptor = descriptor,
                .hook_plan = hook,
                .first_writer_pre =
                    BuildDepthFirstWriterPreObservation(image, rendering_serial),
            };
        const auto ordinal_capture = HostPasses::
            PlanPpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCapture(
                ordinal_capture_descriptor);
        if (!ordinal_capture.capture_pre) {
            if (ordinal_capture.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                ordinal_capture.loss.Any()) {
                upstream_sample_tee_predecessor_writer_pre_depth = {
                    .status = ordinal_capture.status,
                    .loss = ordinal_capture.loss,
                };
            }
            return;
        }
        upstream_sample_tee_route_absent_ordinal_capture_evidence.ObservePredecessor({
            .generic_hook_descriptor = generic_hook_descriptor,
            .generic_hook_plan = generic_hook_plan,
            .ordinal_capture_descriptor = ordinal_capture_descriptor,
            .ordinal_capture_plan = ordinal_capture,
            .ordering_descriptor = occurrence_ordering_descriptor,
            .ordering_plan = occurrence_ordering_plan,
        });
        const auto collision = HostPasses::ObservePpUpstreamSampleTeeProducerPreDepthCollision({
            .open = upstream_sample_tee_predecessor_writer_pre_open,
            .open_writer = upstream_sample_tee_predecessor_writer_pre_intent,
            .incoming_writer = intent,
            .prior_attempted = upstream_sample_tee_predecessor_writer_pre_attempted,
            .prior_bound = upstream_sample_tee_predecessor_writer_pre_bound,
        });
        if (!collision.accept_incoming) {
            upstream_sample_tee_predecessor_writer_pre_collision = collision.collision;
            if (collision.clear_prior_binding) {
                upstream_sample_tee_predecessor_writer_pre_bound = false;
            }
            return;
        }
        upstream_sample_tee_predecessor_writer_pre_attempted = true;
        upstream_sample_tee_predecessor_writer_pre_intent = intent;
        upstream_sample_tee_predecessor_writer_pre_ordinal_capture_descriptor =
            ordinal_capture_descriptor;
        upstream_sample_tee_predecessor_writer_pre_ordinal_capture_plan = ordinal_capture;
        if (!upstream_feedback_slot) {
            const auto token = slots.TryAcquire();
            if (!token) {
                upstream_sample_tee_predecessor_writer_pre_depth = {
                    .status = FinalGuestSurfaceStatus::BusyLoss,
                    .loss = {.busy = 1},
                };
                return;
            }
            upstream_feedback_slot = *token;
            upstream_feedback_base_prefix_initialized = false;
        }
        const bool source_2d = image.info.type == AmdGpu::ImageType::Color2D ||
                               image.info.type == AmdGpu::ImageType::Color2DArray;
        const u32 bytes_per_texel = RawTerminalScopeDepthBytesPerTexel(image.info.pixel_format);
        const std::array view{HostPasses::PpUpstreamInputContentView{
            .width = image.info.size.width,
            .height = image.info.size.height,
            .bytes_per_texel = bytes_per_texel,
            .base_mip = 0,
            .mip_count = 1,
            .base_layer = 0,
            .layer_count = 1,
            .color = true,
            .type_2d = source_2d,
            .uniform_state = image.backing && image.backing->subresource_states.empty(),
        }};
        const auto mapped = HostPasses::PlanPpUpstreamInputContent({
            .enabled = true,
            .logical_width = mapping.logical_width,
            .logical_height = mapping.logical_height,
            .base_offset = HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthTailOffset,
            .selector = config.watch_ordinals,
            .views = view,
            .buffer_alignment = 16,
            .max_regions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals,
            .max_bytes = HostPasses::PpUpstreamSampleTeeProducerPreDepthTailOffset,
        });
        auto tail = HostPasses::PlanPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTail({
            .enabled = true,
            .slot_present = static_cast<bool>(upstream_feedback_slot),
            .slot = upstream_feedback_slot.slot,
            .slot_stride = HostPasses::PpUpstreamInputSnapshotBytes,
            .reserved_tail_bytes =
                HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthTailBytes,
            .buffer_alignment = 16,
            .plane = mapped.input_count == 1 ? mapped.inputs[0]
                                             : HostPasses::PpUpstreamInputContentPlane{},
            .source_2d = source_2d,
            .source_samples = image.backing ? image.backing->num_samples : 0,
            .depth_format_supported = bytes_per_texel == 4,
            .barrier_supported = image.backing && image.backing->subresource_states.empty() &&
                                 image.backing->state.layout != vk::ImageLayout::eUndefined,
            .deferred_shared_release = true,
        });
        if (!HostPasses::
                IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(tail) ||
            !scheduler.IsRendering()) {
            upstream_sample_tee_predecessor_writer_pre_depth =
                tail.status == FinalGuestSurfaceStatus::Complete
                    ? HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan{
                          .status = FinalGuestSurfaceStatus::InvalidationLoss,
                          .loss = {.invalidation = 1}}
                    : tail;
            return;
        }
        const auto aspect_restore = HostPasses::PlanPpUpstreamSampleTeeDepthCopyAspectRestore(
            static_cast<u32>(static_cast<VkImageAspectFlags>(image.aspect_mask)),
            static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT));
        const u64 absolute_slot = static_cast<u64>(upstream_feedback_slot.slot) * slot_stride;
        if (!aspect_restore.exact ||
            absolute_slot + tail.plane.plane_offset + tail.plane.plane_bytes >
                std::numeric_limits<u32>::max()) {
            upstream_sample_tee_predecessor_writer_pre_depth = {
                .status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1},
            };
            return;
        }
        const HostPasses::PpUpstreamSampleTeeBarrierPlan pre_barrier{
            .src_stages = HostPasses::PpUpstreamSampleTeeStage::AllCommands |
                          HostPasses::PpUpstreamSampleTeeStage::Host,
            .src_access = HostPasses::PpUpstreamSampleTeeAccess::MemoryWrite |
                          HostPasses::PpUpstreamSampleTeeAccess::HostRead,
            .dst_stages = HostPasses::PpUpstreamSampleTeeStage::Transfer,
            .dst_access = HostPasses::PpUpstreamSampleTeeAccess::TransferWrite,
            .offset = static_cast<u32>(absolute_slot + tail.plane.plane_offset),
            .bytes = tail.plane.plane_bytes,
        };
        const HostPasses::PpUpstreamSampleTeeBarrierPlan host_read{
            .src_stages = HostPasses::PpUpstreamSampleTeeStage::Transfer,
            .src_access = HostPasses::PpUpstreamSampleTeeAccess::TransferWrite,
            .dst_stages = HostPasses::PpUpstreamSampleTeeStage::Host,
            .dst_access = HostPasses::PpUpstreamSampleTeeAccess::HostRead,
            .offset = static_cast<u32>(absolute_slot + tail.plane.plane_offset),
            .bytes = tail.plane.plane_bytes,
        };
        const auto mapped_pre = HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(pre_barrier);
        const auto mapped_host = HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(host_read);
        if (!mapped_pre.valid || !mapped_host.valid) {
            upstream_sample_tee_predecessor_writer_pre_depth = {
                .status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1},
            };
            return;
        }
        const auto old_state = image.backing->state;
        const u64 pre_host_instance = scheduler.HostRenderingInstanceId();
        const auto cmdbuf = scheduler.CommandBuffer();
        scheduler.EndRendering();
        const vk::BufferMemoryBarrier2 buffer_pre{
            .srcStageMask = mapped_pre.src_stages,
            .srcAccessMask = mapped_pre.src_access,
            .dstStageMask = mapped_pre.dst_stages,
            .dstAccessMask = mapped_pre.dst_access,
            .buffer = download.Handle(),
            .offset = mapped_pre.offset,
            .size = mapped_pre.bytes,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &buffer_pre,
        });
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                      range, cmdbuf);
        auto* resolved = texture_cache.ResolveDiagnosticColorScopeImage(intent.link);
        const bool live =
            resolved == &image && resolved->backing &&
            resolved->backing->subresource_states.empty() && resolved->backing->num_samples == 1 &&
            bool{resolved->aspect_mask & vk::ImageAspectFlagBits::eDepth} &&
            resolved->EnsureDiagnosticBackingGeneration() == intent.backing_generation &&
            resolved->DiagnosticMutationEpoch() == intent.mutation_epoch &&
            resolved->image_uid == intent.link.uid;
        if (live) {
            std::array<vk::BufferImageCopy, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> copies{};
            for (u32 index = 0; index < tail.plane.region_count; ++index) {
                const auto& region = tail.plane.regions[index];
                copies[index] = {
                    .bufferOffset = absolute_slot + tail.plane.plane_offset + region.buffer_offset,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                         .mipLevel = region.mip_level,
                                         .baseArrayLayer = tail.plane.base_layer,
                                         .layerCount = 1},
                    .imageOffset = {static_cast<s32>(region.x), static_cast<s32>(region.y), 0},
                    .imageExtent = {region.width, region.height, 1},
                };
            }
            cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     download.Handle(), tail.plane.region_count, copies.data());
        } else {
            tail = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        const vk::ImageMemoryBarrier2 restore{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = old_state.pl_stage,
            .dstAccessMask = old_state.access_mask,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = old_state.layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.GetImage(),
            .subresourceRange = {.aspectMask = vk::ImageAspectFlags{static_cast<VkImageAspectFlags>(
                                     aspect_restore.restore_aspect)},
                                 .baseMipLevel = range.base.level,
                                 .levelCount = range.extent.levels,
                                 .baseArrayLayer = range.base.layer,
                                 .layerCount = range.extent.layers},
        };
        const vk::BufferMemoryBarrier2 buffer_host{
            .srcStageMask = mapped_host.src_stages,
            .srcAccessMask = mapped_host.src_access,
            .dstStageMask = mapped_host.dst_stages,
            .dstAccessMask = mapped_host.dst_access,
            .buffer = download.Handle(),
            .offset = mapped_host.offset,
            .size = mapped_host.bytes,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &buffer_host,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &restore,
        });
        HostPasses::RestorePpUpstreamInputUniformTracker(
            image.backing->state, image.backing->subresource_states, old_state);
        if (!scheduler.ResumeRenderingForDiagnostic(state, rendering_serial)) {
            tail = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        upstream_sample_tee_predecessor_writer_pre_depth = tail;
        if (tail.status == FinalGuestSurfaceStatus::Complete) {
            upstream_sample_tee_predecessor_writer_pre_expected_serial = rendering_serial;
            upstream_sample_tee_predecessor_writer_pre_host_instance = pre_host_instance;
            upstream_sample_tee_predecessor_writer_pre_open = true;
        }
    }

    void ObserveUpstreamSampleTeeProducerPreDepth(
        VideoCore::ImageId image_id, VideoCore::Image& image, const RenderState& state,
        u64 rendering_serial, const VideoCore::ImageColorScopeDrawDescriptor& draw,
        const VideoCore::SubresourceRange& range,
        bool effective_depth_write) {
        const PpTerminalScopeDrawSelector observed{
            .kind = draw.kind,
            .indexed = draw.indexed,
            .element_count = draw.element_count,
            .instance_count = draw.instance_count,
            .sampled_images = draw.sampled_images,
            .storage_writes = draw.storage_writes,
        };
        const auto observed_shape = VideoCore::MakeImageDepthWriterShape(draw);
        const VideoCore::ImageColorScopePrivateLink link{.id = image_id, .uid = image.image_uid};
        const auto intent = BuildDepthPredecessorIntent(
            link, image, range, observed, draw.fragment_pgm_hash, effective_depth_write);
        const auto immediate_predecessor =
            BuildCurrentDepthPredecessorEndpoint(image_id, image, range);
        const auto first_writer_pre =
            BuildDepthFirstWriterPreObservation(image, rendering_serial);
        const u64 target_pre_host_instance = scheduler.HostRenderingInstanceId();
        if (!config.content.capture_upstream_sample_tee) {
            return;
        }
        const bool current = image.backing &&
                             image.EnsureDiagnosticBackingGeneration() == intent.backing_generation &&
                             image.DiagnosticMutationEpoch() == intent.mutation_epoch;
        const HostPasses::PpUpstreamSampleTeeProducerPreDepthHookDescriptor hook_descriptor{
            .enabled = true,
            .sample_tee_enabled = true,
            .capture_upstream_depth_pre_post = config.content.capture_upstream_depth_pre_post,
            .upstream_feedback_armed = upstream_feedback_armed,
            .pre_draw = true,
            .learned_hint = upstream_sample_tee_producer_pre_hint,
            .observed_writer = intent,
            .effective_depth_write = effective_depth_write,
            .actual_depth_aspect = bool{image.aspect_mask & vk::ImageAspectFlagBits::eDepth},
            .canonical_range = HostPasses::IsCanonicalPpDepthPredecessorRange(range),
            .source_still_live = current,
            .source_identity_exact = current,
            .source_backing_exact = current,
            .source_mutation_exact = current,
        };
        const auto hook =
            HostPasses::PlanPpUpstreamSampleTeeProducerPreDepthHook(hook_descriptor);
        using OccurrenceOrderingStage = HostPasses::
            PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingStage;
        constexpr std::array occurrence_ordering_stages{
            OccurrenceOrderingStage::SnapshotPredecessorAndFirstWriter,
            OccurrenceOrderingStage::GenericHook,
            OccurrenceOrderingStage::GenericEvidence,
            OccurrenceOrderingStage::OccurrencePre,
            OccurrenceOrderingStage::PredecessorOrdinalGateAndCopy,
            OccurrenceOrderingStage::TargetOrdinalGateAndCopy,
            OccurrenceOrderingStage::Draw,
            OccurrenceOrderingStage::Mark,
            OccurrenceOrderingStage::OccurrencePost,
        };
        u32 observed_key_occurrence{};
        if (upstream_sample_tee_scope_writer_trace_frame.exact &&
            HostPasses::MatchesPpUpstreamSampleTeeScopeWriterTraceKey(
                upstream_sample_tee_scope_writer_trace_frame.writer_shape,
                observed_shape)) {
            observed_key_occurrence =
                upstream_sample_tee_scope_writer_trace_key_occurrence_count;
            if (upstream_sample_tee_scope_writer_trace_key_occurrence_count !=
                std::numeric_limits<u32>::max()) {
                ++upstream_sample_tee_scope_writer_trace_key_occurrence_count;
            }
        }
        const bool scope_writer_trace_match =
            HostPasses::ClassifyPpUpstreamSampleTeeScopeWriterTraceMatch({
                .frame = upstream_sample_tee_scope_writer_trace_frame,
                .observed_hint = intent.hint,
                .observed_shape = observed_shape,
                .observed_key_occurrence = observed_key_occurrence,
            }) == HostPasses::PpUpstreamSampleTeeScopeWriterTraceMatch::Capture;
        if (upstream_sample_tee_scope_writer_trace_frame.exact &&
            upstream_sample_tee_scope_writer_trace_live_pre_draws !=
                std::numeric_limits<u32>::max()) {
            ++upstream_sample_tee_scope_writer_trace_live_pre_draws;
        }
        if (scope_writer_trace_match) {
            if (upstream_sample_tee_scope_writer_trace_key_matches !=
                std::numeric_limits<u32>::max()) {
                ++upstream_sample_tee_scope_writer_trace_key_matches;
            }
            if (upstream_feedback_armed &&
                upstream_sample_tee_scope_writer_trace_armed_key_matches !=
                    std::numeric_limits<u32>::max()) {
                ++upstream_sample_tee_scope_writer_trace_armed_key_matches;
            }
        }
        const bool scope_writer_trace_copy =
            scope_writer_trace_match && !upstream_sample_tee_scope_writer_trace_captured;
        const bool predecessor_ordinal_copy =
            !upstream_sample_tee_scope_writer_trace_frame.exact &&
            upstream_sample_tee_producer_pre_ordinal_arm.valid &&
            first_writer_pre.depth_writer_count ==
                upstream_sample_tee_producer_pre_ordinal_arm.predecessor_scope_writer_count;
        const bool target_ordinal_copy =
            !upstream_sample_tee_scope_writer_trace_frame.exact &&
            upstream_sample_tee_producer_pre_ordinal_arm.valid &&
            first_writer_pre.depth_writer_count ==
                upstream_sample_tee_producer_pre_ordinal_arm.target_scope_writer_count;
        const u32 projected_copy_count = static_cast<u32>(scope_writer_trace_copy) +
                                         static_cast<u32>(predecessor_ordinal_copy) +
                                         static_cast<u32>(target_ordinal_copy);
        u64 projected_post_host_instance = target_pre_host_instance;
        for (u32 index = 0;
             index < projected_copy_count &&
             projected_post_host_instance != std::numeric_limits<u64>::max();
             ++index) {
            ++projected_post_host_instance;
        }
        const HostPasses::
            PpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrderingDescriptor
                occurrence_ordering_descriptor{
                .stages = occurrence_ordering_stages,
                .pre_host_rendering_instance_id = target_pre_host_instance,
                .post_host_rendering_instance_id = projected_post_host_instance,
                .predecessor_ordinal_copy = predecessor_ordinal_copy,
                .target_ordinal_copy = target_ordinal_copy,
            };
        const auto occurrence_ordering = HostPasses::
            PlanPpUpstreamSampleTeeProducerPreDepthOrdinalOccurrenceOrdering(
                occurrence_ordering_descriptor);
        upstream_sample_tee_producer_pre_gate_evidence.ObserveProducerPreDraw(
            HostPasses::AdaptPpUpstreamSampleTeeProducerPreDepthPreDrawEvidence(
                hook_descriptor, hook));
        upstream_sample_tee_producer_pre_occurrences.ObservePre({
            .hook_descriptor = hook_descriptor,
            .hook_plan = hook,
            .guest_rendering_serial = rendering_serial,
            .host_rendering_instance_id = target_pre_host_instance,
            .immediately_before_draw = occurrence_ordering.exact,
            .private_immediate_predecessor_endpoint = immediate_predecessor,
            .first_writer_pre = first_writer_pre,
        });
        upstream_sample_tee_scope_load_candidate_evidence.ObserveTarget({
            .target_guest_rendering_serial = rendering_serial,
            .target_pre_draw_host_rendering_instance_id =
                scheduler.HostRenderingInstanceId(),
            .hook_descriptor = hook_descriptor,
            .hook_plan = hook,
            .live_target_intent = intent,
            .first_writer_pre = first_writer_pre,
        });
        ObserveUpstreamSampleTeePredecessorWriterPreDepth(
            image_id, image, state, rendering_serial, observed, draw.fragment_pgm_hash, range,
            effective_depth_write, hook_descriptor, hook, occurrence_ordering_descriptor,
            occurrence_ordering, observed_shape, observed_key_occurrence);
        if (upstream_sample_tee_scope_writer_trace_frame.exact) {
            return;
        }
        const bool pair_ordinal_capture = upstream_sample_tee_producer_pre_ordinal_arm.valid;
        const auto& selected_producer_capture_arm =
            pair_ordinal_capture ? upstream_sample_tee_producer_pre_ordinal_arm
                                 : upstream_sample_tee_selected_producer_ordinal_arm;
        if (!selected_producer_capture_arm.valid) {
            return;
        }
        auto ordinal_hook_descriptor = hook_descriptor;
        ordinal_hook_descriptor.learned_hint =
            selected_producer_capture_arm.target_hint;
        const auto ordinal_hook =
            HostPasses::PlanPpUpstreamSampleTeeProducerPreDepthHook(
                ordinal_hook_descriptor);
        const HostPasses::PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureDescriptor
            ordinal_capture_descriptor{
                .arm = selected_producer_capture_arm,
                .hook_descriptor = ordinal_hook_descriptor,
                .hook_plan = ordinal_hook,
                .first_writer_pre = first_writer_pre,
            };
        HostPasses::PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan ordinal_capture{};
        if (pair_ordinal_capture) {
            ordinal_capture = HostPasses::PlanPpUpstreamSampleTeeProducerPreDepthOrdinalCapture(
                ordinal_capture_descriptor);
        } else {
            const bool matching_occurrence = HostPasses::
                IsPpUpstreamSampleTeeProducerPreDepthSelectedProducerOccurrenceMatch(
                    selected_producer_capture_arm, ordinal_hook_descriptor);
            const u32 observed_occurrence_index =
                upstream_sample_tee_selected_producer_occurrence_count;
            if (matching_occurrence &&
                upstream_sample_tee_selected_producer_occurrence_count !=
                    std::numeric_limits<u32>::max()) {
                ++upstream_sample_tee_selected_producer_occurrence_count;
            }
            ordinal_capture = HostPasses::
                PlanPpUpstreamSampleTeeProducerPreDepthSelectedProducerOccurrenceCapture({
                    .arm = selected_producer_capture_arm,
                    .hook_descriptor = ordinal_hook_descriptor,
                    .hook_plan = ordinal_hook,
                    .observed_occurrence_index = observed_occurrence_index,
                });
        }
        if (!ordinal_capture.capture_pre) {
            if (ordinal_capture.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
                ordinal_capture.loss.Any()) {
                upstream_sample_tee_producer_pre_depth = {
                    .status = ordinal_capture.status,
                    .loss = ordinal_capture.loss,
                };
            }
            return;
        }
        if (pair_ordinal_capture) {
            upstream_sample_tee_route_absent_ordinal_capture_evidence.ObserveTarget({
                .generic_hook_descriptor = hook_descriptor,
                .generic_hook_plan = hook,
                .ordinal_capture_descriptor = ordinal_capture_descriptor,
                .ordinal_capture_plan = ordinal_capture,
                .ordering_descriptor = occurrence_ordering_descriptor,
                .ordering_plan = occurrence_ordering,
                .predecessor_post_host_rendering_instance_id =
                    upstream_sample_tee_predecessor_writer_pre_post.exact
                        ? upstream_sample_tee_predecessor_writer_pre_post.output
                              .host_rendering_instance_id
                        : 0,
            });
        }
        const auto collision = HostPasses::ObservePpUpstreamSampleTeeProducerPreDepthCollision({
            .open = upstream_sample_tee_producer_pre_open,
            .open_writer = upstream_sample_tee_producer_pre_intent,
            .incoming_writer = intent,
            .prior_attempted = upstream_sample_tee_producer_pre_attempted,
            .prior_bound = upstream_sample_tee_producer_pre_bound,
        });
        if (!collision.accept_incoming) {
            upstream_sample_tee_producer_pre_collision = collision.collision;
            if (collision.clear_prior_binding) {
                upstream_sample_tee_producer_pre_bound = false;
            }
            return;
        }
        upstream_sample_tee_producer_pre_attempted = true;
        upstream_sample_tee_producer_pre_intent = intent;
        upstream_sample_tee_producer_pre_ordinal_capture_descriptor =
            ordinal_capture_descriptor;
        upstream_sample_tee_producer_pre_ordinal_capture_plan = ordinal_capture;
        upstream_sample_tee_producer_pre_diagnostic_immediate_predecessor =
            immediate_predecessor;
        if (!upstream_feedback_slot) {
            const auto token = slots.TryAcquire();
            if (!token) {
                upstream_sample_tee_producer_pre_depth = {
                    .status = FinalGuestSurfaceStatus::BusyLoss,
                    .loss = {.busy = 1},
                };
                return;
            }
            upstream_feedback_slot = *token;
            upstream_feedback_base_prefix_initialized = false;
        }
        const bool source_2d = image.info.type == AmdGpu::ImageType::Color2D ||
                               image.info.type == AmdGpu::ImageType::Color2DArray;
        const u32 bytes_per_texel = RawTerminalScopeDepthBytesPerTexel(image.info.pixel_format);
        const std::array view{HostPasses::PpUpstreamInputContentView{
            .width = image.info.size.width,
            .height = image.info.size.height,
            .bytes_per_texel = bytes_per_texel,
            .base_mip = 0,
            .mip_count = 1,
            .base_layer = 0,
            .layer_count = 1,
            .color = true,
            .type_2d = source_2d,
            .uniform_state = image.backing && image.backing->subresource_states.empty(),
        }};
        const auto mapped = HostPasses::PlanPpUpstreamInputContent({
            .enabled = true,
            .logical_width = mapping.logical_width,
            .logical_height = mapping.logical_height,
            .base_offset = HostPasses::PpUpstreamSampleTeeProducerPreDepthPrefixBytes,
            .selector = config.watch_ordinals,
            .views = view,
            .buffer_alignment = 16,
            .max_regions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals,
            .max_bytes = HostPasses::PpUpstreamInputSnapshotBytes,
        });
        auto tail = HostPasses::PlanPpUpstreamSampleTeeProducerPreDepthDownloadTail({
            .enabled = true,
            .slot_present = static_cast<bool>(upstream_feedback_slot),
            .slot = upstream_feedback_slot.slot,
            .slot_stride = HostPasses::PpUpstreamInputSnapshotBytes,
            .reserved_tail_bytes = HostPasses::PpUpstreamSampleTeeProducerPreDepthTailBytes,
            .buffer_alignment = 16,
            .plane = mapped.input_count == 1 ? mapped.inputs[0]
                                             : HostPasses::PpUpstreamInputContentPlane{},
            .source_2d = source_2d,
            .source_samples = image.backing ? image.backing->num_samples : 0,
            .depth_format_supported = bytes_per_texel == 4,
            .barrier_supported = image.backing && image.backing->subresource_states.empty() &&
                                 image.backing->state.layout != vk::ImageLayout::eUndefined,
            .deferred_shared_release = true,
        });
        if (!HostPasses::IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(tail) ||
            !scheduler.IsRendering()) {
            upstream_sample_tee_producer_pre_depth =
                tail.status == FinalGuestSurfaceStatus::Complete
                    ? HostPasses::
                          PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan{.status =
                                                                                  FinalGuestSurfaceStatus::
                                                                                      InvalidationLoss,
                                                                              .loss =
                                                                                  {.invalidation =
                                                                                       1}}
                    : tail;
            return;
        }
        bool watched_regions_inside_render_area = true;
        for (u32 index = 0; index < tail.plane.region_count; ++index) {
            const auto& region = tail.plane.regions[index];
            watched_regions_inside_render_area &=
                static_cast<u64>(region.x) + region.width <= state.width &&
                static_cast<u64>(region.y) + region.height <= state.height;
        }
        upstream_sample_tee_producer_pre_scope_plan =
            HostPasses::PlanPpUpstreamSampleTeeProducerPreDepthScopeEntryProvenance({
                .entry = upstream_sample_tee_producer_pre_scope_entry,
                .producer_pre_plan = tail,
                .target_intent = intent,
                .first_writer_pre = first_writer_pre,
                .pre_draw_host_rendering_instance_id = scheduler.HostRenderingInstanceId(),
                .render_width = state.width,
                .render_height = state.height,
                .render_layers = state.num_layers,
                .watched_regions_inside_render_area =
                    watched_regions_inside_render_area,
            });
        if (upstream_sample_tee_producer_pre_scope_entry_collision) {
            upstream_sample_tee_producer_pre_scope_owned = {
                .status = FinalGuestSurfaceStatus::GapLoss,
                .loss = {.gap = 1},
            };
        } else if (!upstream_sample_tee_producer_pre_scope_plan.record_metadata) {
            upstream_sample_tee_producer_pre_scope_owned = {
                .status = upstream_sample_tee_producer_pre_scope_plan.status,
                .loss = upstream_sample_tee_producer_pre_scope_plan.loss,
            };
        }
        const auto aspect_restore = HostPasses::PlanPpUpstreamSampleTeeDepthCopyAspectRestore(
            static_cast<u32>(static_cast<VkImageAspectFlags>(image.aspect_mask)),
            static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT));
        if (!aspect_restore.exact) {
            upstream_sample_tee_producer_pre_depth = {
                .status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1},
            };
            return;
        }
        const u64 absolute_slot = static_cast<u64>(upstream_feedback_slot.slot) * slot_stride;
        const auto ordering = HostPasses::PlanPpUpstreamSampleTeeProducerPreDepthDownloadOrdering({
            .layout = tail,
            .absolute_slot_offset = absolute_slot,
            .slot_acquired_before_tail = true,
            .tail_copy_recorded = true,
        });
        const auto tail_barrier =
            HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(ordering.tail_pre_copy);
        const auto host_barrier =
            HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(ordering.host_read);
        if (!tail_barrier.valid || !host_barrier.valid) {
            upstream_sample_tee_producer_pre_depth = {
                .status = FinalGuestSurfaceStatus::InvalidationLoss,
                .loss = {.invalidation = 1},
            };
            return;
        }
        const auto old_state = image.backing->state;
        const u64 pre_host_instance = scheduler.HostRenderingInstanceId();
        const auto cmdbuf = scheduler.CommandBuffer();
        scheduler.EndRendering();
        const vk::BufferMemoryBarrier2 buffer_pre{
            .srcStageMask = tail_barrier.src_stages,
            .srcAccessMask = tail_barrier.src_access,
            .dstStageMask = tail_barrier.dst_stages,
            .dstAccessMask = tail_barrier.dst_access,
            .buffer = download.Handle(),
            .offset = tail_barrier.offset,
            .size = tail_barrier.bytes,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &buffer_pre,
        });
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                      range, cmdbuf);
        auto* resolved = texture_cache.ResolveDiagnosticColorScopeImage(intent.link);
        const bool live =
            resolved == &image && resolved->backing &&
            resolved->backing->subresource_states.empty() && resolved->backing->num_samples == 1 &&
            bool{resolved->aspect_mask & vk::ImageAspectFlagBits::eDepth} &&
            resolved->EnsureDiagnosticBackingGeneration() == intent.backing_generation &&
            resolved->DiagnosticMutationEpoch() == intent.mutation_epoch &&
            resolved->image_uid == intent.link.uid;
        if (live) {
            std::array<vk::BufferImageCopy, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> copies{};
            for (u32 index = 0; index < tail.plane.region_count; ++index) {
                const auto& region = tail.plane.regions[index];
                copies[index] = {
                    .bufferOffset = absolute_slot + tail.plane.plane_offset + region.buffer_offset,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                         .mipLevel = region.mip_level,
                                         .baseArrayLayer = tail.plane.base_layer,
                                         .layerCount = 1},
                    .imageOffset = {static_cast<s32>(region.x), static_cast<s32>(region.y), 0},
                    .imageExtent = {region.width, region.height, 1},
                };
            }
            cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     download.Handle(), tail.plane.region_count, copies.data());
        } else {
            tail = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        const vk::ImageMemoryBarrier2 restore{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = old_state.pl_stage,
            .dstAccessMask = old_state.access_mask,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = old_state.layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.GetImage(),
            .subresourceRange = {.aspectMask = vk::ImageAspectFlags{static_cast<VkImageAspectFlags>(
                                     aspect_restore.restore_aspect)},
                                 .baseMipLevel = range.base.level,
                                 .levelCount = range.extent.levels,
                                 .baseArrayLayer = range.base.layer,
                                 .layerCount = range.extent.layers},
        };
        const vk::BufferMemoryBarrier2 buffer_host{
            .srcStageMask = host_barrier.src_stages,
            .srcAccessMask = host_barrier.src_access,
            .dstStageMask = host_barrier.dst_stages,
            .dstAccessMask = host_barrier.dst_access,
            .buffer = download.Handle(),
            .offset = host_barrier.offset,
            .size = host_barrier.bytes,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &buffer_host,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &restore,
        });
        HostPasses::RestorePpUpstreamInputUniformTracker(
            image.backing->state, image.backing->subresource_states, old_state);
        if (!scheduler.ResumeRenderingForDiagnostic(state, rendering_serial)) {
            tail = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1}};
        }
        upstream_sample_tee_producer_pre_depth = tail;
        if (tail.status == FinalGuestSurfaceStatus::Complete) {
            upstream_sample_tee_producer_pre_expected_serial = rendering_serial;
            upstream_sample_tee_producer_pre_host_instance = pre_host_instance;
            upstream_sample_tee_producer_pre_open = true;
        }
    }

    void ObserveUpstreamSampleTeeProducerPreDepthPost(
        VideoCore::ImageId image_id, VideoCore::Image& image, const RenderState& state,
        const VideoCore::SubresourceRange& range, bool effective_depth_write) {
        const bool main_open = upstream_sample_tee_producer_pre_open &&
                               upstream_sample_tee_producer_pre_intent.link.id == image_id &&
                               upstream_sample_tee_producer_pre_intent.range == range;
        const bool predecessor_open = upstream_sample_tee_predecessor_writer_pre_open &&
                                      upstream_sample_tee_predecessor_writer_pre_intent.link.id ==
                                          image_id &&
                                      upstream_sample_tee_predecessor_writer_pre_intent.range ==
                                          range;
        const bool trace_open = upstream_sample_tee_scope_writer_trace_open &&
                                upstream_sample_tee_scope_writer_trace_intent.link.id == image_id &&
                                upstream_sample_tee_scope_writer_trace_intent.range == range;
        const bool occurrence_open = upstream_sample_tee_producer_pre_occurrences.AwaitingPost();
        if (!main_open && !predecessor_open && !trace_open && !occurrence_open) {
            return;
        }
        const auto post_endpoint = BuildCurrentDepthPredecessorEndpoint(image_id, image, range);
        if (occurrence_open) {
            upstream_sample_tee_producer_pre_occurrences.ObservePost({
                .endpoint = post_endpoint,
                .live_guest_rendering_serial = scheduler.RenderingSerial(),
                .live_host_rendering_instance_id = scheduler.HostRenderingInstanceId(),
                .immediately_after_mark = true,
            });
        }
        if (predecessor_open) {
            upstream_sample_tee_predecessor_writer_pre_post =
                HostPasses::CompletePpUpstreamSampleTeeProducerPreDepthPost({
                    .pre = upstream_sample_tee_predecessor_writer_pre_intent,
                    .post = post_endpoint,
                    .expected_guest_rendering_serial =
                        upstream_sample_tee_predecessor_writer_pre_expected_serial,
                    .pre_host_rendering_instance_id =
                        upstream_sample_tee_predecessor_writer_pre_host_instance,
                    .live_post_host_rendering_instance_id = scheduler.HostRenderingInstanceId(),
                    .diagnostic_resume = scheduler.HostRenderingOrigin() ==
                                         HostRenderingInstanceOrigin::DiagnosticResume,
                    .effective_depth_write = effective_depth_write,
                });
            upstream_sample_tee_predecessor_writer_pre_open = false;
            if (!upstream_sample_tee_predecessor_writer_pre_post.exact) {
                upstream_sample_tee_predecessor_writer_pre_depth = {
                    .status = upstream_sample_tee_predecessor_writer_pre_post.status,
                    .loss = upstream_sample_tee_predecessor_writer_pre_post.loss,
                };
            }
        }
        if (main_open) {
            upstream_sample_tee_producer_pre_post =
                HostPasses::CompletePpUpstreamSampleTeeProducerPreDepthPost({
                    .pre = upstream_sample_tee_producer_pre_intent,
                    .post = post_endpoint,
                    .expected_guest_rendering_serial =
                        upstream_sample_tee_producer_pre_expected_serial,
                    .pre_host_rendering_instance_id =
                        upstream_sample_tee_producer_pre_host_instance,
                    .live_post_host_rendering_instance_id = scheduler.HostRenderingInstanceId(),
                    .diagnostic_resume = scheduler.HostRenderingOrigin() ==
                                         HostRenderingInstanceOrigin::DiagnosticResume,
                    .effective_depth_write = effective_depth_write,
                });
            const auto& scope_draw = upstream_sample_tee_producer_pre_intent.hint.draw;
            upstream_sample_tee_producer_pre_scope_post =
                BuildDepthFirstWriterPostObservation(
                    upstream_sample_tee_producer_pre_scope_plan.first_writer_pre, image,
                    {.kind = scope_draw.kind,
                     .indexed = scope_draw.indexed,
                     .element_count = scope_draw.element_count,
                     .instance_count = scope_draw.instance_count,
                     .fragment_pgm_hash =
                         upstream_sample_tee_producer_pre_intent.hint.fragment_pgm_hash,
                     .sampled_images = scope_draw.sampled_images,
                     .storage_writes = scope_draw.storage_writes},
                    upstream_sample_tee_producer_pre_expected_serial);
            upstream_sample_tee_producer_pre_open = false;
            if (!upstream_sample_tee_producer_pre_post.exact) {
                upstream_sample_tee_producer_pre_depth = {
                    .status = upstream_sample_tee_producer_pre_post.status,
                    .loss = upstream_sample_tee_producer_pre_post.loss,
                };
            }
        }
        if (trace_open) {
            const auto trace_post =
                HostPasses::CompletePpUpstreamSampleTeeProducerPreDepthPost({
                    .pre = upstream_sample_tee_scope_writer_trace_intent,
                    .post = post_endpoint,
                    .expected_guest_rendering_serial =
                        upstream_sample_tee_scope_writer_trace_expected_serial,
                    .pre_host_rendering_instance_id =
                        upstream_sample_tee_scope_writer_trace_host_instance,
                    .live_post_host_rendering_instance_id = scheduler.HostRenderingInstanceId(),
                    .diagnostic_resume = scheduler.HostRenderingOrigin() ==
                                         HostRenderingInstanceOrigin::DiagnosticResume,
                    .effective_depth_write = effective_depth_write,
                });
            upstream_sample_tee_scope_writer_trace_open = false;
            if (!trace_post.exact ||
                !CaptureUpstreamSampleTeeScopeWriterTracePlane(
                    image, state, upstream_sample_tee_scope_writer_trace_expected_serial, range,
                    post_endpoint.writer,
                    upstream_sample_tee_scope_writer_trace_tail.post_plane)) {
                upstream_sample_tee_scope_writer_trace_tail = {
                    .status = FinalGuestSurfaceStatus::InvalidationLoss,
                    .loss = {.invalidation = 1},
                };
            } else if (upstream_sample_tee_scope_writer_trace_post_complete !=
                       std::numeric_limits<u32>::max()) {
                ++upstream_sample_tee_scope_writer_trace_post_complete;
            }
        }
    }

private:
    void RecordUpstreamSampleTeeFailure(FinalGuestSurfaceStatus status,
                                        FinalGuestSurfaceLoss loss) noexcept {
        if (!upstream_sample_tee_attempted) {
            return;
        }
        if (status == FinalGuestSurfaceStatus::Complete || !loss.Any()) {
            status = FinalGuestSurfaceStatus::InvalidationLoss;
            loss = {.invalidation = 1};
        }
        if (upstream_sample_tee_status == FinalGuestSurfaceStatus::AlreadyConsumed) {
            upstream_sample_tee_status = status;
            upstream_sample_tee_loss = loss;
        }
        upstream_sample_tee_copy = false;
        upstream_sample_tee_controller.status = upstream_sample_tee_status;
        upstream_sample_tee_controller.loss = upstream_sample_tee_loss;
        upstream_sample_tee_controller.active = false;
        upstream_sample_tee_controller.ready = false;
    }

    [[nodiscard]] bool ReserveDepthPredecessorCrossScopeSource(
        VideoCore::ImageColorScopePrivateLink link, VideoCore::Image& image,
        const PpTerminalScopeDrawSelector& draw,
        const VideoCore::SubresourceRange& range, u64 fragment_pgm_hash,
        u64 rendering_serial) {
        if (!upstream_feedback_armed || depth_predecessor_cross_scope_capture_closed ||
            open_depth_predecessor_cross_scope_source_draw.valid ||
            open_depth_predecessor_cross_scope_target_draw.valid ||
            open_depth_predecessor_cross_scope_tail_draw.valid ||
            upstream_feedback_slot) {
            RejectCrossScopeCapture();
            return false;
        }
        const auto plan =
            MakeUpstreamFeedbackPlan(image, vk::ImageAspectFlagBits::eDepth);
        if (!upstream_feedback_frame.Configure(plan)) {
            RejectCrossScopeCapture();
            return false;
        }
        const auto reservation = upstream_feedback_frame.Preview(link, draw);
        if (!HostPasses::IsExactPpDepthPredecessorCrossScopeSourceReservation(
                reservation)) {
            RejectCrossScopeCapture();
            return false;
        }
        open_depth_predecessor_cross_scope_source_draw = {
            .link = link,
            .range = range,
            .draw = draw,
            .plan = plan,
            .fragment_pgm_hash = fragment_pgm_hash,
            .rendering_serial = rendering_serial,
            .candidate_index = reservation.candidate_index,
            .valid = true,
        };
        return true;
    }

    [[nodiscard]] bool RestartDepthPredecessorCrossScopeSource(
        VideoCore::ImageId image_id, VideoCore::ImageColorScopePrivateLink link,
        VideoCore::Image& image, const PpTerminalScopeDrawSelector& draw,
        const VideoCore::SubresourceRange& range, u64 fragment_pgm_hash,
        u64 rendering_serial) {
        using RestartEvent = HostPasses::
            PpDepthPredecessorCrossScopeSourceRestartEvidenceEvent;
        using RestartReason =
            HostPasses::PpDepthPredecessorCrossScopeSourceRestartReason;
        const auto record_attempt = [&](RestartReason reason, bool pre_exact,
                                        bool retained_slot_exact,
                                        bool copy_mode_exact) {
            depth_predecessor_cross_scope_outcome.source_restart_evidence =
                HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeSourceRestartEvidence(
                        depth_predecessor_cross_scope_outcome
                            .source_restart_evidence,
                        {
                            .event = RestartEvent::Attempt,
                            .evidence = {
                                .reason = reason,
                                .completed = false,
                                .pre_exact = pre_exact,
                                .retained_slot_exact = retained_slot_exact,
                                .copy_mode_exact = copy_mode_exact,
                            },
                        });
        };
        const auto invalidate = [&](RestartReason reason) {
            const auto invalidated =
                depth_predecessor_cross_scope.BeginSourcePostRestart({});
            AccumulateCrossScopeOutcome(invalidated);
            depth_predecessor_cross_scope_resume = {};
            depth_predecessor_cross_scope_outcome.source_plane_recorded = false;
            record_attempt(reason, false, static_cast<bool>(upstream_feedback_slot),
                           false);
            return false;
        };

        if (!upstream_feedback_armed ||
            depth_predecessor_cross_scope_capture_closed ||
            open_depth_predecessor_cross_scope_source_draw.valid ||
            open_depth_predecessor_cross_scope_target_draw.valid ||
            open_depth_predecessor_cross_scope_tail_draw.valid ||
            !upstream_feedback_slot ||
            !depth_predecessor_cross_scope_resume.valid ||
            !depth_predecessor_cross_scope.CanBeginSourcePostRestart()) {
            return invalidate(RestartReason::RollingClosed);
        }

        const auto prior_resume = depth_predecessor_cross_scope_resume;
        const auto plan =
            MakeUpstreamFeedbackPlan(image, vk::ImageAspectFlagBits::eDepth);
        if (!upstream_feedback_frame.Configure(plan)) {
            return invalidate(RestartReason::FrameRestartMismatch);
        }
        const auto current =
            BuildCurrentDepthPredecessorEndpoint(image_id, image, range);
        const HostPasses::PpDepthPredecessorCrossScopeRenderingCheckpoint
            checkpoint{
                .guest_rendering_serial = scheduler.RenderingSerial(),
                .host_rendering_instance_id =
                    scheduler.HostRenderingInstanceId(),
                .rendering_active = scheduler.IsRendering(),
                .regular_origin =
                    scheduler.HostRenderingOrigin() ==
                    HostRenderingInstanceOrigin::Regular,
                .diagnostic_origin =
                    scheduler.HostRenderingOrigin() ==
                    HostRenderingInstanceOrigin::DiagnosticResume,
            };
        const u32 mask_before = upstream_feedback_frame.RecordedPlaneMask(
            prior_resume.source_candidate.candidate_index);
        const auto preview = upstream_feedback_frame.Preview(link, draw);
        const u32 mask_after = upstream_feedback_frame.RecordedPlaneMask(
            prior_resume.source_candidate.candidate_index);
        const auto observation = HostPasses::
            PpDepthPredecessorCrossScopeSourceRestartPreObservation{
                .stored_source_post = prior_resume.source_post,
                .stored_source_candidate = prior_resume.source_candidate,
                .prior_resume = prior_resume,
                .current_source_writer = current.writer,
                .repeated_source_checkpoint = checkpoint,
                .frame_output = link,
                .frame_preview = preview,
                .expected_generation =
                    depth_predecessor_cross_scope_outcome.result.generation,
                .candidate_capacity =
                    HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates,
                .recorded_plane_mask_before_preview = mask_before,
                .recorded_plane_mask_after_preview = mask_after,
                .retained_slot_present =
                    static_cast<bool>(upstream_feedback_slot),
                .target_pre_attempted = false,
            };
        const auto classified = HostPasses::
            ClassifyPpDepthPredecessorCrossScopeSourceRestartPre(observation);
        const auto reason =
            !classified.stored_witness_exact
                ? RestartReason::StoredWitnessMismatch
            : !classified.current_writer_exact ||
                    !classified.mutation_successor_available
                ? RestartReason::CurrentWriterMismatch
            : !classified.rendering_lineage_exact
                ? RestartReason::RenderingLineageMismatch
            : !classified.frame_restart_exact
                ? RestartReason::FrameRestartMismatch
            : !classified.rolling_window_exact
                ? RestartReason::RollingClosed
                : RestartReason::Exact;
        const auto copy_plan = HostPasses::
            PlanPpDepthPredecessorCrossScopeSourcePlaneCopy(true);
        record_attempt(reason, classified.exact,
                       static_cast<bool>(upstream_feedback_slot) &&
                           mask_before == 0x1u && mask_after == 0,
                       copy_plan.reuse_retained_slot &&
                           !copy_plan.first_copy);
        const auto opened =
            depth_predecessor_cross_scope.BeginSourcePostRestart(observation);
        AccumulateCrossScopeOutcome(opened);
        depth_predecessor_cross_scope_resume = {};
        depth_predecessor_cross_scope_outcome.source_plane_recorded = false;
        if (!classified.exact || IsCrossScopeFailure(opened.decision) ||
            !opened.source_restart_open ||
            opened.role !=
                HostPasses::PpDepthPredecessorCrossScopePairRole::SourcePost) {
            return false;
        }
        open_depth_predecessor_cross_scope_source_draw = {
            .link = link,
            .range = range,
            .draw = draw,
            .plan = plan,
            .restart_prior_resume = prior_resume,
            .fragment_pgm_hash = fragment_pgm_hash,
            .rendering_serial = rendering_serial,
            .candidate_index = preview.candidate_index,
            .restart = true,
            .valid = true,
        };
        return true;
    }

    [[nodiscard]] bool RecordDepthPredecessorCrossScopeSourcePost(
        VideoCore::ImageId image_id, VideoCore::Image& image,
        const RenderState& state, u64 rendering_serial,
        const VideoCore::SubresourceRange& range,
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash,
        bool effective_depth_write) {
        const auto open = open_depth_predecessor_cross_scope_source_draw;
        const bool open_exact =
            open.valid && effective_depth_write && open.link.Valid() &&
            open.link.id == image_id && open.link.uid == image.image_uid &&
            open.range == range && open.rendering_serial == rendering_serial &&
            MatchesPpTerminalScopeDraw(open.draw, draw) &&
            open.fragment_pgm_hash == fragment_pgm_hash;
        const bool slot_state_exact =
            open.restart ? static_cast<bool>(upstream_feedback_slot)
                         : !upstream_feedback_slot;
        const auto record_restart_failure = [&](HostPasses::
                                                     PpDepthPredecessorCrossScopeSourceRestartReason
                                                         reason) {
            if (!open.restart) {
                return;
            }
            depth_predecessor_cross_scope_outcome.source_restart_evidence =
                HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeSourceRestartEvidence(
                        depth_predecessor_cross_scope_outcome
                            .source_restart_evidence,
                        {
                            .event = HostPasses::
                                PpDepthPredecessorCrossScopeSourceRestartEvidenceEvent::
                                    Failure,
                            .evidence = {
                                .reason = reason,
                                .pre_exact = true,
                                .retained_slot_exact =
                                    static_cast<bool>(upstream_feedback_slot),
                                .copy_mode_exact = true,
                            },
                        });
        };
        if (!open_exact || !slot_state_exact) {
            record_restart_failure(HostPasses::
                                       PpDepthPredecessorCrossScopeSourceRestartReason::
                                           RollingClosed);
            RejectCrossScopeCapture();
            return false;
        }
        const auto completion = upstream_feedback_frame.Complete(open.link, draw);
        if (!HostPasses::IsExactPpDepthPredecessorCrossScopeSourceCompletion(
                completion, open.candidate_index)) {
            record_restart_failure(HostPasses::
                                       PpDepthPredecessorCrossScopeSourceRestartReason::
                                           CandidateMismatch);
            RejectCrossScopeCapture();
            return false;
        }
        const auto endpoint = BuildCurrentDepthPredecessorEndpoint(
            image_id, image, range);
        const auto plan =
            MakeUpstreamFeedbackPlan(image, vk::ImageAspectFlagBits::eDepth);
        if (plan != open.plan || !upstream_feedback_frame.Configure(plan)) {
            record_restart_failure(HostPasses::
                                       PpDepthPredecessorCrossScopeSourceRestartReason::
                                           FrameRestartMismatch);
            RejectCrossScopeCapture();
            return false;
        }
        const auto copy_plan = HostPasses::
            PlanPpDepthPredecessorCrossScopeSourcePlaneCopy(open.restart);
        if (!open.restart) {
            const auto token = slots.TryAcquire();
            if (!token) {
                RejectCrossScopeCapture();
                return false;
            }
            upstream_feedback_slot = *token;
        } else if (!copy_plan.reuse_retained_slot ||
                   copy_plan.first_copy || !upstream_feedback_slot) {
            record_restart_failure(HostPasses::
                                       PpDepthPredecessorCrossScopeSourceRestartReason::
                                           FrameRestartMismatch);
            RejectCrossScopeCapture();
            return false;
        }
        const auto record = RecordUpstreamFeedbackPlane(
            image, state, 0, rendering_serial, completion.candidate_index, false,
            copy_plan.first_copy, plan, nullptr, {}, nullptr,
            vk::ImageAspectFlagBits::eDepth);
        const auto disposition =
            HostPasses::PlanPpUpstreamFeedbackPlaneRecordDisposition({
                .tee_enabled = false,
                .tee_attempted = false,
                .base_state =
                    HostPasses::PpUpstreamSampleTeeBaseCaptureState::Pending,
                .record = record,
            });
        if (disposition.fanout.reject_base_capture || !disposition.mark_copied) {
            upstream_feedback_frame.Reject(disposition.fanout.status,
                                           disposition.fanout.loss);
            record_restart_failure(HostPasses::
                                       PpDepthPredecessorCrossScopeSourceRestartReason::
                                           FrameRestartMismatch);
            RejectCrossScopeCapture();
            return false;
        }
        upstream_feedback_frame.MarkCopied(completion.candidate_index, false);
        const HostPasses::PpDepthPredecessorCandidateStamp candidate{
            .valid = true,
            .generation = depth_predecessor_cross_scope_outcome.result.generation,
            .candidate_index = completion.candidate_index,
            .recorded_plane_mask = upstream_feedback_frame.RecordedPlaneMask(
                completion.candidate_index),
            .writer = endpoint.writer,
        };
        const auto witness =
            HostPasses::BuildPpDepthPredecessorCrossScopeResumeWitness({
                .endpoint = endpoint,
                .candidate = candidate,
                .expected_generation =
                    depth_predecessor_cross_scope_outcome.result.generation,
                .candidate_capacity =
                    HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates,
                .resume_guest_rendering_serial = scheduler.RenderingSerial(),
                .resume_host_rendering_instance_id =
                    scheduler.HostRenderingInstanceId(),
                .after_producer_mark = endpoint.producer_marked,
                .source_plane_copied =
                    candidate.recorded_plane_mask == 0x1u,
                .resume_rendering_active = scheduler.IsRendering(),
                .resume_diagnostic_origin =
                    scheduler.HostRenderingOrigin() ==
                    HostRenderingInstanceOrigin::DiagnosticResume,
            });
        HostPasses::PpDepthPredecessorCrossScopePairResult observed{};
        if (open.restart) {
            const auto classified = HostPasses::
                ClassifyPpDepthPredecessorCrossScopeSourceRestartPost({
                    .prior_resume = open.restart_prior_resume,
                    .endpoint = endpoint,
                    .candidate = candidate,
                    .new_resume = witness,
                    .expected_generation =
                        depth_predecessor_cross_scope_outcome.result.generation,
                    .candidate_capacity =
                        HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates,
                    .expected_candidate_index = open.candidate_index,
                });
            const bool copy_exact = copy_plan.reuse_retained_slot &&
                                    !copy_plan.first_copy &&
                                    static_cast<bool>(upstream_feedback_slot);
            const bool exact = classified.exact && copy_exact;
            const auto reason =
                !classified.endpoint_identity_exact ||
                        !classified.mutation_successor_exact ||
                        !classified.endpoint_lineage_exact
                    ? HostPasses::
                          PpDepthPredecessorCrossScopeSourceRestartReason::
                              EndpointMismatch
                : !classified.candidate_exact
                    ? HostPasses::
                          PpDepthPredecessorCrossScopeSourceRestartReason::
                              CandidateMismatch
                : !classified.resume_successor_exact
                    ? HostPasses::
                          PpDepthPredecessorCrossScopeSourceRestartReason::
                              ResumeMismatch
                : !copy_exact
                    ? HostPasses::
                          PpDepthPredecessorCrossScopeSourceRestartReason::
                              FrameRestartMismatch
                    : HostPasses::
                          PpDepthPredecessorCrossScopeSourceRestartReason::Exact;
            depth_predecessor_cross_scope_outcome.source_restart_evidence =
                HostPasses::
                    AccumulatePpDepthPredecessorCrossScopeSourceRestartEvidence(
                        depth_predecessor_cross_scope_outcome
                            .source_restart_evidence,
                        {
                            .event = exact
                                         ? HostPasses::
                                               PpDepthPredecessorCrossScopeSourceRestartEvidenceEvent::
                                                   Completion
                                         : HostPasses::
                                               PpDepthPredecessorCrossScopeSourceRestartEvidenceEvent::
                                                   Failure,
                            .evidence = {
                                .reason = reason,
                                .completed = exact,
                                .pre_exact = true,
                                .endpoint_exact =
                                    classified.endpoint_identity_exact &&
                                    classified.mutation_successor_exact &&
                                    classified.endpoint_lineage_exact,
                                .candidate_exact = classified.candidate_exact,
                                .resume_exact =
                                    classified.resume_successor_exact,
                                .retained_slot_exact =
                                    static_cast<bool>(upstream_feedback_slot),
                                .copy_mode_exact = copy_exact,
                                .exact = exact,
                            },
                        });
            observed = depth_predecessor_cross_scope
                           .ObserveSourcePostRestartComplete(endpoint, candidate,
                                                             witness);
        } else {
            observed = depth_predecessor_cross_scope.ObserveSourcePostComplete(
                endpoint, candidate);
        }
        AccumulateCrossScopeOutcome(observed);
        if (!witness.valid || IsCrossScopeFailure(observed.decision) ||
            observed.role !=
                HostPasses::PpDepthPredecessorCrossScopePairRole::TargetPre) {
            if (open.restart && !witness.valid) {
                record_restart_failure(HostPasses::
                                           PpDepthPredecessorCrossScopeSourceRestartReason::
                                               ResumeMismatch);
            }
            RejectCrossScopeCapture();
            return false;
        }
        depth_predecessor_cross_scope_resume = witness;
        depth_predecessor_cross_scope_outcome.source_plane_recorded = true;
        open_depth_predecessor_cross_scope_source_draw = {};
        return true;
    }

    [[nodiscard]] bool RecordDepthPredecessorCrossScopeTargetPre(
        VideoCore::Image& image, const RenderState& state, u64 rendering_serial,
        const HostPasses::PpDepthPredecessorWriterIntent& intent,
        const PpTerminalScopeDrawSelector& draw, u64 fragment_pgm_hash,
        const HostPasses::PpUpstreamDepthFirstWriterPreObservation& first_writer_pre) {
        const auto& witness = depth_predecessor_cross_scope_resume;
        if (!witness.valid || !upstream_feedback_slot ||
            upstream_feedback_frame.RecordedPlaneMask(
                witness.source_candidate.candidate_index) != 0x1u) {
            RejectCrossScopeCapture();
            return false;
        }
        const auto plan =
            MakeUpstreamFeedbackPlan(image, vk::ImageAspectFlagBits::eDepth);
        if (!upstream_feedback_frame.Configure(plan)) {
            RejectCrossScopeCapture();
            return false;
        }
        const auto record = RecordUpstreamFeedbackPlane(
            image, state, 0, rendering_serial,
            witness.source_candidate.candidate_index, true, false, plan, nullptr,
            {}, nullptr, vk::ImageAspectFlagBits::eDepth);
        const auto disposition =
            HostPasses::PlanPpUpstreamFeedbackPlaneRecordDisposition({
                .tee_enabled = false,
                .tee_attempted = false,
                .base_state =
                    HostPasses::PpUpstreamSampleTeeBaseCaptureState::Pending,
                .record = record,
            });
        if (disposition.fanout.reject_base_capture || !disposition.mark_copied) {
            upstream_feedback_frame.Reject(disposition.fanout.status,
                                           disposition.fanout.loss);
            RejectCrossScopeCapture();
            return false;
        }
        upstream_feedback_frame.MarkCopied(witness.source_candidate.candidate_index,
                                           true);
        auto completed = witness.source_candidate;
        completed.recorded_plane_mask = upstream_feedback_frame.RecordedPlaneMask(
            witness.source_candidate.candidate_index);
        const auto observed =
            depth_predecessor_cross_scope.ObserveTargetPreComplete(completed);
        AccumulateCrossScopeOutcome(observed);
        if (IsCrossScopeFailure(observed.decision) || !observed.candidate_saved ||
            observed.role !=
                HostPasses::PpDepthPredecessorCrossScopePairRole::Tail) {
            RejectCrossScopeCapture();
            return false;
        }
        open_depth_predecessor_cross_scope_target_draw = {
            .link = intent.link,
            .range = intent.range,
            .draw = draw,
            .intent = intent,
            .first_writer_pre = first_writer_pre,
            .fragment_pgm_hash = fragment_pgm_hash,
            .valid = true,
        };
        depth_predecessor_cross_scope_outcome.target_plane_recorded = true;
        return true;
    }

    void ObserveUpstreamFeedbackPreDraw(
        VideoCore::ImageColorScopePrivateLink link, VideoCore::Image& image,
        const RenderState& state, u32 attachment_index, u64 rendering_serial,
        const PpTerminalScopeDrawSelector& draw,
        std::span<const HostPasses::PpUpstreamInputContentView> upstream_views,
        std::span<VideoCore::Image* const> sampled_images,
        std::span<const VideoCore::ImageColorScopeTrackedView> sampled_views,
        std::span<const VideoCore::ImageColorScopeProducerObservation> sampled_scopes,
        std::span<const PpUpstreamSampledExtentObservation> sampled_extents,
        std::span<const HostPasses::PpUpstreamSampleRouteObservation> sample_routes,
        std::span<const HostPasses::PpUpstreamFeedbackCubeLiveObservation>
            diagnostic_sampled_cube_bindings,
        const Shader::Info* fragment_info,
        const HostPasses::PpUpstreamSampleTeeOutputPlan& tee_outputs,
        vk::PipelineLayout pipeline_layout,
        vk::ImageAspectFlagBits copy_aspect = vk::ImageAspectFlagBits::eColor,
        u32* copied_candidate = nullptr, bool prevalidated_hint = false,
        bool require_rolling_restart = false) {
        if (!config.content.capture_upstream_pre_post || !upstream_feedback_armed ||
            (!prevalidated_hint &&
             !MatchesPpTerminalScopeUpstreamTarget(
                 config.content, draw, fragment_info ? fragment_info->pgm_hash : 0))) {
            return;
        }
        const auto& diagnostic_sampled_images = sampled_images;
        const auto plan = MakeUpstreamFeedbackPlan(image, copy_aspect);
        if (!upstream_feedback_frame.Configure(plan)) {
            if (prevalidated_hint) {
                if (IsDepthPredecessorStructuralMultiHopRuntime()) {
                    RejectTwoHopCapture();
                } else {
                    depth_predecessor_hint_outcome.duplicate_collision = true;
                    depth_predecessor_hint_capture_closed = true;
                    open_depth_predecessor_hint_draw = {};
                }
            }
            return;
        }
        const auto action = upstream_feedback_frame.Preview(link, draw);
        if (require_rolling_restart &&
            !HostPasses::IsExactPpDepthPredecessorRawPrefixRestart(action)) {
            RejectTwoHopCapture();
            return;
        }
        if (prevalidated_hint && action.restart &&
            action.candidate_index < depth_candidate_predecessors.size()) {
            depth_candidate_inputs[action.candidate_index] = {};
            depth_candidate_inputs_valid[action.candidate_index] = false;
            depth_candidate_predecessors[action.candidate_index] = {};
            auto& restart_count = IsDepthPredecessorStructuralMultiHopRuntime()
                                      ? depth_predecessor_two_hop_outcome.restart_count
                                      : depth_predecessor_hint_outcome.restart_count;
            if (restart_count != std::numeric_limits<u32>::max()) {
                ++restart_count;
            }
        }
        if (!action.capture) {
            if (prevalidated_hint &&
                action.status != FinalGuestSurfaceStatus::AlreadyConsumed) {
                if (IsDepthPredecessorStructuralMultiHopRuntime()) {
                    RejectTwoHopCapture();
                } else {
                    depth_predecessor_hint_outcome.duplicate_collision = true;
                    depth_predecessor_hint_capture_closed = true;
                    open_depth_predecessor_hint_draw = {};
                }
            }
            return;
        }
        const auto input_action = config.content.capture_upstream_input_content
                                      ? upstream_input_gate.Preview(action.candidate_index)
                                      : HostPasses::PpUpstreamInputCaptureAction{};
        const bool capture_inputs = input_action.capture;
        const bool tee_attempted = config.content.capture_upstream_sample_tee && capture_inputs;
        bool tee_active = tee_attempted;
        upstream_sample_tee_attempted |= tee_attempted;
        if (tee_attempted) {
            upstream_sample_tee_producer_trace = PlanPpUpstreamSampleTeeProducerTrace({
                .enabled = true,
                .selected_input_index = config.content.upstream_sample_route_input_index,
                .current_views = sampled_views,
                .producer_scopes = sampled_scopes,
            });
        }
        bool input_selection_completed{};
        const auto complete_input_selection = [&](bool copied) {
            if (!capture_inputs || input_selection_completed) {
                return HostPasses::PpUpstreamInputCaptureAction{};
            }
            input_selection_completed = true;
            return upstream_input_gate.Complete(action.candidate_index, copied);
        };
        const auto reject_shared = [&](FinalGuestSurfaceStatus status,
                                       const FinalGuestSurfaceLoss& loss) {
            if (tee_attempted) {
                RecordUpstreamSampleTeeFailure(status, loss);
            }
            upstream_feedback_frame.Reject(status, loss);
            static_cast<void>(complete_input_selection(false));
        };
        if (capture_inputs) {
            upstream_input_plan = HostPasses::PlanPpUpstreamInputContent({
                .enabled = true,
                .allow_all_unavailable = config.content.capture_upstream_sample_tee,
                .logical_width = mapping.logical_width,
                .logical_height = mapping.logical_height,
                .base_offset = plan.total_bytes,
                .selector = config.watch_ordinals,
                .views = upstream_views,
                .buffer_alignment = 16,
                .max_regions = HostPasses::PpUpstreamInputContentPlan::MaxInputs *
                               HostPasses::PpUpstreamInputContentPlane::MaxRegions,
                .max_bytes = config.content.capture_upstream_sample_tee
                                 ? HostPasses::PpUpstreamSampleTeeDownloadPrefixBytes
                                 : HostPasses::PpUpstreamInputSnapshotBytes,
            });
            if (!HostPasses::IsPpUpstreamInputContentPlanRecordable(upstream_input_plan)) {
                reject_shared(upstream_input_plan.status, upstream_input_plan.loss);
                return;
            }
        }
        if (config.content.capture_upstream_sample_routes) {
            upstream_sample_routes = HostPasses::PlanPpUpstreamSampleRoutes({
                .enabled = true,
                .selected_input_index = config.content.upstream_sample_route_input_index,
                .input_count = static_cast<u32>(upstream_views.size()),
                .observations = sample_routes,
            });
            if (upstream_sample_routes.status != FinalGuestSurfaceStatus::Complete ||
                upstream_sample_routes.loss.Any()) {
                reject_shared(upstream_sample_routes.status, upstream_sample_routes.loss);
                return;
            }
        }
        HostPasses::PpUpstreamSampleTeeControllerPlan tee_controller{};
        HostPasses::PpUpstreamSampleTeeCandidateRecord tee_candidate{};
        HostPasses::PpUpstreamSampleTeeCapturePlan tee_capture{};
        HostPasses::PpUpstreamSampleTeeCommandPlan tee_commands{};
        HostPasses::PpUpstreamSampleTeeControllerState tee_lifecycle{};
        UpstreamSampleTeePreparation tee_preparation{};
        if (tee_active) {
            tee_controller = HostPasses::PlanPpUpstreamSampleTeeController({
                .enabled = true,
                .candidate_selected = true,
                .direct_draw = draw.kind == VideoCore::ImageColorScopeDrawKind::Direct,
                .fragment_stage_present = fragment_info != nullptr,
                .observed_candidate_index = action.candidate_index,
                .configured_candidate_index = config.content.upstream_candidate_index,
                .outputs = tee_outputs,
            });
            if (!tee_controller.plan_commands ||
                tee_controller.status != FinalGuestSurfaceStatus::Complete ||
                tee_controller.loss.Any()) {
                RecordUpstreamSampleTeeFailure(tee_controller.status, tee_controller.loss);
                tee_active = false;
            }
        }
        if (tee_active) {
            const auto tee_shader = HostPasses::PlanPpUpstreamSampleTeeShader({
                .enabled = true,
                .routes = upstream_sample_routes,
                .eligible_results_per_resource =
                    std::span<const u32>{fragment_info->pp_upstream_sample_tee_results},
            });
            const auto tee_window = HostPasses::PlanPpUpstreamSampleTeeWindow({
                .enabled = true,
                .target_width = image.info.size.width,
                .target_height = image.info.size.height,
                .logical_width = mapping.logical_width,
                .logical_height = mapping.logical_height,
                .logical_ordinal = config.watch_ordinals.ordinals[0],
            });
            tee_candidate = HostPasses::RecordPpUpstreamSampleTeeCandidate(
                upstream_sample_tee_candidate,
                {
                    .enabled = true,
                    .configured_candidate_index = config.content.upstream_candidate_index,
                    .observed_candidate_index = action.candidate_index,
                    .shader = tee_shader,
                    .window = tee_window,
                });
            if (!tee_candidate.ready || tee_candidate.status != FinalGuestSurfaceStatus::Complete ||
                tee_candidate.loss.Any()) {
                RecordUpstreamSampleTeeFailure(tee_candidate.status, tee_candidate.loss);
                tee_active = false;
            }
        }
        if (!upstream_feedback_slot) {
            const auto token = slots.TryAcquire();
            if (!token) {
                reject_shared(FinalGuestSurfaceStatus::BusyLoss, {.busy = 1});
                return;
            }
            upstream_feedback_slot = *token;
            upstream_feedback_base_prefix_initialized = false;
        }
        if (tee_active) {
            tee_capture = HostPasses::PlanPpUpstreamSampleTeeCapture({
                .enabled = true,
                .slot_present = static_cast<bool>(upstream_feedback_slot),
                .slot = upstream_feedback_slot.slot,
                .generation = upstream_feedback_slot.generation,
                .candidate_index = action.candidate_index,
                .shader = tee_candidate.shader,
                .window = tee_candidate.window,
            });
            tee_commands = HostPasses::PlanPpUpstreamSampleTeeCommands({
                .enabled = true,
                .binding = upstream_sample_tee_binding,
                .capture = tee_capture,
                .non_coherent_atom_size = instance.NonCoherentAtomSize(),
                .output_candidate_count = tee_controller.output_candidate_count,
                .direct_draw = draw.kind == VideoCore::ImageColorScopeDrawKind::Direct,
                .rendering_active = scheduler.IsRendering(),
                .observed_rendering_serial = scheduler.RenderingSerial(),
                .expected_rendering_serial = rendering_serial,
            });
            tee_preparation = {
                .capture = tee_capture,
                .commands = tee_commands,
                .reuse_barrier =
                    HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(tee_commands.reuse_barrier),
                .fragment_barrier = HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(
                    tee_commands.fragment_barrier),
                .host_barrier =
                    HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(tee_commands.host_barrier),
                .push_stages = HostPasses::MapPpUpstreamSampleTeePushStagesToVulkan(
                    tee_commands.partial_push_stages),
            };
            if (tee_capture.status != FinalGuestSurfaceStatus::Complete || tee_capture.loss.Any() ||
                tee_commands.status != FinalGuestSurfaceStatus::Complete ||
                tee_commands.loss.Any() || !tee_preparation.reuse_barrier.valid ||
                !tee_preparation.fragment_barrier.valid || !tee_preparation.host_barrier.valid ||
                !tee_preparation.push_stages.valid || !pipeline_layout) {
                const auto status = tee_commands.status != FinalGuestSurfaceStatus::Complete ||
                                            tee_commands.loss.Any()
                                        ? tee_commands.status
                                        : FinalGuestSurfaceStatus::InvalidationLoss;
                const auto loss = tee_commands.status != FinalGuestSurfaceStatus::Complete ||
                                          tee_commands.loss.Any()
                                      ? tee_commands.loss
                                      : FinalGuestSurfaceLoss{.invalidation = 1};
                RecordUpstreamSampleTeeFailure(status, loss);
                tee_active = false;
            }
            if (tee_active) {
                tee_lifecycle = HostPasses::BeginPpUpstreamSampleTeeControllerLifecycle(
                    tee_controller, tee_commands);
                if (!tee_lifecycle.active ||
                    tee_lifecycle.status != FinalGuestSurfaceStatus::Complete ||
                    tee_lifecycle.loss.Any()) {
                    RecordUpstreamSampleTeeFailure(tee_lifecycle.status, tee_lifecycle.loss);
                    tee_active = false;
                }
            }
        }
        if (tee_active) {
            const u32 selected_input = config.content.upstream_sample_route_input_index;
            if (selected_input < sampled_images.size() && sampled_images[selected_input] &&
                selected_input < sampled_views.size() && selected_input < sampled_extents.size()) {
                auto& selected_image = *sampled_images[selected_input];
                const auto& tracked = sampled_views[selected_input];
                auto resolved_format = static_cast<vk::Format>(tracked.format);
                vk::ImageAspectFlags resolved_aspect = selected_image.aspect_mask;
                if (resolved_aspect & vk::ImageAspectFlagBits::eDepth &&
                    LiverpoolToVK::IsFormatDepthCompatible(resolved_format)) {
                    resolved_format = selected_image.info.pixel_format;
                    resolved_aspect = vk::ImageAspectFlagBits::eDepth;
                }
                resolved_format =
                    instance.GetSupportedFormat(resolved_format, selected_image.format_features);
                upstream_sample_tee_source = HostPasses::PlanPpUpstreamSampleTeeSourceObservation({
                    .enabled = true,
                    .routes = upstream_sample_routes,
                    .selected_input_index = selected_input,
                    .selected_candidate_index = config.content.upstream_candidate_index,
                    .candidate = tee_candidate,
                    .binding = tracked,
                    .resolved_view_format = static_cast<u32>(resolved_format),
                    .resolved_view_aspect =
                        static_cast<u32>(static_cast<VkImageAspectFlags>(resolved_aspect)),
                    .producer = selected_image.PeekDiagnosticProducer().classification,
                    .sampled_physical_image_count =
                        static_cast<u32>(diagnostic_sampled_images.size()),
                    .cube_observations = diagnostic_sampled_cube_bindings,
                });
                if (upstream_sample_tee_source.valid) {
                    upstream_sample_tee_transfer = MakePpUpstreamSampleTeeTransferObservation(
                        selected_image, tracked, resolved_aspect, sampled_extents[selected_input]);
                }
                const auto& transfer = upstream_sample_tee_transfer;
                const bool nested_depth_copy =
                    transfer.kind == VideoCore::ImageTransferKind::CopyImage &&
                    transfer.origin == VideoCore::ImageTransferOrigin::ComputeCopy &&
                    transfer.source_transfer_kind == VideoCore::ImageTransferKind::CopyImage &&
                    transfer.source_transfer_origin ==
                        VideoCore::ImageTransferOrigin::DepthOverlap &&
                    transfer.source_is_depth;
                if (nested_depth_copy) {
                    const auto image_token = selected_image.PeekDiagnosticTransferSourceToken();
                    const HostPasses::PpUpstreamSampleTeeTransferredDepthSourceToken token{
                        .link = image_token.link,
                        .backing_generation = image_token.backing_generation,
                        .mutation_epoch = image_token.mutation_epoch,
                    };
                    auto* source = texture_cache.ResolveDiagnosticColorScopeImage(token.link);
                    const bool live = source != nullptr;
                    const u64 source_backing =
                        live && source->backing ? source->EnsureDiagnosticBackingGeneration() : 0;
                    const u64 source_mutation = live ? source->DiagnosticMutationEpoch() : 0;
                    const bool token_current =
                        HostPasses::IsCurrentPpUpstreamSampleTeeTransferredDepthSourceToken(
                            token,
                            live ? VideoCore::ImageColorScopePrivateLink{
                                       .id = token.link.id, .uid = source->image_uid}
                                 : VideoCore::ImageColorScopePrivateLink{},
                            source_backing, source_mutation, live);
                    const bool source_2d =
                        live && (source->info.type == AmdGpu::ImageType::Color2D ||
                                 source->info.type == AmdGpu::ImageType::Color2DArray);
                    const u32 source_bytes =
                        live ? RawTerminalScopeDepthBytesPerTexel(source->info.pixel_format) : 0;
                    const bool barrier_supported =
                        live && source->backing && source_2d &&
                        source->backing->subresource_states.empty() &&
                        source->backing->state.layout != vk::ImageLayout::eUndefined &&
                        bool{source->aspect_mask & vk::ImageAspectFlagBits::eDepth};
                    const u64 tee_record_end =
                        sizeof(HostPasses::PpUpstreamSampleTeeReadbackHeader) +
                        static_cast<u64>(tee_candidate.window.pixel_count) *
                            sizeof(HostPasses::PpUpstreamSampleTeePixelRecord);
                    const u64 source_base = Common::AlignUp<u64>(tee_record_end, 16);
                    tee_preparation.transferred_depth_source =
                        HostPasses::PlanPpUpstreamSampleTeeTransferredDepthSource({
                            .enabled = true,
                            .outer_kind = transfer.kind,
                            .outer_origin = transfer.origin,
                            .source_kind = transfer.source_transfer_kind,
                            .source_origin = transfer.source_transfer_origin,
                            .source_token = token,
                            .logical_width = mapping.logical_width,
                            .logical_height = mapping.logical_height,
                            .source_width = live ? source->info.size.width : 0,
                            .source_height = live ? source->info.size.height : 0,
                            .bytes_per_texel = source_bytes,
                            .base_offset = source_base <= std::numeric_limits<u32>::max()
                                               ? static_cast<u32>(source_base)
                                               : 0,
                            .selector = config.watch_ordinals,
                            .buffer_alignment = 16,
                            .max_regions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals,
                            .max_bytes = Shader::PpUpstreamSampleTeeSlotStrideBytes,
                            .base_mip = 0,
                            .mip_count = 1,
                            .base_layer = 0,
                            .layer_count = 1,
                            .source_samples = transfer.source_samples,
                            .old_samples = transfer.old_samples,
                            .new_samples = transfer.new_samples,
                            .range_exact = transfer.range_exact,
                            .request_matches_encoded = transfer.request_matches_encoded,
                            .full_xy = transfer.full_xy,
                            .source_transfer_exact = transfer.source_transfer_exact,
                            .source_is_depth = transfer.source_is_depth,
                            .depth_format_supported = source_bytes == 4,
                            .source_still_live = live,
                            .source_identity_exact = token_current,
                            .source_backing_exact = token_current,
                            .source_mutation_exact = token_current,
                            .shared_slot_present = static_cast<bool>(upstream_feedback_slot),
                            .barrier_supported = barrier_supported,
                            .deferred_shared_release = true,
                        });
                }
                const HostPasses::PpUpstreamSampleTeeSelectedDepthToken selected_token{
                    .link = tracked.link,
                    .backing_generation = tracked.backing_generation,
                    .mutation_epoch = tracked.mutation_epoch,
                };
                auto* live_selected =
                    texture_cache.ResolveDiagnosticColorScopeImage(selected_token.link);
                const bool selected_live = live_selected == &selected_image;
                const u64 selected_backing =
                    selected_live && live_selected->backing
                        ? live_selected->EnsureDiagnosticBackingGeneration()
                        : 0;
                const u64 selected_mutation =
                    selected_live ? live_selected->DiagnosticMutationEpoch() : 0;
                const bool selected_current =
                    HostPasses::IsCurrentPpUpstreamSampleTeeSelectedDepthToken(
                        selected_token,
                        selected_live
                            ? VideoCore::ImageColorScopePrivateLink{
                                  .id = selected_token.link.id, .uid = live_selected->image_uid}
                            : VideoCore::ImageColorScopePrivateLink{},
                        selected_backing, selected_mutation, selected_live);
                const u32 selected_depth_bytes =
                    selected_live
                        ? RawTerminalScopeDepthBytesPerTexel(live_selected->info.pixel_format)
                        : 0;
                const bool selected_barrier_supported =
                    selected_live && live_selected->backing &&
                    live_selected->backing->subresource_states.empty() &&
                    live_selected->backing->state.layout != vk::ImageLayout::eUndefined &&
                    bool{live_selected->aspect_mask & vk::ImageAspectFlagBits::eDepth};
                const u64 tee_record_end =
                    sizeof(HostPasses::PpUpstreamSampleTeeReadbackHeader) +
                    static_cast<u64>(tee_candidate.window.pixel_count) *
                        sizeof(HostPasses::PpUpstreamSampleTeePixelRecord);
                const bool transfer_occupies_slot =
                    HostPasses::IsPpUpstreamSampleTeeTransferredDepthSourcePlanRecordable(
                        tee_preparation.transferred_depth_source);
                const auto transferred_prefix =
                    transfer_occupies_slot ? tee_preparation.transferred_depth_source
                                           : HostPasses::
                                                 PpUpstreamSampleTeeTransferredDepthSourcePlan{};
                const u64 selected_base =
                    transfer_occupies_slot ? transferred_prefix.total_bytes : tee_record_end;
                tee_preparation.selected_depth =
                    HostPasses::PlanPpUpstreamSampleTeeSelectedDepth({
                        .enabled = true,
                        .routes = upstream_sample_routes,
                        .selected_source = upstream_sample_tee_source,
                        .selected_token = selected_token,
                        .selected_input_index = selected_input,
                        .selected_candidate_index = config.content.upstream_candidate_index,
                        .candidate = tee_candidate,
                        .logical_width = mapping.logical_width,
                        .logical_height = mapping.logical_height,
                        .tee_occupied_bytes =
                            tee_record_end <= std::numeric_limits<u32>::max()
                                ? static_cast<u32>(tee_record_end)
                                : 0,
                        .transferred_depth_source = transferred_prefix,
                        .base_offset = selected_base <= std::numeric_limits<u32>::max()
                                           ? static_cast<u32>(selected_base)
                                           : 0,
                        .selector = config.watch_ordinals,
                        .buffer_alignment = 16,
                        .max_regions = FinalGuestSurfaceWatchOrdinals::MaxOrdinals,
                        .max_bytes = Shader::PpUpstreamSampleTeeSlotStrideBytes,
                        .depth_format_supported = selected_depth_bytes == 4,
                        .actual_image_depth_aspect =
                            selected_live &&
                            bool{selected_image.aspect_mask & vk::ImageAspectFlagBits::eDepth},
                        .source_still_live = selected_live,
                        .source_identity_exact = selected_current,
                        .source_backing_exact = selected_current,
                        .source_mutation_exact = selected_current,
                        .shared_slot_present = static_cast<bool>(upstream_feedback_slot),
                        .barrier_supported = selected_barrier_supported,
                        .capture_pre_draw = true,
                        .deferred_shared_release = true,
                    });
                if (HostPasses::IsPpUpstreamSampleTeeSelectedDepthPlanRecordable(
                        tee_preparation.selected_depth)) {
                    if (upstream_sample_tee_selected_consumer_count !=
                        std::numeric_limits<u32>::max()) {
                        ++upstream_sample_tee_selected_consumer_count;
                    }
                    const VideoCore::SubresourceRange selected_range{
                        {.level = tracked.base_mip, .layer = tracked.base_layer},
                        {.levels = tracked.mip_count, .layers = tracked.layer_count}};
                    const auto selected_endpoint = BuildCurrentDepthPredecessorEndpoint(
                        selected_token.link.id, selected_image, selected_range);
                    const HostPasses::PpUpstreamSampleTeeProducerPreDepthLearnDescriptor
                        learn_descriptor{
                        .enabled = true,
                        .sample_tee_enabled = true,
                        .selected_token = selected_token,
                        .selected_endpoint = selected_endpoint,
                        .selected_source_still_live = selected_live,
                        .selected_identity_exact = selected_current,
                        .selected_backing_exact = selected_current,
                        .selected_mutation_exact = selected_current,
                        .selected_route_exact = upstream_sample_routes.route_count == 1 &&
                                                upstream_sample_routes.routes[0] ==
                                                    upstream_sample_tee_source.route,
                        .selected_view_exact = tracked.valid && !tracked.view_conflict,
                        .selected_depth_aspect =
                            bool{selected_image.aspect_mask & vk::ImageAspectFlagBits::eDepth},
                        .selected_depth_write = selected_endpoint.producer_marked,
                    };
                    const auto learned =
                        HostPasses::LearnPpUpstreamSampleTeeProducerPreDepthHint(
                            learn_descriptor);
                    upstream_sample_tee_producer_pre_gate_evidence.ObserveLearn(
                        HostPasses::AdaptPpUpstreamSampleTeeProducerPreDepthLearnEvidence(
                            learn_descriptor, learned));
                    upstream_sample_tee_producer_pre_occurrences.ObserveSelected(
                        {.learn_descriptor = learn_descriptor, .learned_hint = learned});
                    if (learned.valid) {
                        if (upstream_sample_tee_exact_selected_consumer_count !=
                            std::numeric_limits<u32>::max()) {
                            ++upstream_sample_tee_exact_selected_consumer_count;
                        }
                        upstream_sample_tee_producer_pre_observed_hint = learned;
                    } else {
                        upstream_sample_tee_selected_drift = true;
                    }
                    if (learned.valid && !upstream_sample_tee_producer_pre_collision &&
                        upstream_sample_tee_producer_pre_post.exact &&
                        HostPasses::
                            IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(
                                upstream_sample_tee_producer_pre_depth)) {
                        HostPasses::PpUpstreamInputContentPlane selected_semantic{};
                        const auto& selected_plane = tee_preparation.selected_depth.plane;
                        selected_semantic.region_count = selected_plane.region_count;
                        selected_semantic.plane_offset = selected_plane.plane_offset;
                        selected_semantic.plane_bytes = selected_plane.plane_bytes;
                        selected_semantic.bytes_per_texel = selected_plane.bytes_per_texel;
                        selected_semantic.base_mip = selected_plane.base_mip;
                        selected_semantic.mip_count = selected_plane.mip_count;
                        selected_semantic.base_layer = selected_plane.base_layer;
                        selected_semantic.status = selected_plane.status;
                        selected_semantic.loss = selected_plane.loss;
                        for (u32 index = 0; index < selected_plane.region_count; ++index) {
                            selected_semantic.regions[index] = selected_plane.regions[index];
                        }
                        const HostPasses::PpUpstreamSampleTeeProducerPreDepthBindDescriptor
                            binding_descriptor{
                                .post = upstream_sample_tee_producer_pre_post,
                                .selected_token = selected_token,
                                .live_selected_endpoint = selected_endpoint,
                                .producer_pre_plane =
                                    upstream_sample_tee_producer_pre_depth.plane,
                                .selected_depth_plane = selected_semantic,
                                .selected_source_still_live = selected_live,
                            };
                        const auto binding =
                            HostPasses::BindPpUpstreamSampleTeeProducerPreDepthToSelected(
                                binding_descriptor);
                        upstream_sample_tee_producer_pre_selected_binding_descriptor =
                            binding_descriptor;
                        upstream_sample_tee_producer_pre_selected_binding = binding;
                        upstream_sample_tee_producer_pre_bound = binding.publish_producer_pre;
                        if (upstream_sample_tee_producer_pre_scope_plan.record_metadata) {
                            upstream_sample_tee_producer_pre_scope_owned = HostPasses::
                                BindPpUpstreamSampleTeeProducerPreDepthScopeEntryProvenance({
                                    .planned = upstream_sample_tee_producer_pre_scope_plan,
                                    .producer_post = upstream_sample_tee_producer_pre_post,
                                    .selected_binding = binding,
                                    .first_writer_post =
                                        upstream_sample_tee_producer_pre_scope_post,
                                    .post_observation_count =
                                        upstream_sample_tee_producer_pre_post.exact ? 1u : 0u,
                                    .selected_binding_count =
                                        upstream_sample_tee_exact_selected_consumer_count,
                                });
                        }
                        if (!binding.publish_producer_pre) {
                            upstream_sample_tee_producer_pre_depth = {
                                .status = binding.status,
                                .loss = binding.loss,
                            };
                        }
                    }
                } else {
                    upstream_sample_tee_selected_drift = true;
                }
            }
        }
        if (upstream_sample_tee_producer_pre_depth.copy) {
            upstream_sample_tee_producer_pre_depth.base_total_bytes = plan.total_bytes;
            upstream_sample_tee_producer_pre_depth.input_total_bytes =
                capture_inputs ? upstream_input_plan.total_bytes : plan.total_bytes;
            upstream_sample_tee_producer_pre_depth =
                HostPasses::ValidatePpUpstreamSampleTeeProducerPreDepthDownloadTail(
                    upstream_sample_tee_producer_pre_depth);
        }
        auto record = RecordUpstreamFeedbackPlane(
            image, state, attachment_index, rendering_serial, action.candidate_index, false,
            !upstream_feedback_base_prefix_initialized, plan,
            capture_inputs ? &upstream_input_plan : nullptr, sampled_images,
            tee_active ? &tee_preparation : nullptr, copy_aspect);
        auto disposition = HostPasses::PlanPpUpstreamFeedbackPlaneRecordDisposition({
            .tee_enabled = config.content.capture_upstream_sample_tee,
            .tee_attempted = tee_active,
            .input_selection_pending = capture_inputs && !input_selection_completed,
            .base_state = HostPasses::PpUpstreamSampleTeeBaseCaptureState::Pending,
            .record = record,
        });
        if (disposition.fanout.record_tee_failure) {
            RecordUpstreamSampleTeeFailure(disposition.fanout.status, disposition.fanout.loss);
        }
        if (disposition.retry_without_tee) {
            tee_active = false;
            record = RecordUpstreamFeedbackPlane(
                image, state, attachment_index, rendering_serial, action.candidate_index, false,
                !upstream_feedback_base_prefix_initialized, plan,
                capture_inputs ? &upstream_input_plan : nullptr, sampled_images,
                nullptr, copy_aspect);
            disposition = HostPasses::PlanPpUpstreamFeedbackPlaneRecordDisposition({
                .tee_enabled = config.content.capture_upstream_sample_tee,
                .tee_attempted = false,
                .input_selection_pending = capture_inputs && !input_selection_completed,
                .base_state = HostPasses::PpUpstreamSampleTeeBaseCaptureState::Pending,
                .record = record,
            });
            if (disposition.fanout.record_tee_failure) {
                RecordUpstreamSampleTeeFailure(disposition.fanout.status,
                                               disposition.fanout.loss);
            }
        }
        if (disposition.fanout.reject_base_capture) {
            upstream_feedback_frame.Reject(disposition.fanout.status, disposition.fanout.loss);
        }
        if (disposition.fanout.complete_input_selection) {
            static_cast<void>(
                complete_input_selection(disposition.fanout.input_selection_succeeded));
        }
        if (!disposition.mark_copied) {
            return;
        }
        upstream_feedback_frame.MarkCopied(action.candidate_index, false);
        upstream_feedback_base_prefix_initialized = true;
        if (tee_active) {
            tee_lifecycle = HostPasses::ObservePpUpstreamSampleTeeControllerLifecycle(
                tee_lifecycle,
                {.event = HostPasses::PpUpstreamSampleTeeControllerEvent::Initialized,
                 .observed_candidate_index = action.candidate_index,
                 .observed_serial = scheduler.RenderingSerial(),
                 .succeeded = scheduler.IsRendering()});
            if (!tee_lifecycle.active ||
                tee_lifecycle.status != FinalGuestSurfaceStatus::Complete ||
                tee_lifecycle.loss.Any()) {
                RecordUpstreamSampleTeeFailure(tee_lifecycle.status, tee_lifecycle.loss);
                tee_active = false;
            }
        }
        if (tee_active) {
            const std::array<u32, 2> tee_push{tee_capture.control, tee_capture.base_dword};
            scheduler.CommandBuffer().pushConstants(
                pipeline_layout, tee_preparation.push_stages.stages,
                tee_commands.partial_push_offset, tee_commands.partial_push_bytes,
                tee_push.data());
            tee_lifecycle = HostPasses::ObservePpUpstreamSampleTeeControllerLifecycle(
                tee_lifecycle,
                {.event = HostPasses::PpUpstreamSampleTeeControllerEvent::Armed,
                 .observed_candidate_index = action.candidate_index,
                 .observed_serial = scheduler.RenderingSerial(),
                 .succeeded = true});
            const auto completion = HostPasses::PlanPpUpstreamSampleTeeControllerCompletion({
                .controller = tee_controller,
                .upstream_copy_recorded = true,
                .slot_initialized = tee_lifecycle.initialized,
                .rendering_resumed =
                    scheduler.IsRendering() && scheduler.RenderingSerial() == rendering_serial,
                .late_fragment_push_recorded = tee_lifecycle.armed,
            });
            if (!completion.record_candidate ||
                completion.status != FinalGuestSurfaceStatus::Complete || completion.loss.Any()) {
                RecordUpstreamSampleTeeFailure(completion.status, completion.loss);
                tee_active = false;
            }
        }
        if (capture_inputs) {
            const auto completed = complete_input_selection(true);
            upstream_input_ready = completed.publish;
            if (!upstream_input_ready) {
                upstream_feedback_frame.Reject(completed.status, completed.loss);
                if (tee_attempted) {
                    RecordUpstreamSampleTeeFailure(completed.status, completed.loss);
                }
                return;
            }
        }
        if (tee_active) {
            upstream_sample_tee_transfer = {};
            upstream_sample_tee_transferred_depth_source = {};
            upstream_sample_tee_selected_depth = {};
            const u32 selected_input = config.content.upstream_sample_route_input_index;
            if (selected_input < sampled_images.size() && sampled_images[selected_input] &&
                selected_input < sampled_views.size() && selected_input < sampled_extents.size()) {
                auto& source_image = *sampled_images[selected_input];
                const auto& tracked = sampled_views[selected_input];
                auto resolved_format = static_cast<vk::Format>(tracked.format);
                vk::ImageAspectFlags resolved_aspect = source_image.aspect_mask;
                if (resolved_aspect & vk::ImageAspectFlagBits::eDepth &&
                    LiverpoolToVK::IsFormatDepthCompatible(resolved_format)) {
                    resolved_format = source_image.info.pixel_format;
                    resolved_aspect = vk::ImageAspectFlagBits::eDepth;
                }
                if (resolved_aspect & vk::ImageAspectFlagBits::eStencil &&
                    LiverpoolToVK::IsFormatStencilCompatible(resolved_format)) {
                    resolved_format = source_image.info.pixel_format;
                    resolved_aspect = vk::ImageAspectFlagBits::eStencil;
                }
                resolved_format =
                    instance.GetSupportedFormat(resolved_format, source_image.format_features);
                upstream_sample_tee_source = HostPasses::PlanPpUpstreamSampleTeeSourceObservation({
                    .enabled = true,
                    .routes = upstream_sample_routes,
                    .selected_input_index = selected_input,
                    .selected_candidate_index = config.content.upstream_candidate_index,
                    .candidate = tee_candidate,
                    .binding = tracked,
                    .resolved_view_format = static_cast<u32>(resolved_format),
                    .resolved_view_aspect =
                        static_cast<u32>(static_cast<VkImageAspectFlags>(resolved_aspect)),
                    .producer = source_image.PeekDiagnosticProducer().classification,
                    .sampled_physical_image_count =
                        static_cast<u32>(diagnostic_sampled_images.size()),
                    .cube_observations = diagnostic_sampled_cube_bindings,
                });
                if (upstream_sample_tee_source.valid) {
                    upstream_sample_tee_transfer = MakePpUpstreamSampleTeeTransferObservation(
                        source_image, tracked, resolved_aspect, sampled_extents[selected_input]);
                }
            }
            upstream_sample_tee_candidate = tee_candidate;
            upstream_sample_tee_capture = tee_capture;
            upstream_sample_tee_commands = tee_commands;
            upstream_sample_tee_controller = tee_lifecycle;
            upstream_sample_tee_transferred_depth_source =
                tee_preparation.transferred_depth_source;
            upstream_sample_tee_selected_depth = tee_preparation.selected_depth;
        }
        if (copied_candidate) {
            *copied_candidate = action.candidate_index;
        }
    }

    [[nodiscard]] bool ObserveUpstreamFeedbackPostDraw(VideoCore::ImageColorScopePrivateLink link,
                                                       VideoCore::Image& image,
                                                       const RenderState& state,
                                                       u32 attachment_index, u64 rendering_serial,
                                                       const PpTerminalScopeDrawSelector& draw,
                                                       u64 fragment_pgm_hash,
                                                       vk::ImageAspectFlagBits copy_aspect =
                                                           vk::ImageAspectFlagBits::eColor,
                                                       u32* copied_candidate = nullptr,
                                                       bool prevalidated_hint = false) {
        if (!config.content.capture_upstream_pre_post || !upstream_feedback_armed ||
            (!prevalidated_hint &&
             !MatchesPpTerminalScopeUpstreamTarget(config.content, draw, fragment_pgm_hash))) {
            return true;
        }
        const auto action = upstream_feedback_frame.Complete(link, draw);
        if (!action.capture) {
            if (prevalidated_hint) {
                if (IsDepthPredecessorStructuralMultiHopRuntime()) {
                    RejectTwoHopCapture();
                } else {
                    depth_predecessor_hint_outcome.duplicate_collision = true;
                    depth_predecessor_hint_capture_closed = true;
                    open_depth_predecessor_hint_draw = {};
                }
            }
            if (upstream_sample_tee_controller.active) {
                RecordUpstreamSampleTeeFailure(
                    action.status == FinalGuestSurfaceStatus::AlreadyConsumed
                        ? FinalGuestSurfaceStatus::GapLoss
                        : action.status,
                    action.loss.Any() ? action.loss : FinalGuestSurfaceLoss{.gap = 1});
            }
            return true;
        }
        const auto plan = MakeUpstreamFeedbackPlan(image, copy_aspect);
        if (!upstream_feedback_frame.Configure(plan)) {
            if (upstream_sample_tee_controller.active) {
                RecordUpstreamSampleTeeFailure(FinalGuestSurfaceStatus::InvalidationLoss,
                                               {.invalidation = 1});
            }
            return true;
        }
        if (!upstream_feedback_slot) {
            upstream_feedback_frame.Reject(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
            if (upstream_sample_tee_controller.active) {
                RecordUpstreamSampleTeeFailure(FinalGuestSurfaceStatus::GapLoss, {.gap = 1});
            }
            return true;
        }
        bool tee_active = upstream_sample_tee_controller.active;
        UpstreamSampleTeePreparation tee_post{};
        HostPasses::PpUpstreamSampleTeePostDrawPlan tee_post_plan{};
        if (tee_active) {
            if (draw.kind != VideoCore::ImageColorScopeDrawKind::Direct ||
                upstream_sample_tee_controller.candidate_index != action.candidate_index ||
                upstream_sample_tee_candidate.candidate_index != action.candidate_index ||
                upstream_sample_tee_capture.header.candidate_index != action.candidate_index ||
                upstream_sample_tee_capture.header.generation !=
                    upstream_feedback_slot.generation ||
                upstream_sample_tee_capture.slot_offset !=
                    upstream_feedback_slot.slot * Shader::PpUpstreamSampleTeeSlotStrideBytes) {
                RecordUpstreamSampleTeeFailure(FinalGuestSurfaceStatus::InvalidationLoss,
                                               {.invalidation = 1});
                tee_active = false;
            }
        }
        if (tee_active) {
            tee_post_plan = HostPasses::PlanPpUpstreamSampleTeePostDraw({
                .enabled = true,
                .controller = upstream_sample_tee_controller,
                .candidate = upstream_sample_tee_candidate,
                .capture = upstream_sample_tee_capture,
                .commands = upstream_sample_tee_commands,
                .observed_candidate_index = action.candidate_index,
                .observed_serial = rendering_serial,
                .direct_draw_executed = true,
            });
            tee_post = {
                .capture = upstream_sample_tee_capture,
                .commands = upstream_sample_tee_commands,
                .host_barrier =
                    HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(tee_post_plan.host_barrier),
            };
            if (tee_post_plan.status != FinalGuestSurfaceStatus::Complete ||
                tee_post_plan.loss.Any() || !tee_post_plan.issue_host_barrier ||
                !tee_post_plan.resume_rendering || !tee_post.host_barrier.valid ||
                tee_post.host_barrier.offset != upstream_sample_tee_capture.slot_offset ||
                tee_post.host_barrier.bytes != upstream_sample_tee_capture.slot_bytes) {
                RecordUpstreamSampleTeeFailure(tee_post_plan.status, tee_post_plan.loss);
                tee_active = false;
            }
        }
        auto record = RecordUpstreamFeedbackPlane(
            image, state, attachment_index, rendering_serial, action.candidate_index, true, false,
            plan, nullptr, {}, tee_active ? &tee_post : nullptr, copy_aspect);
        auto disposition = HostPasses::PlanPpUpstreamFeedbackPlaneRecordDisposition({
            .tee_enabled = config.content.capture_upstream_sample_tee,
            .tee_attempted = tee_active,
            .base_state = HostPasses::PpUpstreamSampleTeeBaseCaptureState::Pending,
            .record = record,
        });
        if (disposition.fanout.record_tee_failure) {
            RecordUpstreamSampleTeeFailure(disposition.fanout.status, disposition.fanout.loss);
        }
        if (disposition.retry_without_tee) {
            tee_active = false;
            record = RecordUpstreamFeedbackPlane(image, state, attachment_index, rendering_serial,
                                                 action.candidate_index, true, false, plan, nullptr,
                                                 {}, nullptr, copy_aspect);
            disposition = HostPasses::PlanPpUpstreamFeedbackPlaneRecordDisposition({
                .tee_enabled = config.content.capture_upstream_sample_tee,
                .tee_attempted = false,
                .base_state = HostPasses::PpUpstreamSampleTeeBaseCaptureState::Pending,
                .record = record,
            });
            if (disposition.fanout.record_tee_failure) {
                RecordUpstreamSampleTeeFailure(disposition.fanout.status,
                                               disposition.fanout.loss);
            }
        }
        if (disposition.fanout.reject_base_capture) {
            upstream_feedback_frame.Reject(disposition.fanout.status, disposition.fanout.loss);
        }
        if (!disposition.mark_copied) {
            return true;
        }
        upstream_feedback_frame.MarkCopied(action.candidate_index, true);
        if (copied_candidate) {
            *copied_candidate = action.candidate_index;
        }
        if (tee_active) {
            const bool resumed =
                scheduler.IsRendering() && scheduler.RenderingSerial() == rendering_serial;
            const auto completion = HostPasses::CompletePpUpstreamSampleTeePostDraw({
                .plan = tee_post_plan,
                .observed_candidate_index = action.candidate_index,
                .observed_serial = scheduler.RenderingSerial(),
                .host_barrier_recorded = true,
                .rendering_resumed = resumed,
            });
            if (!completion.ready || completion.status != FinalGuestSurfaceStatus::Complete ||
                completion.loss.Any()) {
                RecordUpstreamSampleTeeFailure(completion.status, completion.loss);
            } else {
                upstream_sample_tee_controller = completion.controller;
                upstream_sample_tee_status = FinalGuestSurfaceStatus::Complete;
                upstream_sample_tee_loss = {};
                upstream_sample_tee_copy = true;
            }
        }
        return true;
    }

    [[nodiscard]] HostPasses::PpUpstreamFeedbackPlaneRecordResult RecordUpstreamFeedbackPlane(
        VideoCore::Image& image, const RenderState& state, u32 attachment_index,
        u64 rendering_serial, u32 candidate_index, bool post, bool first_copy,
        const HostPasses::PpUpstreamFeedbackPrePostPlan& plan,
        const HostPasses::PpUpstreamInputContentPlan* input_plan = nullptr,
        std::span<VideoCore::Image* const> sampled_images = {},
        UpstreamSampleTeePreparation* tee_preparation = nullptr,
        vk::ImageAspectFlagBits copy_aspect = vk::ImageAspectFlagBits::eColor) {
        const auto failure = [](HostPasses::PpUpstreamSampleTeeFailureScope scope,
                                FinalGuestSurfaceStatus status, FinalGuestSurfaceLoss loss) {
            if (status == FinalGuestSurfaceStatus::Complete || !loss.Any()) {
                status = FinalGuestSurfaceStatus::InvalidationLoss;
                loss = {.invalidation = 1};
            }
            return HostPasses::PpUpstreamFeedbackPlaneRecordResult{
                .failure_scope = scope,
                .status = status,
                .loss = loss,
            };
        };
        const bool color_capture = copy_aspect == vk::ImageAspectFlagBits::eColor;
        const bool depth_capture = copy_aspect == vk::ImageAspectFlagBits::eDepth;
        const bool attachment_valid =
            (color_capture && attachment_index < state.num_color_attachments) ||
            (depth_capture && state.depth_stencil_attachment.has_depth &&
             static_cast<bool>(image.aspect_mask & vk::ImageAspectFlagBits::eDepth));
        if (!upstream_feedback_slot || !attachment_valid || !scheduler.IsRendering() ||
            !image.backing ||
            image.backing->state.layout == vk::ImageLayout::eUndefined) {
            return failure(HostPasses::PpUpstreamSampleTeeFailureScope::SharedCapture,
                           FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
        }
        const auto expected_plan = MakeUpstreamFeedbackPlan(image, copy_aspect);
        if (plan != expected_plan || plan.status != FinalGuestSurfaceStatus::Complete ||
            plan.loss.Any() || !plan.copy || plan.total_bytes > slot_stride) {
            return failure(HostPasses::PpUpstreamSampleTeeFailureScope::SharedCapture,
                           FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
        }
        if (input_plan) {
            if (!HostPasses::IsPpUpstreamInputContentPlanRecordable(*input_plan) ||
                input_plan->total_bytes > slot_stride ||
                input_plan->input_count > sampled_images.size()) {
                return failure(HostPasses::PpUpstreamSampleTeeFailureScope::SharedCapture,
                               FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
            }
            for (u32 input_index = 0; input_index < input_plan->input_count; ++input_index) {
                if ((input_plan->capture_mask & (1u << input_index)) == 0) {
                    continue;
                }
                const auto* sampled = sampled_images[input_index];
                const auto& input = input_plan->inputs[input_index];
                const u32 bytes_per_texel =
                    sampled && !sampled->info.props.is_block && sampled->info.num_bits % 8 == 0
                        ? sampled->info.num_bits / 8
                        : 0;
                if (!sampled || sampled == &image || !sampled->backing ||
                    sampled->backing->state.layout == vk::ImageLayout::eUndefined ||
                    !sampled->backing->subresource_states.empty() ||
                    sampled->aspect_mask != vk::ImageAspectFlagBits::eColor ||
                    sampled->backing->num_samples != 1 ||
                    bytes_per_texel != input.bytes_per_texel || input.mip_count == 0 ||
                    input.base_mip + input.mip_count > sampled->info.resources.levels ||
                    input.base_layer >= sampled->info.resources.layers) {
                    return failure(HostPasses::PpUpstreamSampleTeeFailureScope::SharedCapture,
                                   FinalGuestSurfaceStatus::InvalidationLoss,
                                   {.invalidation = 1});
                }
            }
        }
        VideoCore::Buffer* tee_buffer{};
        VideoCore::Image* transferred_depth_source{};
        VideoCore::Image* selected_depth_image{};
        if (tee_preparation) {
            tee_buffer = upstream_sample_tee_buffer.get();
            const auto& capture = tee_preparation->capture;
            const auto& commands = tee_preparation->commands;
            const auto& selected_barrier =
                post ? tee_preparation->host_barrier : tee_preparation->fragment_barrier;
            if (!tee_buffer || !capture.initialize || !capture.arm ||
                capture.status != FinalGuestSurfaceStatus::Complete || capture.loss.Any() ||
                commands.status != FinalGuestSurfaceStatus::Complete || commands.loss.Any() ||
                !commands.initialize || !commands.arm || !commands.issue_draw ||
                !commands.require_post_barrier || !commands.require_preserved_serial ||
                !selected_barrier.valid || selected_barrier.offset != capture.slot_offset ||
                selected_barrier.bytes != capture.slot_bytes ||
                (!post && (!tee_preparation->reuse_barrier.valid ||
                           tee_preparation->reuse_barrier.offset != capture.slot_offset ||
                           tee_preparation->reuse_barrier.bytes != capture.slot_bytes)) ||
                static_cast<u64>(capture.slot_offset) + capture.slot_bytes >
                    tee_buffer->SizeBytes()) {
                return failure(HostPasses::PpUpstreamSampleTeeFailureScope::TeeOnly,
                               FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
            }
            auto& depth_plan = tee_preparation->transferred_depth_source;
            if (depth_plan.status == FinalGuestSurfaceStatus::Complete) {
                if (!HostPasses::IsPpUpstreamSampleTeeTransferredDepthSourcePlanRecordable(
                        depth_plan)) {
                    depth_plan = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                                  .loss = {.invalidation = 1}};
                } else {
                    transferred_depth_source =
                        texture_cache.ResolveDiagnosticColorScopeImage(depth_plan.source_token.link);
                    const bool current =
                        transferred_depth_source && transferred_depth_source->backing &&
                        HostPasses::IsCurrentPpUpstreamSampleTeeTransferredDepthSourceToken(
                            depth_plan.source_token,
                            {.id = depth_plan.source_token.link.id,
                             .uid = transferred_depth_source->image_uid},
                            transferred_depth_source->EnsureDiagnosticBackingGeneration(),
                            transferred_depth_source->DiagnosticMutationEpoch(), true);
                    if (!current) {
                        transferred_depth_source = nullptr;
                        depth_plan = {.status = FinalGuestSurfaceStatus::GapLoss,
                                      .loss = {.gap = 1}};
                    }
                }
            }
            auto& selected_plan = tee_preparation->selected_depth;
            if (selected_plan.status == FinalGuestSurfaceStatus::Complete) {
                if (!HostPasses::IsPpUpstreamSampleTeeSelectedDepthPlanRecordable(
                        selected_plan)) {
                    selected_plan = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                                     .loss = {.invalidation = 1}};
                } else {
                    selected_depth_image = texture_cache.ResolveDiagnosticColorScopeImage(
                        selected_plan.selected_token.link);
                    const bool selected_2d =
                        selected_depth_image &&
                        (selected_depth_image->info.type == AmdGpu::ImageType::Color2D ||
                         selected_depth_image->info.type == AmdGpu::ImageType::Color2DArray);
                    const bool current =
                        selected_depth_image && selected_depth_image->backing && selected_2d &&
                        selected_depth_image->backing->subresource_states.empty() &&
                        selected_depth_image->backing->state.layout !=
                            vk::ImageLayout::eUndefined &&
                        selected_depth_image->backing->num_samples == 1 &&
                        bool{selected_depth_image->aspect_mask &
                             vk::ImageAspectFlagBits::eDepth} &&
                        RawTerminalScopeDepthBytesPerTexel(
                            selected_depth_image->info.pixel_format) == 4 &&
                        HostPasses::IsCurrentPpUpstreamSampleTeeSelectedDepthToken(
                            selected_plan.selected_token,
                            {.id = selected_plan.selected_token.link.id,
                             .uid = selected_depth_image->image_uid},
                            selected_depth_image->EnsureDiagnosticBackingGeneration(),
                            selected_depth_image->DiagnosticMutationEpoch(), true);
                    if (!current) {
                        selected_depth_image = nullptr;
                        selected_plan = {.status = FinalGuestSurfaceStatus::GapLoss,
                                         .loss = {.gap = 1}};
                    }
                }
            }
            if (depth_plan.status == FinalGuestSurfaceStatus::Complete &&
                selected_plan.status == FinalGuestSurfaceStatus::Complete) {
                const auto overlaps = [](u32 first_offset, u32 first_bytes, u32 second_offset,
                                         u32 second_bytes) {
                    return static_cast<u64>(first_offset) <
                               static_cast<u64>(second_offset) + second_bytes &&
                           static_cast<u64>(second_offset) <
                               static_cast<u64>(first_offset) + first_bytes;
                };
                if (overlaps(depth_plan.plane.plane_offset, depth_plan.plane.plane_bytes,
                             selected_plan.plane.plane_offset,
                             selected_plan.plane.plane_bytes)) {
                    selected_depth_image = nullptr;
                    selected_plan = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                                     .loss = {.invalidation = 1}};
                }
            }
        }
        const auto split = PlanPpTerminalScopeRenderingSplit(
            scheduler.IsRendering(), scheduler.RenderingSerial(), rendering_serial);
        if (split.status != FinalGuestSurfaceStatus::Complete) {
            return failure(HostPasses::PpUpstreamSampleTeeFailureScope::SharedCapture,
                           split.status, split.loss);
        }
        std::array<vk::BufferImageCopy, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> copies{};
        for (u32 index = 0; index < plan.region_count; ++index) {
            const auto copy = HostPasses::ResolvePpUpstreamFeedbackPrePostCopy(
                plan, candidate_index, post, index);
            if (!copy.copy) {
                return failure(HostPasses::PpUpstreamSampleTeeFailureScope::SharedCapture,
                               copy.status, copy.loss);
            }
            copies[index] = {
                .bufferOffset = upstream_feedback_slot.slot * slot_stride + copy.buffer_offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = copy_aspect,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = {static_cast<s32>(copy.x), static_cast<s32>(copy.y), 0},
                .imageExtent = {copy.width, copy.height, 1},
            };
        }
        const u64 slot_offset = upstream_feedback_slot.slot * slot_stride;
        const auto cmdbuf = scheduler.CommandBuffer();
        scheduler.EndRendering();
        const vk::BufferMemoryBarrier2 buffer_pre{
            .srcStageMask = first_copy ? vk::PipelineStageFlagBits2::eAllCommands |
                                             vk::PipelineStageFlagBits2::eHost
                                       : vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask =
                first_copy ? vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eHostRead
                           : vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = download.Handle(),
            .offset = slot_offset,
            .size = config.content.capture_upstream_sample_tee
                        ? HostPasses::PpUpstreamSampleTeeDownloadPrefixBytes
                        : slot_stride,
        };
        std::array<vk::BufferMemoryBarrier2, 2> buffer_pre_barriers{buffer_pre};
        u32 buffer_pre_count = 1;
        if (tee_preparation && !post) {
            const auto& barrier = tee_preparation->reuse_barrier;
            buffer_pre_barriers[buffer_pre_count++] = {
                .srcStageMask = barrier.src_stages,
                .srcAccessMask = barrier.src_access,
                .dstStageMask = barrier.dst_stages,
                .dstAccessMask = barrier.dst_access,
                .buffer = tee_buffer->Handle(),
                .offset = barrier.offset,
                .size = barrier.bytes,
            };
        }
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = buffer_pre_count,
            .pBufferMemoryBarriers = buffer_pre_barriers.data(),
        });
        if (tee_preparation && !post) {
            const auto& capture = tee_preparation->capture;
            cmdbuf.fillBuffer(tee_buffer->Handle(), capture.clear_offset, capture.clear_bytes, 0);
            cmdbuf.updateBuffer(tee_buffer->Handle(), capture.slot_offset, sizeof(capture.header),
                                &capture.header);
            auto& depth_plan = tee_preparation->transferred_depth_source;
            if (transferred_depth_source &&
                depth_plan.status == FinalGuestSurfaceStatus::Complete) {
                const auto aspect_restore =
                    HostPasses::PlanPpUpstreamSampleTeeDepthCopyAspectRestore(
                        static_cast<u32>(
                            static_cast<VkImageAspectFlags>(transferred_depth_source->aspect_mask)),
                        static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT));
                if (!aspect_restore.exact) {
                    depth_plan = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                                  .loss = {.invalidation = 1}};
                } else {
                    const auto old_source_state = transferred_depth_source->backing->state;
                    const VideoCore::SubresourceRange source_range{
                        {.level = depth_plan.plane.base_mip, .layer = depth_plan.plane.base_layer},
                        {.levels = depth_plan.plane.mip_count, .layers = 1}};
                    transferred_depth_source->Transit(vk::ImageLayout::eTransferSrcOptimal,
                                                      vk::AccessFlagBits2::eTransferRead,
                                                      source_range, cmdbuf);
                    auto* current_source = texture_cache.ResolveDiagnosticColorScopeImage(
                        depth_plan.source_token.link);
                    const bool current =
                        current_source == transferred_depth_source && current_source->backing &&
                        HostPasses::IsCurrentPpUpstreamSampleTeeTransferredDepthSourceToken(
                            depth_plan.source_token,
                            {.id = depth_plan.source_token.link.id,
                             .uid = current_source->image_uid},
                            current_source->EnsureDiagnosticBackingGeneration(),
                            current_source->DiagnosticMutationEpoch(), true);
                    if (current) {
                        const auto pre_copy_barrier =
                            HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(
                                HostPasses::
                                    PlanPpUpstreamSampleTeeTransferredDepthSourcePreCopyBarrier(
                                        depth_plan, capture.slot_offset));
                        std::array<vk::BufferImageCopy, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>
                            depth_copies{};
                        for (u32 index = 0; index < depth_plan.plane.region_count; ++index) {
                            const auto& region = depth_plan.plane.regions[index];
                            depth_copies[index] = {
                                .bufferOffset = capture.slot_offset +
                                                depth_plan.plane.plane_offset +
                                                region.buffer_offset,
                                .bufferRowLength = 0,
                                .bufferImageHeight = 0,
                                .imageSubresource =
                                    {
                                        .aspectMask = vk::ImageAspectFlagBits::eDepth,
                                        .mipLevel = region.mip_level,
                                        .baseArrayLayer = depth_plan.plane.base_layer,
                                        .layerCount = 1,
                                    },
                                .imageOffset = {static_cast<s32>(region.x),
                                                static_cast<s32>(region.y), 0},
                                .imageExtent = {region.width, region.height, 1},
                            };
                        }
                        if (pre_copy_barrier.valid) {
                            const vk::BufferMemoryBarrier2 barrier{
                                .srcStageMask = pre_copy_barrier.src_stages,
                                .srcAccessMask = pre_copy_barrier.src_access,
                                .dstStageMask = pre_copy_barrier.dst_stages,
                                .dstAccessMask = pre_copy_barrier.dst_access,
                                .buffer = tee_buffer->Handle(),
                                .offset = pre_copy_barrier.offset,
                                .size = pre_copy_barrier.bytes,
                            };
                            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                                .bufferMemoryBarrierCount = 1,
                                .pBufferMemoryBarriers = &barrier,
                            });
                            cmdbuf.copyImageToBuffer(
                                current_source->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                tee_buffer->Handle(), depth_plan.plane.region_count,
                                depth_copies.data());
                        } else {
                            depth_plan = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                                          .loss = {.invalidation = 1}};
                        }
                    } else {
                        depth_plan = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                                      .loss = {.invalidation = 1}};
                    }
                    const vk::ImageMemoryBarrier2 source_restore{
                        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
                        .dstStageMask = old_source_state.pl_stage,
                        .dstAccessMask = old_source_state.access_mask,
                        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                        .newLayout = old_source_state.layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = transferred_depth_source->GetImage(),
                        .subresourceRange =
                            {
                                .aspectMask = vk::ImageAspectFlags{static_cast<VkImageAspectFlags>(
                                    aspect_restore.restore_aspect)},
                                .baseMipLevel = source_range.base.level,
                                .levelCount = source_range.extent.levels,
                                .baseArrayLayer = source_range.base.layer,
                                .layerCount = source_range.extent.layers,
                            },
                    };
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .imageMemoryBarrierCount = 1,
                        .pImageMemoryBarriers = &source_restore,
                    });
                    HostPasses::RestorePpUpstreamInputUniformTracker(
                        transferred_depth_source->backing->state,
                        transferred_depth_source->backing->subresource_states, old_source_state);
                }
            }
            auto& selected_plan = tee_preparation->selected_depth;
            if (selected_depth_image && selected_plan.status == FinalGuestSurfaceStatus::Complete) {
                const auto aspect_restore =
                    HostPasses::PlanPpUpstreamSampleTeeDepthCopyAspectRestore(
                        static_cast<u32>(
                            static_cast<VkImageAspectFlags>(selected_depth_image->aspect_mask)),
                        static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT));
                if (!aspect_restore.exact) {
                    selected_plan = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                                     .loss = {.invalidation = 1}};
                } else {
                    const auto old_selected_state = selected_depth_image->backing->state;
                    const VideoCore::SubresourceRange selected_range{
                        {.level = selected_plan.plane.base_mip,
                         .layer = selected_plan.plane.base_layer},
                        {.levels = selected_plan.plane.mip_count, .layers = 1}};
                    selected_depth_image->Transit(vk::ImageLayout::eTransferSrcOptimal,
                                                  vk::AccessFlagBits2::eTransferRead,
                                                  selected_range, cmdbuf);
                    auto* current_selected = texture_cache.ResolveDiagnosticColorScopeImage(
                        selected_plan.selected_token.link);
                    const bool current =
                        current_selected == selected_depth_image && current_selected->backing &&
                        current_selected->backing->subresource_states.empty() &&
                        current_selected->backing->num_samples == 1 &&
                        bool{current_selected->aspect_mask & vk::ImageAspectFlagBits::eDepth} &&
                        RawTerminalScopeDepthBytesPerTexel(current_selected->info.pixel_format) ==
                            4 &&
                        HostPasses::IsCurrentPpUpstreamSampleTeeSelectedDepthToken(
                            selected_plan.selected_token,
                            {.id = selected_plan.selected_token.link.id,
                             .uid = current_selected->image_uid},
                            current_selected->EnsureDiagnosticBackingGeneration(),
                            current_selected->DiagnosticMutationEpoch(), true);
                    if (current) {
                        const auto pre_copy_barrier =
                            HostPasses::MapPpUpstreamSampleTeeBarrierToVulkan(
                                HostPasses::PlanPpUpstreamSampleTeeSelectedDepthPreCopyBarrier(
                                    selected_plan, capture.slot_offset));
                        std::array<vk::BufferImageCopy, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>
                            selected_copies{};
                        for (u32 index = 0; index < selected_plan.plane.region_count; ++index) {
                            const auto& region = selected_plan.plane.regions[index];
                            selected_copies[index] = {
                                .bufferOffset = capture.slot_offset +
                                                selected_plan.plane.plane_offset +
                                                region.buffer_offset,
                                .bufferRowLength = 0,
                                .bufferImageHeight = 0,
                                .imageSubresource =
                                    {
                                        .aspectMask = vk::ImageAspectFlagBits::eDepth,
                                        .mipLevel = region.mip_level,
                                        .baseArrayLayer = selected_plan.plane.base_layer,
                                        .layerCount = 1,
                                    },
                                .imageOffset = {static_cast<s32>(region.x),
                                                static_cast<s32>(region.y), 0},
                                .imageExtent = {region.width, region.height, 1},
                            };
                        }
                        if (pre_copy_barrier.valid) {
                            const vk::BufferMemoryBarrier2 barrier{
                                .srcStageMask = pre_copy_barrier.src_stages,
                                .srcAccessMask = pre_copy_barrier.src_access,
                                .dstStageMask = pre_copy_barrier.dst_stages,
                                .dstAccessMask = pre_copy_barrier.dst_access,
                                .buffer = tee_buffer->Handle(),
                                .offset = pre_copy_barrier.offset,
                                .size = pre_copy_barrier.bytes,
                            };
                            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                                .bufferMemoryBarrierCount = 1,
                                .pBufferMemoryBarriers = &barrier,
                            });
                            cmdbuf.copyImageToBuffer(
                                current_selected->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                tee_buffer->Handle(), selected_plan.plane.region_count,
                                selected_copies.data());
                        } else {
                            selected_plan = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                                             .loss = {.invalidation = 1}};
                        }
                    } else {
                        selected_plan = {.status = FinalGuestSurfaceStatus::InvalidationLoss,
                                         .loss = {.invalidation = 1}};
                    }
                    const vk::ImageMemoryBarrier2 selected_restore{
                        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
                        .dstStageMask = old_selected_state.pl_stage,
                        .dstAccessMask = old_selected_state.access_mask,
                        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                        .newLayout = old_selected_state.layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = selected_depth_image->GetImage(),
                        .subresourceRange =
                            {
                                .aspectMask = vk::ImageAspectFlags{static_cast<VkImageAspectFlags>(
                                    aspect_restore.restore_aspect)},
                                .baseMipLevel = selected_range.base.level,
                                .levelCount = selected_range.extent.levels,
                                .baseArrayLayer = selected_range.base.layer,
                                .layerCount = selected_range.extent.layers,
                            },
                    };
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .imageMemoryBarrierCount = 1,
                        .pImageMemoryBarriers = &selected_restore,
                    });
                    HostPasses::RestorePpUpstreamInputUniformTracker(
                        selected_depth_image->backing->state,
                        selected_depth_image->backing->subresource_states, old_selected_state);
                }
            }
        }
        const auto old_state = image.backing->state;
        const VideoCore::SubresourceRange range{{.level = 0, .layer = 0},
                                                {.levels = 1, .layers = 1}};
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                      range, cmdbuf);
        cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                 download.Handle(), plan.region_count, copies.data());
        if (input_plan) {
            for (u32 input_index = 0; input_index < input_plan->input_count; ++input_index) {
                if ((input_plan->capture_mask & (1u << input_index)) == 0) {
                    continue;
                }
                auto& sampled = *sampled_images[input_index];
                const auto& input = input_plan->inputs[input_index];
                const auto sampled_old_state = sampled.backing->state;
                const VideoCore::SubresourceRange sampled_range{
                    {.level = input.base_mip, .layer = input.base_layer},
                    {.levels = input.mip_count, .layers = 1}};
                sampled.Transit(vk::ImageLayout::eTransferSrcOptimal,
                                vk::AccessFlagBits2::eTransferRead, sampled_range, cmdbuf);
                std::array<vk::BufferImageCopy, HostPasses::PpUpstreamInputContentPlane::MaxRegions>
                    input_copies{};
                for (u32 region_index = 0; region_index < input.region_count; ++region_index) {
                    const auto& region = input.regions[region_index];
                    input_copies[region_index] = {
                        .bufferOffset = slot_offset + input.plane_offset + region.buffer_offset,
                        .bufferRowLength = 0,
                        .bufferImageHeight = 0,
                        .imageSubresource =
                            {
                                .aspectMask = vk::ImageAspectFlagBits::eColor,
                                .mipLevel = region.mip_level,
                                .baseArrayLayer = input.base_layer,
                                .layerCount = 1,
                            },
                        .imageOffset = {static_cast<s32>(region.x), static_cast<s32>(region.y), 0},
                        .imageExtent = {region.width, region.height, 1},
                    };
                }
                cmdbuf.copyImageToBuffer(sampled.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                         download.Handle(), input.region_count,
                                         input_copies.data());
                const vk::ImageMemoryBarrier2 sampled_restore{
                    .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                    .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
                    .dstStageMask = sampled_old_state.pl_stage,
                    .dstAccessMask = sampled_old_state.access_mask,
                    .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .newLayout = sampled_old_state.layout,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = sampled.GetImage(),
                    .subresourceRange =
                        {
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = input.base_mip,
                            .levelCount = input.mip_count,
                            .baseArrayLayer = input.base_layer,
                            .layerCount = 1,
                        },
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &sampled_restore,
                });
                HostPasses::RestorePpUpstreamInputUniformTracker(
                    sampled.backing->state, sampled.backing->subresource_states, sampled_old_state);
            }
        }
        const vk::ImageMemoryBarrier2 restore{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = old_state.pl_stage,
            .dstAccessMask = old_state.access_mask,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = old_state.layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.GetImage(),
            .subresourceRange =
                {
                    .aspectMask = image.aspect_mask,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        const vk::BufferMemoryBarrier2 buffer_post{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
            .buffer = download.Handle(),
            .offset = slot_offset,
            .size = slot_stride,
        };
        std::array<vk::BufferMemoryBarrier2, 2> buffer_post_barriers{buffer_post};
        u32 buffer_post_count = 1;
        if (tee_preparation) {
            const auto& barrier =
                post ? tee_preparation->host_barrier : tee_preparation->fragment_barrier;
            buffer_post_barriers[buffer_post_count++] = {
                .srcStageMask = barrier.src_stages,
                .srcAccessMask = barrier.src_access,
                .dstStageMask = barrier.dst_stages,
                .dstAccessMask = barrier.dst_access,
                .buffer = tee_buffer->Handle(),
                .offset = barrier.offset,
                .size = barrier.bytes,
            };
        }
        const std::array image_barriers{restore};
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = buffer_post_count,
            .pBufferMemoryBarriers = buffer_post_barriers.data(),
            .imageMemoryBarrierCount = static_cast<u32>(image_barriers.size()),
            .pImageMemoryBarriers = image_barriers.data(),
        });
        image.backing->state = old_state;
        if (!scheduler.ResumeRenderingForDiagnostic(state, rendering_serial)) {
            return failure(HostPasses::PpUpstreamSampleTeeFailureScope::SharedCapture,
                           FinalGuestSurfaceStatus::InvalidationLoss, {.invalidation = 1});
        }
        return {.recorded = true, .status = FinalGuestSurfaceStatus::Complete};
    }

    void RecordPlane(Entry& entry, VideoCore::Image& image, const RenderState& state,
                     u32 attachment_index, u64 rendering_serial, u32 plane, bool finalize = false,
                     std::span<VideoCore::Image* const> sampled_images = {}) {
        if (entry.status != FinalGuestSurfaceStatus::Complete || entry.loss.Any()) {
            return;
        }
        if (entry.plan.status != FinalGuestSurfaceStatus::Complete || !entry.plan.copy ||
            attachment_index >= state.num_color_attachments || !scheduler.IsRendering()) {
            entry.status = FinalGuestSurfaceStatus::InvalidationLoss;
            entry.loss = {.invalidation = 1};
            return;
        }
        if (plane == 5 && config.content.capture_sampled_input_content) {
            u32 pointer_mask{};
            u32 backing_mask{};
            u32 view_match_mask{};
            u32 alias_mask{};
            const auto mip_dimension = [](u32 dimension, u32 level) {
                return level < 32 ? std::max(1u, dimension >> level) : 0u;
            };
            for (u32 input_index = 0; input_index < entry.input_content_plan.input_count;
                 ++input_index) {
                const u32 bit = 1u << input_index;
                VideoCore::Image* sampled =
                    input_index < sampled_images.size() ? sampled_images[input_index] : nullptr;
                if (sampled) {
                    pointer_mask |= bit;
                }
                if (sampled && sampled->backing) {
                    backing_mask |= bit;
                }
                const auto& expected = entry.sampled_inputs.inputs[input_index].view;
                const auto& planned = entry.input_content_plan.inputs[input_index];
                if (sampled && sampled->backing && expected.copy_eligible &&
                    expected.status == FinalGuestSurfaceStatus::Complete && !expected.loss.Any() &&
                    planned.base_mip == expected.base_mip &&
                    planned.base_layer == expected.base_layer &&
                    planned.format == expected.format &&
                    sampled->aspect_mask == vk::ImageAspectFlagBits::eColor &&
                    sampled->backing->num_samples == expected.samples &&
                    sampled->backing->subresource_states.empty() &&
                    sampled->backing->state.layout != vk::ImageLayout::eUndefined &&
                    mip_dimension(sampled->info.size.width, planned.base_mip) == expected.width &&
                    mip_dimension(sampled->info.size.height, planned.base_mip) == expected.height) {
                    view_match_mask |= bit;
                }
                if (entry.sampled_inputs.inputs[input_index].aliases_output || sampled == &image) {
                    alias_mask |= bit;
                }
            }
            const auto copy_decision = PlanPpTerminalScopeSampledInputCopyDecision({
                .enabled = true,
                .input_count = entry.input_content_plan.input_count,
                .capture_mask = entry.input_content_plan.capture_mask,
                .pointer_mask = pointer_mask,
                .backing_mask = backing_mask,
                .view_match_mask = view_match_mask,
                .alias_mask = alias_mask,
            });
            if (!copy_decision.copy) {
                entry.status = copy_decision.status;
                entry.loss = copy_decision.loss;
                return;
            }
        }
        const auto slot_decision = PlanPpTerminalScopePlaneSlot(plane, bool{entry.slot});
        if (slot_decision.status != FinalGuestSurfaceStatus::Complete) {
            entry.status = slot_decision.status;
            entry.loss = slot_decision.loss;
            return;
        }
        if (slot_decision.acquire) {
            const auto token = slots.TryAcquire();
            if (!token) {
                entry.status = FinalGuestSurfaceStatus::BusyLoss;
                entry.loss = {.busy = 1};
                return;
            }
            entry.slot = *token;
        }
        const auto split = PlanPpTerminalScopeRenderingSplit(
            scheduler.IsRendering(), scheduler.RenderingSerial(), rendering_serial);
        if (split.status != FinalGuestSurfaceStatus::Complete) {
            entry.status = split.status;
            entry.loss = split.loss;
            return;
        }
        const u64 slot_offset = entry.slot.slot * slot_stride;
        const auto cmdbuf = scheduler.CommandBuffer();
        scheduler.EndRendering();
        if (slot_decision.acquire) {
            const vk::BufferMemoryBarrier2 pre_barrier{
                .srcStageMask =
                    vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eHost,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eHostRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .buffer = download.Handle(),
                .offset = slot_offset,
                .size = slot_stride,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &pre_barrier,
            });
        } else if (slot_decision.requires_write_barrier) {
            const vk::BufferMemoryBarrier2 overwrite_barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .buffer = download.Handle(),
                .offset = slot_offset,
                .size = slot_stride,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &overwrite_barrier,
            });
        }
        const auto old_state = image.backing->state;
        const VideoCore::SubresourceRange range{{.level = 0, .layer = 0},
                                                {.levels = 1, .layers = 1}};
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                      range, cmdbuf);
        std::array<vk::BufferImageCopy, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> copies{};
        const u64 plane_offset = plane == 0   ? entry.plan.first_plane_offset
                                 : plane == 1 ? entry.plan.second_plane_offset
                                 : plane == 2 ? entry.plan.consumer_plane_offset
                                 : plane == 3 ? entry.plan.output_plane_offset
                                 : plane == 4 ? entry.plan.pre_first_plane_offset
                                 : plane == 5 ? entry.plan.predecessor_pre_plane_offset
                                              : entry.plan.predecessor_post_plane_offset;
        for (u32 index = 0; index < entry.plan.region_count; ++index) {
            const auto& region = entry.plan.regions[index];
            copies[index] = {
                .bufferOffset = slot_offset + plane_offset + region.buffer_offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = {static_cast<s32>(region.x), static_cast<s32>(region.y), 0},
                .imageExtent = {region.width, region.height, 1},
            };
        }
        cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                 download.Handle(), entry.plan.region_count, copies.data());
        const vk::ImageMemoryBarrier2 restore{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = old_state.pl_stage,
            .dstAccessMask = old_state.access_mask,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = old_state.layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.GetImage(),
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &restore,
        });
        image.backing->state = old_state;
        if (plane == 5 && config.content.capture_sampled_input_content) {
            for (u32 input_index = 0; input_index < entry.input_content_plan.input_count;
                 ++input_index) {
                if ((entry.input_content_plan.capture_mask & (1u << input_index)) == 0) {
                    continue;
                }
                auto& sampled = *sampled_images[input_index];
                const auto& input = entry.input_content_plan.inputs[input_index];
                const auto sampled_old_state = sampled.backing->state;
                const VideoCore::SubresourceRange sampled_range{
                    {.level = input.base_mip, .layer = input.base_layer},
                    {.levels = 1, .layers = 1}};
                sampled.Transit(vk::ImageLayout::eTransferSrcOptimal,
                                vk::AccessFlagBits2::eTransferRead, sampled_range, cmdbuf);
                std::array<vk::BufferImageCopy, FinalGuestSurfaceWatchOrdinals::MaxOrdinals>
                    input_copies{};
                for (u32 region_index = 0; region_index < input.region_count; ++region_index) {
                    const auto& region = input.regions[region_index];
                    input_copies[region_index] = {
                        .bufferOffset = slot_offset + input.plane_offset + region.buffer_offset,
                        .bufferRowLength = 0,
                        .bufferImageHeight = 0,
                        .imageSubresource =
                            {
                                .aspectMask = vk::ImageAspectFlagBits::eColor,
                                .mipLevel = input.base_mip,
                                .baseArrayLayer = input.base_layer,
                                .layerCount = 1,
                            },
                        .imageOffset = {static_cast<s32>(region.x), static_cast<s32>(region.y), 0},
                        .imageExtent = {region.width, region.height, 1},
                    };
                }
                cmdbuf.copyImageToBuffer(sampled.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                         download.Handle(), input.region_count,
                                         input_copies.data());
                const vk::ImageMemoryBarrier2 sampled_restore{
                    .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                    .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
                    .dstStageMask = sampled_old_state.pl_stage,
                    .dstAccessMask = sampled_old_state.access_mask,
                    .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .newLayout = sampled_old_state.layout,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = sampled.GetImage(),
                    .subresourceRange =
                        {
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = input.base_mip,
                            .levelCount = 1,
                            .baseArrayLayer = input.base_layer,
                            .layerCount = 1,
                        },
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &sampled_restore,
                });
                sampled.backing->state = sampled_old_state;
            }
        }
        if (finalize) {
            const vk::BufferMemoryBarrier2 post_barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                .dstAccessMask = vk::AccessFlagBits2::eHostRead,
                .buffer = download.Handle(),
                .offset = slot_offset,
                .size = slot_stride,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &post_barrier,
            });
        }
        if (!scheduler.ResumeRenderingForDiagnostic(state, rendering_serial)) {
            entry.status = FinalGuestSurfaceStatus::InvalidationLoss;
            entry.loss = {.invalidation = 1};
        } else {
            entry.recorded_plane_mask |= 1u << plane;
        }
    }

    [[nodiscard]] static PpTerminalScopeContentHistoryLayout MakeUpstreamFeedbackHistoryLayout(
        const HostPasses::PpUpstreamFeedbackPrePostPlan& plan,
        const HostPasses::PpUpstreamInputContentPlan& input_plan,
        const HostPasses::PpUpstreamInputContentCompactResult& input_compact,
        const HostPasses::PpUpstreamSampleRouteSet& sample_routes) noexcept {
        PpTerminalScopeContentHistoryLayout layout{
            .region_count = plan.region_count,
            .plane_bytes = plan.plane_bytes,
            .second_plane_offset = plan.plane_bytes,
            .total_bytes = plan.plane_bytes * 2,
            .input_count = input_plan.input_count,
            .input_capture_mask = input_compact.capture_mask | input_compact.alias_mask,
            .input_unavailable_mask = input_compact.unavailable_mask,
            .input_alias_mask = input_compact.alias_mask,
            .plane_mask = 0x3,
            .bytes_per_pixel = plan.bytes_per_pixel,
            .format = plan.format,
            .raw_compare = plan.raw_compare,
            .upstream_sample_routes = sample_routes,
        };
        for (u32 index = 0; index < plan.region_count; ++index) {
            layout.regions[index] = {
                .logical_ordinal = plan.regions[index].logical_ordinal,
                .buffer_offset = plan.regions[index].buffer_offset,
                .byte_size = plan.regions[index].byte_size,
            };
        }
        const u32 input_base = layout.total_bytes;
        for (u32 input_index = 0; input_index < input_plan.input_count; ++input_index) {
            const u32 bit = 1u << input_index;
            auto& destination = layout.input_planes[input_index];
            const auto& source = input_plan.inputs[input_index];
            if ((input_compact.alias_mask & bit) != 0) {
                destination = {
                    .region_count = plan.region_count,
                    .plane_offset = 0,
                    .plane_bytes = plan.plane_bytes,
                    .format = plan.format,
                    .bytes_per_texel = plan.bytes_per_pixel,
                    .raw_compare = plan.raw_compare,
                    .status = FinalGuestSurfaceStatus::Complete,
                };
                for (u32 region_index = 0; region_index < plan.region_count; ++region_index) {
                    destination.regions[region_index] = {
                        .logical_ordinal = plan.regions[region_index].logical_ordinal,
                        .buffer_offset = plan.regions[region_index].buffer_offset,
                        .byte_size = plan.regions[region_index].byte_size,
                    };
                }
                continue;
            }
            if ((input_compact.capture_mask & bit) == 0) {
                destination.status = source.status;
                destination.loss = source.loss;
                continue;
            }
            destination = {
                .region_count = source.region_count,
                .plane_offset = input_base + input_compact.plane_offsets[input_index],
                .plane_bytes = source.plane_bytes,
                .format = FinalGuestSurfaceFormat::Unsupported,
                .bytes_per_texel = source.bytes_per_texel,
                .raw_compare = true,
                .status = FinalGuestSurfaceStatus::Complete,
            };
            for (u32 region_index = 0; region_index < source.region_count; ++region_index) {
                destination.regions[region_index] = {
                    .logical_ordinal = source.regions[region_index].logical_ordinal,
                    .buffer_offset = source.regions[region_index].buffer_offset,
                    .byte_size = source.regions[region_index].byte_size,
                };
            }
        }
        layout.total_bytes += static_cast<u32>(input_compact.bytes.size());
        return layout;
    }

    void TakeUpstreamFeedbackAtFlip(u64 sequence, u64 process_time_us) {
        LOG_INFO(Render, "FGSCPRW s={} v={} c={} h={:#x}", sequence,
                 target_sampled_depth_writer_valid, target_sampled_depth_writer_collision,
                 target_sampled_depth_writer_hash);
        target_sampled_depth_writer_hash = 0;
        target_sampled_depth_writer_valid = false;
        target_sampled_depth_writer_collision = false;
        const auto& ff = target_fixed_function_state;
        LOG_INFO(Render,
                 "FGSCFF s={} v={} c={} de={} dw={} df={} cf={} cp={} ze={} ke={} zo={} ad={} cv={} me={} ns={} mm={:#x} lo={} be={} cs={} cd={} co={} as={} adf={} ao={} wm={:#x} sm={:#x} rm={}",
                 sequence, target_fixed_function_state_valid,
                 target_fixed_function_state_collision, ff.depth_enable, ff.depth_write_enable,
                 ff.depth_compare, ff.color_on_depth_fail, ff.no_color_on_depth_pass,
                 ff.z_export_enable, ff.kill_enable, ff.z_order, ff.alpha_to_mask_disable,
                 ff.coverage_to_mask_enable, ff.mask_export_enable, ff.num_samples, ff.mrt_mask,
                 ff.logic_op, ff.blend_enable, ff.color_src_factor, ff.color_dst_factor,
                 ff.color_blend_op, ff.alpha_src_factor, ff.alpha_dst_factor, ff.alpha_blend_op,
                 ff.write_mask, ff.shader_mask, ff.round_mode);
        target_fixed_function_state = {};
        target_fixed_function_state_valid = false;
        target_fixed_function_state_collision = false;
        const auto scope_writer_trace_frame_for_pending =
            upstream_sample_tee_scope_writer_trace_frame;
        const auto scope_writer_trace_tail_for_pending =
            upstream_sample_tee_scope_writer_trace_tail;
        const auto scope_load_candidates =
            upstream_sample_tee_scope_load_candidate_evidence.TakeWithPrivateTraceArm(
                {.sequence = sequence});
        LOG_INFO(Render, "{}",
                 HostPasses::FormatPpUpstreamSampleTeeScopeWriterTraceGate({
                     .sequence = sequence,
                     .frame_exact = upstream_sample_tee_scope_writer_trace_frame.exact,
                     .incoming_arm_exact = scope_load_candidates.private_trace_arm.exact,
                     .incoming_writer_count =
                         scope_load_candidates.private_trace_arm.writer_count,
                     .incoming_hints_valid = std::ranges::all_of(
                         std::span{scope_load_candidates.private_trace_arm.writer_hints.data(),
                                   std::min<u32>(
                                       scope_load_candidates.private_trace_arm.writer_count,
                                       scope_load_candidates.private_trace_arm.writer_hints.size())},
                         [](const auto& hint) { return hint.Valid(); }),
                     .incoming_shapes_valid = std::ranges::all_of(
                         std::span{scope_load_candidates.private_trace_arm.writer_shapes.data(),
                                   std::min<u32>(
                                       scope_load_candidates.private_trace_arm.writer_count,
                                       scope_load_candidates.private_trace_arm.writer_shapes.size())},
                         [](const auto& shape) { return shape.Valid(); }),
                     .live_pre_draws = upstream_sample_tee_scope_writer_trace_live_pre_draws,
                     .key_matches = upstream_sample_tee_scope_writer_trace_key_matches,
                     .armed_key_matches =
                         upstream_sample_tee_scope_writer_trace_armed_key_matches,
                     .capture_attempts =
                         upstream_sample_tee_scope_writer_trace_capture_attempts,
                     .pre_complete = upstream_sample_tee_scope_writer_trace_pre_complete,
                     .post_complete = upstream_sample_tee_scope_writer_trace_post_complete,
                     .tee_attempted = upstream_sample_tee_attempted,
                 }));
        upstream_sample_tee_scope_writer_trace_live_pre_draws = 0;
        upstream_sample_tee_scope_writer_trace_key_occurrence_count = 0;
        upstream_sample_tee_scope_writer_trace_key_matches = 0;
        upstream_sample_tee_scope_writer_trace_armed_key_matches = 0;
        upstream_sample_tee_scope_writer_trace_capture_attempts = 0;
        upstream_sample_tee_scope_writer_trace_pre_complete = 0;
        upstream_sample_tee_scope_writer_trace_post_complete = 0;
        const auto scope_writer_trace_advance =
            HostPasses::AdvancePpUpstreamSampleTeeScopeWriterTraceFrame({
                .current = upstream_sample_tee_scope_writer_trace_frame,
                .incoming_arm = scope_load_candidates.private_trace_arm,
                .cycle_index = upstream_sample_tee_scope_writer_trace_cycle_index,
                .current_captured = upstream_sample_tee_scope_writer_trace_captured,
            });
        upstream_sample_tee_scope_writer_trace_frame = scope_writer_trace_advance.frame;
        upstream_sample_tee_scope_writer_trace_cycle_index =
            scope_writer_trace_advance.next_cycle_index;
        upstream_sample_tee_scope_writer_trace_tail = {};
        upstream_sample_tee_scope_writer_trace_intent = {};
        upstream_sample_tee_scope_writer_trace_expected_serial = 0;
        upstream_sample_tee_scope_writer_trace_host_instance = 0;
        upstream_sample_tee_scope_writer_trace_captured = false;
        upstream_sample_tee_scope_writer_trace_open = false;
        upstream_sample_tee_scope_load_candidate_evidence.BeginFrame(
            scheduler.RenderingSerial());
        LOG_INFO(
            Render, "{}",
            HostPasses::FormatPpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidence(
                scope_load_candidates.report));
        auto producer_pre_occurrences =
            upstream_sample_tee_producer_pre_occurrences.TakeWithSelectedPredecessorPair(
                {.sequence = sequence});
        LOG_INFO(Render, "{}",
                 HostPasses::FormatPpUpstreamSampleTeeProducerPreDepthOccurrenceReport(
                     producer_pre_occurrences.report));
        upstream_sample_tee_selected_predecessor_pair =
            std::move(producer_pre_occurrences.private_pair);
        const bool selected_producer_occurrence_frame_exact = HostPasses::
            IsCompletePpUpstreamSampleTeeProducerPreDepthSelectedProducerOccurrenceFrame(
                upstream_sample_tee_selected_producer_ordinal_arm,
                upstream_sample_tee_selected_producer_occurrence_count);
        if (upstream_sample_tee_selected_producer_ordinal_arm.valid &&
            !selected_producer_occurrence_frame_exact) {
            upstream_sample_tee_producer_pre_collision = true;
            upstream_sample_tee_producer_pre_bound = false;
        }
        upstream_sample_tee_selected_producer_occurrence_count = 0;
        upstream_sample_tee_selected_producer_ordinal_arm =
            std::move(producer_pre_occurrences.private_selected_producer_arm);
        const auto route_absent_ordinal_capture =
            upstream_sample_tee_route_absent_ordinal_capture_evidence.Take({
                .sequence = sequence,
            });
        LOG_INFO(Render, "{}",
                 HostPasses::
                     FormatPpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureReport(
                         route_absent_ordinal_capture));
        const u32 fresh_selected_pair_count =
            producer_pre_occurrences.report.learned_predecessor_valid_count;
        const bool predecessor_writer_pre_active =
            upstream_sample_tee_predecessor_writer_pre_hint.valid ||
            upstream_sample_tee_predecessor_writer_pre_attempted ||
            upstream_sample_tee_predecessor_writer_pre_depth.copy ||
            upstream_sample_tee_predecessor_writer_pre_depth.status !=
                FinalGuestSurfaceStatus::AlreadyConsumed;
        const bool scope_entry_active =
            upstream_sample_tee_producer_pre_scope_entry.valid ||
            upstream_sample_tee_producer_pre_scope_entry_collision;
        const auto reset_scope_entry = [this] {
            upstream_sample_tee_producer_pre_scope_entry =
                HostPasses::ResetPpUpstreamSampleTeeProducerPreDepthScopeEntry(
                    upstream_sample_tee_producer_pre_scope_entry);
            upstream_sample_tee_producer_pre_scope_entry_collision = false;
            upstream_sample_tee_producer_pre_scope_plan = {};
            upstream_sample_tee_producer_pre_scope_post = {};
            upstream_sample_tee_producer_pre_scope_owned = {};
        };
        if (!config.content.capture_upstream_pre_post ||
            (!upstream_feedback_armed && !predecessor_writer_pre_active)) {
            if (scope_entry_active) {
                auto frame_entry = upstream_sample_tee_producer_pre_scope_entry;
                if (upstream_sample_tee_producer_pre_scope_entry_collision) {
                    HostPasses::PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation
                        collision{};
                    collision.preserved_entry = frame_entry;
                    collision.status = FinalGuestSurfaceStatus::GapLoss;
                    collision.loss.gap = 1;
                    collision.collision = true;
                    frame_entry = collision;
                }
                (void)HostPasses::
                    FinishPpUpstreamSampleTeeProducerPreDepthScopeEntryResourceFrame({
                        .entry = frame_entry,
                        .target_attempted = upstream_sample_tee_producer_pre_attempted,
                        .upstream_feedback_armed = upstream_feedback_armed,
                        .outer_slot_present = static_cast<bool>(upstream_feedback_slot),
                });
                reset_scope_entry();
            }
            const auto ordinal_frame = HostPasses::
                FinishPpUpstreamSampleTeeProducerPreDepthOrdinalContentFrame({
                    .sequence = sequence,
                    .prior_arm = upstream_sample_tee_producer_pre_ordinal_arm,
                    .inactive_flip_count =
                        upstream_sample_tee_producer_pre_ordinal_inactive_flip_count,
                    .fresh_selected_pair = upstream_sample_tee_selected_predecessor_pair,
                     .fresh_selected_pair_count = fresh_selected_pair_count,
                     .frame_inactive = true,
                     .route_absent_ordinal_capture = route_absent_ordinal_capture,
                 });
            upstream_sample_tee_producer_pre_ordinal_arm = ordinal_frame.next_arm;
            upstream_sample_tee_producer_pre_ordinal_inactive_flip_count =
                ordinal_frame.next_inactive_flip_count;
            LOG_INFO(
                Render, "{}",
                HostPasses::FormatPpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameReport(
                    ordinal_frame.report));
            upstream_sample_tee_predecessor_writer_pre_ordinal_capture_descriptor = {};
            upstream_sample_tee_predecessor_writer_pre_ordinal_capture_plan = {};
            upstream_sample_tee_producer_pre_ordinal_capture_descriptor = {};
            upstream_sample_tee_producer_pre_ordinal_capture_plan = {};
            upstream_sample_tee_producer_pre_diagnostic_immediate_predecessor = {};
            upstream_sample_tee_producer_pre_selected_binding_descriptor = {};
            upstream_sample_tee_selected_predecessor_pair = {};
            return;
        }
        const auto slot = upstream_feedback_slot;
        upstream_feedback_slot = {};
        if (HostPasses::ShouldUseConfiguredDepthCandidateFallback(
                config.content.capture_upstream_depth_pre_post,
                config.content.capture_upstream_depth_predecessor_hint) &&
            upstream_feedback_frame.SelectCandidate(config.content.upstream_candidate_index)) {
            LOG_INFO(Render, "FGSCTUS sel=self_flip");
        }
        const auto tee_selected_candidate_index =
            upstream_sample_tee_controller.candidate_index;
        const auto tee_selected_recorded_plane_mask =
            upstream_feedback_frame.RecordedPlaneMask(tee_selected_candidate_index);
        const HostPasses::PpUpstreamFeedbackTeeAuthorizedTargetPrePostReceipt
            tee_target_pre_post_receipt{
                .outer_slot = slot,
                .download_offset = slot ? slot.slot * slot_stride : 0,
                .download_bytes = slot_stride,
                .rendering_serial = upstream_sample_tee_controller.serial,
                .selected_candidate_index = tee_selected_candidate_index,
                .recorded_plane_mask = tee_selected_recorded_plane_mask,
            };
        const auto selection_take = HostPasses::TakePpUpstreamFeedbackSelectionAuthority(
            upstream_feedback_selection_authority, upstream_feedback_frame, bool{slot});
        LOG_INFO(Render, "{}",
                 HostPasses::FormatPpUpstreamFeedbackSelectionAuthorityReport(
                     HostPasses::MakePpUpstreamFeedbackSelectionAuthorityReport(
                         sequence, selection_take.authority)));
        LOG_INFO(Render, "{}",
                 HostPasses::FormatPpUpstreamFeedbackProducerAdmissionReport(
                     HostPasses::MakePpUpstreamFeedbackProducerAdmissionReport(
                         sequence, selection_take.authority.producer_admission)));
        const auto& take_evaluation = selection_take.evaluation;
        LOG_INFO(Render, "{}",
                 HostPasses::FormatPpUpstreamFeedbackPrePostTakeReport(
                     HostPasses::MakePpUpstreamFeedbackPrePostTakeReport(sequence,
                                                                         take_evaluation)));
        auto take = take_evaluation.take;
        const auto tee_outcome = HostPasses::ResolvePpUpstreamSampleTeeCaptureOutcome(
            {
                .status = upstream_sample_tee_status,
                .loss = upstream_sample_tee_loss,
                .copy = upstream_sample_tee_copy,
            },
            {
                .status = take.status,
                .loss = take.loss,
                .copy = take.copy,
            });
        if (upstream_sample_tee_selected_consumer_count != 0) {
            const auto& selected = upstream_sample_tee_selected_depth;
            auto* live_selected = texture_cache.ResolveDiagnosticColorScopeImage(
                selected.selected_token.link);
            const bool selected_current =
                live_selected && live_selected->backing &&
                HostPasses::IsCurrentPpUpstreamSampleTeeSelectedDepthToken(
                    selected.selected_token,
                    {.id = selected.selected_token.link.id,
                     .uid = live_selected->image_uid},
                    live_selected->EnsureDiagnosticBackingGeneration(),
                    live_selected->DiagnosticMutationEpoch(), true);
            upstream_sample_tee_selected_drift |=
                !HostPasses::IsPpUpstreamSampleTeeSelectedDepthPlanRecordable(selected) ||
                !upstream_sample_tee_source.valid ||
                upstream_sample_tee_source.binding.link != selected.selected_token.link ||
                upstream_sample_tee_source.binding.backing_generation !=
                    selected.selected_token.backing_generation ||
                upstream_sample_tee_source.binding.mutation_epoch !=
                    selected.selected_token.mutation_epoch ||
                !selected_current;
        }
        if (upstream_sample_tee_predecessor_writer_pre_depth.copy) {
            upstream_sample_tee_predecessor_writer_pre_depth.base_total_bytes =
                take.plan.total_bytes;
            upstream_sample_tee_predecessor_writer_pre_depth.input_total_bytes =
                upstream_input_ready ? upstream_input_plan.total_bytes : take.plan.total_bytes;
            upstream_sample_tee_predecessor_writer_pre_depth = HostPasses::
                ValidatePpUpstreamSampleTeePredecessorWriterPreDepthDownloadTail(
                    upstream_sample_tee_predecessor_writer_pre_depth);
        }
        const auto ordinal_tails = HostPasses::PlanPpUpstreamSampleTeeDepthDownloadTails({
            .predecessor_writer_pre = upstream_sample_tee_predecessor_writer_pre_depth,
            .producer_pre = upstream_sample_tee_producer_pre_depth,
        });
        const bool ordinal_selected_exact =
            !upstream_sample_tee_selected_drift &&
            upstream_sample_tee_selected_consumer_count ==
                upstream_sample_tee_exact_selected_consumer_count;
        upstream_sample_tee_producer_pre_ordinal_binding = HostPasses::
            BindPpUpstreamSampleTeeProducerPreDepthOrdinalContentToFreshSelectedPair({
                .prior_arm = upstream_sample_tee_producer_pre_ordinal_arm,
                .fresh_selected_pair = upstream_sample_tee_selected_predecessor_pair,
                .fresh_selected_pair_count = fresh_selected_pair_count,
                .combined_tails = ordinal_tails,
                .predecessor_pre_plan = upstream_sample_tee_predecessor_writer_pre_depth,
                .predecessor_pre_intent = upstream_sample_tee_predecessor_writer_pre_intent,
                .predecessor_post = upstream_sample_tee_predecessor_writer_pre_post,
                .predecessor_scope_writer_count =
                    upstream_sample_tee_producer_pre_ordinal_arm
                        .predecessor_scope_writer_count,
                .predecessor_pre_observation_count =
                    upstream_sample_tee_predecessor_writer_pre_attempted ? 1u : 0u,
                .predecessor_post_observation_count =
                    upstream_sample_tee_predecessor_writer_pre_post.exact ? 1u : 0u,
                .predecessor_open = upstream_sample_tee_predecessor_writer_pre_open,
                .predecessor_collision = upstream_sample_tee_predecessor_writer_pre_collision,
                .target_pre_plan = upstream_sample_tee_producer_pre_depth,
                .target_pre_intent = upstream_sample_tee_producer_pre_intent,
                .target_post = upstream_sample_tee_producer_pre_post,
                .target_scope_writer_count =
                    upstream_sample_tee_producer_pre_ordinal_arm.target_scope_writer_count,
                .target_pre_observation_count =
                    upstream_sample_tee_producer_pre_attempted ? 1u : 0u,
                .target_post_observation_count =
                    upstream_sample_tee_producer_pre_post.exact ? 1u : 0u,
                .target_open = upstream_sample_tee_producer_pre_open,
                .target_collision = upstream_sample_tee_producer_pre_collision,
                .selected_binding = upstream_sample_tee_producer_pre_selected_binding,
                .selected_binding_count = upstream_sample_tee_exact_selected_consumer_count,
                .later_route_exact = ordinal_selected_exact,
                .later_view_exact = ordinal_selected_exact,
                .later_selected_token_current = ordinal_selected_exact,
            });
        HostPasses::PpUpstreamSampleTeeProducerPreDepthOrdinalDirectSelectedFirstFailure
            direct_selected_first_failure{};
        if (fresh_selected_pair_count == 0 &&
            upstream_sample_tee_selected_consumer_count != 0) {
            const auto direct_selected_evaluation = HostPasses::
                EvaluatePpUpstreamSampleTeeProducerPreDepthOrdinalContentToDirectSelected({
                    .sequence = sequence,
                    .prior_arm = upstream_sample_tee_producer_pre_ordinal_arm,
                    .fresh_selected_pair_count = fresh_selected_pair_count,
                    .combined_tails = ordinal_tails,
                    .predecessor_pre_plan =
                        upstream_sample_tee_predecessor_writer_pre_depth,
                    .predecessor_pre_intent =
                        upstream_sample_tee_predecessor_writer_pre_intent,
                    .predecessor_post = upstream_sample_tee_predecessor_writer_pre_post,
                    .predecessor_scope_writer_count =
                        upstream_sample_tee_producer_pre_ordinal_arm
                            .predecessor_scope_writer_count,
                    .predecessor_pre_observation_count =
                        upstream_sample_tee_predecessor_writer_pre_attempted ? 1u : 0u,
                    .predecessor_post_observation_count =
                        upstream_sample_tee_predecessor_writer_pre_post.exact ? 1u : 0u,
                    .predecessor_ordinal_capture_descriptor =
                        upstream_sample_tee_predecessor_writer_pre_ordinal_capture_descriptor,
                    .predecessor_ordinal_capture_plan =
                        upstream_sample_tee_predecessor_writer_pre_ordinal_capture_plan,
                    .predecessor_ordinal_capture_count =
                        upstream_sample_tee_predecessor_writer_pre_attempted ? 1u : 0u,
                    .predecessor_open = upstream_sample_tee_predecessor_writer_pre_open,
                    .predecessor_collision =
                        upstream_sample_tee_predecessor_writer_pre_collision,
                    .target_pre_plan = upstream_sample_tee_producer_pre_depth,
                    .target_pre_intent = upstream_sample_tee_producer_pre_intent,
                    .target_pre_diagnostic_immediate_predecessor =
                        upstream_sample_tee_producer_pre_diagnostic_immediate_predecessor,
                    .target_pre_diagnostic_immediate_predecessor_count =
                        upstream_sample_tee_producer_pre_attempted ? 1u : 0u,
                    .target_post = upstream_sample_tee_producer_pre_post,
                    .target_scope_writer_count =
                        upstream_sample_tee_producer_pre_ordinal_arm.target_scope_writer_count,
                    .target_pre_observation_count =
                        upstream_sample_tee_producer_pre_attempted ? 1u : 0u,
                    .target_post_observation_count =
                        upstream_sample_tee_producer_pre_post.exact ? 1u : 0u,
                    .target_ordinal_capture_descriptor =
                        upstream_sample_tee_producer_pre_ordinal_capture_descriptor,
                    .target_ordinal_capture_plan =
                        upstream_sample_tee_producer_pre_ordinal_capture_plan,
                    .target_ordinal_capture_count =
                        upstream_sample_tee_producer_pre_attempted ? 1u : 0u,
                    .target_open = upstream_sample_tee_producer_pre_open,
                    .target_collision = upstream_sample_tee_producer_pre_collision,
                    .selected_binding_descriptor =
                        upstream_sample_tee_producer_pre_selected_binding_descriptor,
                    .selected_binding = upstream_sample_tee_producer_pre_selected_binding,
                    .selected_binding_count =
                        upstream_sample_tee_exact_selected_consumer_count,
                    .occurrence_report = producer_pre_occurrences.report,
                    .route_absent_ordinal_capture = route_absent_ordinal_capture,
                    .later_route_exact = ordinal_selected_exact,
                    .later_view_exact = ordinal_selected_exact,
                    .later_selected_token_current = ordinal_selected_exact,
                });
            upstream_sample_tee_producer_pre_ordinal_binding =
                direct_selected_evaluation.binding;
            direct_selected_first_failure = direct_selected_evaluation.first_failure;
        }
        upstream_sample_tee_predecessor_writer_pre_bound =
            upstream_sample_tee_producer_pre_ordinal_binding.publish_predecessor_pre;
        const auto ordinal_frame = HostPasses::
            FinishPpUpstreamSampleTeeProducerPreDepthOrdinalContentFrame({
                .sequence = sequence,
                .prior_arm = upstream_sample_tee_producer_pre_ordinal_arm,
                .inactive_flip_count =
                    upstream_sample_tee_producer_pre_ordinal_inactive_flip_count,
                .fresh_selected_pair = upstream_sample_tee_selected_predecessor_pair,
                .fresh_selected_pair_count = fresh_selected_pair_count,
                .frame_active = upstream_sample_tee_predecessor_writer_pre_attempted ||
                                upstream_sample_tee_producer_pre_attempted,
                .predecessor_capture_count =
                    upstream_sample_tee_predecessor_writer_pre_attempted ? 1u : 0u,
                .predecessor_post_count =
                    upstream_sample_tee_predecessor_writer_pre_post.exact ? 1u : 0u,
                .target_capture_count = upstream_sample_tee_producer_pre_attempted ? 1u : 0u,
                .target_post_count = upstream_sample_tee_producer_pre_post.exact ? 1u : 0u,
                .ordinal_binding = upstream_sample_tee_producer_pre_ordinal_binding,
                .direct_selected_first_failure = direct_selected_first_failure,
                .predecessor_collision = upstream_sample_tee_predecessor_writer_pre_collision,
                 .target_collision = upstream_sample_tee_producer_pre_collision,
                 .occurrence_report = producer_pre_occurrences.report,
                 .route_absent_ordinal_capture = route_absent_ordinal_capture,
             });
        const bool ordinal_retry_scheduled =
            ordinal_frame.disposition == HostPasses::
                PpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameDisposition::
                    RetryScheduled;
        const auto ordinal_retry_cleanup = ordinal_retry_scheduled
            ? HostPasses::DiscardPpUpstreamSampleTeeProducerPreDepthOrdinalCaptureForRetry({
                  .frame = ordinal_frame,
                  .outer_slot = slot,
                  .predecessor_writer_pre_depth =
                      upstream_sample_tee_predecessor_writer_pre_depth,
                  .producer_pre_depth = upstream_sample_tee_producer_pre_depth,
                  .ordinal_content_binding = upstream_sample_tee_producer_pre_ordinal_binding,
                  .ordinal_depth_download_tails = ordinal_tails,
              })
            : HostPasses::PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureRetryDiscard{};
        const bool ordinal_retry_exact =
            ordinal_retry_scheduled &&
            ordinal_retry_cleanup.status == FinalGuestSurfaceStatus::Complete &&
            !ordinal_retry_cleanup.loss.Any() &&
            ordinal_retry_cleanup.release_only_outer_slot;
        upstream_sample_tee_producer_pre_ordinal_arm =
            ordinal_retry_scheduled && !ordinal_retry_exact
                ? HostPasses::PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm{}
                : ordinal_frame.next_arm;
        upstream_sample_tee_producer_pre_ordinal_inactive_flip_count =
            ordinal_retry_scheduled && !ordinal_retry_exact
                ? 0u
                : ordinal_frame.next_inactive_flip_count;
        LOG_INFO(
            Render, "{}",
            HostPasses::FormatPpUpstreamSampleTeeProducerPreDepthOrdinalContentFrameReport(
                ordinal_frame.report));
        if (!ordinal_retry_exact && !upstream_sample_tee_predecessor_writer_pre_bound &&
            upstream_sample_tee_predecessor_writer_pre_attempted) {
            upstream_sample_tee_predecessor_writer_pre_depth = {
                .status = upstream_sample_tee_producer_pre_ordinal_binding.status,
                .loss = upstream_sample_tee_producer_pre_ordinal_binding.loss,
            };
        }
        auto producer_pre_for_pending = upstream_sample_tee_producer_pre_depth;
        if (upstream_sample_tee_producer_pre_collision) {
            producer_pre_for_pending = {
                .status = FinalGuestSurfaceStatus::GapLoss,
                .loss = {.gap = 1},
            };
        }
        const auto producer_pre_gate =
            upstream_sample_tee_producer_pre_gate_evidence.Take({
                .sequence = sequence,
                .prior_hint_valid = upstream_sample_tee_producer_pre_hint.valid,
            });
        LOG_INFO(Render, "{}",
                 HostPasses::FormatPpUpstreamSampleTeeProducerPreDepthGateEvidence(
                     producer_pre_gate));
        const auto producer_pre_frame = HostPasses::FinishPpUpstreamSampleTeeProducerPreDepthFrame({
            .prior_hint = upstream_sample_tee_producer_pre_hint,
            .observed_selected_hint = upstream_sample_tee_producer_pre_observed_hint,
            .selected_consumer_count = upstream_sample_tee_selected_consumer_count,
            .exact_selected_consumer_count = upstream_sample_tee_exact_selected_consumer_count,
            .producer_pre_attempted = upstream_sample_tee_producer_pre_attempted,
            .producer_pre_open = upstream_sample_tee_producer_pre_open,
            .producer_pre_bound = upstream_sample_tee_producer_pre_bound,
            .selected_route_exact = !upstream_sample_tee_selected_drift &&
                                    upstream_sample_tee_selected_consumer_count ==
                                        upstream_sample_tee_exact_selected_consumer_count,
            .selected_view_exact = !upstream_sample_tee_selected_drift &&
                                   upstream_sample_tee_selected_consumer_count ==
                                       upstream_sample_tee_exact_selected_consumer_count,
            .selected_link_and_uid_exact = !upstream_sample_tee_selected_drift &&
                                           upstream_sample_tee_selected_consumer_count ==
                                               upstream_sample_tee_exact_selected_consumer_count,
            .selected_backing_exact = !upstream_sample_tee_selected_drift &&
                                      upstream_sample_tee_selected_consumer_count ==
                                          upstream_sample_tee_exact_selected_consumer_count,
        });
        const bool producer_present =
            producer_pre_for_pending.status != FinalGuestSurfaceStatus::AlreadyConsumed ||
            producer_pre_for_pending.loss.Any() || producer_pre_for_pending.copy ||
            producer_pre_for_pending.plane.region_count != 0;
        const bool producer_specific_loss =
            producer_pre_for_pending.status != FinalGuestSurfaceStatus::Complete &&
            producer_pre_for_pending.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
            producer_pre_for_pending.loss.Any();
        if (!ordinal_retry_exact && producer_present && !upstream_sample_tee_producer_pre_bound &&
            !producer_specific_loss) {
            producer_pre_for_pending = {
                .status = producer_pre_frame.status == FinalGuestSurfaceStatus::Complete
                              ? FinalGuestSurfaceStatus::GapLoss
                              : producer_pre_frame.status,
                .loss = producer_pre_frame.loss.Any() ? producer_pre_frame.loss
                                                      : FinalGuestSurfaceLoss{.gap = 1},
            };
        }
        upstream_sample_tee_producer_pre_hint = producer_pre_frame.next_hint;
        auto predecessor_writer_pre_for_pending =
            upstream_sample_tee_predecessor_writer_pre_depth;
        if (upstream_sample_tee_predecessor_writer_pre_collision) {
            predecessor_writer_pre_for_pending = {
                .status = FinalGuestSurfaceStatus::GapLoss,
                .loss = {.gap = 1},
            };
        }
        const bool predecessor_writer_pre_bound_for_flip =
            upstream_sample_tee_producer_pre_ordinal_binding.publish_predecessor_pre;
        const bool predecessor_present =
            predecessor_writer_pre_for_pending.status !=
                FinalGuestSurfaceStatus::AlreadyConsumed ||
            predecessor_writer_pre_for_pending.loss.Any() ||
            predecessor_writer_pre_for_pending.copy ||
            predecessor_writer_pre_for_pending.plane.region_count != 0;
        const bool predecessor_specific_loss =
            predecessor_writer_pre_for_pending.status != FinalGuestSurfaceStatus::Complete &&
            predecessor_writer_pre_for_pending.status !=
                FinalGuestSurfaceStatus::AlreadyConsumed &&
            predecessor_writer_pre_for_pending.loss.Any();
        if (!ordinal_retry_exact && predecessor_present &&
            !predecessor_writer_pre_bound_for_flip &&
            !predecessor_specific_loss) {
            predecessor_writer_pre_for_pending = {
                .status = upstream_sample_tee_producer_pre_ordinal_binding.status ==
                                  FinalGuestSurfaceStatus::Complete
                              ? FinalGuestSurfaceStatus::GapLoss
                              : upstream_sample_tee_producer_pre_ordinal_binding.status,
                .loss = upstream_sample_tee_producer_pre_ordinal_binding.loss.Any()
                            ? upstream_sample_tee_producer_pre_ordinal_binding.loss
                            : FinalGuestSurfaceLoss{.gap = 1},
            };
        }
        auto ordinal_binding_for_pending = upstream_sample_tee_producer_pre_ordinal_binding;
        auto ordinal_tails_for_pending = ordinal_tails;
        bool release_only_outer_slot{};
        if (ordinal_retry_exact) {
            producer_pre_for_pending = ordinal_retry_cleanup.producer_pre_depth;
            predecessor_writer_pre_for_pending =
                ordinal_retry_cleanup.predecessor_writer_pre_depth;
            ordinal_binding_for_pending = ordinal_retry_cleanup.ordinal_content_binding;
            ordinal_tails_for_pending = ordinal_retry_cleanup.ordinal_depth_download_tails;
            release_only_outer_slot = ordinal_retry_cleanup.release_only_outer_slot;
        }
        auto tee_attempt = release_only_outer_slot
            ? HostPasses::DetachPpUpstreamSampleTeeDeferredReadback({
                  .enabled = true,
                  .outer_slot = ordinal_retry_cleanup.outer_slot,
                  .release_only_outer_slot = true,
              })
            : HostPasses::DetachPpUpstreamSampleTeeDeferredReadback({
                  .enabled = config.content.capture_upstream_sample_tee,
                  .attempted = upstream_sample_tee_attempted,
                  .outer_slot = slot,
                  .record = upstream_sample_tee_candidate,
                  .capture = upstream_sample_tee_capture,
                  .commands = upstream_sample_tee_commands,
                  .controller = upstream_sample_tee_controller,
                  .selected_candidate_index = config.content.upstream_candidate_index,
                  .capture_status = tee_outcome.status,
                  .capture_loss = tee_outcome.loss,
                  .capture_has_copy = tee_outcome.copy,
                  .mapped_bytes = upstream_sample_tee_buffer
                                      ? upstream_sample_tee_buffer->mapped_data.size()
                                      : 0,
                  .transferred_depth_source = upstream_sample_tee_transferred_depth_source,
                  .selected_depth = upstream_sample_tee_selected_depth,
                  .producer_pre_depth = producer_pre_for_pending,
                  .predecessor_writer_pre_depth = predecessor_writer_pre_for_pending,
                  .ordinal_content_binding = ordinal_binding_for_pending,
                  .ordinal_depth_download_tails = ordinal_tails_for_pending,
              });
        PendingUpstreamFeedback pending{
            .sequence = sequence,
            .process_time_us = process_time_us,
            .take = take,
            .take_evaluation = take_evaluation,
            .input_plan = upstream_input_plan,
            .sample_routes = upstream_sample_routes,
            .producer_trace = upstream_sample_tee_producer_trace,
            .tee_source = upstream_sample_tee_source,
            .tee_transfer = upstream_sample_tee_transfer,
            .transferred_depth_source = upstream_sample_tee_transferred_depth_source,
            .producer_pre_scope_entry =
                std::move(upstream_sample_tee_producer_pre_scope_owned),
            .depth_predecessor =
                take.candidate_index < depth_candidate_predecessors.size()
                    ? depth_candidate_predecessors[take.candidate_index]
                    : PendingUpstreamFeedback::DepthPredecessorObservation{},
            .input_ready = upstream_input_ready,
            .tee_attempt = std::move(tee_attempt),
            .tee_target_pre_post_receipt = tee_target_pre_post_receipt,
            .scope_writer_trace_frame = scope_writer_trace_frame_for_pending,
            .scope_writer_trace_tail = scope_writer_trace_tail_for_pending,
            .slot_offset = slot ? slot.slot * slot_stride : 0,
        };
        upstream_input_plan = {};
        upstream_input_ready = false;
        upstream_sample_routes = {};
        upstream_sample_tee_producer_trace = {};
        upstream_sample_tee_source = {};
        upstream_sample_tee_transfer = {};
        upstream_sample_tee_transferred_depth_source = {};
        upstream_sample_tee_selected_depth = {};
        upstream_sample_tee_producer_pre_observed_hint = {};
        upstream_sample_tee_producer_pre_intent = {};
        upstream_sample_tee_producer_pre_post = {};
        upstream_sample_tee_producer_pre_selected_binding_descriptor = {};
        upstream_sample_tee_producer_pre_selected_binding = {};
        upstream_sample_tee_producer_pre_ordinal_binding = {};
        upstream_sample_tee_predecessor_writer_pre_ordinal_capture_descriptor = {};
        upstream_sample_tee_predecessor_writer_pre_ordinal_capture_plan = {};
        upstream_sample_tee_producer_pre_ordinal_capture_descriptor = {};
        upstream_sample_tee_producer_pre_ordinal_capture_plan = {};
        upstream_sample_tee_producer_pre_diagnostic_immediate_predecessor = {};
        upstream_sample_tee_selected_predecessor_pair = {};
        upstream_sample_tee_producer_pre_depth = {};
        upstream_sample_tee_producer_pre_scope_entry =
            HostPasses::ResetPpUpstreamSampleTeeProducerPreDepthScopeEntry(
                upstream_sample_tee_producer_pre_scope_entry);
        upstream_sample_tee_producer_pre_scope_plan = {};
        upstream_sample_tee_producer_pre_scope_post = {};
        upstream_sample_tee_producer_pre_scope_owned = {};
        upstream_sample_tee_predecessor_writer_pre_observed_hint = {};
        upstream_sample_tee_predecessor_writer_pre_intent = {};
        upstream_sample_tee_predecessor_writer_pre_post = {};
        upstream_sample_tee_predecessor_writer_pre_depth = {};
        upstream_sample_tee_producer_pre_expected_serial = 0;
        upstream_sample_tee_producer_pre_host_instance = 0;
        upstream_sample_tee_predecessor_writer_pre_expected_serial = 0;
        upstream_sample_tee_predecessor_writer_pre_host_instance = 0;
        upstream_sample_tee_selected_consumer_count = 0;
        upstream_sample_tee_exact_selected_consumer_count = 0;
        upstream_sample_tee_predecessor_writer_pre_exact_load_count = 0;
        upstream_sample_tee_producer_pre_open = false;
        upstream_sample_tee_producer_pre_attempted = false;
        upstream_sample_tee_producer_pre_bound = false;
        upstream_sample_tee_producer_pre_collision = false;
        upstream_sample_tee_producer_pre_scope_entry_collision = false;
        upstream_sample_tee_predecessor_writer_pre_open = false;
        upstream_sample_tee_predecessor_writer_pre_attempted = false;
        upstream_sample_tee_predecessor_writer_pre_bound = false;
        upstream_sample_tee_predecessor_writer_pre_collision = false;
        upstream_sample_tee_selected_drift = false;
        upstream_feedback_base_prefix_initialized = false;
        upstream_sample_tee_candidate = {};
        upstream_sample_tee_capture = {};
        upstream_sample_tee_commands = {};
        upstream_sample_tee_controller = {};
        upstream_sample_tee_attempted = false;
        upstream_sample_tee_status = FinalGuestSurfaceStatus::AlreadyConsumed;
        upstream_sample_tee_loss = {};
        upstream_sample_tee_copy = false;
        upstream_feedback_armed = false;
        scheduler.DeferOperation([this, pending = std::move(pending)]() mutable {
            ConsumeUpstreamFeedback(std::move(pending));
        });
    }

    void ConsumeUpstreamFeedback(PendingUpstreamFeedback pending) {
        std::vector<std::byte> compact_bytes{};
        HostPasses::PpUpstreamInputContentCompactResult input_compact{
            .status = FinalGuestSurfaceStatus::Complete,
        };
        const HostPasses::PpUpstreamInputPublication take_snapshot{
            .status = pending.take.status,
            .loss = pending.take.loss,
            .copy = pending.take.copy,
        };
        const auto publication_evaluation = HostPasses::EvaluatePpUpstreamInputPublication(
            config.content.capture_upstream_input_content, pending.input_ready,
            pending.take.candidate_index, config.content.upstream_candidate_index,
            pending.take.status, pending.take.loss, pending.take.copy);
        const auto& publication = publication_evaluation.publication;
        pending.take.status = publication.status;
        pending.take.loss = publication.loss;
        pending.take.copy = publication.copy;
        HostPasses::PpUpstreamSampleTeeProducerPreDepthOwned producer_pre_owned{};
        const auto& producer_pre_plan = pending.tee_attempt.producer_pre_depth;
        const bool producer_pre_recordable =
            HostPasses::IsPpUpstreamSampleTeeProducerPreDepthDownloadTailRecordable(
                producer_pre_plan) &&
            pending.tee_attempt.outer_slot &&
            producer_pre_plan.slot == pending.tee_attempt.outer_slot.slot;
        const bool producer_pre_specific_loss =
            producer_pre_plan.status != FinalGuestSurfaceStatus::Complete &&
            producer_pre_plan.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
            producer_pre_plan.loss.Any();
        HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthOwned
            predecessor_writer_pre_owned{};
        const auto& predecessor_writer_pre_plan =
            pending.tee_attempt.predecessor_writer_pre_depth;
        const bool predecessor_writer_pre_recordable =
            HostPasses::
                IsPpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailRecordable(
                    predecessor_writer_pre_plan) &&
            pending.tee_attempt.outer_slot &&
            predecessor_writer_pre_plan.slot == pending.tee_attempt.outer_slot.slot;
        const bool predecessor_writer_pre_specific_loss =
            predecessor_writer_pre_plan.status != FinalGuestSurfaceStatus::Complete &&
            predecessor_writer_pre_plan.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
            predecessor_writer_pre_plan.loss.Any();
        HostPasses::PpUpstreamSampleTeeScopeWriterTraceOwned scope_writer_trace_owned{};
        const auto& scope_writer_trace_tail = pending.scope_writer_trace_tail;
        const bool scope_writer_trace_recordable =
            pending.scope_writer_trace_frame.exact &&
            scope_writer_trace_tail.status == FinalGuestSurfaceStatus::Complete &&
            !scope_writer_trace_tail.loss.Any() && scope_writer_trace_tail.copy_pre &&
            scope_writer_trace_tail.copy_post && pending.tee_attempt.outer_slot;
        const bool scope_writer_trace_specific_loss =
            pending.scope_writer_trace_frame.exact &&
            scope_writer_trace_tail.status != FinalGuestSurfaceStatus::Complete &&
            scope_writer_trace_tail.status != FinalGuestSurfaceStatus::AlreadyConsumed &&
            scope_writer_trace_tail.loss.Any();
        const bool base_need_download_invalidation =
            pending.take.copy || producer_pre_recordable || predecessor_writer_pre_recordable ||
            scope_writer_trace_recordable;
        const auto tee_authorized_target_pre_post_plan =
            HostPasses::PlanPpUpstreamFeedbackTeeAuthorizedTargetPrePost({
                .take = pending.take_evaluation,
                .tee_attempt = pending.tee_attempt,
                .receipt = pending.tee_target_pre_post_receipt,
                .download_invalidation_already_planned = base_need_download_invalidation,
                .input_count = pending.input_plan.input_count,
                .sample_routes = pending.sample_routes,
            });
        HostPasses::PpUpstreamFeedbackPrePostCompactResult
            tee_authorized_target_pre_post_compact{};
        const bool need_download_invalidation =
            base_need_download_invalidation ||
            tee_authorized_target_pre_post_plan.additional_download_invalidation_count != 0;
        bool download_invalidated{};
        std::span<const std::byte> download_slot{};
        if (need_download_invalidation && pending.tee_attempt.outer_slot) {
            const u64 mapped_bytes = download.mapped_data.size();
            const bool mapped_range_valid = pending.slot_offset <= mapped_bytes &&
                                            slot_stride <= mapped_bytes - pending.slot_offset &&
                                            pending.take.plan.total_bytes <=
                                                HostPasses::
                                                    PpUpstreamSampleTeeDownloadPrefixBytes;
            download_invalidated =
                mapped_range_valid &&
                download.InvalidateMappedRange(pending.slot_offset, slot_stride);
            if (mapped_range_valid) {
                download_slot = std::as_bytes(download.mapped_data)
                                    .subspan(pending.slot_offset, slot_stride);
            }
            if (pending.take.copy && !download_invalidated) {
                pending.take.status = FinalGuestSurfaceStatus::InvalidationLoss;
                pending.take.loss = {.invalidation = 1};
                pending.take.copy = false;
            }
            if (pending.take.copy && download_invalidated) {
                const auto compact = HostPasses::CompactPpUpstreamFeedbackPrePost(
                    pending.take.plan, pending.take.candidate_index,
                    download_slot.first(pending.take.plan.total_bytes));
                if (compact.status != FinalGuestSurfaceStatus::Complete || compact.loss.Any()) {
                    pending.take.status = compact.status;
                    pending.take.loss = compact.loss;
                    pending.take.copy = false;
                } else {
                    compact_bytes = compact.bytes;
                    if (config.content.capture_upstream_input_content) {
                        input_compact = HostPasses::CompactPpUpstreamInputContent(
                            pending.input_plan, download_slot);
                        if (input_compact.status != FinalGuestSurfaceStatus::Complete ||
                            input_compact.loss.Any()) {
                            pending.take.status = input_compact.status;
                            pending.take.loss = input_compact.loss;
                            pending.take.copy = false;
                            compact_bytes.clear();
                        } else {
                            compact_bytes.insert(compact_bytes.end(), input_compact.bytes.begin(),
                                                 input_compact.bytes.end());
                        }
                    }
                }
            }
            if (tee_authorized_target_pre_post_plan.provisionally_compact &&
                download_invalidated) {
                tee_authorized_target_pre_post_compact =
                    HostPasses::CompactPpUpstreamFeedbackPrePost(
                        pending.take_evaluation.take.plan,
                        pending.tee_attempt.selected_candidate_index,
                        download_slot.first(pending.take_evaluation.take.plan.total_bytes));
            }
        }
        if (producer_pre_recordable) {
            producer_pre_owned = HostPasses::CopyPpUpstreamSampleTeeProducerPreDepth(
                producer_pre_plan, download_invalidated, download_slot);
        } else if (producer_pre_specific_loss) {
            producer_pre_owned.status = producer_pre_plan.status;
            producer_pre_owned.loss = producer_pre_plan.loss;
            producer_pre_owned.shared_release_deferred = true;
        } else if (producer_pre_plan.status != FinalGuestSurfaceStatus::AlreadyConsumed ||
                   producer_pre_plan.copy || producer_pre_plan.plane.region_count != 0) {
            producer_pre_owned.status = FinalGuestSurfaceStatus::InvalidationLoss;
            producer_pre_owned.loss = {.invalidation = 1};
            producer_pre_owned.shared_release_deferred = true;
        }
        if (predecessor_writer_pre_recordable) {
            predecessor_writer_pre_owned =
                HostPasses::CopyPpUpstreamSampleTeePredecessorWriterPreDepth(
                    predecessor_writer_pre_plan, download_invalidated, download_slot);
        } else if (predecessor_writer_pre_specific_loss) {
            predecessor_writer_pre_owned.status = predecessor_writer_pre_plan.status;
            predecessor_writer_pre_owned.loss = predecessor_writer_pre_plan.loss;
            predecessor_writer_pre_owned.shared_release_deferred = true;
        } else if (predecessor_writer_pre_plan.status !=
                       FinalGuestSurfaceStatus::AlreadyConsumed ||
                   predecessor_writer_pre_plan.copy ||
                   predecessor_writer_pre_plan.plane.region_count != 0) {
            predecessor_writer_pre_owned.status =
                FinalGuestSurfaceStatus::InvalidationLoss;
            predecessor_writer_pre_owned.loss = {.invalidation = 1};
            predecessor_writer_pre_owned.shared_release_deferred = true;
        }
        if (scope_writer_trace_recordable) {
            scope_writer_trace_owned = HostPasses::CopyPpUpstreamSampleTeeScopeWriterTrace(
                scope_writer_trace_tail, download_invalidated, download_slot);
        } else if (scope_writer_trace_specific_loss) {
            scope_writer_trace_owned.status = scope_writer_trace_tail.status;
            scope_writer_trace_owned.loss = scope_writer_trace_tail.loss;
            scope_writer_trace_owned.shared_release_deferred = true;
        } else if (pending.scope_writer_trace_frame.exact) {
            scope_writer_trace_owned.status = FinalGuestSurfaceStatus::InvalidationLoss;
            scope_writer_trace_owned.loss = {.invalidation = 1};
            scope_writer_trace_owned.shared_release_deferred = true;
        }
        if (producer_pre_owned.status == FinalGuestSurfaceStatus::Complete &&
            !producer_pre_owned.loss.Any()) {
            producer_pre_owned.scope_entry = std::move(pending.producer_pre_scope_entry);
        }
        std::span<const std::byte> tee_mapped{};
        if (upstream_sample_tee_buffer) {
            tee_mapped = std::as_bytes(upstream_sample_tee_buffer->mapped_data);
        }
        auto consumed = HostPasses::ConsumePpUpstreamSampleTeeDeferredReadback(
            pending.tee_attempt,
            HostPasses::PpUpstreamInputPublication{
                .status = pending.take.status,
                .loss = pending.take.loss,
                .copy = pending.take.copy,
            },
            tee_mapped,
            [this](u64 offset, u64 bytes) {
                return upstream_sample_tee_buffer &&
                       upstream_sample_tee_buffer->InvalidateMappedRange(offset, bytes);
            },
            [this](FinalGuestSurfaceReadbackSlotPool::Token token) {
                return slots.ReleaseAfterCpuConsume(token);
            });
        const HostPasses::PpUpstreamFeedbackTeeAuthorizedTargetPrePostDownloadObservation
            tee_authorized_target_pre_post_download{
                .outer_slot = pending.tee_target_pre_post_receipt.outer_slot,
                .offset = pending.tee_target_pre_post_receipt.download_offset,
                .bytes = pending.tee_target_pre_post_receipt.download_bytes,
                .attempted = need_download_invalidation,
                .succeeded = download_invalidated,
            };
        auto tee_authorized_target_pre_post =
            HostPasses::AuthorizePpUpstreamFeedbackTeeAuthorizedTargetPrePost(
                tee_authorized_target_pre_post_plan,
                std::move(tee_authorized_target_pre_post_compact),
                tee_authorized_target_pre_post_download, pending.tee_attempt, consumed.tee);
        const auto tee_target_same_frame_fact =
            HostPasses::ReducePpUpstreamFeedbackTeeTargetSameFrame(
                pending.sequence, tee_authorized_target_pre_post);
        const auto fragment_export_target_relation =
            HostPasses::ReducePpUpstreamFeedbackFragmentExportTargetRelation(
                pending.sequence, config.content.capture_upstream_fragment_export_tee,
                pending.tee_attempt.selected_candidate_index,
                tee_authorized_target_pre_post, consumed.tee);
        if (config.content.capture_upstream_fragment_export_tee) {
            LOG_INFO(Render, "{}",
                     HostPasses::FormatPpUpstreamFeedbackFragmentExportTargetRelation(
                         fragment_export_target_relation));
        }
        if (pending.scope_writer_trace_frame.exact) {
            const bool tee_authoritative =
                consumed.tee.observe_owned && consumed.tee.publication.publish &&
                consumed.tee.lifecycle.finalized &&
                !consumed.tee.lifecycle.lifecycle_invalid;
            const auto scope_writer_trace =
                HostPasses::ReducePpUpstreamSampleTeeScopeWriterTrace({
                    .sequence = pending.sequence,
                    .frame = pending.scope_writer_trace_frame,
                    .tail = pending.scope_writer_trace_tail,
                    .owned = std::move(scope_writer_trace_owned),
                    .tee_authoritative = tee_authoritative,
                });
            LOG_INFO(Render, "{}",
                     HostPasses::FormatPpUpstreamSampleTeeScopeWriterTrace(
                         scope_writer_trace));
        }
        if (pending.tee_attempt.attempted) {
            const auto fragment_buffer_dependencies =
                HostPasses::ReducePpUpstreamFeedbackFragmentBufferDependencies(
                    consumed.tee, pending.tee_attempt.selected_candidate_index);
            LOG_INFO(Render, "{}",
                     HostPasses::FormatPpUpstreamFeedbackFragmentBufferDependencies(
                         pending.sequence, fragment_buffer_dependencies));
            const auto cube_reduction = HostPasses::ReducePpUpstreamFeedbackCubeOwnedReadback(
                consumed.tee, pending.tee_attempt.selected_candidate_index);
            const auto cube_query = HostPasses::MakePpUpstreamFeedbackCubeQueryReport(
                pending.sequence, pending.tee_source, cube_reduction);
            if (cube_query.sequence != 0) {
                LOG_INFO(Render, "{}",
                         HostPasses::FormatPpUpstreamFeedbackCubeQueryReport(cube_query));
            }
        }
        consumed.tee.producer_pre_depth = std::move(producer_pre_owned);
        consumed.tee.predecessor_writer_pre_depth =
            std::move(predecessor_writer_pre_owned);
        pending.tee_attempt.outer_slot = {};
        pending.take.status = consumed.base.status;
        pending.take.loss = consumed.base.loss;
        pending.take.copy = consumed.base.copy;
        if (!pending.take.copy) {
            compact_bytes.clear();
            input_compact = {};
        }
        if (config.content.capture_upstream_depth_pre_post && pending.take.copy &&
            pending.take.plan.depth && pending.take.plan.raw_compare) {
            const auto& plan = pending.take.plan;
            bool valid = plan.bytes_per_pixel != 0 && plan.plane_bytes != 0 &&
                         compact_bytes.size() >= static_cast<u64>(plan.plane_bytes) * 2;
            u64 covered_pixels{};
            u64 changed_pixels{};
            for (u32 region_index = 0; valid && region_index < plan.region_count; ++region_index) {
                const auto& region = plan.regions[region_index];
                const u64 pre_offset = region.buffer_offset;
                const u64 post_offset = static_cast<u64>(plan.plane_bytes) + region.buffer_offset;
                valid = region.byte_size != 0 && region.byte_size % plan.bytes_per_pixel == 0 &&
                        pre_offset + region.byte_size <= plan.plane_bytes &&
                        post_offset + region.byte_size <= compact_bytes.size();
                for (u32 byte = 0; valid && byte < region.byte_size;
                     byte += plan.bytes_per_pixel) {
                    bool changed{};
                    for (u32 component = 0; component < plan.bytes_per_pixel; ++component) {
                        changed |= compact_bytes[pre_offset + byte + component] !=
                                   compact_bytes[post_offset + byte + component];
                    }
                    ++covered_pixels;
                    changed_pixels += changed;
                }
            }
            LOG_INFO(Render, "FGSCTUD s={} v={} px={} ch={}", pending.sequence, valid,
                     covered_pixels, changed_pixels);
        }
        if (config.content.capture_upstream_depth_pre_post) {
            const auto& predecessor = pending.depth_predecessor;
            LOG_INFO(Render,
                     "FGSCTUP s={} pc={} tx={} tk={} to={} dx={} dss={} dk={} di={} de={} dn={} ds={} "
                     "dw={} sfv={} sfk={} sfi={} sfe={} sfn={} sfs={} sfw={} sfc={} sfov={} "
                     "sfhv={} sfdc={} sfsc={} dfd={} ddisc={} dd={} dsw={} dfa={} dc={} sc={} "
                     "tpe={} tfw={} tsl={} tsc={} dbv={} dbj={} dbw={} dbpr={} dbpt={} dbpe={} "
                     "dbar={} dbsi={} dbld={} dbse={} dbsk={} dbsh={} dbxh={} dbix={} dbbd={} "
                     "dbdx={} dbrw={}",
                     pending.sequence, static_cast<u32>(predecessor.producer),
                     predecessor.transfer_exact, static_cast<u32>(predecessor.transfer_kind),
                     static_cast<u32>(predecessor.transfer_origin),
                     predecessor.depth_stencil_exact, predecessor.depth_stencil_same_scope,
                     static_cast<u32>(predecessor.depth_stencil_kind), predecessor.indexed,
                     predecessor.element_count, predecessor.instance_count,
                     predecessor.sampled_images, predecessor.storage_writes,
                     predecessor.scope_first_depth_valid,
                     static_cast<u32>(predecessor.scope_first_depth_kind),
                     predecessor.scope_first_depth_indexed,
                     predecessor.scope_first_depth_element_count,
                     predecessor.scope_first_depth_instance_count,
                      predecessor.scope_first_depth_sampled_images,
                      predecessor.scope_first_depth_storage_writes,
                     predecessor.scope_depth_writer_count,
                     predecessor.scope_depth_writer_count_overflow,
                     predecessor.scope_depth_history_valid, predecessor.scope_depth_clear,
                      predecessor.scope_stencil_clear,
                     predecessor.fragment_writes_depth, predecessor.fragment_has_discard,
                      predecessor.depth_write, predecessor.stencil_write,
                      predecessor.feedback_alias, predecessor.depth_clear,
                     predecessor.stencil_clear, predecessor.target_post_exact,
                     predecessor.target_first_depth_writer, predecessor.target_scope_load,
                     predecessor.target_scope_clear, predecessor.depth_boundary_valid,
                     predecessor.depth_boundary_join_exact,
                     predecessor.depth_boundary_writer_exact,
                     predecessor.depth_boundary_pre_rendering,
                     predecessor.depth_boundary_post_transit_rendering,
                     predecessor.depth_boundary_pre_external_rendering,
                     predecessor.depth_boundary_after_rendering,
                     predecessor.depth_boundary_split_interval,
                     predecessor.depth_boundary_target_load,
                     predecessor.depth_boundary_state_exact,
                     predecessor.depth_boundary_skipped_write,
                     predecessor.depth_boundary_same_host,
                     predecessor.depth_boundary_cross_host,
                     predecessor.depth_boundary_instrumented_cross_host,
                     predecessor.depth_boundary_before_diagnostic,
                     predecessor.depth_boundary_diagnostic_excluded,
                     predecessor.depth_boundary_barrier_recognized_write);
        }
        if (consumed.release_attempted && !consumed.release_succeeded) {
            LOG_ERROR(Render, "PpUpstreamFeedbackPrePost slot_release_loss=1");
        }
        pending.producer_trace = FinalizePpUpstreamSampleTeeProducerTrace({
            .trace = pending.producer_trace,
            .input_count = pending.input_plan.input_count,
            .capture_mask = input_compact.capture_mask,
            .unavailable_mask = input_compact.unavailable_mask,
            .alias_mask = input_compact.alias_mask,
            .status = pending.take.status,
            .loss = pending.take.loss,
            .copy = pending.take.copy,
        });
        if (pending.producer_trace.status != FinalGuestSurfaceStatus::AlreadyConsumed) {
            const auto source_input =
                pending.producer_trace.source_link_in_current_inputs
                    ? std::to_string(pending.producer_trace.source_input_index)
                    : std::string{"-"};
            LOG_INFO(Render,
                     "FGSCTTA s={} i={} p={} ss={} sv={} sb={} se={} sr={} su={} sa={} sx={} os={} "
                     "ov={} ob={} oe={} or={} ou={} oa={} ox={} d={} t={} tr={} ch={} st={} lm={}",
                     pending.sequence, pending.producer_trace.selected_input_index, source_input,
                     pending.producer_trace.source_subresource_matches,
                     pending.producer_trace.source_view_matches,
                     pending.producer_trace.source_backing_matches,
                     pending.producer_trace.source_epoch_matches,
                     pending.producer_trace.source_raw_captured,
                     pending.producer_trace.source_unavailable, pending.producer_trace.source_alias,
                     pending.producer_trace.exact_source_raw_plane,
                     pending.producer_trace.output_subresource_matches,
                     pending.producer_trace.output_view_matches,
                     pending.producer_trace.output_backing_matches,
                     pending.producer_trace.output_epoch_matches,
                     pending.producer_trace.output_raw_captured,
                     pending.producer_trace.output_unavailable, pending.producer_trace.output_alias,
                     pending.producer_trace.exact_output_raw_plane,
                     pending.producer_trace.ancestry.depth,
                     static_cast<u32>(pending.producer_trace.ancestry.terminal),
                     pending.producer_trace.ancestry.truncated,
                     HostPasses::FormatImageColorScopeAncestry(pending.producer_trace.ancestry),
                     static_cast<u32>(pending.producer_trace.status),
                     FinalGuestSurfaceLossMask(pending.producer_trace.loss, 0));
        }
        if (consumed.tee.observe_owned && pending.tee_source.valid) {
            LOG_INFO(Render,
                     "FGSCTTP s={} k={} o={} v={} x={} rx={} rq={} xy={} ao={} ro={} rc={} m={}/{} "
                     "l={}/{} sm={}/{}/{} ev={} ec={} em={} req={}/{} back={}/{} sp={} stk={} "
                     "sto={} ste={} sc={} si={} sb={} smc={} smd={} sd={} nsp={} nstk={} nsto={} "
                     "nste={} nsd={} dsx={} dsk={} dsi={} dse={} dsn={} dss={} dsw={} dsd={} "
                     "dssw={} dsdir={} sa={} sid={} dph={:#x} dmrt={} dfd={} ddisc={} dai={} "
                     "dab={} daa={} dar={} daf={} dcu={} dpt={} dpe={} dqc={} dcm={} dff={} "
                     "sfv={} sfk={} sfi={} sfe={} sfn={} sfs={} sfw={} sfpt={} sfpe={} "
                     "sfqc={} sfcm={} sfff={} sfph={:#x}",
                     pending.sequence, static_cast<u32>(pending.tee_transfer.kind),
                     static_cast<u32>(pending.tee_transfer.origin),
                     pending.tee_transfer.observed, pending.tee_transfer.exact,
                     pending.tee_transfer.range_exact,
                     pending.tee_transfer.request_matches_encoded, pending.tee_transfer.full_xy,
                     pending.tee_transfer.aspect_overlap, pending.tee_transfer.range_overlap,
                     pending.tee_transfer.range_covers, pending.tee_transfer.base_mip,
                     pending.tee_transfer.mip_count, pending.tee_transfer.base_layer,
                     pending.tee_transfer.layer_count, pending.tee_transfer.source_samples,
                     pending.tee_transfer.old_samples, pending.tee_transfer.new_samples,
                     pending.tee_transfer.extent_valid, pending.tee_transfer.extent_conflict,
                     pending.tee_transfer.extent_matches, pending.tee_transfer.requested_width,
                     pending.tee_transfer.requested_height, pending.tee_transfer.backing_width,
                     pending.tee_transfer.backing_height,
                     static_cast<u32>(pending.tee_transfer.source_producer),
                     static_cast<u32>(pending.tee_transfer.source_transfer_kind),
                     static_cast<u32>(pending.tee_transfer.source_transfer_origin),
                     pending.tee_transfer.source_transfer_exact,
                     pending.tee_transfer.source_identity_comparable,
                     pending.tee_transfer.source_image_changed,
                     pending.tee_transfer.source_backing_changed,
                     pending.tee_transfer.source_mutation_comparable,
                     pending.tee_transfer.source_mutation_changed,
                     pending.tee_transfer.source_is_depth,
                     static_cast<u32>(pending.tee_transfer.nested_source_producer),
                     static_cast<u32>(pending.tee_transfer.nested_source_transfer_kind),
                     static_cast<u32>(pending.tee_transfer.nested_source_transfer_origin),
                     pending.tee_transfer.nested_source_transfer_exact,
                     pending.tee_transfer.nested_source_is_depth,
                     pending.tee_transfer.source_depth_stencil_exact,
                     static_cast<u32>(pending.tee_transfer.source_depth_stencil_kind),
                     pending.tee_transfer.source_depth_stencil_indexed,
                     pending.tee_transfer.source_depth_stencil_element_count,
                     pending.tee_transfer.source_depth_stencil_instance_count,
                     pending.tee_transfer.source_depth_stencil_sampled_images,
                     pending.tee_transfer.source_depth_stencil_storage_writes,
                     pending.tee_transfer.source_depth_write,
                     pending.tee_transfer.source_stencil_write,
                     pending.tee_transfer.source_depth_stencil_direct,
                     pending.tee_transfer.selected_aspect,
                     pending.tee_transfer.selected_is_depth,
                     pending.tee_transfer.source_depth_stencil_fragment_pgm_hash,
                     pending.tee_transfer.source_depth_stencil_fragment_mrt_mask,
                     pending.tee_transfer.source_depth_stencil_fragment_writes_depth,
                     pending.tee_transfer.source_depth_stencil_fragment_has_discard,
                     pending.tee_transfer.source_depth_stencil_sampled_id_alias,
                     pending.tee_transfer.source_depth_stencil_sampled_backing_alias,
                     pending.tee_transfer.source_depth_stencil_sampled_aspect_alias,
                     pending.tee_transfer.source_depth_stencil_sampled_range_alias,
                     pending.tee_transfer.source_depth_stencil_sampled_feedback_alias,
                     pending.tee_transfer.source_depth_stencil_user_clip_plane_mask,
                     pending.tee_transfer.source_depth_stencil_guest_primitive_type,
                     pending.tee_transfer.source_depth_stencil_effective_primitive_type,
                     pending.tee_transfer.source_depth_stencil_quad_list_converted,
                     pending.tee_transfer.source_depth_stencil_cull_mode,
                     pending.tee_transfer.source_depth_stencil_front_face,
                     pending.tee_transfer.source_depth_stencil_scope_first_valid,
                     static_cast<u32>(pending.tee_transfer.source_depth_stencil_scope_first_kind),
                     pending.tee_transfer.source_depth_stencil_scope_first_indexed,
                     pending.tee_transfer.source_depth_stencil_scope_first_element_count,
                     pending.tee_transfer.source_depth_stencil_scope_first_instance_count,
                     pending.tee_transfer.source_depth_stencil_scope_first_sampled_images,
                     pending.tee_transfer.source_depth_stencil_scope_first_storage_writes,
                     pending.tee_transfer.source_depth_stencil_scope_first_guest_primitive_type,
                     pending.tee_transfer.source_depth_stencil_scope_first_effective_primitive_type,
                     pending.tee_transfer.source_depth_stencil_scope_first_quad_list_converted,
                     pending.tee_transfer.source_depth_stencil_scope_first_cull_mode,
                     pending.tee_transfer.source_depth_stencil_scope_first_front_face,
                     pending.tee_transfer.source_depth_stencil_scope_first_fragment_pgm_hash);
        }
        auto history_layout = MakeUpstreamFeedbackHistoryLayout(
            pending.take.plan, pending.input_plan, input_compact, pending.sample_routes);
        std::span<const std::byte> history_bytes = compact_bytes;
        auto history_status = pending.take.status;
        auto history_loss = pending.take.loss;
        if (tee_authorized_target_pre_post.publish) {
            history_layout = tee_authorized_target_pre_post.layout;
            history_bytes = tee_authorized_target_pre_post.bytes;
            history_status = tee_authorized_target_pre_post.status;
            history_loss = tee_authorized_target_pre_post.loss;
        }
        {
            std::scoped_lock lock{reducer_mutex};
            upstream_feedback_reducer.ObserveContent(pending.sequence, history_layout,
                                                     history_bytes, history_status, history_loss,
                                                     tee_target_same_frame_fact);
            if (upstream_sample_tee_reducer && consumed.tee.observe_owned) {
                upstream_sample_tee_reducer->ObserveOwned(
                    pending.sequence, consumed.tee.record, std::move(consumed.tee.owned),
                    pending.tee_source, std::move(consumed.tee.transferred_depth_source),
                    std::move(consumed.tee.selected_depth),
                    std::move(consumed.tee.producer_pre_depth),
                    std::move(consumed.tee.predecessor_writer_pre_depth));
            }
            LogUpstreamSampleTeeCalibratedOutput();
            LogUpstreamFeedbackReports();
            LogUpstreamFeedbackCalibratedCoverage();
        }
        if (config.window.Contains(pending.sequence)) {
            ++upstream_feedback_selected_frames;
            ++upstream_feedback_emitted_frames;
            upstream_feedback_complete_frames +=
                pending.take.status == FinalGuestSurfaceStatus::Complete &&
                !pending.take.loss.Any();
            upstream_feedback_loss_frames +=
                pending.take.status != FinalGuestSurfaceStatus::Complete || pending.take.loss.Any();
            const auto publication_report =
                HostPasses::MakePpUpstreamInputPublicationFrameReport({
                    .sequence = pending.sequence,
                    .candidate_index = pending.take.candidate_index,
                    .recorded_plane_mask = pending.take.recorded_plane_mask,
                    .region_count = pending.take.plan.region_count,
                    .take = take_snapshot,
                    .evaluation = publication_evaluation,
                    .final = {.status = pending.take.status,
                              .loss = pending.take.loss,
                              .copy = pending.take.copy},
                });
            LOG_INFO(Render, "{}",
                     HostPasses::FormatPpUpstreamInputPublicationFrameReport(
                         publication_report));
            if (config.window.IsFinal(pending.sequence)) {
                LOG_INFO(Render,
                         "FGSCTUC s={} selected={} emitted={} complete={} loss={} regions={}",
                         pending.sequence, upstream_feedback_selected_frames,
                         upstream_feedback_emitted_frames, upstream_feedback_complete_frames,
                         upstream_feedback_loss_frames, pending.take.plan.region_count);
            }
        }
    }

    void LogUpstreamFeedbackReports() {
        for (const auto& report : upstream_feedback_reducer.TakeReports()) {
            LOG_INFO(
                Render,
                "FGSCTUT q={} abc={}/{}/{} b={} a={} be={} bs={} bx={} ae={} as={} ax={} st={} "
                "lm={} {}",
                report.request_ordinal, report.sequences[0], report.sequences[1],
                report.sequences[2],
                FormatPpTerminalScopeOrdinalList(report.first_localized_visual_return_ordinals),
                FormatPpTerminalScopeOrdinalList(report.second_localized_visual_return_ordinals),
                FormatPpTerminalScopeOrdinalList(report.first_aba_ordinals),
                FormatPpTerminalScopeOrdinalList(report.first_stable_ordinals),
                FormatPpTerminalScopeOrdinalList(report.first_ambiguous_ordinals),
                FormatPpTerminalScopeOrdinalList(report.second_aba_ordinals),
                FormatPpTerminalScopeOrdinalList(report.second_stable_ordinals),
                FormatPpTerminalScopeOrdinalList(report.second_ambiguous_ordinals),
                static_cast<u32>(report.status), FinalGuestSurfaceLossMask(report.loss, 0),
                FormatPpUpstreamFeedbackTeeTargetSameFrameSuffix(report));
            if (config.content.capture_upstream_input_content) {
                std::string inputs = "FGSCTUI q=" + std::to_string(report.request_ordinal) +
                                     " im=" + std::to_string(report.input_count) + '/' +
                                     std::to_string(report.input_capture_mask) + '/' +
                                     std::to_string(report.input_unavailable_mask) + '/' +
                                     std::to_string(report.input_alias_mask);
                for (u32 input_index = 0; input_index < report.input_count; ++input_index) {
                    if ((report.input_capture_mask & (1u << input_index)) == 0) {
                        continue;
                    }
                    inputs +=
                        " z" + std::to_string(input_index) + "a=" +
                        FormatPpTerminalScopeOrdinalList(
                            report.sampled_input_aba_ordinals[input_index]) +
                        " z" + std::to_string(input_index) + "s=" +
                        FormatPpTerminalScopeOrdinalList(
                            report.sampled_input_stable_ordinals[input_index]) +
                        " z" + std::to_string(input_index) + "x=" +
                        FormatPpTerminalScopeOrdinalList(
                            report.sampled_input_ambiguous_ordinals[input_index]) +
                        " z" + std::to_string(input_index) + "y=" +
                        FormatPpTerminalScopeOrdinalList(
                            report.sampled_input_localized_visual_return_ordinals[input_index]);
                }
                inputs += " st=" + std::to_string(static_cast<u32>(report.status)) +
                          " lm=" + std::to_string(FinalGuestSurfaceLossMask(report.loss, 0));
                LOG_INFO(Render, "{}", inputs);
            }
            if (config.content.capture_upstream_sample_routes) {
                LOG_INFO(Render, "FGSCTUR q={} i={} r={} sm={} sg={} st={} lm={}",
                         report.request_ordinal, report.upstream_sample_route_input_index,
                         report.upstream_sample_route_count,
                         report.upstream_sample_route_sampler_count,
                         report.upstream_sample_route_stage_mask, static_cast<u32>(report.status),
                         FinalGuestSurfaceLossMask(report.loss, 0));
            }
        }
    }

    void LogUpstreamSampleTeeCalibratedOutput(bool force = false) {
        if (!upstream_sample_tee_reducer || upstream_sample_tee_coverage_logged) {
            return;
        }
        if (!force && !upstream_sample_tee_reducer->TerminalOutputReady()) {
            return;
        }
        auto output = HostPasses::FinalizePpUpstreamSampleTeeCalibratedOutput(
            *upstream_sample_tee_reducer);
        for (const auto& report : output.reports) {
            LOG_INFO(Render, "{}", HostPasses::FormatPpUpstreamSampleTeeCalibratedReport(report));
            if (report.transferred_depth_source.status !=
                FinalGuestSurfaceStatus::AlreadyConsumed) {
                LOG_INFO(
                    Render, "{}",
                    HostPasses::FormatPpUpstreamSampleTeeTransferredDepthSourceReport(report));
            }
            if (report.selected_depth.status != FinalGuestSurfaceStatus::AlreadyConsumed) {
                LOG_INFO(Render, "{}",
                         HostPasses::FormatPpUpstreamSampleTeeSelectedDepthReport(report));
            }
            if (report.producer_pre_depth.status !=
                FinalGuestSurfaceStatus::AlreadyConsumed) {
                LOG_INFO(Render, "{}",
                         HostPasses::FormatPpUpstreamSampleTeeProducerPreDepthReport(report));
            }
            if (report.predecessor_writer_pre_producer_pre_depth.status !=
                FinalGuestSurfaceStatus::AlreadyConsumed) {
                LOG_INFO(Render, "{}",
                         HostPasses::
                             FormatPpUpstreamSampleTeePredecessorWriterPreProducerPreDepthReport(
                                 report));
            }
            if (report.producer_pre_scope_entry.status !=
                FinalGuestSurfaceStatus::AlreadyConsumed) {
                LOG_INFO(
                    Render, "{}",
                    HostPasses::FormatPpUpstreamSampleTeeProducerPreDepthScopeEntryReport(report));
            }
            if (report.producer_pre_selected_depth.status !=
                FinalGuestSurfaceStatus::AlreadyConsumed) {
                LOG_INFO(Render, "{}",
                         HostPasses::
                             FormatPpUpstreamSampleTeeProducerPreSelectedDepthReport(report));
            }
        }
        LOG_INFO(Render, "{}",
                 HostPasses::FormatPpUpstreamSampleTeeCalibratedCoverage(output.coverage));
        upstream_sample_tee_coverage_logged = true;
    }

    void LogUpstreamFeedbackCalibratedCoverage(bool force = false) {
        if (upstream_feedback_coverage_logged) {
            return;
        }
        const auto coverage = upstream_feedback_reducer.GetCoverage(config.expected_calibrations);
        if (!force && !coverage.ready) {
            return;
        }
        LOG_INFO(Render, "FGSCTUTC c={} o={} e={}/{}/{} lm={}", coverage.calibrations,
                 coverage.outside, coverage.eligible, coverage.emitted, coverage.complete,
                 coverage.loss);
        upstream_feedback_coverage_logged = true;
    }

    void Defer(Pending pending) {
        scheduler.DeferOperation(
            [this, pending = std::move(pending)]() mutable { Consume(std::move(pending)); });
    }

    void Consume(Pending pending) {
        std::span<const std::byte> bytes{};
        if (pending.status == FinalGuestSurfaceStatus::Complete && pending.slot) {
            if (!download.InvalidateMappedRange(pending.slot_offset, slot_stride) ||
                pending.slot_offset + pending.layout.total_bytes > download.mapped_data.size()) {
                pending.status = FinalGuestSurfaceStatus::InvalidationLoss;
                pending.loss = {.invalidation = 1};
            } else {
                bytes = std::as_bytes(
                    download.mapped_data.subspan(pending.slot_offset, pending.layout.total_bytes));
            }
        }
        {
            std::scoped_lock lock{reducer_mutex};
            reducer.ObserveContent(pending.sequence, pending.layout, bytes, pending.status,
                                   pending.loss);
            LogReports();
            LogCalibratedCoverage();
        }
        if (config.window.Contains(pending.sequence)) {
            ++selected_frames;
            ++emitted_frames;
            complete_frames +=
                pending.status == FinalGuestSurfaceStatus::Complete && !pending.loss.Any();
            loss_frames +=
                pending.status != FinalGuestSurfaceStatus::Complete || pending.loss.Any();
            LOG_INFO(
                Render, "{} {} {}",
                FormatPpTerminalScopeContentReport(MakePpTerminalScopeContentReport(
                    pending.sequence, pending.status, pending.loss, pending.draw_count,
                    pending.layout.region_count, pending.consumer_observations,
                    pending.consumer_phase_mask, pending.consumer_shape_matches,
                    pending.consumer_frozen, pending.layout.plane_mask, pending.lineage_hops,
                    pending.lineage_status, pending.lineage_loss)),
                FormatPpTerminalScopePredecessor(pending.predecessor),
                FormatPpTerminalScopeSampledInputs(pending.sampled_inputs) +
                    (config.content.capture_upstream_inputs
                         ? " " + FormatPpTerminalScopeSampledInputs(pending.upstream_inputs, 'u')
                         : ""));
            if (config.window.IsFinal(pending.sequence)) {
                LOG_INFO(Render,
                         "FGSCTSC s={} selected={} emitted={} complete={} loss={} regions={}",
                         pending.sequence, selected_frames, emitted_frames, complete_frames,
                         loss_frames, pending.layout.region_count);
                if (discovery_coverage_emission.Observe(config.window, pending.sequence)) {
                    LOG_INFO(Render, "{}",
                             FormatPpTerminalScopeDiscoveryCoverage(discovery_coverage));
                    LOG_INFO(Render, "{}",
                             FormatPpTerminalScopeProgressCoverage(progress_coverage));
                }
            }
        }
        if (pending.slot && !slots.ReleaseAfterCpuConsume(pending.slot)) {
            LOG_ERROR(Render, "PpTerminalScopeContent slot_release_loss=1");
        }
    }

    void LogReports() {
        for (const auto& report : reducer.TakeReports()) {
            LOG_INFO(Render, "{}", FormatPpTerminalScopeCalibratedReport(report));
        }
    }

    void LogCalibratedCoverage(bool force = false) {
        if (calibrated_coverage_logged) {
            return;
        }
        const auto coverage = reducer.GetCoverage(config.expected_calibrations);
        if (!force && !coverage.ready) {
            return;
        }
        LOG_INFO(Render, "{}", FormatPpTerminalScopeCalibratedCoverage(coverage));
        calibrated_coverage_logged = true;
    }

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    PpTerminalScopeRuntimeConfig config{};
    PpTerminalScopeContentReducer reducer;
    FinalGuestSurfaceScreenshotCalibration calibration;
    PpTerminalScopeUpstreamInputRegistry upstream_registry;
    HostPasses::PpUpstreamFeedbackPrePostFrame upstream_feedback_frame;
    HostPasses::PpUpstreamFeedbackSelectionAuthorityFrame
        upstream_feedback_selection_authority{};
    HostPasses::PpDepthPredecessorHintLifecycle depth_predecessor_hint;
    HostPasses::PpDepthPredecessorTwoHopLifecycle depth_predecessor_two_hop;
    HostPasses::PpDepthPredecessorCrossScopePairLifecycle depth_predecessor_cross_scope;
    HostPasses::PpDepthPredecessorCrossScopeArmedP0RepeatProbe
        depth_predecessor_cross_scope_p0_repeat_probe{};
    HostPasses::PpDepthPredecessorCrossScopeArmedP0RepeatProbeConfiguration
        depth_predecessor_cross_scope_p0_repeat_probe_configuration{};
    std::array<PpTerminalScopeSampledInputs,
               HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates>
        depth_candidate_inputs{};
    std::array<bool, HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates>
        depth_candidate_inputs_valid{};
    std::array<PendingUpstreamFeedback::DepthPredecessorObservation,
               HostPasses::PpUpstreamFeedbackPrePostPlan::MaxCandidates>
        depth_candidate_predecessors{};
    u64 depth_candidate_generation{};
    DepthPredecessorHintOutcome depth_predecessor_hint_outcome{};
    DepthPredecessorTwoHopOutcome depth_predecessor_two_hop_outcome{};
    DepthPredecessorCrossScopeOutcome depth_predecessor_cross_scope_outcome{};
    HostPasses::PpDepthPredecessorRuntimeFramePlan depth_predecessor_runtime_plan{};
    HostPasses::PpDepthPredecessorCrossScopeBootstrapSnapshot
        depth_predecessor_cross_scope_bootstrap{};
    HostPasses::PpDepthPredecessorCrossScopeResumeWitness
        depth_predecessor_cross_scope_resume{};
    OpenDepthPredecessorHintDraw open_depth_predecessor_hint_draw{};
    OpenDepthPredecessorTwoHopP1Draw open_depth_predecessor_two_hop_p1_draw{};
    OpenDepthPredecessorCrossScopeSourceDraw
        open_depth_predecessor_cross_scope_source_draw{};
    OpenDepthPredecessorCrossScopeTargetDraw
        open_depth_predecessor_cross_scope_target_draw{};
    OpenDepthPredecessorCrossScopeTailDraw
        open_depth_predecessor_cross_scope_tail_draw{};
    OpenDepthPredecessorCrossScopeArmedP0RepeatDraw
        open_depth_predecessor_cross_scope_p0_repeat_draw{};
    bool depth_predecessor_cross_scope_p0_repeat_probe_configured{};
    bool depth_predecessor_hint_capture_closed{};
    bool depth_predecessor_cross_scope_frame_open{};
    bool depth_predecessor_cross_scope_capture_closed{};
    HostPasses::PpUpstreamInputCaptureGate upstream_input_gate;
    PpTerminalScopeContentReducer upstream_feedback_reducer;
    FinalGuestSurfaceReadbackSlotPool slots{};
    u64 slot_stride{};
    VideoCore::Buffer download;
    std::unique_ptr<VideoCore::Buffer> upstream_sample_tee_buffer;
    std::unique_ptr<HostPasses::PpUpstreamSampleTeeCalibratedReducer>
        upstream_sample_tee_reducer;
    HostPasses::PpUpstreamSampleTeeBindingPlan upstream_sample_tee_binding{};
    u32 upstream_sample_tee_binding_count{};
    std::array<std::unique_ptr<Entry>, MaxTargets> entries{};
    MappingTemplate mapping{};
    PpTerminalScopeDiscoveryCoverage discovery_coverage{};
    PpTerminalScopeProgressCoverage progress_coverage{};
    PpTerminalScopeDiscoveryCoverageEmissionGate discovery_coverage_emission{};
    FinalGuestSurfaceReadbackSlotPool::Token upstream_feedback_slot{};
    HostPasses::PpUpstreamInputContentPlan upstream_input_plan{};
    HostPasses::PpUpstreamSampleRouteSet upstream_sample_routes{};
    PpUpstreamSampleTeeProducerTrace upstream_sample_tee_producer_trace{};
    HostPasses::PpUpstreamSampleTeeSourceObservation upstream_sample_tee_source{};
    PpUpstreamSampleTeeTransferObservation upstream_sample_tee_transfer{};
    HostPasses::PpUpstreamSampleTeeTransferredDepthSourcePlan
        upstream_sample_tee_transferred_depth_source{};
    HostPasses::PpUpstreamSampleTeeSelectedDepthPlan upstream_sample_tee_selected_depth{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthHint
        upstream_sample_tee_producer_pre_hint{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthHint
        upstream_sample_tee_producer_pre_observed_hint{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthGateEvidence
        upstream_sample_tee_producer_pre_gate_evidence{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthOccurrenceEvidence
        upstream_sample_tee_producer_pre_occurrences{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthRouteAbsentOrdinalCaptureEvidence
        upstream_sample_tee_route_absent_ordinal_capture_evidence{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthSelectedPredecessorPair
        upstream_sample_tee_selected_predecessor_pair{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm
        upstream_sample_tee_producer_pre_ordinal_arm{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthSelectedPairOrdinalArm
        upstream_sample_tee_selected_producer_ordinal_arm{};
    u32 upstream_sample_tee_selected_producer_occurrence_count{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthOrdinalContentBinding
        upstream_sample_tee_producer_pre_ordinal_binding{};
    u32 upstream_sample_tee_producer_pre_ordinal_inactive_flip_count{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthScopeLoadCandidateEvidence
        upstream_sample_tee_scope_load_candidate_evidence{};
    HostPasses::PpUpstreamSampleTeeScopeWriterTraceFramePlan
        upstream_sample_tee_scope_writer_trace_frame{};
    HostPasses::PpUpstreamSampleTeeScopeWriterTraceTailPlan
        upstream_sample_tee_scope_writer_trace_tail{};
    HostPasses::PpDepthPredecessorWriterIntent
        upstream_sample_tee_scope_writer_trace_intent{};
    u64 upstream_sample_tee_scope_writer_trace_expected_serial{};
    u64 upstream_sample_tee_scope_writer_trace_host_instance{};
    u32 upstream_sample_tee_scope_writer_trace_cycle_index{};
    u32 upstream_sample_tee_scope_writer_trace_live_pre_draws{};
    u32 upstream_sample_tee_scope_writer_trace_key_occurrence_count{};
    u32 upstream_sample_tee_scope_writer_trace_key_matches{};
    u32 upstream_sample_tee_scope_writer_trace_armed_key_matches{};
    u32 upstream_sample_tee_scope_writer_trace_capture_attempts{};
    u32 upstream_sample_tee_scope_writer_trace_pre_complete{};
    u32 upstream_sample_tee_scope_writer_trace_post_complete{};
    bool upstream_sample_tee_scope_writer_trace_captured{};
    bool upstream_sample_tee_scope_writer_trace_open{};
    HostPasses::PpDepthPredecessorWriterIntent
        upstream_sample_tee_producer_pre_intent{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthPost
        upstream_sample_tee_producer_pre_post{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthBinding
        upstream_sample_tee_producer_pre_selected_binding{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthBindDescriptor
        upstream_sample_tee_producer_pre_selected_binding_descriptor{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthDownloadTailPlan
        upstream_sample_tee_producer_pre_depth{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthScopeEntryObservation
        upstream_sample_tee_producer_pre_scope_entry{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthScopeEntryPlan
        upstream_sample_tee_producer_pre_scope_plan{};
    HostPasses::PpUpstreamDepthFirstWriterPostObservation
        upstream_sample_tee_producer_pre_scope_post{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthScopeEntryOwned
        upstream_sample_tee_producer_pre_scope_owned{};
    HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthHint
        upstream_sample_tee_predecessor_writer_pre_hint{};
    HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthHint
        upstream_sample_tee_predecessor_writer_pre_observed_hint{};
    HostPasses::PpDepthPredecessorWriterIntent
        upstream_sample_tee_predecessor_writer_pre_intent{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthPost
        upstream_sample_tee_predecessor_writer_pre_post{};
    HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthDownloadTailPlan
        upstream_sample_tee_predecessor_writer_pre_depth{};
    HostPasses::PpUpstreamSampleTeePredecessorWriterPreDepthOrdinalCaptureDescriptor
        upstream_sample_tee_predecessor_writer_pre_ordinal_capture_descriptor{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan
        upstream_sample_tee_predecessor_writer_pre_ordinal_capture_plan{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthOrdinalCaptureDescriptor
        upstream_sample_tee_producer_pre_ordinal_capture_descriptor{};
    HostPasses::PpUpstreamSampleTeeProducerPreDepthOrdinalCapturePlan
        upstream_sample_tee_producer_pre_ordinal_capture_plan{};
    HostPasses::PpDepthPredecessorEndpointStamp
        upstream_sample_tee_producer_pre_diagnostic_immediate_predecessor{};
    u64 upstream_sample_tee_producer_pre_expected_serial{};
    u64 upstream_sample_tee_producer_pre_host_instance{};
    u64 upstream_sample_tee_predecessor_writer_pre_expected_serial{};
    u64 upstream_sample_tee_predecessor_writer_pre_host_instance{};
    u32 upstream_sample_tee_selected_consumer_count{};
    u32 upstream_sample_tee_exact_selected_consumer_count{};
    u32 upstream_sample_tee_predecessor_writer_pre_exact_load_count{};
    bool upstream_sample_tee_producer_pre_open{};
    bool upstream_sample_tee_producer_pre_attempted{};
    bool upstream_sample_tee_producer_pre_bound{};
    bool upstream_sample_tee_producer_pre_collision{};
    bool upstream_sample_tee_producer_pre_scope_entry_collision{};
    bool upstream_sample_tee_predecessor_writer_pre_open{};
    bool upstream_sample_tee_predecessor_writer_pre_attempted{};
    bool upstream_sample_tee_predecessor_writer_pre_bound{};
    bool upstream_sample_tee_predecessor_writer_pre_collision{};
    bool upstream_sample_tee_selected_drift{};
    bool upstream_feedback_base_prefix_initialized{};
    HostPasses::PpUpstreamSampleTeeCandidateRecord upstream_sample_tee_candidate{};
    HostPasses::PpUpstreamSampleTeeCapturePlan upstream_sample_tee_capture{};
    HostPasses::PpUpstreamSampleTeeCommandPlan upstream_sample_tee_commands{};
    HostPasses::PpUpstreamSampleTeeControllerState upstream_sample_tee_controller{};
    bool upstream_sample_tee_attempted{};
    FinalGuestSurfaceStatus upstream_sample_tee_status{
        FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss upstream_sample_tee_loss{};
    bool upstream_sample_tee_copy{};
    bool upstream_input_ready{};
    std::mutex reducer_mutex{};
    u32 selected_frames{};
    u32 emitted_frames{};
    u32 complete_frames{};
    u32 loss_frames{};
    u32 upstream_feedback_selected_frames{};
    u32 upstream_feedback_emitted_frames{};
    u32 upstream_feedback_complete_frames{};
    u32 upstream_feedback_loss_frames{};
    bool calibrated_coverage_logged{};
    bool upstream_sample_tee_coverage_logged{};
    bool upstream_feedback_armed{};
    bool upstream_feedback_coverage_logged{};
    u64 target_sampled_depth_writer_hash{};
    bool target_sampled_depth_writer_valid{};
    bool target_sampled_depth_writer_collision{};
    PpUpstreamTargetFixedFunctionState target_fixed_function_state{};
    bool target_fixed_function_state_valid{};
    bool target_fixed_function_state_collision{};
};

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : Rasterizer{instance_, scheduler_, liverpool_,
                 ReadPpTerminalScopeRuntimeConfig().value_or(PpTerminalScopeRuntimeConfig{})} {}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_, PpTerminalScopeRuntimeConfig config)
    : instance{instance_}, scheduler{scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool,
                     HostPasses::PlanPpUpstreamSampleTeePipelinePolicy(
                         config.content.enabled && config.content.capture_upstream_sample_tee,
                         config.content.upstream_sample_tee_fragment_pgm_hash,
                         config.content.capture_upstream_fragment_export_tee)} {
    if (config.content.enabled) {
        pp_terminal_scope_content =
            std::make_unique<PpTerminalScopeContentRuntime>(instance, scheduler, texture_cache,
                                                            config);
    }
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
}

Rasterizer::~Rasterizer() = default;

void Rasterizer::ObservePpTerminalScopeFlip(
    u64 sequence, u64 process_time_us, const VideoCore::ImageColorScopeProducerObservation& scope,
    u32 final_source_width, u32 final_source_height, u32 logical_width, u32 logical_height) {
    if (!pp_terminal_scope_content) {
        return;
    }
    auto* target = texture_cache.ResolveDiagnosticColorScopeImage(scope.sampled_input_image);
    pp_terminal_scope_content->ObserveFlip(sequence, process_time_us, scope.sampled_input_image,
                                           target, final_source_width, final_source_height,
                                           logical_width, logical_height);
}

void Rasterizer::CalibratePpTerminalScopeScreenshots(
    const FinalGuestSurfaceFrameDiagnosticStamp& stamp,
    FinalGuestSurfacePresentationMapping mapping, u32 count, u64 fallback_process_time_us) {
    if (pp_terminal_scope_content) {
        pp_terminal_scope_content->Calibrate(stamp, mapping, count, fallback_process_time_us);
    }
}

bool Rasterizer::HasPpTerminalScopeContent() const noexcept {
    return pp_terminal_scope_content != nullptr;
}

bool Rasterizer::HasPpTerminalFinalBackingJoin() const noexcept {
    return pp_terminal_scope_content && pp_terminal_scope_content->HasFinalBackingJoin();
}

void Rasterizer::ObservePpTerminalFinalBacking(u64 sequence, FinalGuestSurfaceStage stage,
                                               const FinalGuestSurfaceTilePlan& plan,
                                               std::span<const std::byte> bytes,
                                               FinalGuestSurfaceStatus status,
                                               FinalGuestSurfaceLoss loss) {
    if (pp_terminal_scope_content) {
        pp_terminal_scope_content->ObserveFinalBacking(sequence, stage, plan, bytes, status, loss);
    }
}

void Rasterizer::FinalizePpTerminalScopeContent() {
    if (pp_terminal_scope_content) {
        pp_terminal_scope_content->Finalize();
    }
}

void Rasterizer::CpSync() {
    scheduler.EndRendering();
    auto cmdbuf = scheduler.CommandBuffer();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    diagnostic_depth_write_boundary = {};

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const auto& regs = liverpool->regs;
    const auto quad_list_plan = VideoCore::PlanQuadListDraw(
        regs.primitive_type, regs.stage_enable.raw, (regs.enable_primitive_restart & 1) != 0,
        is_indexed, regs.num_indices);
    auto pipeline_result = pipeline_cache.GetGraphicsPipeline(quad_list_plan.primitive_type);
    const GraphicsPipeline* pipeline = pipeline_result.pipeline;
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline, &pipeline_result.resources)) {
        return;
    }

    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, pipeline_result.vertex_inputs, buffer_barriers);
    u32 index_count = regs.num_indices;
    if (quad_list_plan.convert) {
        index_count = buffer_cache.BindQuadListIndexBuffer(is_indexed, index_offset);
        if (index_count == 0) {
            ResetBindings();
            return;
        }
        is_indexed = true;
    } else if (is_indexed) {
        buffer_cache.BindIndexBuffer(index_offset, buffer_barriers);
    }

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    CheckpointDepthWriteBoundaryBeforeExternalBegin();
    scheduler.BeginRendering(state);
    CheckpointDepthWriteBoundaryAfterExternalBegin();

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Graphics, pipeline->Handle());

    const auto stages = pipeline->GetStages();
    const u32 fragment_index = static_cast<u32>(Shader::LogicalStage::Fragment);
    const Shader::Info* fragment_info =
        fragment_index < stages.size() ? stages[fragment_index] : nullptr;
    if (fragment_info && fragment_info->pgm_hash == 0xdbae5023) {
        LOG_INFO(Render, "FGSCDPW fh={:#x} vh={:#x}", fragment_info->pgm_hash,
                 vs_info.pgm_hash);
    }
    if (fragment_info &&
        (fragment_info->pgm_hash == 0x14bd455a || fragment_info->pgm_hash == 0x144934aa)) {
        LOG_INFO(Render, "FGSCUWR fh={:#x} vh={:#x}", fragment_info->pgm_hash,
                 vs_info.pgm_hash);
    }
    const auto effective_cull_mode =
        LiverpoolToVK::IsPrimitiveCulled(quad_list_plan.primitive_type)
            ? regs.polygon_control.CullingMode()
            : AmdGpu::CullMode::None;
    const auto& diagnostic_dynamic_state = scheduler.GetDynamicState();

    const VideoCore::ImageColorScopeDrawDescriptor diagnostic_draw{
        .kind = VideoCore::ImageColorScopeDrawKind::Direct,
        .indexed = is_indexed,
        .element_count = regs.num_indices,
        .instance_count = regs.num_instances.NumInstances(),
        .fragment_pgm_hash = fragment_info ? fragment_info->pgm_hash : 0,
        .user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable,
        .guest_primitive_type = static_cast<u8>(regs.primitive_type),
        .effective_primitive_type = static_cast<u8>(quad_list_plan.primitive_type),
        .cull_mode = static_cast<u8>(effective_cull_mode),
        .front_face = static_cast<u8>(regs.polygon_control.front_face),
        .fragment_mrt_mask = static_cast<u8>(fragment_info ? fragment_info->mrt_mask : 0),
        .depth_compare_op = static_cast<u8>(diagnostic_dynamic_state.depth_compare_op),
        .quad_list_converted = quad_list_plan.convert,
        .fragment_writes_depth =
            fragment_info && fragment_info->stores.GetAny(Shader::IR::Attribute::Depth),
        .fragment_has_discard = fragment_info && fragment_info->has_discard,
        .depth_test_enabled = diagnostic_dynamic_state.depth_test_enabled,
        .depth_bounds_test_enabled = diagnostic_dynamic_state.depth_bounds_test_enabled,
        .depth_bias_enabled = diagnostic_dynamic_state.depth_bias_enabled,
        .stencil_test_enabled = diagnostic_dynamic_state.stencil_test_enabled,
        .sampled_bindings = diagnostic_sampled_bindings,
        .sampled_images = static_cast<u32>(diagnostic_sampled_images.size()),
        .storage_writes = diagnostic_storage_writes,
    };
    if (!CapturePpTerminalScopePreDraw(pipeline, state, diagnostic_draw)) {
        ResetBindings();
        return;
    }

    // Diagnostic pre-draw work can resume on a command buffer where neither the dynamic vertex
    // layout nor its buffers have been bound. Restore the exact guest state before either draw.
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16> post_capture_vertex_barriers;
    buffer_cache.BindVertexBuffers(*pipeline, pipeline_result.vertex_inputs,
                                   post_capture_vertex_barriers);
    ASSERT_MSG(post_capture_vertex_barriers.empty(),
               "Post-capture vertex rebind unexpectedly required a new buffer barrier");

    if (is_indexed) {
        cmdbuf.drawIndexed(index_count, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }

    MarkEncodedImageProducers(state, diagnostic_draw);

    ResetBindings();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;
    diagnostic_depth_write_boundary = {};

    const auto& regs = liverpool->regs;

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    auto pipeline_result = pipeline_cache.GetGraphicsPipeline(liverpool->regs.primitive_type);
    const GraphicsPipeline* pipeline = pipeline_result.pipeline;
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline, &pipeline_result.resources)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, pipeline_result.vertex_inputs, buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(0, buffer_barriers);
    }

    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }
    if (count_buffer) {
        if (auto barrier = count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                    vk::PipelineStageFlagBits2::eDrawIndirect)) {
            buffer_barriers.emplace_back(*barrier);
        }
    }

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    CheckpointDepthWriteBoundaryBeforeExternalBegin();
    scheduler.BeginRendering(state);
    CheckpointDepthWriteBoundaryAfterExternalBegin();

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Graphics, pipeline->Handle());

    const auto stages = pipeline->GetStages();
    const u32 fragment_index = static_cast<u32>(Shader::LogicalStage::Fragment);
    const Shader::Info* fragment_info =
        fragment_index < stages.size() ? stages[fragment_index] : nullptr;
    const auto effective_cull_mode =
        LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
            ? regs.polygon_control.CullingMode()
            : AmdGpu::CullMode::None;
    const auto& diagnostic_dynamic_state = scheduler.GetDynamicState();

    const VideoCore::ImageColorScopeDrawDescriptor diagnostic_draw{
        .kind = VideoCore::ImageColorScopeDrawKind::Indirect,
        .indexed = is_indexed,
        .fragment_pgm_hash = fragment_info ? fragment_info->pgm_hash : 0,
        .user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable,
        .guest_primitive_type = static_cast<u8>(regs.primitive_type),
        .effective_primitive_type = static_cast<u8>(regs.primitive_type),
        .cull_mode = static_cast<u8>(effective_cull_mode),
        .front_face = static_cast<u8>(regs.polygon_control.front_face),
        .fragment_mrt_mask = static_cast<u8>(fragment_info ? fragment_info->mrt_mask : 0),
        .depth_compare_op = static_cast<u8>(diagnostic_dynamic_state.depth_compare_op),
        .fragment_writes_depth =
            fragment_info && fragment_info->stores.GetAny(Shader::IR::Attribute::Depth),
        .fragment_has_discard = fragment_info && fragment_info->has_discard,
        .depth_test_enabled = diagnostic_dynamic_state.depth_test_enabled,
        .depth_bounds_test_enabled = diagnostic_dynamic_state.depth_bounds_test_enabled,
        .depth_bias_enabled = diagnostic_dynamic_state.depth_bias_enabled,
        .stencil_test_enabled = diagnostic_dynamic_state.stencil_test_enabled,
        .sampled_bindings = diagnostic_sampled_bindings,
        .sampled_images = static_cast<u32>(diagnostic_sampled_images.size()),
        .storage_writes = diagnostic_storage_writes,
    };
    if (!CapturePpTerminalScopePreDraw(pipeline, state, diagnostic_draw)) {
        ResetBindings();
        return;
    }

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_base, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_base,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
    }

    MarkEncodedImageProducers(state, diagnostic_draw);

    ResetBindings();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Compute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);

    MarkEncodedStorageImageProducers();

    ResetBindings();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(PipelineBindPoint::Compute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);

    MarkEncodedStorageImageProducers();

    ResetBindings();
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

bool Rasterizer::BindResources(
    const Pipeline* pipeline,
    const std::array<Shader::StageResourceSnapshot, MaxShaderStages>* resource_snapshots) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();
    if (pp_terminal_scope_content) {
        pp_terminal_scope_content->ResetUpstreamSampleTeeBinding();
    }
    if (VideoCore::IsPpSourceProducerTrackingEnabled()) {
        diagnostic_sampled_images.clear();
        diagnostic_sampled_cube_bindings.clear();
        diagnostic_sampled_routes.clear();
        diagnostic_sampled_view_bindings.clear();
        diagnostic_sampler_pair_bindings.clear();
        static_cast<void>(diagnostic_sampled_view_binding.Take());
        diagnostic_sampled_bindings = 0;
        diagnostic_storage_writes = 0;
    }

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        const auto stage_index = static_cast<u32>(stage->l_stage);
        const Shader::StageResourceSnapshot captured =
            resource_snapshots ? Shader::StageResourceSnapshot{}
                               : Shader::StageResourceSnapshot::Capture(*stage);
        const auto& resources = resource_snapshots ? (*resource_snapshots)[stage_index] : captured;
        if (!resources.valid || resources.buffers.size() != stage->buffers.size() ||
            resources.images.size() != stage->images.size() ||
            resources.samplers.size() != stage->samplers.size() ||
            resources.fmasks.size() != stage->fmasks.size()) {
            return false;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        stage->PushUd(binding, push_data);
        if (!BindBuffers(*stage, resources, binding, push_data)) {
            return false;
        }
        BindTextures(*stage, resources, binding);
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        // We only use fault buffer for DMA right now.
        buffer_cache.SynchronizeDmaBuffers();
        fault_process_pending = true;
    }

    return true;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    const VideoCore::ImageId src_image_id = desc0.is_written ? image1_id : image0_id;
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
        dst_image.CopyImage(src_image, VideoCore::ImageTransferOrigin::ComputeCopy, src_image_id);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    dst_image.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    image1.flags |= VideoCore::ImageFlagBits::GpuModified;
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::BindBuffers(const Shader::Info& stage,
                             const Shader::StageResourceSnapshot& resources,
                             Shader::Backend::Bindings& binding, Shader::PushData& push_data) {
    buffer_bindings.clear();
    for (u32 index = 0; index < stage.buffers.size(); ++index) {
        const auto& desc = stage.buffers[index];
        const auto vsharp = resources.buffers[index];
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 binding_size = desc.GetBindingSize(vsharp);
            const u64 size = memory->ClampRangeSize(vsharp.base_address, binding_size);
            const auto buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            buffer_bindings.emplace_back(buffer_id, vsharp, size);
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
        }
    }

    // Second pass to re-bind buffers that were updated after binding
    for (u32 i = 0; i < buffer_bindings.size(); i++) {
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        // Buffer is not from the cache, either a special buffer or unbound.
        if (!buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::PpUpstreamSampleTee) {
                auto* tee_buffer = pp_terminal_scope_content
                                       ? pp_terminal_scope_content->UpstreamSampleTeeBuffer()
                                       : nullptr;
                const auto tee_binding = HostPasses::PlanPpUpstreamSampleTeeBinding({
                    .enabled = true,
                    .stage = stage.l_stage,
                    .buffer_type = desc.buffer_type,
                    .is_written = desc.is_written,
                    .descriptor_bytes = vsharp.GetSize(),
                    .buffer_present = tee_buffer != nullptr,
                    .buffer_bytes = tee_buffer ? tee_buffer->SizeBytes() : 0,
                    .mapped_bytes = tee_buffer ? tee_buffer->mapped_data.size() : 0,
                });
                if (!tee_binding.bind) {
                    return false;
                }
                if (!pp_terminal_scope_content->RetainUpstreamSampleTeeBinding(tee_binding)) {
                    return false;
                }
                buffer_infos.emplace_back(tee_buffer->Handle(), 0, tee_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id);
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            ASSERT(adjust % 4 == 0);
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            const vk::AccessFlags2 access = vk::AccessFlagBits2::eShaderRead |
                                            (desc.is_written
                                                 ? vk::AccessFlagBits2::eShaderWrite
                                                 : vk::AccessFlags2{});
            if (auto barrier =
                    vk_buffer->GetBarrier(access, vk::PipelineStageFlagBits2::eAllCommands)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (desc.is_written && desc.is_formatted) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eUniformBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }

    return true;
}

void Rasterizer::BindTextures(const Shader::Info& stage,
                              const Shader::StageResourceSnapshot& resources,
                              Shader::Backend::Bindings& binding) {
    struct PendingSampledViewBinding {
        HostPasses::PpUpstreamFeedbackSampledViewBindingSelection selection{};
        VideoCore::ImageSubresourceLookupFact initial_lookup{};
        VideoCore::ImageSubresourceLookupFact rebound_lookup{};
        VideoCore::ImageSubresourceLookupFact selected_lookup_origin{};
        VideoCore::ImageViewCreationObservation creation{};
        VideoCore::ImageRefreshObservation refresh{};
        vk::ImageView final_view{};
        bool view_owner_exact{};
        vk::ImageLayout final_layout{vk::ImageLayout::eUndefined};
        u32 operation_use_mask{};
        bool resource_depth{};
        bool refresh_forwarded{};
        bool observed{};
        Shader::PpUpstreamSampleTeeCubeBindingFact cube_binding{};
    };
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the size of each consecutive array with the number of bindings consumed.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;
    boost::container::static_vector<u32, Shader::NUM_IMAGES> image_resource_indices;
    boost::container::static_vector<u32, Shader::NUM_IMAGES> image_array_elements;
    boost::container::static_vector<PendingSampledViewBinding, Shader::NUM_IMAGES>
        pending_sampled_view_bindings;
    std::array<bool, Shader::NUM_IMAGES> image_lod0_only{};

    for (u32 image_resource_index = 0; image_resource_index < stage.images.size();
         ++image_resource_index) {
        const auto& image_desc = stage.images[image_resource_index];
        const auto tsharp = resources.images[image_resource_index];
        image_lod0_only[image_resource_index] = tsharp.base_level == 0 && tsharp.last_level == 0;
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_resource_indices.emplace_back(image_resource_index);
            image_array_elements.emplace_back(0);
            pending_sampled_view_bindings.emplace_back();
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings =
            image_desc.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex
                ? tsharp.last_level - tsharp.base_level + 1
                : 1;

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::tuple{tsharp, image_desc});
            image_resource_indices.emplace_back(image_resource_index);
            image_array_elements.emplace_back(i);
            auto& pending = pending_sampled_view_bindings.emplace_back();
            pending.cube_binding = Shader::MakePpUpstreamSampleTeeCubeBindingFact(
                stage.is_pp_upstream_sample_tee_target, stage.l_stage,
                image_resource_index, i, image_desc, tsharp);

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            const bool observe_sampled_lookup =
                desc.type != VideoCore::TextureCache::BindingType::Storage &&
                VideoCore::IsPpSourceProducerTrackingEnabled();
            VideoCore::ImageSubresourceLookupObservation initial_lookup{};
            image_id = texture_cache.FindImage(
                desc, false, observe_sampled_lookup ? &initial_lookup : nullptr);
            if (observe_sampled_lookup) {
                pending.initial_lookup = initial_lookup.fact;
            }
            pending.selection.initial_direct = image_id;
            auto* image = &texture_cache.GetImage(image_id);
            if (VideoCore::ShouldUseAssociatedDepthForView(desc.view_info.format)) {
                if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                    pending.selection.initial_associated = depth_image_id;
                    // If this image has an associated depth image, it's a stencil attachment.
                    // Redirect the access to the actual depth-stencil buffer.
                    image_id = depth_image_id;
                    image = &texture_cache.GetImage(image_id);
                }
            }
            pending.selection.initial_selected = image_id;
            pending.operation_use_mask =
                stage.pp_upstream_image_operation_use_masks[image_resource_index];
            pending.resource_depth = image_desc.is_depth;
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    boost::container::static_vector<AmdGpu::Sampler, Shader::NUM_SAMPLERS> effective_samplers;
    for (u32 sampler_index = 0; sampler_index < stage.samplers.size(); ++sampler_index) {
        const auto& sampler = stage.samplers[sampler_index];
        auto ssharp = resources.samplers[sampler_index];
        if (sampler.disable_aniso) {
            const auto& tsharp = resources.images[sampler.associated_image];
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        effective_samplers.emplace_back(ssharp);
    }

    // Second pass to re-bind images that were updated after binding
    for (u32 flat_index = 0; flat_index < image_bindings.size(); ++flat_index) {
        auto& [image_id, desc] = image_bindings[flat_index];
        auto& pending = pending_sampled_view_bindings[flat_index];
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                pending.selection.rebind_requested = true;
                const bool observe_sampled_lookup =
                    !is_storage && VideoCore::IsPpSourceProducerTrackingEnabled();
                VideoCore::ImageSubresourceLookupObservation rebound_lookup{};
                image_id = texture_cache.FindImage(
                    desc, false, observe_sampled_lookup ? &rebound_lookup : nullptr);
                if (observe_sampled_lookup) {
                    pending.rebound_lookup = rebound_lookup.fact;
                }
                pending.selection.rebound_direct = image_id;
            }
            pending.selection.final_selected = image_id;

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            const bool observe_sampled_transit =
                !is_storage && VideoCore::IsPpSourceProducerTrackingEnabled();
            const bool selected_by_direct_lookup =
                image_id == (pending.selection.rebind_requested
                                 ? pending.selection.rebound_direct
                                 : pending.selection.initial_direct);
            const auto& authoritative_lookup = pending.selection.rebind_requested
                                                   ? pending.rebound_lookup
                                                   : pending.initial_lookup;
            const bool selected_by_unique_perfect_lookup =
                selected_by_direct_lookup &&
                authoritative_lookup.path == VideoCore::ImageSubresourceLookupPath::Perfect &&
                authoritative_lookup.perfect_candidate_count == 1 &&
                !authoritative_lookup.overflowed;
            if (observe_sampled_transit && selected_by_unique_perfect_lookup) {
                pending.selected_lookup_origin =
                    image.PeekDiagnosticSubresourceLookupProvenance();
            }
            VideoCore::ImageTextureObservation texture_observation{};
            auto& image_view = texture_cache.FindTexture(
                image_id, desc, observe_sampled_transit ? &texture_observation : nullptr);
            pending.refresh_forwarded = observe_sampled_transit;
            pending.refresh = texture_observation.refresh;
            pending.view_owner_exact = texture_observation.view_owner_exact;
            pending.creation = image_view.CreationObservation();
            pending.final_view = *image_view.image_view;
            pending.observed = observe_sampled_transit;
            VideoCore::ImageTransitBarrierReceipt sampled_transit_receipt{};
            auto* sampled_transit =
                observe_sampled_transit ? &sampled_transit_receipt : nullptr;

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                const bool feedback =
                    instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target;
                image.Transit(feedback ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                       : vk::ImageLayout::eGeneral,
                              feedback ? AttachmentFeedbackLoopTextureAccess()
                                       : vk::AccessFlagBits2::eShaderRead |
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead,
                              {}, {}, sampled_transit);
            } else {
                if (is_storage) {
                    image.Transit(vk::ImageLayout::eGeneral,
                                  vk::AccessFlagBits2::eShaderRead |
                                      vk::AccessFlagBits2::eShaderWrite,
                                  desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead,
                                  desc.view_info.range, {}, sampled_transit);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;
            if (is_storage) {
                storage_written_images.emplace_back(image_id);
                if (VideoCore::IsPpSourceProducerTrackingEnabled()) {
                    ++diagnostic_storage_writes;
                }
            } else if (VideoCore::IsPpSourceProducerTrackingEnabled()) {
                ++diagnostic_sampled_bindings;
                const auto sampled = std::ranges::find_if(
                    diagnostic_sampled_images,
                    [image_id](const auto& candidate) { return candidate.id == image_id; });
                const bool first_sampled_binding = sampled == diagnostic_sampled_images.end();
                if (sampled == diagnostic_sampled_images.end()) {
                    auto sampled_aspect = image.aspect_mask;
                    if (sampled_aspect & vk::ImageAspectFlagBits::eDepth &&
                        LiverpoolToVK::IsFormatDepthCompatible(image_view.info.format)) {
                        sampled_aspect = vk::ImageAspectFlagBits::eDepth;
                    }
                    if (sampled_aspect & vk::ImageAspectFlagBits::eStencil &&
                        LiverpoolToVK::IsFormatStencilCompatible(image_view.info.format)) {
                        sampled_aspect = vk::ImageAspectFlagBits::eStencil;
                    }
                    diagnostic_sampled_images.emplace_back(DiagnosticSampledImage{
                        .id = image_id,
                        .view = image_view.info,
                        .backing_generation = image.EnsureDiagnosticBackingGeneration(),
                        .aspect_mask = static_cast<u32>(
                            static_cast<VkImageAspectFlags>(sampled_aspect)),
                        .requested_width = desc.info.size.width,
                        .requested_height = desc.info.size.height,
                    });
                } else {
                    auto sampled_aspect = image.aspect_mask;
                    if (sampled_aspect & vk::ImageAspectFlagBits::eDepth &&
                        LiverpoolToVK::IsFormatDepthCompatible(image_view.info.format)) {
                        sampled_aspect = vk::ImageAspectFlagBits::eDepth;
                    }
                    if (sampled_aspect & vk::ImageAspectFlagBits::eStencil &&
                        LiverpoolToVK::IsFormatStencilCompatible(image_view.info.format)) {
                        sampled_aspect = vk::ImageAspectFlagBits::eStencil;
                    }
                    sampled->view_conflict |= sampled->view != image_view.info;
                    sampled->extent_conflict |=
                        sampled->requested_width != desc.info.size.width ||
                        sampled->requested_height != desc.info.size.height;
                    sampled->backing_conflict |=
                        sampled->backing_generation != image.EnsureDiagnosticBackingGeneration();
                    sampled->aspect_conflict |=
                        sampled->aspect_mask != static_cast<u32>(
                                                       static_cast<VkImageAspectFlags>(sampled_aspect));
                }
                const auto selected = std::ranges::find_if(
                    diagnostic_sampled_images,
                    [image_id](const auto& candidate) { return candidate.id == image_id; });
                ASSERT(selected != diagnostic_sampled_images.end());
                auto bound_view = MakeImageColorScopeTrackedView(
                    {.id = image_id, .uid = image.image_uid}, image, image_view.info,
                    selected->view_conflict);
                bound_view.aspect_mask = selected->aspect_mask;
                bound_view.valid &= !selected->view_conflict && !selected->backing_conflict &&
                                    !selected->aspect_conflict;
                const auto sampled_transit_fact =
                    HostPasses::MakePpUpstreamFeedbackSampledImageTransitFact(
                        sampled_transit_receipt, bound_view,
                        image.PeekDiagnosticDepthStencilProducer());
                if (first_sampled_binding) {
                    static_cast<void>(selected->transit.ObserveFirst(sampled_transit_fact));
                } else {
                    static_cast<void>(
                        selected->transit.ObserveDuplicate(sampled_transit_fact));
                }
                const u32 resource_index = image_resource_indices[flat_index];
                HostPasses::PpUpstreamSampleRoute route{
                    .logical_stage = static_cast<u32>(stage.l_stage),
                    .image_resource_index = resource_index,
                    .array_element = image_array_elements[flat_index],
                    .view =
                        {
                            .type = static_cast<u32>(image_view.info.type),
                            .format = static_cast<u32>(image_view.info.format),
                            .base_mip = image_view.info.range.base.level,
                            .mip_count = image_view.info.range.extent.levels,
                            .base_layer = image_view.info.range.base.layer,
                            .layer_count = image_view.info.range.extent.layers,
                            .mapping = {static_cast<u32>(image_view.info.mapping.r),
                                        static_cast<u32>(image_view.info.mapping.g),
                                        static_cast<u32>(image_view.info.mapping.b),
                                        static_cast<u32>(image_view.info.mapping.a)},
                            .min_lod = image_view.info.min_lod,
                            .storage = image_view.info.is_storage,
                        },
                };
                for (u32 sampler_index = 0; sampler_index < stage.samplers.size();
                     ++sampler_index) {
                    if (stage.samplers[sampler_index].associated_image != resource_index) {
                        continue;
                    }
                    if (route.sampler_count == route.MaxSamplers) {
                        route.valid = false;
                        break;
                    }
                    const auto& ssharp = effective_samplers[sampler_index];
                    route.samplers[route.sampler_count++] = {
                        .raw0 = ssharp.raw0,
                        .raw1 = ssharp.raw1,
                    };
                }
                if (diagnostic_sampled_routes.size() == diagnostic_sampled_routes.capacity()) {
                    route.valid = false;
                } else {
                    diagnostic_sampled_routes.emplace_back(
                        HostPasses::PpUpstreamSampleRouteObservation{
                            .input_index = static_cast<u32>(
                                std::distance(diagnostic_sampled_images.begin(), selected)),
                            .route = route,
                        });
                }
            }

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            pending.final_layout = image.backing->state.layout;
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        if (!is_storage && VideoCore::IsPpSourceProducerTrackingEnabled()) {
            const std::span<const vk::DescriptorImageInfo> descriptors{
                &image_infos[image_info_idx], array_size};
            for (u32 element = 0; element < array_size; ++element) {
                const auto& pending = pending_sampled_view_bindings[image_binding_idx + element];
                if (!pending.observed) {
                    continue;
                }
                const auto fact =
                    HostPasses::MakePpUpstreamFeedbackSampledViewBindingFact({
                        .operation_use_mask = pending.operation_use_mask,
                        .resource_depth = pending.resource_depth,
                        .selection = pending.selection,
                        .creation = pending.creation,
                        .refresh = pending.refresh,
                        .refresh_forwarded = pending.refresh_forwarded,
                        .final_view = pending.final_view,
                        .view_owner_exact = pending.view_owner_exact,
                        .final_layout = pending.final_layout,
                        .descriptor_infos = descriptors,
                        .selected_descriptor_index = element,
                        .write = &set_write,
                        .expected_descriptor_type = vk::DescriptorType::eSampledImage,
                        .expected_binding = set_write.dstBinding,
                        .expected_descriptor_count = array_size,
                        .expected_array_element = 0,
                        .observed = true,
                    });
                const auto subresource_lookup =
                    HostPasses::MakePpUpstreamFeedbackImageSubresourceLookupBindingFact({
                        .initial_lookup = pending.initial_lookup,
                        .rebound_lookup = pending.rebound_lookup,
                        .selected_origin = pending.selected_lookup_origin,
                        .rebind_requested = pending.selection.rebind_requested,
                    });
                diagnostic_sampled_view_bindings.emplace_back(DiagnosticSampledViewBinding{
                    .image_id = image_bindings[image_binding_idx + element].first,
                    .logical_stage = static_cast<u32>(stage.l_stage),
                    .image_resource_index =
                        image_resource_indices[image_binding_idx + element],
                    .fact = fact,
                    .subresource_lookup = subresource_lookup,
                });
                if (HostPasses::ShouldRecordPpUpstreamFeedbackCubeLiveObservation(
                        pending.cube_binding)) {
                    diagnostic_sampled_cube_bindings.emplace_back(
                        HostPasses::PpUpstreamFeedbackCubeLiveObservation{
                            .physical_image = image_bindings[image_binding_idx + element].first,
                            .binding = pending.cube_binding,
                            .view = fact,
                        });
                }
            }
        }

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (u32 sampler_index = 0; sampler_index < stage.samplers.size(); ++sampler_index) {
        const auto& ssharp = effective_samplers[sampler_index];
        VideoCore::SamplerCacheObservation sampler_cache{};
        const bool observe_sampler = VideoCore::IsPpSourceProducerTrackingEnabled();
        const auto vk_sampler = texture_cache.GetSampler(
            ssharp, liverpool->regs.ta_bc_base, observe_sampler ? &sampler_cache : nullptr);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
        if (observe_sampler) {
            const auto& sampler_resource = stage.samplers[sampler_index];
            const u64 association_mask =
                stage.pp_upstream_sampler_image_association_masks[sampler_index];
            const u64 disable_aniso_mask =
                stage.pp_upstream_sampler_image_disable_aniso_masks[sampler_index];
            const u64 keep_aniso_mask =
                stage.pp_upstream_sampler_image_keep_aniso_masks[sampler_index];
            const bool shared_sampler = std::popcount(association_mask) > 1;
            const u64 associated_disable_mask = association_mask & disable_aniso_mask;
            const bool shared_intent_conflict =
                shared_sampler && associated_disable_mask != 0 &&
                associated_disable_mask != association_mask;
            bool have_adjusted_outcome{};
            bool first_adjusted_outcome{};
            bool shared_outcome_conflict{};
            for (u32 image_resource_index = 0; image_resource_index < Shader::NUM_IMAGES;
                 ++image_resource_index) {
                const u64 image_bit = u64{1} << image_resource_index;
                if ((associated_disable_mask & image_bit) == 0) {
                    continue;
                }
                const bool outcome = image_lod0_only[image_resource_index];
                if (!have_adjusted_outcome) {
                    have_adjusted_outcome = true;
                    first_adjusted_outcome = outcome;
                } else {
                    shared_outcome_conflict |= outcome != first_adjusted_outcome;
                }
            }
            for (u32 image_resource_index = 0; image_resource_index < Shader::NUM_IMAGES;
                 ++image_resource_index) {
                const u64 image_bit = u64{1} << image_resource_index;
                const u32 operation_use_mask =
                    stage.pp_upstream_sampler_image_operation_use_masks[sampler_index]
                                                                         [image_resource_index];
                const bool association_present = (association_mask & image_bit) != 0;
                const bool adjustment_intended = (disable_aniso_mask & image_bit) != 0;
                const bool pair_intent_conflict =
                    adjustment_intended && (keep_aniso_mask & image_bit) != 0;
                if (operation_use_mask == 0 && !association_present && !adjustment_intended) {
                    continue;
                }
                const auto adjustment_outcome =
                    HostPasses::ClassifyPpUpstreamFeedbackSamplerAdjustment({
                        .full_image_resource_index = image_resource_index,
                        .legacy_associated_image = sampler_resource.associated_image,
                        .disable_aniso = adjustment_intended,
                        .image_single_level = image_lod0_only[image_resource_index],
                        .shared_intent_conflict = shared_intent_conflict,
                        .shared_outcome_conflict = shared_outcome_conflict,
                        .pair_intent_conflict = pair_intent_conflict,
                    });
                const auto fact = HostPasses::MakePpUpstreamFeedbackSamplerBindingFact({
                    .logical_stage = static_cast<u32>(stage.l_stage),
                    .sampler_resource_index = sampler_index,
                    .image_resource_index = image_resource_index,
                    .pair_operation_use_mask = operation_use_mask,
                    .effective_sampler = &ssharp,
                    .association_present = association_present,
                    .adjustment_intended = adjustment_intended,
                    .adjustment_outcome = adjustment_outcome,
                    .inline_sampler = static_cast<bool>(sampler_resource.is_inline_sampler),
                    .shared_sampler = shared_sampler,
                    .creation = sampler_cache.fact.creation,
                    .cache = sampler_cache.fact,
                    .returned_sampler = vk_sampler,
                    .descriptor_info = &image_infos.back(),
                    .write = &set_write,
                    .expected_binding = set_write.dstBinding,
                    .observed = true,
                });
                diagnostic_sampler_pair_bindings.emplace_back(
                    HostPasses::PpUpstreamFeedbackSamplerPairObservation{
                        .key = {
                            .logical_stage = static_cast<u32>(stage.l_stage),
                            .sampler_resource_index = sampler_index,
                            .image_resource_index = image_resource_index,
                        },
                        .fact = fact,
                    });
            }
        }
    }
}

void Rasterizer::CheckpointDepthWriteBoundaryBeforeExternalBegin() {
    if (diagnostic_depth_write_boundary.active) {
        diagnostic_depth_write_boundary.pre_external_begin_rendering = scheduler.IsRendering();
        diagnostic_depth_write_boundary.pre_external_host_rendering_instance =
            scheduler.HostRenderingInstanceId();
        diagnostic_depth_write_boundary.pre_external_guest_rendering_serial =
            scheduler.RenderingSerial();
        diagnostic_depth_write_boundary.pre_external_diagnostic_resume =
            scheduler.HostRenderingOrigin() ==
            HostRenderingInstanceOrigin::DiagnosticResume;
    }
}

void Rasterizer::CheckpointDepthWriteBoundaryAfterExternalBegin() {
    if (!diagnostic_depth_write_boundary.active) {
        return;
    }
    diagnostic_depth_write_boundary.after_rendering = scheduler.IsRendering();
    diagnostic_depth_write_boundary.host_rendering_instance_after =
        scheduler.HostRenderingInstanceId();
    diagnostic_depth_write_boundary.guest_rendering_serial_after = scheduler.RenderingSerial();
    diagnostic_depth_write_boundary.after_diagnostic_resume =
        scheduler.HostRenderingOrigin() == HostRenderingInstanceOrigin::DiagnosticResume;
}

bool Rasterizer::CapturePpTerminalScopePreDraw(
    const GraphicsPipeline* pipeline, const RenderState& state,
    const VideoCore::ImageColorScopeDrawDescriptor& draw) {
    if (!pp_terminal_scope_content || !VideoCore::IsPpSourceProducerTrackingEnabled()) {
        return true;
    }
    const auto& regs = liverpool->regs;
    const u64 rendering_serial = scheduler.RenderingSerial();
    u32 active_attachment_mask{};
    const u32 attachment_span = std::min<u32>(state.num_color_attachments, 8);
    for (u32 index = 0; index < attachment_span; ++index) {
        if (cb_descs[index].first) {
            active_attachment_mask |= 1u << index;
        }
    }
    const auto tee_outputs = HostPasses::PlanPpUpstreamSampleTeeOutputs({
        .enabled = pp_terminal_scope_content->CapturesUpstreamSampleTee(),
        .attachment_span = state.num_color_attachments,
        .active_attachment_mask = active_attachment_mask,
    });
    const Shader::Info* fragment_info{};
    if (pipeline) {
        const auto stages = pipeline->GetStages();
        const u32 fragment_index = static_cast<u32>(Shader::LogicalStage::Fragment);
        if (fragment_index < stages.size()) {
            fragment_info = stages[fragment_index];
        }
    }
    if (pp_terminal_scope_content->CapturesUpstreamSampleTee() && db_desc.first &&
        state.depth_stencil_attachment.has_depth) {
        auto& depth_image = texture_cache.GetImage(db_desc.first);
        pp_terminal_scope_content->ObserveUpstreamSampleTeeProducerPreDepthScopeEntryBegin(
            db_desc.first, depth_image, state, db_desc.second.view_info.range,
            rendering_serial,
            {.kind = draw.kind,
             .indexed = draw.indexed,
             .element_count = draw.element_count,
             .instance_count = draw.instance_count,
             .sampled_images = draw.sampled_images,
             .storage_writes = draw.storage_writes},
            fragment_info ? fragment_info->pgm_hash : 0,
            IsEffectiveDepthWrite(state, scheduler.GetDynamicState()));
    }
    const bool classify_inputs = pp_terminal_scope_content->ShouldClassifyInputs(draw);
    std::array<PpTerminalScopeSampledInput, PpTerminalScopeSampledInputs::MaxInputs>
        sampled_input_candidates{};
    std::array<VideoCore::Image*, PpTerminalScopeSampledInputs::MaxInputs> sampled_input_images{};
    std::array<VideoCore::ImageColorScopePrivateLink, PpTerminalScopeSampledInputs::MaxInputs>
        sampled_input_links{};
    std::array<VideoCore::ImageColorScopeTrackedView, PpTerminalScopeSampledInputs::MaxInputs>
        sampled_input_tracked_views{};
    std::array<VideoCore::ImageColorScopeProducerObservation,
               PpTerminalScopeSampledInputs::MaxInputs>
        sampled_input_scopes{};
    std::array<HostPasses::PpUpstreamFeedbackProducerDepthConsumerInput,
               PpTerminalScopeSampledInputs::MaxInputs>
        sampled_input_depth_consumers{};
    std::array<HostPasses::PpUpstreamInputContentView, PpTerminalScopeSampledInputs::MaxInputs>
        upstream_input_views{};
    std::array<PpUpstreamSampledExtentObservation, PpTerminalScopeSampledInputs::MaxInputs>
        sampled_input_extents{};
    if (classify_inputs &&
        diagnostic_sampled_images.size() <= PpTerminalScopeSampledInputs::MaxInputs) {
        for (u32 index = 0; index < diagnostic_sampled_images.size(); ++index) {
            const auto& sampled = diagnostic_sampled_images[index];
            auto& sampled_image = texture_cache.GetImage(sampled.id);
            sampled_input_images[index] = &sampled_image;
            sampled_input_links[index] = {.id = sampled.id, .uid = sampled_image.image_uid};
            sampled_input_tracked_views[index] = MakeImageColorScopeTrackedView(
                sampled_input_links[index], sampled_image, sampled.view, sampled.view_conflict);
            sampled_input_tracked_views[index].aspect_mask = sampled.aspect_mask;
            sampled_input_tracked_views[index].valid &= !sampled.aspect_conflict;
            sampled_input_scopes[index] = sampled_image.ObserveDiagnosticColorScope();
            if (sampled_image.info.props.is_depth) {
                sampled_input_depth_consumers[index] = {
                    .selected_view = sampled_input_tracked_views[index],
                    .current_writer = sampled_image.PeekDiagnosticDepthStencilProducer(),
                    .transit = sampled.transit.Peek(),
                    .live_state = sampled_image.ObserveReadState(
                        vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                        vk::AccessFlagBits2::eShaderRead, sampled.view.range),
                };
            }
            sampled_input_extents[index] = {
                .requested_width = sampled.requested_width,
                .requested_height = sampled.requested_height,
                .backing_width = sampled_image.info.size.width,
                .backing_height = sampled_image.info.size.height,
                .conflict = sampled.extent_conflict,
                .valid = sampled.requested_width != 0 && sampled.requested_height != 0 &&
                         sampled_image.info.size.width != 0 && sampled_image.info.size.height != 0,
            };
            sampled_input_candidates[index].producer = ClassifyPpTerminalScopePredecessor(
                sampled_image.PeekDiagnosticProducer(), sampled_input_scopes[index]);
            const auto& range = sampled.view.range;
            const auto mip_dimension = [](u32 dimension, u32 level) {
                return level < 32 ? std::max(1u, dimension >> level) : 0u;
            };
            const bool type_2d = sampled.view.type == AmdGpu::ImageType::Color2D ||
                                 sampled.view.type == AmdGpu::ImageType::Color2DArray;
            sampled_input_candidates[index].view = ClassifyPpTerminalScopeSampledInputView({
                .width = mip_dimension(sampled_image.info.size.width, range.base.level),
                .height = mip_dimension(sampled_image.info.size.height, range.base.level),
                .format = TerminalScopeFormat(sampled.view.format),
                .samples = sampled_image.backing ? sampled_image.backing->num_samples : 0,
                .base_mip = range.base.level,
                .mip_count = range.extent.levels,
                .base_layer = range.base.layer,
                .layer_count = range.extent.layers,
                .color = sampled_image.aspect_mask == vk::ImageAspectFlagBits::eColor,
                .type_2d = type_2d,
                .uniform_state =
                    sampled_image.backing && sampled_image.backing->subresource_states.empty(),
                .view_conflict = sampled.view_conflict,
            });
            upstream_input_views[index] = {
                .width = mip_dimension(sampled_image.info.size.width, range.base.level),
                .height = mip_dimension(sampled_image.info.size.height, range.base.level),
                .bytes_per_texel =
                    !sampled_image.info.props.is_block && sampled_image.info.num_bits % 8 == 0
                        ? sampled_image.info.num_bits / 8
                        : 0,
                .base_mip = range.base.level,
                .mip_count = range.extent.levels,
                .base_layer = range.base.layer,
                .layer_count = range.extent.layers,
                .color = sampled_image.aspect_mask == vk::ImageAspectFlagBits::eColor,
                .type_2d = type_2d,
                .uniform_state =
                    sampled_image.backing && sampled_image.backing->subresource_states.empty(),
                .view_conflict = sampled.view_conflict,
            };
        }
    }
    bool sampled_view_binding_selected{};
    const auto select_sampled_view_binding = [&] {
        if (sampled_view_binding_selected) {
            return;
        }
        sampled_view_binding_selected = true;
        static_cast<void>(diagnostic_sampled_view_binding.Take());
        if (diagnostic_sampled_images.size() != 1) {
            return;
        }
        const auto sole_image_id = diagnostic_sampled_images.front().id;
        boost::container::static_vector<
            HostPasses::PpUpstreamFeedbackSamplerPairKey,
            Shader::NUM_SAMPLERS * Shader::NUM_IMAGES * Shader::MaxStageTypes>
            selected_pairs;
        for (const auto& sampler_binding : diagnostic_sampler_pair_bindings) {
            const bool selects_sole_image = std::ranges::any_of(
                diagnostic_sampled_view_bindings, [&](const auto& view_binding) {
                    return view_binding.image_id == sole_image_id &&
                           view_binding.logical_stage == sampler_binding.key.logical_stage &&
                           view_binding.image_resource_index ==
                               sampler_binding.key.image_resource_index;
                });
            if (selects_sole_image) {
                selected_pairs.emplace_back(sampler_binding.key);
            }
        }
        const auto sampler_selection =
            HostPasses::SelectPpUpstreamFeedbackSamplerBindingsForSoleImage({
                .single_physical_image_authoritative = true,
                .bindings = diagnostic_sampler_pair_bindings,
                .selected_pairs = selected_pairs,
            });
        for (const auto& view_binding : diagnostic_sampled_view_bindings) {
            if (view_binding.image_id != sole_image_id) {
                continue;
            }
            if (diagnostic_sampled_view_binding.Peek().binding_count == 0) {
                static_cast<void>(diagnostic_sampled_view_binding.ObserveFirst(
                    view_binding.fact, view_binding.subresource_lookup,
                    sampler_selection.fact));
            } else {
                static_cast<void>(diagnostic_sampled_view_binding.ObserveDuplicate(
                    view_binding.fact, view_binding.subresource_lookup,
                    sampler_selection.fact));
            }
        }
    };
    for (u32 index = 0; index < state.num_color_attachments; ++index) {
        const auto image_id = cb_descs[index].first;
        if (!image_id) {
            continue;
        }
        auto& image = texture_cache.GetImage(image_id);
        PpTerminalScopeSampledInputs sampled_inputs{};
        if (classify_inputs) {
            if (diagnostic_sampled_images.size() > PpTerminalScopeSampledInputs::MaxInputs) {
                sampled_inputs.status = FinalGuestSurfaceStatus::CapacityLoss;
                sampled_inputs.loss.tile_capacity = 1;
            } else {
                auto candidates = sampled_input_candidates;
                for (u32 input = 0; input < diagnostic_sampled_images.size(); ++input) {
                    const auto& sampled = diagnostic_sampled_images[input];
                    candidates[input].aliases_output = sampled.id == image_id;
                    upstream_input_views[input].aliases_output = sampled.id == image_id;
                    if (candidates[input].aliases_output) {
                        const auto& sampled_image = texture_cache.GetImage(sampled.id);
                        const auto sampled_layout = sampled_image.backing
                                                        ? sampled_image.backing->state.layout
                                                        : vk::ImageLayout::eUndefined;
                        const auto attachment_layout = state.color_attachments[index].image_layout;
                        candidates[input].feedback = ClassifyPpTerminalScopeFeedbackLoop({
                            .logical_alias = true,
                            .exact_subresource =
                                sampled.view.range == cb_descs[index].second.view_info.range,
                            .extension_supported =
                                instance.IsAttachmentFeedbackLoopLayoutSupported(),
                            .sampled_feedback_layout =
                                sampled_layout ==
                                vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                            .attachment_feedback_layout =
                                attachment_layout ==
                                vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                            .sampled_general_layout = sampled_layout == vk::ImageLayout::eGeneral,
                            .attachment_general_layout =
                                attachment_layout == vk::ImageLayout::eGeneral,
                            .dynamic_feedback_enabled = attachment_feedback_loop,
                        });
                    }
                }
                sampled_inputs = ClassifyPpTerminalScopeSampledInputs(
                    draw.sampled_images,
                    std::span{candidates.data(), diagnostic_sampled_images.size()});
            }
        }
        if (sampled_inputs.count == 1 && diagnostic_sampled_images.size() == 1) {
            select_sampled_view_binding();
        }
        const auto sampled_view_binding =
            sampled_inputs.count == 1 && diagnostic_sampled_images.size() == 1
                ? diagnostic_sampled_view_binding.Peek()
                : HostPasses::PpUpstreamFeedbackSampledViewBindingObservation{};
        pp_terminal_scope_content->ObservePreDraw(
            image_id, image, state, index, rendering_serial, draw, sampled_inputs,
            std::span{sampled_input_images.data(), sampled_inputs.count},
            std::span{sampled_input_links.data(), sampled_inputs.count},
            std::span{sampled_input_tracked_views.data(), sampled_inputs.count},
            std::span{sampled_input_scopes.data(), sampled_inputs.count},
            std::span{sampled_input_depth_consumers.data(), sampled_inputs.count},
            sampled_view_binding,
            std::span{upstream_input_views.data(), sampled_inputs.count},
            std::span{sampled_input_extents.data(), sampled_inputs.count},
            std::span<const HostPasses::PpUpstreamSampleRouteObservation>{
                diagnostic_sampled_routes.data(), diagnostic_sampled_routes.size()},
            std::span<const HostPasses::PpUpstreamFeedbackCubeLiveObservation>{
                diagnostic_sampled_cube_bindings.data(),
                diagnostic_sampled_cube_bindings.size()},
            MakePpUpstreamTargetFixedFunctionState(regs, pipeline),
            fragment_info, tee_outputs, pipeline ? pipeline->GetLayout() : vk::PipelineLayout{});
    }
    const auto& depth_range = db_desc.second.view_info.range;
    if (pp_terminal_scope_content->CapturesUpstreamSampleTee() && db_desc.first &&
        state.depth_stencil_attachment.has_depth) {
        auto& depth_image = texture_cache.GetImage(db_desc.first);
        pp_terminal_scope_content->ObserveUpstreamSampleTeeProducerPreDepth(
            db_desc.first, depth_image, state, rendering_serial, draw, depth_range,
            IsEffectiveDepthWrite(state, scheduler.GetDynamicState()));
    }
    if (pp_terminal_scope_content->CapturesUpstreamDepthPrePost() && db_desc.first &&
        state.depth_stencil_attachment.has_depth) {
        PpTerminalScopeSampledInputs sampled_inputs{};
        if (classify_inputs) {
            if (diagnostic_sampled_images.size() > PpTerminalScopeSampledInputs::MaxInputs) {
                sampled_inputs.status = FinalGuestSurfaceStatus::CapacityLoss;
                sampled_inputs.loss.tile_capacity = 1;
            } else {
                auto candidates = sampled_input_candidates;
                for (u32 input = 0; input < diagnostic_sampled_images.size(); ++input) {
                    candidates[input].aliases_output =
                        diagnostic_sampled_images[input].id == db_desc.first;
                }
                sampled_inputs = ClassifyPpTerminalScopeSampledInputs(
                    draw.sampled_images,
                    std::span{candidates.data(), diagnostic_sampled_images.size()});
            }
        }
        auto& depth_image = texture_cache.GetImage(db_desc.first);
        const VideoCore::ImageColorScopePrivateLink depth_link{
            .id = db_desc.first,
            .uid = depth_image.image_uid,
        };
        const u64 depth_backing_generation =
            depth_image.backing ? depth_image.EnsureDiagnosticBackingGeneration() : 0;
        const auto& boundary = diagnostic_depth_write_boundary;
        const bool guest_serial_advanced =
            boundary.guest_rendering_serial_before != std::numeric_limits<u64>::max() &&
            boundary.guest_rendering_serial_after ==
                boundary.guest_rendering_serial_before + 1;
        const auto depth_write_boundary = ClassifyPpTerminalScopeDepthWriteBoundary({
            .tracked = boundary.active && boundary.tracked,
            .uniform_state = boundary.uniform_state,
            .image_identity_exact = boundary.link == depth_link,
            .backing_exact = boundary.backing_generation != 0 &&
                             boundary.backing_generation == depth_backing_generation,
            .range_exact = boundary.range == depth_range,
            .encoded_depth_writer = boundary.encoded_depth_writer,
            .writer_image_identity_exact = boundary.writer_image_identity_exact,
            .writer_backing_exact = boundary.writer_backing_exact,
            .writer_range_exact = boundary.writer_range_exact,
            .writer_mutation_exact = boundary.writer_mutation_exact &&
                                     boundary.mutation_epoch != 0 &&
                                     boundary.mutation_epoch ==
                                         depth_image.DiagnosticMutationEpoch(),
            .writer_guest_serial_exact = boundary.writer_guest_serial_exact,
            .writer_host_instance_exact = boundary.writer_host_instance_exact,
            .pre_rendering = boundary.pre_rendering,
            .post_transit_rendering = boundary.post_transit_rendering,
            .pre_external_begin_rendering = boundary.pre_external_begin_rendering,
            .after_rendering = boundary.after_rendering,
            .target_guest_serial_exact = boundary.guest_rendering_serial_after != 0 &&
                                         boundary.guest_rendering_serial_after ==
                                             rendering_serial,
            .guest_serial_advanced = guest_serial_advanced,
            .target_depth_load = state.depth_stencil_attachment.has_depth &&
                                 !state.depth_stencil_attachment.depth_clear,
            .state_exact = boundary.state_exact,
            .source_depth_stencil_attachment_write =
                boundary.source_depth_stencil_attachment_write,
            .source_barrier_recognized_write = boundary.source_barrier_recognized_write,
            .before_diagnostic_resume = boundary.before_diagnostic_resume,
            .after_diagnostic_resume = boundary.after_diagnostic_resume,
            .host_rendering_instance_before = boundary.host_rendering_instance_before,
            .host_rendering_instance_after = boundary.host_rendering_instance_after,
        });
        pp_terminal_scope_content->ObserveDepthUpstreamPreDraw(
            db_desc.first, depth_image, state, rendering_serial, draw, sampled_inputs,
            depth_write_boundary,
            {
                .current_link = boundary.link,
                .current_range = boundary.range,
                .current_backing_generation = boundary.backing_generation,
                .current_mutation_epoch = boundary.mutation_epoch,
                .boundary_checkpoints = {
                    .source_pre_transit = {
                        .guest_rendering_serial =
                            boundary.guest_rendering_serial_before,
                        .host_rendering_instance_id =
                            boundary.host_rendering_instance_before,
                        .rendering_active = boundary.pre_rendering,
                        .regular_origin =
                            !boundary.before_diagnostic_resume,
                        .diagnostic_origin = boundary.before_diagnostic_resume,
                    },
                    .pre_external_begin = {
                        .guest_rendering_serial =
                            boundary.pre_external_guest_rendering_serial,
                        .host_rendering_instance_id =
                            boundary.pre_external_host_rendering_instance,
                        .rendering_active =
                            boundary.pre_external_begin_rendering,
                        .regular_origin =
                            !boundary.pre_external_diagnostic_resume,
                        .diagnostic_origin =
                            boundary.pre_external_diagnostic_resume,
                    },
                },
                .boundary_after_guest_rendering_serial =
                    boundary.guest_rendering_serial_after,
                .boundary_after_host_rendering_instance_id =
                    boundary.host_rendering_instance_after,
                .live_target_guest_rendering_serial = scheduler.RenderingSerial(),
                .live_target_host_rendering_instance_id =
                    scheduler.HostRenderingInstanceId(),
                .raw_boundary_active = boundary.active,
                .raw_boundary_tracked = boundary.tracked,
                .boundary_after_rendering_active = boundary.after_rendering,
                .boundary_after_diagnostic_origin =
                    boundary.after_diagnostic_resume,
                .live_target_rendering_active = scheduler.IsRendering(),
                .live_target_diagnostic_origin =
                    scheduler.HostRenderingOrigin() ==
                    HostRenderingInstanceOrigin::DiagnosticResume,
            },
            depth_range,
            IsEffectiveDepthWrite(state, scheduler.GetDynamicState()), fragment_info);
    }
    diagnostic_depth_write_boundary = {};
    return scheduler.IsRendering();
}

void Rasterizer::MarkEncodedImageProducers(const RenderState& state,
                                           VideoCore::ImageColorScopeDrawDescriptor draw) {
    if (!VideoCore::IsPpSourceProducerTrackingEnabled()) {
        return;
    }
    const u64 rendering_serial = scheduler.RenderingSerial();
    bool writes_color_output{};
    bool input_alias{};
    VideoCore::Image* sampled_input_image{};
    VideoCore::ImageId sole_output_id{};
    u32 sole_output_attachment_index{};
    u32 output_count{};
    for (u32 index = 0; index < state.num_color_attachments; ++index) {
        const auto image_id = cb_descs[index].first;
        if (image_id) {
            writes_color_output = true;
            ++output_count;
            if (output_count == 1) {
                sole_output_id = image_id;
                sole_output_attachment_index = index;
            }
            input_alias |= diagnostic_sampled_images.size() == 1 &&
                           diagnostic_sampled_images.front().id == image_id;
        }
    }
    if (pp_terminal_scope_content) {
        for (u32 index = 0; index < state.num_color_attachments; ++index) {
            const auto image_id = cb_descs[index].first;
            if (!image_id) {
                continue;
            }
            auto& image = texture_cache.GetImage(image_id);
            static_cast<void>(pp_terminal_scope_content->ObserveEncodedUpstreamPostDraw(
                image_id, image, state, index, rendering_serial, draw));
        }
    }
    if (writes_color_output && diagnostic_sampled_images.size() == 1) {
        auto& input_image = texture_cache.GetImage(diagnostic_sampled_images.front().id);
        sampled_input_image = &input_image;
        draw.sampled_input_image = {
            .id = diagnostic_sampled_images.front().id,
            .uid = input_image.image_uid,
        };
        draw.sampled_input_view = MakeImageColorScopeTrackedView(
            draw.sampled_input_image, input_image, diagnostic_sampled_images.front().view,
            diagnostic_sampled_images.front().view_conflict);
        const auto input = input_image.ObserveDiagnosticSampledInputProducer();
        draw.sampled_input_producer = input.classification;
        draw.sampled_input_fresh = input.produced_since_last_observation;
        draw.sampled_input_alias = input_alias;
        draw.sampled_input_valid = true;
        if (input.classification == VideoCore::ImageProducerClass::ColorAttachment) {
            const auto input_scope = input_image.ObserveDiagnosticColorScope();
            draw.sampled_input_scope_draw_count = input_scope.draw_count;
            draw.sampled_input_scope_last_draw = input_scope.last_draw;
            draw.sampled_input_scope_indexed = input_scope.indexed;
            draw.sampled_input_scope_element_count = input_scope.element_count;
            draw.sampled_input_scope_instance_count = input_scope.instance_count;
            draw.sampled_input_scope_sampled_images = input_scope.sampled_images;
            draw.sampled_input_scope_storage_writes = input_scope.storage_writes;
            draw.sampled_input_scope_clear_at_begin = input_scope.clear_at_begin;
            draw.sampled_input_scope_valid = input_scope.valid;
            draw.sampled_input_scope_overflow = input_scope.overflow;
            draw.sampled_input_scope_input_producer = input_scope.sampled_input_producer;
            draw.sampled_input_scope_input_fresh = input_scope.sampled_input_fresh;
            draw.sampled_input_scope_input_alias = input_scope.sampled_input_alias;
            draw.sampled_input_scope_input_valid = input_scope.sampled_input_valid;
            draw.ancestry =
                VideoCore::BuildImageColorScopeAncestry(input, true, input_alias, &input_scope);
        } else {
            draw.ancestry =
                VideoCore::BuildImageColorScopeAncestry(input, true, input_alias, nullptr);
        }
    }
    if (pp_terminal_scope_content) {
        VideoCore::Image* output_image{};
        VideoCore::ImageColorScopePrivateLink output_link{};
        if (output_count == 1) {
            output_image = &texture_cache.GetImage(sole_output_id);
            output_link = {.id = sole_output_id, .uid = output_image->image_uid};
        }
        pp_terminal_scope_content->ObserveConsumer(draw, sampled_input_image, output_link,
                                                   output_image, sole_output_attachment_index,
                                                   output_count == 1, state, rendering_serial);
    }
    for (u32 index = 0; index < state.num_color_attachments; ++index) {
        const auto image_id = cb_descs[index].first;
        if (image_id) {
            auto& image = texture_cache.GetImage(image_id);
            image.BeginDiagnosticColorScope(rendering_serial,
                                            state.color_attachments[index].is_clear);
            image.MarkDiagnosticProducer(VideoCore::ImageProducerClass::ColorAttachment);
            auto output_draw = draw;
            output_draw.output_view =
                MakeImageColorScopeTrackedView({.id = image_id, .uid = image.image_uid}, image,
                                               cb_descs[index].second.view_info, false);
            image.MarkDiagnosticColorDraw(rendering_serial, output_draw);
            if (pp_terminal_scope_content) {
                pp_terminal_scope_content->ObserveDraw(image_id, image, state, index,
                                                       rendering_serial, output_draw);
            }
        }
    }
    MarkEncodedStorageImageProducers();
    if (const auto image_id = db_desc.first; image_id) {
        const auto& dynamic_state = scheduler.GetDynamicState();
        const auto stencil_ops_write = [](const StencilOps& ops) {
            return ops.fail_op != vk::StencilOp::eKeep || ops.pass_op != vk::StencilOp::eKeep ||
                   ops.depth_fail_op != vk::StencilOp::eKeep;
        };
        const bool rasterizes = !dynamic_state.rasterizer_discard_enable;
        const bool depth_write = IsEffectiveDepthWrite(state, dynamic_state);
        const bool stencil_write =
            rasterizes && state.depth_stencil_attachment.has_stencil &&
            dynamic_state.stencil_test_enabled &&
            ((dynamic_state.stencil_front_write_mask != 0 &&
              stencil_ops_write(dynamic_state.stencil_front_ops)) ||
             (dynamic_state.stencil_back_write_mask != 0 &&
              stencil_ops_write(dynamic_state.stencil_back_ops)));
        const auto overlaps = [](u32 first_base, u32 first_count, u32 second_base,
                                 u32 second_count) {
            return first_count != 0 && second_count != 0 &&
                   static_cast<u64>(first_base) < static_cast<u64>(second_base) + second_count &&
                   static_cast<u64>(second_base) < static_cast<u64>(first_base) + first_count;
        };
        auto& depth_image = texture_cache.GetImage(image_id);
        const auto sampled = std::ranges::find_if(
            diagnostic_sampled_images,
            [image_id](const auto& candidate) { return candidate.id == image_id; });
        const bool sampled_id_alias = sampled != diagnostic_sampled_images.end();
        const bool sampled_backing_alias =
            sampled_id_alias && !sampled->backing_conflict &&
            sampled->backing_generation == depth_image.EnsureDiagnosticBackingGeneration();
        u32 attachment_aspect{};
        attachment_aspect |= state.depth_stencil_attachment.has_depth
                                 ? static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT)
                                 : 0;
        attachment_aspect |= state.depth_stencil_attachment.has_stencil
                                 ? static_cast<u32>(VK_IMAGE_ASPECT_STENCIL_BIT)
                                 : 0;
        const bool sampled_aspect_alias = sampled_id_alias && !sampled->aspect_conflict &&
                                          (sampled->aspect_mask & attachment_aspect) != 0;
        const auto& db_range = db_desc.second.view_info.range;
        const bool sampled_range_alias =
            sampled_id_alias && !sampled->view_conflict &&
            overlaps(sampled->view.range.base.level, sampled->view.range.extent.levels,
                     db_range.base.level, db_range.extent.levels) &&
            overlaps(sampled->view.range.base.layer, sampled->view.range.extent.layers,
                     db_range.base.layer, db_range.extent.layers);
        depth_image.MarkDiagnosticDepthStencilProducer(
            draw, rendering_serial, scheduler.HostRenderingInstanceId(),
            scheduler.HostRenderingOrigin() ==
                HostRenderingInstanceOrigin::DiagnosticResume,
            {.id = image_id, .uid = depth_image.image_uid}, db_range, attachment_aspect,
            state.depth_stencil_attachment.depth_clear,
            state.depth_stencil_attachment.stencil_clear, depth_write, stencil_write,
            sampled_id_alias,
            sampled_backing_alias,
            sampled_aspect_alias, sampled_range_alias);
        if (depth_write && pp_terminal_scope_content &&
            pp_terminal_scope_content->CapturesUpstreamSampleTee()) {
            pp_terminal_scope_content->ObserveUpstreamSampleTeeProducerPreDepthPost(
                image_id, depth_image, state, db_range, depth_write);
        }
        if (depth_write && pp_terminal_scope_content &&
            pp_terminal_scope_content->CapturesUpstreamDepthPrePost()) {
            static_cast<void>(pp_terminal_scope_content->ObserveEncodedDepthUpstreamPostDraw(
                image_id, depth_image, state, rendering_serial, draw, db_range, depth_write));
        }
    }
}

void Rasterizer::MarkEncodedStorageImageProducers() {
    if (!VideoCore::IsPpSourceProducerTrackingEnabled()) {
        return;
    }
    for (const auto image_id : storage_written_images) {
        if (image_id) {
            texture_cache.GetImage(image_id).MarkDiagnosticProducer(
                VideoCore::ImageProducerClass::StorageImage);
        }
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           AttachmentFeedbackLoopRenderTargetAccess(), {});
            attachment_feedback_loop = true;
        } else {
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear = ShouldClearDepthAttachment(
            regs.depth_render_control.depth_clear_enable, regs.depth_control.depth_enable,
            regs.depth_control.depth_write_enable,
            texture_cache.IsMetaCleared(htile_address, slice));
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        constexpr auto depth_access = vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                      vk::AccessFlagBits2::eDepthStencilAttachmentRead;
        constexpr auto depth_stage = vk::PipelineStageFlagBits2::eAllGraphics |
                                     vk::PipelineStageFlagBits2::eComputeShader;
        if (pp_terminal_scope_content &&
            pp_terminal_scope_content->CapturesUpstreamDepthPrePost() &&
            VideoCore::IsPpSourceProducerTrackingEnabled()) {
            constexpr auto barrier_recognized_write = vk::AccessFlagBits2::eTransferWrite |
                                                      vk::AccessFlagBits2::eShaderWrite |
                                                      vk::AccessFlagBits2::eMemoryWrite;
            const VideoCore::ImageColorScopePrivateLink link{.id = image_id,
                                                              .uid = image.image_uid};
            const u64 backing_generation =
                image.backing ? image.EnsureDiagnosticBackingGeneration() : 0;
            const u64 host_rendering_instance = scheduler.HostRenderingInstanceId();
            const u64 guest_rendering_serial = scheduler.RenderingSerial();
            const auto writer = image.PeekDiagnosticDepthStencilProducer();
            const bool uniform_state =
                image.backing && image.backing->subresource_states.empty();
            const auto old_state = image.backing ? image.backing->state : VideoCore::Image::State{};
            diagnostic_depth_write_boundary = {
                .link = link,
                .range = desc.view_info.range,
                .backing_generation = backing_generation,
                .mutation_epoch = image.DiagnosticMutationEpoch(),
                .host_rendering_instance_before = host_rendering_instance,
                .guest_rendering_serial_before = guest_rendering_serial,
                .active = true,
                .tracked = true,
                .uniform_state = uniform_state,
                .encoded_depth_writer = writer.valid && writer.depth_write &&
                                        (writer.attachment_aspect_mask &
                                         static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT)) != 0,
                .writer_image_identity_exact = writer.link == link,
                .writer_backing_exact = writer.backing_generation != 0 &&
                                        writer.backing_generation == backing_generation,
                .writer_range_exact = writer.range == desc.view_info.range,
                .writer_mutation_exact = writer.mutation_epoch != 0 &&
                                         writer.mutation_epoch == image.DiagnosticMutationEpoch(),
                .writer_guest_serial_exact = writer.rendering_serial != 0 &&
                                             writer.rendering_serial == guest_rendering_serial,
                .writer_host_instance_exact = writer.host_rendering_instance_id != 0 &&
                                              writer.host_rendering_instance_id ==
                                                  host_rendering_instance,
                .pre_rendering = scheduler.IsRendering(),
                .state_exact = uniform_state && old_state.layout == new_layout &&
                               old_state.access_mask == depth_access &&
                               old_state.pl_stage == depth_stage,
                .source_depth_stencil_attachment_write =
                    uniform_state && static_cast<bool>(
                                         old_state.access_mask &
                                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite),
                .source_barrier_recognized_write =
                    uniform_state &&
                    static_cast<bool>(old_state.access_mask & barrier_recognized_write),
                .before_diagnostic_resume =
                    scheduler.HostRenderingOrigin() ==
                    HostRenderingInstanceOrigin::DiagnosticResume,
            };
        }
        image.Transit(new_layout, depth_access, desc.view_info.range);
        if (diagnostic_depth_write_boundary.active) {
            diagnostic_depth_write_boundary.post_transit_rendering =
                scheduler.IsRendering() && scheduler.HostRenderingInstanceId() ==
                                               diagnostic_depth_write_boundary
                                                   .host_rendering_instance_before;
        }

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = {write_image.info.size.width, write_image.info.size.height, 1},
    };
    scheduler.CommandBuffer().copyImage(read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        write_image.GetImage(),
                                        vk::ImageLayout::eTransferDstOptimal, region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size);
    return true;
}

void Rasterizer::ProcessDownloadImages() {
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(pipeline, is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        const auto& sc = regs.stencil_control;
        const auto front_reference = PlanStencilReference(
            sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front,
            regs.depth_control.stencil_ref_func, front);
        const auto back_reference = regs.depth_control.backface_enable
                                        ? PlanStencilReference(
                                              sc.stencil_fail_back, sc.stencil_zpass_back,
                                              sc.stencil_zfail_back,
                                              regs.depth_control.stencil_bf_func, back)
                                        : front_reference;
        if (front_reference.compare_conflict || back_reference.compare_conflict) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_reference.reference, back_reference.reference);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const GraphicsPipeline* pipeline,
                                      const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    const auto primitive_type = pipeline->GetGraphicsKey().prim_type;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
