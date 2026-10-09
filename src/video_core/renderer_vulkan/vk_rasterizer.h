// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/color_attachment_provenance_receipt.h"
#include "video_core/emitted_color_command_ledger.h"
#include "video_core/emitted_color_vertex_receipt.h"
#include "video_core/index_binding_provenance.h"
#include "video_core/indirect_draw_register_provenance.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/descriptor_image_finalization_receipt.h"
#include "video_core/renderer_vulkan/descriptor_image_layout.h"
#include "video_core/renderer_vulkan/draw_drop_diagnostics.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class Scheduler;
class RenderState;
class GraphicsPipeline;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address,
                      VideoCore::IndirectDrawRegisterLocations register_locations = {});

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size);
    bool ReadMemory(VAddr addr, u64 size);
    void ProcessDownloadImages();
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size,
                   VideoCore::GpuMappingKind kind = VideoCore::GpuMappingKind::HostMapped);
    void UnmapMemory(VAddr addr, u64 size,
                     VideoCore::GpuMappingKind kind = VideoCore::GpuMappingKind::HostMapped);

    void CpSync();
    u64 Flush();
    void FlushWithGpuCompletion(Common::UniqueFunction<void, u64>&& on_submitted,
                                Common::UniqueFunction<void>&& wake_owner);
    void Finish();
    void OnSubmit();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

private:
    void RecordIndexBindingProvenance(
        const GraphicsPipeline& pipeline, bool indirect, u32 index_offset,
        VideoCore::IndirectDrawRegisterLocations register_locations = {}) noexcept;
    void RecordEmittedColorCommand(const GraphicsPipeline& pipeline, const RenderState& state,
                                   VideoCore::EmittedColorCommand command) noexcept;
    [[nodiscard]] bool PrepareRenderState(const GraphicsPipeline* pipeline);
    bool BeginRendering(const GraphicsPipeline* pipeline, RenderState& state);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    [[nodiscard]] bool FinalizeUtilityBufferBindings();
    [[nodiscard]] bool FinalizeBufferBindings(VideoCore::BufferCache::BufferBindingBatch& batch);
    void PrepareAddr64PointerResidency(const Shader::Info& stage);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);
    // Buffer preparation can retire and replace texture-cache images. Reconcile pending image
    // bindings at the last non-waiting boundary before descriptor publication.
    [[nodiscard]] bool RebindPendingImageBindings();
    void ObserveDescriptorImageDrawDrop(const DescriptorImageCommandContext& context);
    void ObserveDrawDrop(DrawDropReason reason,
                         const DescriptorImageCommandContext& context) noexcept;
    [[nodiscard]] bool FinalizeImageDescriptors(const DescriptorImageCommandContext& context);

    struct ColorAttachmentProvenanceSample {
        VideoCore::ImageId image_id{};
        u64 image_uid{};
        u32 attachment{};
        vk::Format base_format{vk::Format::eUndefined};
        vk::Format view_format{vk::Format::eUndefined};
        VideoCore::ColorAttachmentProvenanceSamplePlan plan{};
    };
    struct ColorAttachmentProvenanceCommand {
        VideoCore::ColorAttachmentProvenanceCommandToken token{};
        u64 vertex_hash{};
        u64 fragment_hash{};
        u64 pipeline_key_hash{};
        boost::container::small_vector<ColorAttachmentProvenanceSample, 32> samples;
    };
    [[nodiscard]] std::optional<ColorAttachmentProvenanceCommand> PrepareColorAttachmentProvenance(
        const GraphicsPipeline& pipeline, const RenderState& state);
    void CompleteColorAttachmentProvenance(ColorAttachmentProvenanceCommand command,
                                           const RenderState& state);
    void RecordColorAttachmentProvenanceCopies(const ColorAttachmentProvenanceCommand& command,
                                               bool post);

    struct BoundImageToken {
        VideoCore::ImageId image_id;
        u64 image_uid;
    };
    void TrackBoundImage(VideoCore::ImageId image_id);

    void ResetBindings() {
        for (const auto& bound : bound_images) {
            if (auto* image = texture_cache.TryGetImage(bound.image_id, bound.image_uid)) {
                image->binding = {};
            }
        }
        bound_images.clear();
    }

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    PipelineCache pipeline_cache;
    VideoCore::EmittedColorCommandLedger color_command_ledger{{}};
    VideoCore::EmittedColorVertexReceipt emitted_color_vertex_receipt{};
    VideoCore::IndexBindingProvenanceReceipt index_binding_provenance{};
    DescriptorImageFinalizationReceipt descriptor_image_finalization_receipt{};
    DrawDropDiagnostics draw_drop_diagnostics{};
    std::chrono::steady_clock::time_point descriptor_finalization_origin{};
    std::chrono::steady_clock::time_point draw_drop_diagnostics_origin{};
    std::chrono::steady_clock::time_point color_ledger_origin{};
    VideoCore::ColorAttachmentProvenancePolicy color_attachment_provenance_policy{};
    VideoCore::ColorAttachmentProvenanceReceipt color_attachment_provenance_receipt{
        VideoCore::ColorAttachmentProvenancePolicy{}};
    VideoCore::ColorAttachmentProvenanceSlotAllocator color_attachment_provenance_slots{0};
    std::shared_ptr<VideoCore::Buffer> color_attachment_provenance_buffer{};
    std::chrono::steady_clock::time_point color_attachment_provenance_origin{};

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    boost::container::small_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::small_vector<DescriptorImageLayoutBinding, Shader::NUM_IMAGES>
        descriptor_image_layout_bindings;
    boost::container::small_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::small_vector<BoundImageToken, Shader::NUM_IMAGES> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;

    using BufferBindingInfo = std::tuple<VideoCore::BufferId, AmdGpu::Buffer, u64>;
    boost::container::static_vector<BufferBindingInfo, Shader::NUM_BUFFERS> buffer_bindings;
    struct PendingBufferBinding {
        VideoCore::BufferCache::PreparedBufferBinding prepared;
        u32 buffer_info_index;
        u32 push_binding;
        u32 alignment;
        u32 size;
        vk::AccessFlags2 access;
    };
    boost::container::small_vector<PendingBufferBinding, Shader::NUM_BUFFERS>
        pending_buffer_bindings;
    struct PendingUtilityBufferBinding {
        u32 buffer_info_index{};
        u32 alignment{};
        std::vector<u8> bytes;
    };
    boost::container::small_vector<PendingUtilityBufferBinding, Shader::NUM_BUFFERS>
        pending_utility_buffer_bindings;
    bool utility_buffer_bindings_failed{};
    // A malformed guest descriptor must reject the resource binding before ClampRangeSize can
    // assert on an address outside the VMA map.  Keep this separate from utility-buffer failures
    // so the caller can retain the existing special-buffer semantics.
    bool resource_bindings_failed{};
    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::small_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
    struct PendingImageBinding {
        VideoCore::ImageId image_id;
        u64 image_uid;
        VideoCore::ImageViewInfo view_info;
        VideoCore::TextureCache::ImageDesc desc;
        bool uses_integer_coordinates{};
        bool queries_dimensions{};
    };
    boost::container::small_vector<PendingImageBinding, Shader::NUM_IMAGES> pending_image_bindings;
    bool fault_process_pending{};
    bool attachment_feedback_loop{};
    u32 prepared_host_scale{1};
    u32 current_host_scale{1};
};

} // namespace Vulkan
