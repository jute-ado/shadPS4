// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/debug.h"
#include "video_core/renderer_vulkan/vk_device_loss_diagnostics.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_bind_history.h"

namespace Vulkan {

void LogDeviceLossDiagnosticsWithSubmitLockHeld(const Instance& instance,
                                                DeviceLossSource source) {
    if (ShouldReportDiagnosticCheckpoints(source, instance.SupportsDiagnosticCheckpoints())) {
        for (const auto& checkpoint : instance.GetGraphicsQueue().getCheckpointDataNV()) {
            LOG_CRITICAL(Render_Vulkan, "GPU diagnostic checkpoint: stage={}, pipeline={:#018x}",
                         vk::to_string(checkpoint.stage),
                         DecodeDiagnosticCheckpoint(checkpoint.pCheckpointMarker));
        }
    }

    for (const auto& bind : GetPipelineBindHistory().RecentUnique()) {
        LOG_CRITICAL(
            Render_Vulkan,
            "Recent {} pipeline bind: pipeline={:#018x}, shaders=[{:#018x}, {:#018x}, "
            "{:#018x}, {:#018x}, {:#018x}, {:#018x}], command={}, args=[{:#x}, {:#x}, "
            "{:#x}]",
            bind.type == PipelineBindType::Graphics ? "graphics" : "compute", bind.pipeline_hash,
            bind.shader_hashes[0], bind.shader_hashes[1], bind.shader_hashes[2],
            bind.shader_hashes[3], bind.shader_hashes[4], bind.shader_hashes[5],
            PipelineCommandName(bind.command.type), bind.command.arguments[0],
            bind.command.arguments[1], bind.command.arguments[2]);
        for (size_t index = 0; index < bind.buffer_count; ++index) {
            const auto& buffer = bind.buffers[index];
            LOG_CRITICAL(Render_Vulkan,
                         "  buffer[{}]: addr={:#x}, requested={:#x}, bound={:#x}, "
                         "stride={:#x}, records={:#x}, written={}, formatted={}",
                         index, buffer.base_address, buffer.requested_size, buffer.bound_size,
                         buffer.stride, buffer.num_records, buffer.is_written, buffer.is_formatted);
            for (size_t sample_index = 0; sample_index < buffer.sample_count; ++sample_index) {
                LOG_CRITICAL(Render_Vulkan, "    dword[{}]={:#x}", sample_index,
                             buffer.sample_dwords[sample_index]);
            }
        }
        for (size_t index = 0; index < bind.image_count; ++index) {
            const auto& image = bind.images[index];
            LOG_CRITICAL(Render_Vulkan,
                         "  image[{}]: addr={:#x}, extent={}x{}x{}, pitch={}, "
                         "format={:#x}, type={:#x}, written={}",
                         index, image.base_address, image.width, image.height, image.depth,
                         image.pitch, image.data_format, image.type, image.is_written);
        }
    }
}

} // namespace Vulkan
