// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include "video_core/screenshot_request_queue.h"

namespace VideoCore {

/// Loads renderdoc dynamic library module.
void LoadRenderDoc(bool allow_offline_loading = true);

/// Schedules a capture for the next frame presented by the emulator.
void TriggerCapture();

/// Arms a pending capture after shader and image preparation has completed. Transfer-only command
/// buffers must not consume the trigger before useful graphics work is ready.
[[nodiscard]] bool ArmGuestFrameCaptureForPreparedGuestWork();

/// Begins a scheduled capture immediately before presenting a frame.
// Starts capture before a fresh Vulkan command buffer records the next guest GPU work.
[[nodiscard]] bool BeginGuestFrameCapture(void* vulkan_instance);

/// Returns true while a scheduled guest-frame capture is active.
[[nodiscard]] bool IsGuestFrameCaptureActive();

// Counts a submitted guest command buffer and reports when the bounded capture batch is complete.
[[nodiscard]] bool ShouldEndGuestFrameCapture();

/// Ends a capture begun after the previous present, once the next frame is presented.
void EndGuestFrameCapture(void* vulkan_instance);

/// Sets output directory for captures
void SetOutputDir(const std::filesystem::path& path, const std::string& prefix);

/// Returns true when RenderDoc API was loaded and is usable.
bool IsRenderDocLoaded();

/// Queues an in-emulator screenshot request to be consumed by the presenter.
void RequestScreenshot(ScreenshotRequest request,
                       ScreenshotRequestOrigin origin = ScreenshotRequestOrigin::User);

/// Atomically consumes pending "game only" screenshot requests.
ScreenshotRequestBatch ConsumeGameOnlyScreenshotRequests();

/// Atomically consumes pending "with overlays" screenshot requests.
ScreenshotRequestBatch ConsumeWithOverlaysScreenshotRequests();

} // namespace VideoCore
