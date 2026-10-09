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

/// Returns true after a capture has spanned a complete non-reused presented-frame interval.
[[nodiscard]] bool ConsumeCaptureForPresentedFrame(bool is_reusing_frame);

/// Ends a capture begun at trigger time after the next complete presented frame.
void EndPresentedFrameCapture();

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
