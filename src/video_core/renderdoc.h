// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include "video_core/renderdoc_submission_capture.h"
#include "video_core/screenshot_request_queue.h"

namespace VideoCore {

/// Loads renderdoc dynamic library module.
void LoadRenderDoc(bool allow_offline_loading = true);

/// Schedules a capture of the next submission containing recorded guest draw/dispatch work.
void TriggerCapture();

/// Begins a requested capture immediately before the owner submits recorded guest work.
[[nodiscard]] GuestSubmissionCapture BeginGuestWorkCapture(void* vulkan_instance,
                                                           bool has_guest_work);

/// Reports the actual queue result before any fatal check; failed submissions are discarded.
void CompleteGuestWorkCapture(GuestSubmissionCapture& capture, bool submission_succeeded,
                              bool& pending_guest_work);

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
