// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_diagnostic_checkpoint.h"

namespace Vulkan {

class Instance;

// The graphics queue is externally synchronized. Callers must hold Scheduler::submit_mutex.
void LogDeviceLossDiagnosticsWithSubmitLockHeld(const Instance& instance,
                                                DeviceLossSource source);

} // namespace Vulkan
