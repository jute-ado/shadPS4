// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/test_lab_probe.h"

namespace Core {

std::string_view TestLabProbeJson() {
    return R"({
  "protocolVersion": 1,
  "emulator": "shadps4",
  "adapterVersion": "1.0.0",
  "capabilities": [
    "console_profile_ps4",
    "console_profile_ps4_pro",
    "controller_recording",
    "controller_replay",
    "emulator_control",
    "game_frame_screenshot",
    "output_resolution",
    "presented_frame_screenshot",
    "presented_frame_timing_trace",
    "render_resolution",
    "renderdoc_capture",
    "touch_input"
  ]
})";
}

} // namespace Core
