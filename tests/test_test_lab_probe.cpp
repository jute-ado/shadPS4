// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "core/test_lab_probe.h"

TEST(TestLabProbe, EmitsCanonicalProtocolOneCapabilityDocument) {
    const auto document = nlohmann::json::parse(Core::TestLabProbeJson());

    EXPECT_EQ(document.size(), 4);
    EXPECT_EQ(document.at("protocolVersion"), 1);
    EXPECT_EQ(document.at("emulator"), "shadps4");
    EXPECT_EQ(document.at("adapterVersion"), "1.0.0");
    EXPECT_EQ(document.at("capabilities"),
              (std::vector<std::string>{
                  "console_profile_ps4",
                  "console_profile_ps4_pro",
                  "async_graphics_pipeline_compilation",
                  "controller_recording",
                  "controller_replay",
                  "emulator_control",
                  "extra_direct_memory",
                  "game_frame_screenshot",
                  "overlay_controller_replay",
                  "output_resolution",
                  "presented_frame_screenshot",
                  "presented_frame_timing_trace",
                  "render_resolution",
                  "renderdoc_capture",
                  "stereo_pcm16_audio_capture",
                  "touch_input",
                  "vulkan_validation",
              }));
}
