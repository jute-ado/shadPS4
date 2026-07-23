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
                  "controller_recording",
                  "controller_replay",
                  "emulator_control",
                  "game_frame_screenshot",
                  "presented_frame_screenshot",
                  "renderdoc_capture",
                  "touch_input",
              }));
}
