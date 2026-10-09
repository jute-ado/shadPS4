// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <sstream>

#include <gtest/gtest.h>

#include "input/controller_trace.h"

using Libraries::Pad::OrbisPadButtonDataOffset;

TEST(ControllerTrace, WritesCanonicalChangedFullSnapshots) {
    std::ostringstream output;
    Input::ControllerTraceWriter writer{output};
    Input::State neutral;
    writer.Record(neutral, 0);
    writer.Record(neutral, 10);

    Input::State changed;
    changed.buttonsState = OrbisPadButtonDataOffset::Right |
                           OrbisPadButtonDataOffset::Cross |
                           OrbisPadButtonDataOffset::L2;
    changed.axes[std::to_underlying(Input::Axis::LeftX)] = 0;
    changed.axes[std::to_underlying(Input::Axis::RightX)] = 255;
    changed.axes[std::to_underlying(Input::Axis::RightY)] = 127;
    changed.axes[std::to_underlying(Input::Axis::TriggerLeft)] = 255;
    changed.axes[std::to_underlying(Input::Axis::TriggerRight)] = 128;
    changed.touchpad[0] = {.ID = 0, .state = true, .x = 1919, .y = 471};
    writer.Record(changed, 1000);

    EXPECT_EQ(
        output.str(),
        "{\"kind\":\"header\",\"protocolVersion\":1,\"profile\":\"dualshock4\","
        "\"clock\":\"elapsed_milliseconds\"}\n"
        "{\"kind\":\"sample\",\"protocolVersion\":1,\"offset\":0,\"state\":{"
        "\"buttons\":[],\"leftX\":0,\"leftY\":0,\"rightX\":0,\"rightY\":0,"
        "\"leftTrigger\":0,\"rightTrigger\":0,\"touches\":[]}}\n"
        "{\"kind\":\"sample\",\"protocolVersion\":1,\"offset\":1000,\"state\":{"
        "\"buttons\":[\"dpad_right\",\"cross\",\"l2\"],"
        "\"leftX\":-32768,\"leftY\":0,\"rightX\":32767,\"rightY\":-256,"
        "\"leftTrigger\":65535,\"rightTrigger\":32896,"
        "\"touches\":[{\"id\":0,\"x\":65535,\"y\":32802}]}}\n");
}

TEST(ControllerTrace, MakesSubMillisecondChangesStrictlyOrdered) {
    std::ostringstream output;
    Input::ControllerTraceWriter writer{output};
    Input::State state;
    writer.Record(state, 0);
    state.buttonsState = OrbisPadButtonDataOffset::Cross;
    writer.Record(state, 0);
    state.buttonsState = OrbisPadButtonDataOffset::None;
    writer.Record(state, 0);

    const auto text = output.str();
    EXPECT_NE(text.find("\"offset\":0"), std::string::npos);
    EXPECT_NE(text.find("\"offset\":1"), std::string::npos);
    EXPECT_NE(text.find("\"offset\":2"), std::string::npos);
}
