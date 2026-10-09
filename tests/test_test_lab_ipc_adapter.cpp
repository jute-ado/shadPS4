// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/ipc/test_lab_input.h"

using Libraries::Pad::OrbisPadButtonDataOffset;

int main() {
    static_assert(Core::Ipc::ParseTestLabButton("cross") == OrbisPadButtonDataOffset::Cross);
    static_assert(Core::Ipc::ParseTestLabButton("circle") == OrbisPadButtonDataOffset::Circle);
    static_assert(Core::Ipc::ParseTestLabButton("square") == OrbisPadButtonDataOffset::Square);
    static_assert(Core::Ipc::ParseTestLabButton("triangle") == OrbisPadButtonDataOffset::Triangle);
    static_assert(Core::Ipc::ParseTestLabButton("options") == OrbisPadButtonDataOffset::Options);
    static_assert(Core::Ipc::ParseTestLabButton("dpad_up") == OrbisPadButtonDataOffset::Up);
    static_assert(Core::Ipc::ParseTestLabButton("dpad_right") == OrbisPadButtonDataOffset::Right);
    static_assert(Core::Ipc::ParseTestLabButton("dpad_down") == OrbisPadButtonDataOffset::Down);
    static_assert(Core::Ipc::ParseTestLabButton("dpad_left") == OrbisPadButtonDataOffset::Left);
    static_assert(Core::Ipc::ParseTestLabButton("l1") == OrbisPadButtonDataOffset::L1);
    static_assert(Core::Ipc::ParseTestLabButton("l2") == OrbisPadButtonDataOffset::L2);
    static_assert(Core::Ipc::ParseTestLabButton("r1") == OrbisPadButtonDataOffset::R1);
    static_assert(Core::Ipc::ParseTestLabButton("r2") == OrbisPadButtonDataOffset::R2);
    static_assert(Core::Ipc::ParseTestLabButton("l3") == OrbisPadButtonDataOffset::L3);
    static_assert(Core::Ipc::ParseTestLabButton("r3") == OrbisPadButtonDataOffset::R3);
    static_assert(Core::Ipc::ParseTestLabButton("touchpad") == OrbisPadButtonDataOffset::TouchPad);
    static_assert(!Core::Ipc::ParseTestLabButton("Cross"));
    static_assert(!Core::Ipc::ParseTestLabButton(""));

    static_assert(Core::Ipc::ParseTestLabAxis("left_x") == Input::Axis::LeftX);
    static_assert(Core::Ipc::ParseTestLabAxis("left_y") == Input::Axis::LeftY);
    static_assert(Core::Ipc::ParseTestLabAxis("right_x") == Input::Axis::RightX);
    static_assert(Core::Ipc::ParseTestLabAxis("right_y") == Input::Axis::RightY);
    static_assert(Core::Ipc::ParseTestLabAxis("l2") == Input::Axis::TriggerLeft);
    static_assert(Core::Ipc::ParseTestLabAxis("r2") == Input::Axis::TriggerRight);
    static_assert(!Core::Ipc::ParseTestLabAxis("left_z"));
    static_assert(!Core::Ipc::ParseTestLabAxis(""));
    return 0;
}
