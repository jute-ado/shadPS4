// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "input/controller_touch.h"

TEST(ControllerTouch, ValidatesNativeCoordinateBounds) {
    EXPECT_TRUE(Input::IsValidControllerTouch(0, 0, 0));
    EXPECT_TRUE(Input::IsValidControllerTouch(1, 1919, 941));
    EXPECT_FALSE(Input::IsValidControllerTouch(2, 0, 0));
    EXPECT_FALSE(Input::IsValidControllerTouch(0, 1920, 0));
    EXPECT_FALSE(Input::IsValidControllerTouch(0, 0, 942));
}

TEST(ControllerTouch, PackedEventRoundTrips) {
    const auto packed = Input::PackControllerTouch(1919, 941, true);
    const auto event = Input::UnpackControllerTouch(1, packed);

    EXPECT_EQ(event.finger, 1);
    EXPECT_TRUE(event.down);
    EXPECT_EQ(event.x, 1919);
    EXPECT_EQ(event.y, 941);
}
