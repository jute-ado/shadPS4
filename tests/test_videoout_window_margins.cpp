// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/error_codes.h"
#include "core/libraries/videoout/videoout_error.h"
#include "core/libraries/videoout/window_margins.h"

using namespace Libraries::VideoOut;

TEST(VideoOutWindowMargins, RejectsAnInvalidOrClosedPort) {
    EXPECT_EQ(ApplyWindowModeMargins(nullptr, 108, 108), ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE);
}

TEST(VideoOutWindowMargins, StoresMarginsForAnOpenPort) {
    WindowMargins margins{};

    EXPECT_EQ(ApplyWindowModeMargins(&margins, 108, 108), ORBIS_OK);
    EXPECT_EQ(margins.top, 108);
    EXPECT_EQ(margins.bottom, 108);
}

TEST(VideoOutWindowMargins, PreservesTheLatestRequestedValues) {
    WindowMargins margins{.top = 108, .bottom = 108};

    EXPECT_EQ(ApplyWindowModeMargins(&margins, -12, 24), ORBIS_OK);
    EXPECT_EQ(margins.top, -12);
    EXPECT_EQ(margins.bottom, 24);
}
