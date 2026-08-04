// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/startup_resolution_log.h"

TEST(StartupResolutionLog, ReportsGuestFacingInternalDimensions) {
    EXPECT_EQ(Core::FormatInternalScreenResolution(3840, 2160),
              "GPU internalScreenWidth: 3840 internalScreenHeight: 2160");
}
