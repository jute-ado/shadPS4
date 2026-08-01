// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string_view>

#include <gtest/gtest.h>

#include "core/ipc/ipc_capabilities.h"

TEST(IpcCapabilities, OmitsRenderDocWhenRuntimeIsUnavailable) {
    const auto capabilities = Core::Ipc::IpcCapabilities(false);

    EXPECT_EQ(std::ranges::find(capabilities, "ENABLE_RENDERDOC_CAPTURE"), capabilities.end());
}

TEST(IpcCapabilities, AdvertisesRenderDocWhenRuntimeIsLoaded) {
    const auto capabilities = Core::Ipc::IpcCapabilities(true);

    EXPECT_NE(std::ranges::find(capabilities, "ENABLE_RENDERDOC_CAPTURE"), capabilities.end());
}
