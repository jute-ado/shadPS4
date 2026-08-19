// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string_view>

#include <gtest/gtest.h>

#include "core/emulator_settings.h"
#include "core/guest_display_resolution.h"

TEST(GuestDisplayResolution, RecognizesStandardPresetsAndPreservesCustomDimensions) {
    EXPECT_EQ(Core::FindGuestDisplayResolutionPreset(1280, 720), 0u);
    EXPECT_EQ(Core::FindGuestDisplayResolutionPreset(1920, 1080), 1u);
    EXPECT_EQ(Core::FindGuestDisplayResolutionPreset(2560, 1440), 2u);
    EXPECT_EQ(Core::FindGuestDisplayResolutionPreset(3840, 2160), 3u);
    EXPECT_EQ(Core::FindGuestDisplayResolutionPreset(3440, 1440), 4u);

    EXPECT_EQ(Core::ResolveGuestDisplayResolutionPreset(3, {1280, 720}),
              (Core::GuestDisplayResolution{3840, 2160}));
    EXPECT_EQ(Core::ResolveGuestDisplayResolutionPreset(4, {3440, 1440}),
              (Core::GuestDisplayResolution{3440, 1440}));
}

TEST(GuestDisplayResolution, FormatsTheExactGuestFacingDimensions) {
    EXPECT_EQ(Core::FormatGuestDisplayResolution({3840, 2160}),
              "Guest display resolution: 3840 x 2160");
}

TEST(GuestDisplayResolution, InternalDimensionsArePerGameOverrideable) {
    GPUSettings settings;
    const auto fields = settings.GetOverrideableFields();
    const auto has = [&](const char* key) {
        return std::ranges::any_of(
            fields, [key](const OverrideItem& item) { return std::string_view{item.key} == key; });
    };

    EXPECT_TRUE(has("internal_screen_width"));
    EXPECT_TRUE(has("internal_screen_height"));
    EXPECT_TRUE(has("internal_resolution_scale"));
    EXPECT_EQ(settings.internal_resolution_scale.get(), 100u);
}
