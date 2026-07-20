// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "core/emulator_settings.h"

TEST(GPUCompatibilitySettings, CompressedTextureMipCompatibilityDefaultsDisabled) {
    EmulatorSettingsImpl settings;

    EXPECT_FALSE(settings.IsCompressedTextureMipsDisabled());
}

TEST(GPUCompatibilitySettings, CompressedTextureMipCompatibilityRoundTrips) {
    EmulatorSettingsImpl settings;

    settings.SetCompressedTextureMipsDisabled(true);

    EXPECT_TRUE(settings.IsCompressedTextureMipsDisabled());
}

TEST(GPUCompatibilitySettings, CompressedTextureMipCompatibilitySerializesWithGPUSettings) {
    GPUSettings original;
    original.disable_compressed_texture_mips.set(true);

    const nlohmann::json encoded = original;
    ASSERT_TRUE(encoded.contains("disable_compressed_texture_mips"));
    EXPECT_TRUE(encoded["disable_compressed_texture_mips"].get<bool>());

    const auto decoded = encoded.get<GPUSettings>();
    EXPECT_TRUE(decoded.disable_compressed_texture_mips.value);
}

TEST(GPUCompatibilitySettings, CompressedTextureMipCompatibilityIsGameOverrideable) {
    EmulatorSettingsImpl settings;
    const auto fields = settings.GetGPUOverrideableFields();

    EXPECT_TRUE(std::ranges::any_of(fields, [](const OverrideItem& field) {
        return std::string(field.key) == "disable_compressed_texture_mips";
    }));
}
