// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdint>
#include <string_view>

#include <gtest/gtest.h>

#include "core/loader/module_metadata_policy.h"

using namespace Core::Loader;

TEST(ModuleMetadataPolicy, AppendsSharedObjectSuffixAndTerminatesName) {
    std::array<char, 16> destination{};

    WriteModuleFilename(destination, "libFoo");

    EXPECT_EQ(std::string_view(destination.data()), "libFoo.sprx");
    EXPECT_EQ(destination[11], '\0');
}

TEST(ModuleMetadataPolicy, TruncatesStemButPreservesSuffixAndTerminator) {
    std::array<char, 9> destination{};

    WriteModuleFilename(destination, "abcdefgh");

    EXPECT_EQ(std::string_view(destination.data()), "abc.sprx");
    EXPECT_EQ(destination.back(), '\0');
}

TEST(ModuleMetadataPolicy, ClearsUnusedBytesFromPreviousName) {
    std::array<char, 12> destination;
    destination.fill('x');

    WriteModuleFilename(destination, "a");

    EXPECT_EQ(std::string_view(destination.data()), "a.sprx");
    EXPECT_EQ(destination[6], '\0');
    EXPECT_EQ(destination.back(), '\0');
}

TEST(ModuleMetadataPolicy, CopiesAbiFingerprintFromDynamicDataOffset) {
    std::array<std::uint8_t, 32> dynamic_data{};
    for (std::uint8_t index = 0; index < 24; ++index) {
        dynamic_data[index + 4] = static_cast<std::uint8_t>(index + 1);
    }
    std::array<std::uint8_t, 20> fingerprint{};

    EXPECT_TRUE(CopyModuleFingerprint(fingerprint, dynamic_data, 4));
    for (std::uint8_t index = 0; index < fingerprint.size(); ++index) {
        EXPECT_EQ(fingerprint[index], index + 1);
    }
}

TEST(ModuleMetadataPolicy, RejectsTruncatedDynamicFingerprint) {
    std::array<std::uint8_t, 24> dynamic_data{};
    std::array<std::uint8_t, 20> fingerprint;
    const std::array<std::uint8_t, 20> empty_fingerprint{};
    fingerprint.fill(0xff);

    EXPECT_FALSE(CopyModuleFingerprint(fingerprint, dynamic_data, 1));
    EXPECT_EQ(fingerprint, empty_fingerprint);
}
