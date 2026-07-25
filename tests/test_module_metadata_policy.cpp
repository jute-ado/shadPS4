// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
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
