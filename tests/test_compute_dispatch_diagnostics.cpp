// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/compute_dispatch_diagnostics.h"

namespace Vulkan {
namespace {

TEST(ComputeDispatchDiagnostics, CopiesOnlyCompleteDwords) {
    const std::array<u8, 7> bytes{0x78, 0x56, 0x34, 0x12, 0xAA, 0xBB, 0xCC};

    const auto preview = CopyDwordPreview(bytes.data(), bytes.size());

    ASSERT_EQ(preview.size, 1u);
    EXPECT_EQ(preview.words[0], 0x12345678u);
}

TEST(ComputeDispatchDiagnostics, SynchronizesSmallGpuModifiedReadBuffers) {
    EXPECT_TRUE(Vulkan::ShouldSynchronizeDiagnosticPreview(false, false, true, 2704));
    EXPECT_FALSE(Vulkan::ShouldSynchronizeDiagnosticPreview(true, false, true, 2704));
    EXPECT_FALSE(Vulkan::ShouldSynchronizeDiagnosticPreview(false, true, true, 2704));
    EXPECT_FALSE(Vulkan::ShouldSynchronizeDiagnosticPreview(false, false, false, 2704));
    EXPECT_FALSE(Vulkan::ShouldSynchronizeDiagnosticPreview(false, false, true, 4097));
}

TEST(ComputeDispatchDiagnostics, CapsPreviewWithoutReadingPastLimit) {
    std::array<u32, ComputeDiagnosticPreviewDwords + 2> words{};
    for (u32 i = 0; i < words.size(); ++i) {
        words[i] = i + 1;
    }

    const auto preview = CopyDwordPreview(words.data(), sizeof(words));

    EXPECT_EQ(preview.size, ComputeDiagnosticPreviewDwords);
    EXPECT_EQ(preview.words.front(), 1u);
    EXPECT_EQ(preview.words.back(), ComputeDiagnosticPreviewDwords);
}

TEST(ComputeDispatchDiagnostics, EmptyInputProducesEmptyPreview) {
    const auto preview = CopyDwordPreview(nullptr, 64);

    EXPECT_EQ(preview.size, 0u);
}

TEST(ComputeDispatchDiagnostics, SelectsOnlyLargeWorkgroupsForVerbosePreview) {
    EXPECT_TRUE(IsLargeComputeWorkgroup(32, 32, 1));
    EXPECT_TRUE(IsLargeComputeWorkgroup(16, 16, 4));
    EXPECT_FALSE(IsLargeComputeWorkgroup(16, 4, 1));
    EXPECT_FALSE(IsLargeComputeWorkgroup(256, 1, 1));
}

} // namespace
} // namespace Vulkan
