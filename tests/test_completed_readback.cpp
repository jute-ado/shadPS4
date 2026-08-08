// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/completed_readback.h"

namespace VideoCore {
namespace {

TEST(CompletedReadback, InvalidatesNonCoherentMemoryBeforeConsumingBytes) {
    std::vector<std::string_view> operations;

    const auto result = ConsumeCompletedReadback(
        false,
        [&] {
            operations.emplace_back("invalidate");
            return true;
        },
        [&] { operations.emplace_back("convert_and_write"); });

    EXPECT_EQ(result, CompletedReadbackResult::Consumed);
    EXPECT_EQ(operations, (std::vector<std::string_view>{"invalidate", "convert_and_write"}));
}

TEST(CompletedReadback, CoherentMemoryDoesNotRequireInvalidation) {
    int invalidations = 0;
    int consumptions = 0;

    const auto result = ConsumeCompletedReadback(
        true,
        [&] {
            ++invalidations;
            return true;
        },
        [&] { ++consumptions; });

    EXPECT_EQ(result, CompletedReadbackResult::Consumed);
    EXPECT_EQ(invalidations, 0);
    EXPECT_EQ(consumptions, 1);
}

TEST(CompletedReadback, InvalidationFailureSuppressesConversionAndWrite) {
    int invalidations = 0;
    int consumptions = 0;

    const auto result = ConsumeCompletedReadback(
        false,
        [&] {
            ++invalidations;
            return false;
        },
        [&] { ++consumptions; });

    EXPECT_EQ(result, CompletedReadbackResult::InvalidationFailed);
    EXPECT_EQ(invalidations, 1);
    EXPECT_EQ(consumptions, 0);
}

} // namespace
} // namespace VideoCore
