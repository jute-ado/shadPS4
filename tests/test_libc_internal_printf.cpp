// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <string>

#include <gtest/gtest.h>

#include "core/libraries/libc_internal/printf.h"

namespace Libraries::LibcInternal {
namespace {

TEST(LibcInternalPrintf, BoundedFormattingTerminatesWithoutOverwritingTheFollowingByte) {
    struct {
        std::array<char, 5> output;
        char canary;
    } storage{{}, '!'};
    Common::VaList arguments{};

    const int result =
        FormatToBuffer(storage.output.data(), storage.output.size(), "abcdef", &arguments);

    EXPECT_EQ(result, 6);
    EXPECT_STREQ(storage.output.data(), "abcd");
    EXPECT_EQ(storage.canary, '!');
}

TEST(LibcInternalPrintf, ZeroLengthFormattingOnlyComputesTheRequiredSize) {
    char canary = '!';
    Common::VaList arguments{};

    const int result = FormatToBuffer(&canary, 0, "abcdef", &arguments);

    EXPECT_EQ(result, 6);
    EXPECT_EQ(canary, '!');
}

TEST(LibcInternalPrintf, VprintfDoesNotTruncateOutputAtTheLegacyStackBufferSize) {
    const std::string message(300, 'x');
    Common::VaList arguments{};
    testing::internal::CaptureStdout();

    const int result = PrintToStdout(message.c_str(), &arguments);

    EXPECT_EQ(result, message.size());
    EXPECT_EQ(testing::internal::GetCapturedStdout(), message);
}

TEST(LibcInternalPrintf, VprintfMeasuresArgumentsWithoutConsumingTheGuestList) {
    const char text[] = "frame";
    Common::VaRegSave registers{};
    registers.gp[0] = reinterpret_cast<uintptr_t>(text);
    registers.gp[1] = 42;
    Common::VaList arguments{
        .gp_offset = offsetof(Common::VaRegSave, gp),
        .fp_offset = offsetof(Common::VaRegSave, fp),
        .overflow_arg_area = nullptr,
        .reg_save_area = &registers,
    };
    testing::internal::CaptureStdout();

    const int result = PrintToStdout("%s:%d", &arguments);

    EXPECT_EQ(result, 8);
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "frame:42");
}

} // namespace
} // namespace Libraries::LibcInternal
