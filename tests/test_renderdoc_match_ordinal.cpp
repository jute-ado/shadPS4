// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <optional>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include "video_core/renderdoc_capture_state.h"
#include "video_core/renderdoc_shader_selector.h"

namespace {
using VideoCore::RenderDocCaptureState;
using VideoCore::RenderDocShaderSelector;

TEST(RenderDocMatchOrdinal, DefaultAndZeroPreserveTheFirstSubmission) {
    EXPECT_EQ(RenderDocShaderSelector::Parse({}, {}).MatchSkip(), 0);
    const auto zero = RenderDocShaderSelector::Parse({}, "1234", "1", "0");
    ASSERT_TRUE(zero.Valid());
    EXPECT_EQ(zero.MatchSkip(), 0);
    RenderDocCaptureState state;
    ASSERT_TRUE(state.Trigger());
    EXPECT_TRUE(state.TryBeginGuestSubmission(true));
    EXPECT_TRUE(state.FinishGuestSubmission());
}

TEST(RenderDocMatchOrdinal, DecimalSkipHasABoundedExactParse) {
    for (const auto [text, expected] :
         {std::pair{"1", 1U}, std::pair{"16", 16U}, std::pair{"65535", 65535U}}) {
        const auto selector = RenderDocShaderSelector::Parse({}, "1234", "1", text);
        EXPECT_TRUE(selector.Valid());
        EXPECT_EQ(selector.MatchSkip(), expected);
    }
    for (const std::string_view text :
         {"", "-1", "+1", " 1", "1 ", "1x", "0x10", "65536", "9999999999999999999999999999"}) {
        EXPECT_FALSE(RenderDocShaderSelector::Parse({}, "1234", "1", text).Valid()) << text;
    }
}

TEST(RenderDocMatchOrdinal, BeforeTriggerAndNonmatchingCommandsDoNotConsumeAnOrdinal) {
    RenderDocCaptureState state;
    EXPECT_FALSE(state.SelectEligibleCommand(true));
    ASSERT_TRUE(state.Trigger(2));
    EXPECT_FALSE(state.SelectEligibleCommand(false));
    EXPECT_FALSE(state.SelectEligibleCommand(true));
    EXPECT_FALSE(state.SelectEligibleCommand(false));
    EXPECT_FALSE(state.SelectEligibleCommand(true));
    EXPECT_TRUE(state.SelectEligibleCommand(true));
    EXPECT_TRUE(state.TryBeginGuestSubmission(true));
    EXPECT_TRUE(state.FinishGuestSubmission());
    EXPECT_FALSE(state.SelectEligibleCommand(true));
}

TEST(RenderDocMatchOrdinal, PendingOldWorkCannotBeginWhileSkipRemains) {
    RenderDocCaptureState state;
    ASSERT_TRUE(state.Trigger(1));
    EXPECT_FALSE(state.TryBeginGuestSubmission(true));
    EXPECT_FALSE(state.SelectEligibleCommand(true));
    EXPECT_FALSE(state.TryBeginGuestSubmission(true));
    EXPECT_FALSE(state.TryBeginGuestSubmission(false));
    EXPECT_TRUE(state.SelectEligibleCommand(true));
    EXPECT_TRUE(state.TryBeginGuestSubmission(true));
}

TEST(RenderDocMatchOrdinal, RepeatedTriggerCannotResetTheRemainingCount) {
    RenderDocCaptureState state;
    ASSERT_TRUE(state.Trigger(2));
    EXPECT_FALSE(state.SelectEligibleCommand(true));
    EXPECT_FALSE(state.Trigger(99));
    EXPECT_FALSE(state.SelectEligibleCommand(true));
    EXPECT_TRUE(state.SelectEligibleCommand(true));
    EXPECT_TRUE(state.TryBeginGuestSubmission(true));
    EXPECT_FALSE(state.SelectEligibleCommand(true));
    EXPECT_FALSE(state.Trigger(99));
    EXPECT_TRUE(state.FinishGuestSubmission());
    EXPECT_TRUE(state.Trigger(1));
    EXPECT_FALSE(state.SelectEligibleCommand(true));
    EXPECT_TRUE(state.SelectEligibleCommand(true));
}

TEST(RenderDocMatchOrdinal, InvalidCountCannotArmState) {
    RenderDocCaptureState state;
    EXPECT_FALSE(state.Trigger(65536));
    EXPECT_TRUE(state.Trigger(65535));
    EXPECT_FALSE(state.TryBeginGuestSubmission(true));
}

TEST(RenderDocMatchOrdinal, SelectorStillRejectsWrongStageAndColorBeforeCounting) {
    const auto selector = RenderDocShaderSelector::Parse({}, "1234", "1", "1");
    RenderDocCaptureState state;
    ASSERT_TRUE(state.Trigger(1));
    EXPECT_FALSE(state.SelectEligibleCommand(selector.Allows(0x1234, {}, true)));
    EXPECT_FALSE(state.SelectEligibleCommand(selector.Allows({}, 0x1234, false)));
    EXPECT_FALSE(state.SelectEligibleCommand(selector.Allows({}, 0x1111, true)));
    EXPECT_FALSE(state.SelectEligibleCommand(selector.Allows({}, 0x1234, true)));
    EXPECT_TRUE(state.SelectEligibleCommand(selector.Allows({}, 0x1234, true)));
}
} // namespace
