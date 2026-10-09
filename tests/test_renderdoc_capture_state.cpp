// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderdoc_capture_state.h"
#include "video_core/renderdoc_compute_selector.h"
#include "video_core/renderdoc_submission_capture.h"

TEST(RenderDocCaptureState, TriggerSchedulesExactlyOneGuestSubmission) {
    VideoCore::RenderDocCaptureState state;

    EXPECT_TRUE(state.Trigger());
    EXPECT_TRUE(state.TryBeginGuestSubmission(true));
    EXPECT_FALSE(state.TryBeginGuestSubmission(true));
    EXPECT_TRUE(state.FinishGuestSubmission());
    EXPECT_FALSE(state.FinishGuestSubmission());
}

TEST(RenderDocCaptureState, TriggerIsPublishedAcrossThreads) {
    VideoCore::RenderDocCaptureState state;

    std::thread trigger_thread{[&state] { EXPECT_TRUE(state.Trigger()); }};
    trigger_thread.join();

    EXPECT_TRUE(state.TryBeginGuestSubmission(true));
}

TEST(RenderDocCaptureState, RepeatedTriggerDoesNotQueueAnotherCapture) {
    VideoCore::RenderDocCaptureState state;

    ASSERT_TRUE(state.Trigger());
    EXPECT_FALSE(state.Trigger());
    ASSERT_TRUE(state.TryBeginGuestSubmission(true));
    EXPECT_FALSE(state.TryBeginGuestSubmission(true));
}

TEST(RenderDocCaptureState, AClaimedCaptureCannotOverlapAnotherRequest) {
    VideoCore::RenderDocCaptureState state;
    ASSERT_TRUE(state.Trigger());
    ASSERT_TRUE(state.TryBeginGuestSubmission(true));
    EXPECT_FALSE(state.Trigger());
    EXPECT_TRUE(state.FinishGuestSubmission());
    EXPECT_TRUE(state.Trigger());
}

namespace {

using VideoCore::GuestCaptureResult;
using VideoCore::GuestSubmissionCapture;

struct FakeCaptureApi {
    bool active{};
    bool allow_start{true};
    bool end_succeeds{true};
    bool discard_succeeds{true};
    bool throw_on_start{};
    bool throw_on_end{};
    bool throw_on_discard{};
    std::vector<std::string> events;

    VideoCore::RenderDocCaptureApi Api() {
        return {
            .context = this,
            .is_capturing =
                [](void* context) { return static_cast<FakeCaptureApi*>(context)->active; },
            .start =
                [](void* context) {
                    auto& api = *static_cast<FakeCaptureApi*>(context);
                    api.events.emplace_back("start");
                    api.active = api.allow_start;
                    if (api.throw_on_start) {
                        throw std::runtime_error{"start failed after activation"};
                    }
                },
            .end =
                [](void* context) {
                    auto& api = *static_cast<FakeCaptureApi*>(context);
                    api.events.emplace_back("end");
                    if (api.throw_on_end) {
                        throw std::runtime_error{"end failed"};
                    }
                    if (api.end_succeeds) {
                        api.active = false;
                    }
                    return api.end_succeeds;
                },
            .discard =
                [](void* context) {
                    auto& api = *static_cast<FakeCaptureApi*>(context);
                    api.events.emplace_back("discard");
                    if (api.throw_on_discard) {
                        throw std::runtime_error{"discard failed"};
                    }
                    if (api.discard_succeeds) {
                        api.active = false;
                    }
                    return api.discard_succeeds;
                },
        };
    }
};

std::string ReadSource(const char* path) {
    std::ifstream stream{path};
    if (!stream.is_open()) {
        throw std::runtime_error{"Cannot read capture integration source"};
    }
    return {std::istreambuf_iterator<char>{stream}, {}};
}

std::string FunctionBody(const std::string& source, const std::string& signature) {
    const auto signature_begin = source.find(signature);
    if (signature_begin == std::string::npos) {
        throw std::runtime_error{"Missing capture integration function " + signature};
    }
    const auto body_begin = source.find('{', signature_begin);
    unsigned depth{};
    for (auto index = body_begin; index < source.size(); ++index) {
        if (source[index] == '{') {
            ++depth;
        } else if (source[index] == '}' && --depth == 0) {
            return source.substr(body_begin, index - body_begin + 1);
        }
    }
    throw std::runtime_error{"Unterminated capture integration function"};
}

} // namespace

TEST(RenderDocGuestSubmission, UnrequestedWorkDoesNotCallCaptureApi) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    bool work = true;
    GuestSubmissionCapture capture{state, api.Api(), work};
    EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::NotRequested);
    EXPECT_FALSE(work);
    EXPECT_TRUE(api.events.empty());
}

TEST(RenderDocGuestSubmission, PresenterAndCopyOnlySubmissionsCannotConsumeRequest) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    ASSERT_TRUE(state.Trigger());
    for (unsigned index = 0; index < 3; ++index) {
        bool presenter_or_copy_work = false;
        GuestSubmissionCapture capture{state, api.Api(), presenter_or_copy_work};
        EXPECT_EQ(capture.CompleteSubmission(true, presenter_or_copy_work),
                  GuestCaptureResult::NotRequested);
        EXPECT_FALSE(state.Trigger());
    }
    EXPECT_TRUE(api.events.empty());
    bool guest_work = true;
    GuestSubmissionCapture capture{state, api.Api(), guest_work};
    ASSERT_EQ(capture.Result(), GuestCaptureResult::Active);
    api.events.emplace_back("submit guest");
    EXPECT_EQ(capture.CompleteSubmission(true, guest_work), GuestCaptureResult::Captured);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start", "submit guest", "end"}));
}

TEST(RenderDocGuestSubmission, SuccessfulCaptureOwnsRequestUntilEndAndResetsWorkOnce) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    GuestSubmissionCapture capture{state, api.Api(), work};
    EXPECT_TRUE(work);
    EXPECT_FALSE(state.Trigger());
    EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::Captured);
    EXPECT_FALSE(work);
    ASSERT_TRUE(state.Trigger());
    work = true; // A later command must not be cleared by a duplicate completion callback.
    EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::Captured);
    EXPECT_TRUE(work);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start", "end"}));
    EXPECT_TRUE(state.TryBeginGuestSubmission(true));
}

TEST(RenderDocGuestSubmission, MissingRuntimeDoesNotPreventActualSubmission) {
    VideoCore::RenderDocCaptureState state;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    GuestSubmissionCapture capture{state, {}, work};
    EXPECT_EQ(capture.Result(), GuestCaptureResult::ApiUnavailable);
    EXPECT_TRUE(work);
    EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::ApiUnavailable);
    EXPECT_FALSE(work);
    EXPECT_TRUE(state.Trigger());
}

TEST(RenderDocGuestSubmission, ExternalCaptureIsNeverStartedEndedOrDiscarded) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    api.active = true;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    {
        GuestSubmissionCapture capture{state, api.Api(), work};
        EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::ExternalCapture);
    }
    EXPECT_TRUE(api.active);
    EXPECT_TRUE(api.events.empty());
    EXPECT_FALSE(work);
}

TEST(RenderDocGuestSubmission, RefusedStartDoesNotEndOrDiscardAndDoesNotClearUnsubmittedWork) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    api.allow_start = false;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    GuestSubmissionCapture capture{state, api.Api(), work};
    EXPECT_EQ(capture.Result(), GuestCaptureResult::StartRefused);
    EXPECT_TRUE(work);
    EXPECT_EQ(capture.CompleteSubmission(false, work), GuestCaptureResult::StartRefused);
    EXPECT_TRUE(work);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start"}));
}

TEST(RenderDocGuestSubmission, QueueErrorDiscardsBeforeFatalCheckAndPreservesMarker) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    {
        GuestSubmissionCapture capture{state, api.Api(), work};
        api.events.emplace_back("queue error");
        EXPECT_EQ(capture.CompleteSubmission(false, work), GuestCaptureResult::SubmissionFailed);
        api.events.emplace_back("fatal result check");
        EXPECT_FALSE(api.active);
        EXPECT_TRUE(work);
    }
    EXPECT_EQ(api.events,
              (std::vector<std::string>{"start", "queue error", "discard", "fatal result check"}));
    EXPECT_TRUE(state.Trigger());
}

TEST(RenderDocGuestSubmission, ExceptionUnwindsOwnedCaptureWithoutClearingMarker) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    EXPECT_THROW(
        ([&] {
            GuestSubmissionCapture capture{state, api.Api(), work};
            api.events.emplace_back("submit throws");
            throw std::runtime_error{"submission threw"};
        }()),
        std::runtime_error);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start", "submit throws", "discard"}));
    EXPECT_TRUE(work);
    EXPECT_FALSE(api.active);
    EXPECT_TRUE(state.Trigger());
}

TEST(RenderDocGuestSubmission, FailedEndDiscardsButSuccessfulSubmissionStillClearsMarker) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    api.end_succeeds = false;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    GuestSubmissionCapture capture{state, api.Api(), work};
    EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::EndFailed);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start", "end", "discard"}));
    EXPECT_FALSE(work);
    EXPECT_FALSE(api.active);
    EXPECT_FALSE(capture.CleanupFailed());
    EXPECT_TRUE(state.Trigger());
}

TEST(RenderDocGuestSubmission, PartialStartExceptionIsCleanedWithoutDroppingSubmission) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    api.throw_on_start = true;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    GuestSubmissionCapture capture{state, api.Api(), work};
    EXPECT_EQ(capture.Result(), GuestCaptureResult::ApiException);
    EXPECT_TRUE(work);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start", "discard"}));
    EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::ApiException);
    EXPECT_FALSE(work);
}

TEST(RenderDocGuestSubmission, EndExceptionIsContainedAndOwnedCaptureIsDiscarded) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    api.throw_on_end = true;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    GuestSubmissionCapture capture{state, api.Api(), work};
    EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::ApiException);
    EXPECT_FALSE(work);
    EXPECT_FALSE(api.active);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start", "end", "discard"}));
}

TEST(RenderDocGuestSubmission, FailedDiscardIsReportedAndCannotStartOverStillActiveCapture) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    api.discard_succeeds = false;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    {
        GuestSubmissionCapture capture{state, api.Api(), work};
        EXPECT_EQ(capture.CompleteSubmission(false, work), GuestCaptureResult::SubmissionFailed);
        EXPECT_TRUE(capture.CleanupFailed());
    }
    ASSERT_TRUE(state.Trigger());
    GuestSubmissionCapture next{state, api.Api(), work};
    EXPECT_EQ(next.Result(), GuestCaptureResult::ExternalCapture);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start", "discard"}));
}

TEST(RenderDocGuestSubmission, DiscardExceptionCannotTerminateExceptionUnwinding) {
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    api.throw_on_discard = true;
    bool work = true;
    ASSERT_TRUE(state.Trigger());
    EXPECT_THROW(
        ([&] {
            GuestSubmissionCapture capture{state, api.Api(), work};
            throw std::runtime_error{"submission threw"};
        }()),
        std::runtime_error);
    EXPECT_TRUE(work);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start", "discard"}));
    EXPECT_TRUE(state.Trigger());
}

TEST(RenderDocGuestWorkWiring, PresenterCannotStartOrEndTheGuestCapture) {
    const auto source = ReadSource(SHADPS4_PRESENTER_SOURCE_PATH);
    EXPECT_EQ(source.find("BeginNextPresentedFrameCapture("), std::string::npos);
    EXPECT_EQ(source.find("EndPresentedFrameCapture("), std::string::npos);
    EXPECT_EQ(source.find("BeginGuestWorkCapture("), std::string::npos);
    EXPECT_EQ(source.find("CompleteGuestWorkCapture("), std::string::npos);
    EXPECT_EQ(source.find("MarkGuestWork("), std::string::npos);
}

TEST(RenderDocGuestWorkWiring, CaptureBracketsActualQueueSubmitBeforeFatalErrorCheck) {
    const auto body = FunctionBody(ReadSource(SHADPS4_SCHEDULER_SOURCE_PATH),
                                   "Scheduler::SubmitExecutionWithSubmitLockHeld(");
    const auto begin = body.find("VideoCore::BeginGuestWorkCapture(");
    const auto submit = body.find("instance.GetGraphicsQueue().submit(");
    const auto complete = body.find("VideoCore::CompleteGuestWorkCapture(");
    const auto check = body.find("Check<\"submit GPU command buffer\">");
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(submit, std::string::npos);
    ASSERT_NE(complete, std::string::npos);
    ASSERT_NE(check, std::string::npos);
    EXPECT_LT(begin, submit);
    EXPECT_LT(submit, complete);
    EXPECT_LT(complete, check);
    EXPECT_EQ(body.find(".submit("), body.rfind(".submit("));
    EXPECT_EQ(body.find(".Wait("), std::string::npos);
    EXPECT_EQ(body.find(".Finish("), std::string::npos);
    EXPECT_EQ(body.find(".Flush("), std::string::npos);
    EXPECT_NE(body.find("pending_guest_work", begin), std::string::npos);
    EXPECT_NE(body.find("submit_result == vk::Result::eSuccess", complete), std::string::npos);
}

TEST(RenderDocGuestWorkWiring, OnlyActualGuestCommandEmissionMarksWork) {
    const auto source = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    for (const auto* signature : {"Rasterizer::Draw(", "Rasterizer::DrawIndirect(",
                                  "Rasterizer::DispatchDirect(", "Rasterizer::DispatchIndirect("}) {
        SCOPED_TRACE(signature);
        const auto body = FunctionBody(source, signature);
        const auto mark = body.find("scheduler.MarkGuestWork(");
        ASSERT_NE(mark, std::string::npos);
        EXPECT_EQ(body.find("scheduler.MarkGuestWork(", mark + 1), std::string::npos);
        const auto last_draw = body.rfind("cmdbuf.draw");
        const auto last_dispatch = body.rfind("cmdbuf.dispatch");
        const auto emission = last_draw == std::string::npos ? last_dispatch : last_draw;
        ASSERT_NE(emission, std::string::npos);
        EXPECT_LT(emission, mark);
        EXPECT_GT(body.find("ResetBindings();"), mark);
        // Every early exit for filtering, compilation, HLE, or binding is before emission.
        EXPECT_LT(body.rfind("return;"), emission);
    }
}

TEST(RenderDocComputeSelector, UnsetPreservesDrawAndComputeCapture) {
    const auto selector = VideoCore::RenderDocComputeSelector::Parse(std::nullopt);
    EXPECT_TRUE(selector.Valid());
    EXPECT_TRUE(selector.Allows(std::nullopt));
    EXPECT_TRUE(selector.Allows(0x12345678ULL));
}

TEST(RenderDocComputeSelector, ExactHashExcludesDrawsAndOtherDispatches) {
    const auto selector = VideoCore::RenderDocComputeSelector::Parse("0x123456789abcdef0");
    EXPECT_TRUE(selector.Valid());
    EXPECT_FALSE(selector.Allows(std::nullopt));
    EXPECT_FALSE(selector.Allows(0x12345678ULL));
    EXPECT_TRUE(selector.Allows(0x123456789abcdef0ULL));
}

TEST(RenderDocComputeSelector, OptionalPrefixAndHexCaseDoNotChangeIdentity) {
    for (const auto input : {"aBcD09", "0XABCD09", "0xabcd09"}) {
        const auto selector = VideoCore::RenderDocComputeSelector::Parse(input);
        EXPECT_TRUE(selector.Valid()) << input;
        EXPECT_TRUE(selector.Allows(0xabcd09ULL)) << input;
        EXPECT_FALSE(selector.Allows(0xabcd08ULL)) << input;
    }
}

TEST(RenderDocComputeSelector, ZeroAndMaximumAreExactHashesNotWildcards) {
    const auto zero = VideoCore::RenderDocComputeSelector::Parse("0");
    const auto maximum = VideoCore::RenderDocComputeSelector::Parse("ffffffffffffffff");
    EXPECT_TRUE(zero.Valid());
    EXPECT_TRUE(zero.Allows(0));
    EXPECT_FALSE(zero.Allows(std::nullopt));
    EXPECT_FALSE(zero.Allows(1));
    EXPECT_TRUE(maximum.Valid());
    EXPECT_TRUE(maximum.Allows(~u64{0}));
    EXPECT_FALSE(maximum.Allows(0));
}

TEST(RenderDocComputeSelector, MalformedConfiguredValueNeverFallsBackToBroadCapture) {
    for (const auto input : {"", "0x", "-1", "+1", " 1", "1 ", "12g", "12\n", "1,2",
                             "10000000000000000", "0x10000000000000000"}) {
        const auto selector = VideoCore::RenderDocComputeSelector::Parse(input);
        EXPECT_FALSE(selector.Valid()) << input;
        EXPECT_FALSE(selector.Allows(std::nullopt)) << input;
        EXPECT_FALSE(selector.Allows(1)) << input;
        EXPECT_FALSE(selector.Allows(~u64{0})) << input;
    }
}

TEST(RenderDocComputeSelector, NonmatchingSubmissionsRetainRequestUntilMatchingWork) {
    const auto selector = VideoCore::RenderDocComputeSelector::Parse("1234");
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    ASSERT_TRUE(state.Trigger());
    for (const auto shader : {std::optional<u64>{}, std::optional<u64>{0x5678}}) {
        bool work = selector.Allows(shader);
        GuestSubmissionCapture capture{state, api.Api(), work};
        EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::NotRequested);
        EXPECT_FALSE(state.Trigger());
    }
    EXPECT_TRUE(api.events.empty());
    bool work = selector.Allows(0x1234);
    GuestSubmissionCapture capture{state, api.Api(), work};
    EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::Captured);
    EXPECT_EQ(api.events, (std::vector<std::string>{"start", "end"}));
}

TEST(RenderDocComputeSelector, MatchingWorkRemainsEligibleAfterLaterDrawAndResetsOnSubmit) {
    const auto selector = VideoCore::RenderDocComputeSelector::Parse("1234");
    VideoCore::RenderDocCaptureState state;
    FakeCaptureApi api;
    bool work = selector.Allows(0x1234);
    work |= selector.Allows(std::nullopt);
    work |= selector.Allows(0x5678);
    ASSERT_TRUE(state.Trigger());
    GuestSubmissionCapture capture{state, api.Api(), work};
    EXPECT_EQ(capture.CompleteSubmission(true, work), GuestCaptureResult::Captured);
    EXPECT_FALSE(work);
    ASSERT_TRUE(state.Trigger());
    work |= selector.Allows(std::nullopt);
    GuestSubmissionCapture next{state, api.Api(), work};
    EXPECT_EQ(next.CompleteSubmission(true, work), GuestCaptureResult::NotRequested);
}

TEST(RenderDocComputeSelectorWiring, SelectorUsesPrivateEnvironmentAndInvalidValueIsDiagnosed) {
    const auto source = ReadSource(SHADPS4_RENDERDOC_SOURCE_PATH);
    EXPECT_NE(source.find("SHADPS4_RENDERDOC_COMPUTE_HASH"), std::string::npos);
    EXPECT_NE(source.find("RenderDocComputeSelector::Parse("), std::string::npos);
    EXPECT_NE(source.find("Invalid SHADPS4_RENDERDOC_COMPUTE_HASH"), std::string::npos);
    const auto trigger = FunctionBody(source, "void TriggerCapture(");
    const auto validation = trigger.find("if (!GetComputeCaptureSelector().Valid())");
    const auto arm = trigger.find("capture_state.Trigger()");
    ASSERT_NE(validation, std::string::npos);
    ASSERT_NE(arm, std::string::npos);
    EXPECT_LT(validation, arm);
    EXPECT_NE(trigger.substr(validation, arm - validation).find("return;"), std::string::npos);
    const auto body = FunctionBody(source, "IsGuestWorkCaptureEligible(");
    EXPECT_NE(body.find(".Allows(compute_shader_hash)"), std::string::npos);
}

TEST(RenderDocComputeSelectorWiring, OnlyEmittedComputePassesHashAndSchedulerKeepsMatchingWork) {
    const auto source = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    for (const auto* signature : {"Rasterizer::DispatchDirect(", "Rasterizer::DispatchIndirect("}) {
        const auto body = FunctionBody(source, signature);
        const auto mark = body.find("scheduler.MarkGuestWork(cs.pgm_hash);");
        ASSERT_NE(mark, std::string::npos) << signature;
        EXPECT_LT(body.rfind("cmdbuf.dispatch"), mark);
    }
    for (const auto* signature : {"Rasterizer::Draw(", "Rasterizer::DrawIndirect("}) {
        const auto body = FunctionBody(source, signature);
        EXPECT_NE(body.find("scheduler.MarkGuestWork();"), std::string::npos);
    }
    const auto scheduler =
        FunctionBody(ReadSource(SHADPS4_SCHEDULER_SOURCE_PATH), "Scheduler::MarkGuestWork(");
    EXPECT_NE(scheduler.find("pending_guest_work |="), std::string::npos);
    EXPECT_NE(scheduler.find("VideoCore::IsGuestWorkCaptureEligible(compute_shader_hash)"),
              std::string::npos);
    EXPECT_EQ(scheduler.find("Flush("), std::string::npos);
    EXPECT_EQ(scheduler.find("Wait("), std::string::npos);
}
