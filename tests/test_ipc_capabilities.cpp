// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "core/ipc/ipc_capabilities.h"
#include "core/ipc/presented_frame_input_queue.h"

namespace AllocationProbe {

thread_local bool enabled{};
thread_local std::size_t count{};

void Begin() {
    count = 0;
    enabled = true;
}

std::size_t End() {
    enabled = false;
    return count;
}

} // namespace AllocationProbe

void* operator new(std::size_t size) {
    if (AllocationProbe::enabled) {
        ++AllocationProbe::count;
    }
    if (void* allocation = std::malloc(size == 0 ? 1 : size)) {
        return allocation;
    }
    throw std::bad_alloc{};
}

void* operator new[](const std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* allocation) noexcept {
    std::free(allocation);
}

void operator delete[](void* allocation) noexcept {
    ::operator delete(allocation);
}

void operator delete(void* allocation, std::size_t) noexcept {
    ::operator delete(allocation);
}

void operator delete[](void* allocation, std::size_t) noexcept {
    ::operator delete(allocation);
}

namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input);
    std::string source{std::istreambuf_iterator<char>{input}, {}};
    std::erase(source, '\r');
    return source;
}

} // namespace

TEST(IpcCapabilities, OmitsRenderDocWhenRuntimeIsUnavailable) {
    const auto capabilities = Core::Ipc::IpcCapabilities(false);

    EXPECT_EQ(std::ranges::find(capabilities, "ENABLE_RENDERDOC_CAPTURE"), capabilities.end());
}

TEST(IpcCapabilities, AdvertisesRenderDocWhenRuntimeIsLoaded) {
    const auto capabilities = Core::Ipc::IpcCapabilities(true);

    EXPECT_NE(std::ranges::find(capabilities, "ENABLE_RENDERDOC_CAPTURE"), capabilities.end());
}

TEST(IpcCapabilities, AdvertisesPresentedFrameGamepadScheduling) {
    const auto capabilities = Core::Ipc::IpcCapabilities(false);

    EXPECT_NE(std::ranges::find(capabilities, "ENABLE_PRESENTED_FRAME_GAMEPAD"),
              capabilities.end());
}

TEST(IpcCapabilities, AdvertisesPresentedFrameScreenshotSchedulingWithoutRenderDoc) {
    const auto capabilities = Core::Ipc::IpcCapabilities(false);

    EXPECT_NE(std::ranges::find(capabilities, "ENABLE_PRESENTED_FRAME_SCREENSHOT"),
              capabilities.end());
}

TEST(IpcCapabilities, AdvertisesPresentedFrameScreenshotSchedulingWithRenderDoc) {
    const auto capabilities = Core::Ipc::IpcCapabilities(true);

    EXPECT_NE(std::ranges::find(capabilities, "ENABLE_PRESENTED_FRAME_SCREENSHOT"),
              capabilities.end());
}

TEST(PresentedFrameScreenshotQueue, AcceptsOnlyPresentedFramesOneThroughFiftyMillion) {
    Core::Ipc::PresentedFrameInputQueue queue;
    using Event = Core::Ipc::PresentedFrameInputEvent;
    using Kind = Core::Ipc::PresentedFrameInputKind;

    EXPECT_FALSE(queue.Schedule(Event{.presented_frame = 0,
                                      .kind = Kind::ScreenshotWithOverlays}));
    EXPECT_TRUE(queue.Schedule(Event{.presented_frame = 1,
                                     .kind = Kind::ScreenshotWithOverlays}));
    EXPECT_TRUE(queue.Schedule(Event{.presented_frame = 50'000'000,
                                     .kind = Kind::ScreenshotWithOverlays}));
    EXPECT_FALSE(queue.Schedule(Event{.presented_frame = 50'000'001,
                                      .kind = Kind::ScreenshotWithOverlays}));
}

TEST(PresentedFrameScreenshotQueue, IsBoundedAndPreservesSameFrameOrder) {
    Core::Ipc::PresentedFrameInputQueue queue;
    using Event = Core::Ipc::PresentedFrameInputEvent;
    using Kind = Core::Ipc::PresentedFrameInputKind;

    for (u32 index = 0; index < Core::Ipc::MaxPresentedFrameInputEvents; ++index) {
        ASSERT_TRUE(queue.Schedule(Event{.presented_frame = 77,
                                         .kind = Kind::ScreenshotWithOverlays,
                                         .control = index}));
    }
    EXPECT_FALSE(queue.Schedule(Event{.presented_frame = 77,
                                      .kind = Kind::ScreenshotWithOverlays}));

    u32 expected{};
    const auto missed =
        queue.DispatchExact(77, [&](const Event& event) { EXPECT_EQ(event.control, expected++); });
    EXPECT_EQ(missed, 0U);
    EXPECT_EQ(expected, Core::Ipc::MaxPresentedFrameInputEvents);
    EXPECT_EQ(queue.PendingCount(), 0U);
}

TEST(PresentedFrameScreenshotQueue, DropsAMissedTargetInsteadOfCapturingTheNextFrame) {
    Core::Ipc::PresentedFrameInputQueue queue;
    using Event = Core::Ipc::PresentedFrameInputEvent;
    using Kind = Core::Ipc::PresentedFrameInputKind;
    ASSERT_TRUE(queue.Schedule(Event{.presented_frame = 6000,
                                     .kind = Kind::ScreenshotWithOverlays}));

    u32 dispatched{};
    const auto missed = queue.DispatchExact(6001, [&](const Event&) { ++dispatched; });
    EXPECT_EQ(missed, 1U);
    EXPECT_EQ(dispatched, 0U);
    EXPECT_EQ(queue.PendingCount(), 0U);
}

TEST(PresentedFrameScreenshotQueue, ScheduleAndExactDispatchDoNotAllocate) {
    bool accepted{};
    u32 dispatched{};
    AllocationProbe::Begin();
    {
        Core::Ipc::PresentedFrameInputQueue queue;
        accepted = queue.Schedule({
            .presented_frame = 1,
            .kind = Core::Ipc::PresentedFrameInputKind::ScreenshotWithOverlays,
        });
        queue.DispatchExact(1, [&](const auto&) { ++dispatched; });
    }
    const auto allocations = AllocationProbe::End();

    EXPECT_TRUE(accepted);
    EXPECT_EQ(dispatched, 1U);
    EXPECT_EQ(allocations, 0U);
}

TEST(PresentedFrameAutomationClock, ReusedFramesDoNotCreateASecondAutomationFrame) {
    const auto first = Core::Ipc::NextPresentedFrameForAutomation(0, false);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(*first, 1U);
    EXPECT_FALSE(Core::Ipc::NextPresentedFrameForAutomation(*first, true).has_value());

    const auto last = Core::Ipc::NextPresentedFrameForAutomation(49'999'999, false);
    ASSERT_TRUE(last.has_value());
    EXPECT_EQ(*last, 50'000'000U);
    EXPECT_FALSE(Core::Ipc::NextPresentedFrameForAutomation(*last, false).has_value());
}

TEST(PresentedFrameScreenshotQueue, ProductionQueueIsFixedCapacityAndAllocationFree) {
    const std::string queue = ReadSource(SHADPS4_PRESENTED_FRAME_QUEUE_SOURCE_PATH);

    EXPECT_NE(queue.find("std::array<PresentedFrameInputEvent, MaxPresentedFrameInputEvents>"),
              std::string::npos);
    EXPECT_EQ(queue.find("std::vector"), std::string::npos);
    EXPECT_EQ(queue.find("std::deque"), std::string::npos);
}

TEST(PresentedFrameScreenshotQueue, ProductionParserConsumesFrameBeforeScheduling) {
    const std::string ipc = ReadSource(SHADPS4_IPC_SOURCE_PATH);

    const auto command =
        ipc.find("else if (cmd == \"SCREENSHOT_WITH_OVERLAYS_AT_PRESENTED_FRAME\")");
    const auto frame = ipc.find("const u64 presented_frame = next_u64();", command);
    const auto schedule = ipc.find("SchedulePresentedFrameScreenshot(presented_frame)", frame);
    const auto next_command = ipc.find("else if (cmd ==", command + 1);
    EXPECT_NE(command, std::string::npos);
    EXPECT_NE(frame, std::string::npos);
    EXPECT_NE(schedule, std::string::npos);
    EXPECT_NE(next_command, std::string::npos);
    EXPECT_LT(command, frame);
    EXPECT_LT(frame, schedule);
    EXPECT_LT(schedule, next_command);
}

TEST(PresentedFrameScreenshotQueue,
     ProductionDispatchesFromTheAuthoritativeClockImmediatelyBeforeConsumption) {
    const std::string presenter = ReadSource(SHADPS4_PRESENTER_SOURCE_PATH);

    const auto consume_call =
        presenter.find("VideoCore::ConsumeWithOverlaysScreenshotRequests()");
    const auto consume = presenter.rfind("const auto capture_with_overlays =", consume_call);
    const auto dispatch = presenter.rfind("DispatchPresentedFrameScreenshots", consume);
    const auto clock = presenter.rfind("DebugState.GetFrameNum()", dispatch);
    const auto reuse = presenter.find("is_reusing_frame", clock);
    const auto dispatch_end = presenter.find(';', dispatch);
    const auto increment = presenter.find("DebugState.IncFlipFrameNum()", consume_call);
    EXPECT_NE(consume_call, std::string::npos);
    EXPECT_NE(consume, std::string::npos);
    EXPECT_NE(dispatch, std::string::npos);
    EXPECT_NE(clock, std::string::npos);
    EXPECT_NE(reuse, std::string::npos);
    EXPECT_NE(dispatch_end, std::string::npos);
    EXPECT_NE(increment, std::string::npos);
    EXPECT_LT(clock, reuse);
    EXPECT_LT(reuse, dispatch);
    EXPECT_LT(dispatch, consume);
    EXPECT_LT(consume, increment);
    ASSERT_LT(dispatch_end, consume);
    const auto between = presenter.substr(dispatch_end + 1, consume - dispatch_end - 1);
    EXPECT_TRUE(std::ranges::all_of(between, [](const char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '}';
    }));
    EXPECT_EQ(presenter.find("DispatchPresentedFrameScreenshots", consume), std::string::npos);
}

TEST(PresentedFrameInputQueue, DispatchesDueInputsInStableRouteOrder) {
    Core::Ipc::PresentedFrameInputQueue queue;
    using Event = Core::Ipc::PresentedFrameInputEvent;
    using Kind = Core::Ipc::PresentedFrameInputKind;

    EXPECT_TRUE(queue.Schedule(Event{.presented_frame = 600,
                                     .kind = Kind::Button,
                                     .control = 3,
                                     .value = 1}));
    EXPECT_TRUE(queue.Schedule(Event{.presented_frame = 600,
                                     .kind = Kind::Axis,
                                     .control = 5,
                                     .value = 255}));
    EXPECT_TRUE(queue.Schedule(Event{.presented_frame = 601,
                                     .kind = Kind::Button,
                                     .control = 3,
                                     .value = 0}));

    std::vector<Event> dispatched;
    queue.Dispatch(599, [&](const Event& event) { dispatched.push_back(event); });
    EXPECT_TRUE(dispatched.empty());
    queue.Dispatch(600, [&](const Event& event) { dispatched.push_back(event); });
    ASSERT_EQ(dispatched.size(), 2U);
    EXPECT_EQ(dispatched[0].kind, Kind::Button);
    EXPECT_EQ(dispatched[0].value, 1U);
    EXPECT_EQ(dispatched[1].kind, Kind::Axis);
    EXPECT_EQ(dispatched[1].value, 255U);
    queue.Dispatch(601, [&](const Event& event) { dispatched.push_back(event); });
    ASSERT_EQ(dispatched.size(), 3U);
    EXPECT_EQ(dispatched[2].value, 0U);
    EXPECT_EQ(queue.PendingCount(), 0U);
}

TEST(PresentedFrameInputQueue, LateDispatchDoesNotLoseAnInputDuringAHostPause) {
    Core::Ipc::PresentedFrameInputQueue queue;
    using Event = Core::Ipc::PresentedFrameInputEvent;
    using Kind = Core::Ipc::PresentedFrameInputKind;
    ASSERT_TRUE(queue.Schedule(Event{.presented_frame = 100,
                                     .kind = Kind::Button,
                                     .control = 1,
                                     .value = 1}));

    std::vector<Event> dispatched;
    queue.Dispatch(140, [&](const Event& event) { dispatched.push_back(event); });
    ASSERT_EQ(dispatched.size(), 1U);
    EXPECT_EQ(dispatched[0].presented_frame, 100U);
}

TEST(PresentedFrameInputQueue, RejectsMalformedOutOfOrderAndUnboundedSchedules) {
    Core::Ipc::PresentedFrameInputQueue queue;
    using Event = Core::Ipc::PresentedFrameInputEvent;
    using Kind = Core::Ipc::PresentedFrameInputKind;

    EXPECT_FALSE(queue.Schedule(Event{}));
    EXPECT_FALSE(queue.Schedule(Event{.presented_frame = 50'000'001,
                                      .kind = Kind::Button,
                                      .control = 1,
                                      .value = 1}));
    ASSERT_TRUE(queue.Schedule(Event{.presented_frame = 200,
                                     .kind = Kind::Button,
                                     .control = 1,
                                     .value = 1}));
    EXPECT_FALSE(queue.Schedule(Event{.presented_frame = 199,
                                      .kind = Kind::Button,
                                      .control = 1,
                                      .value = 0}));
    for (u32 index = 1; index < Core::Ipc::MaxPresentedFrameInputEvents; ++index) {
        ASSERT_TRUE(queue.Schedule(Event{.presented_frame = 200 + index,
                                         .kind = Kind::Axis,
                                         .control = 0,
                                         .value = 128}));
    }
    EXPECT_FALSE(queue.Schedule(Event{.presented_frame = 10'000,
                                      .kind = Kind::Axis,
                                      .control = 0,
                                      .value = 128}));
}

TEST(PresentedFrameInputQueue, ReusesCapacityAfterPartialDispatch) {
    Core::Ipc::PresentedFrameInputQueue queue;
    using Event = Core::Ipc::PresentedFrameInputEvent;
    using Kind = Core::Ipc::PresentedFrameInputKind;

    for (u32 index = 0; index < Core::Ipc::MaxPresentedFrameInputEvents; ++index) {
        ASSERT_TRUE(queue.Schedule(Event{.presented_frame = index + 1,
                                         .kind = Kind::Button,
                                         .control = 1,
                                         .value = 1}));
    }
    u32 dispatched{};
    queue.Dispatch(Core::Ipc::MaxPresentedFrameInputEvents / 2,
                   [&](const Event&) { ++dispatched; });
    ASSERT_EQ(dispatched, Core::Ipc::MaxPresentedFrameInputEvents / 2);

    for (u32 index = 0; index < Core::Ipc::MaxPresentedFrameInputEvents / 2; ++index) {
        ASSERT_TRUE(queue.Schedule(Event{
            .presented_frame = Core::Ipc::MaxPresentedFrameInputEvents + index + 1,
            .kind = Kind::Axis,
            .control = 0,
            .value = 128}));
    }
    EXPECT_EQ(queue.PendingCount(), Core::Ipc::MaxPresentedFrameInputEvents);
}

TEST(PresentedFrameInputQueue, ProductionIpcQueuesAndDispatchesOnlyAtActualPresentation) {
    const std::string ipc = ReadSource(SHADPS4_IPC_SOURCE_PATH);
    const std::string presenter = ReadSource(SHADPS4_PRESENTER_SOURCE_PATH);

    EXPECT_NE(ipc.find("GAMEPAD_BUTTON_AT_PRESENTED_FRAME"), std::string::npos);
    EXPECT_NE(ipc.find("GAMEPAD_AXIS_AT_PRESENTED_FRAME"), std::string::npos);
    EXPECT_NE(ipc.find("presented_frame_input_queue.Schedule"), std::string::npos);
    const auto increment = presenter.find("DebugState.IncFlipFrameNum()");
    const auto dispatch = presenter.find("DispatchPresentedFrameInputs(presented_frame)", increment);
    EXPECT_NE(increment, std::string::npos);
    EXPECT_NE(dispatch, std::string::npos);
    EXPECT_LT(increment, dispatch);
    EXPECT_EQ(presenter.find("DispatchPresentedFrameInputs", dispatch + 1), std::string::npos);
}
