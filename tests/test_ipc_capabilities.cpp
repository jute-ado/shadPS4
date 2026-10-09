// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "core/ipc/ipc_capabilities.h"
#include "core/ipc/presented_frame_input_queue.h"

namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input);
    return {std::istreambuf_iterator<char>{input}, {}};
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
