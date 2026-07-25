// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>

#include <gtest/gtest.h>

#include "core/libraries/system_gesture/system_gesture_behavior.h"

namespace SystemGesture = Libraries::SystemGesture;
namespace Behavior = Libraries::SystemGesture::Behavior;

TEST(SystemGestureBehavior, OpenAcceptsOnlyTouchPadInput) {
    EXPECT_EQ(Behavior::Open(SystemGesture::InputType::TouchPad), SystemGesture::ValidHandle);
    EXPECT_EQ(Behavior::Open(static_cast<SystemGesture::InputType>(1)),
              SystemGesture::ErrorInvalidArgument);
}

TEST(SystemGestureBehavior, HandleOperationsRejectUnknownHandles) {
    EXPECT_EQ(Behavior::Close(SystemGesture::ValidHandle), SystemGesture::Ok);
    EXPECT_EQ(Behavior::Close(0), SystemGesture::ErrorInvalidHandle);
    EXPECT_EQ(Behavior::UpdatePrimitive(SystemGesture::ValidHandle), SystemGesture::Ok);
    EXPECT_EQ(Behavior::UpdatePrimitive(2), SystemGesture::ErrorInvalidHandle);
}

TEST(SystemGestureBehavior, PrimitiveEventsReportNoTouchAndInitializeOutputs) {
    SystemGesture::PrimitiveTouchEvent event;
    std::memset(&event, 0xff, sizeof(event));
    u32 count = 99;

    EXPECT_EQ(Behavior::GetPrimitiveEvents(SystemGesture::ValidHandle, &event, 1, &count),
              SystemGesture::Ok);
    EXPECT_EQ(count, 0u);
    const SystemGesture::PrimitiveTouchEvent empty_event{};
    EXPECT_EQ(std::memcmp(&event, &empty_event, sizeof(event)), 0);

    EXPECT_EQ(Behavior::GetPrimitiveEvents(SystemGesture::ValidHandle, nullptr, 0, nullptr),
              SystemGesture::ErrorInvalidArgument);
    EXPECT_EQ(Behavior::GetPrimitiveEvents(0, &event, 1, &count),
              SystemGesture::ErrorInvalidHandle);
}

TEST(SystemGestureBehavior, PrimitiveLookupReturnsNotFoundAndInitializesOutput) {
    SystemGesture::PrimitiveTouchEvent event;
    std::memset(&event, 0xff, sizeof(event));

    EXPECT_EQ(Behavior::GetPrimitiveEventByIndex(SystemGesture::ValidHandle, 7, &event),
              SystemGesture::ErrorEventDataNotFound);
    const SystemGesture::PrimitiveTouchEvent empty_event{};
    EXPECT_EQ(std::memcmp(&event, &empty_event, sizeof(event)), 0);
    EXPECT_EQ(Behavior::GetPrimitiveEventById(0, 3, &event),
              SystemGesture::ErrorInvalidHandle);
}

TEST(SystemGestureBehavior, TouchRecognizerCreationAndResetInitializeStorage) {
    SystemGesture::TouchRecognizer recognizer;
    std::ranges::fill(recognizer.reserved, ~u64{});

    EXPECT_EQ(Behavior::CreateRecognizer(SystemGesture::ValidHandle, &recognizer),
              SystemGesture::Ok);
    EXPECT_TRUE(std::ranges::all_of(recognizer.reserved, [](u64 value) { return value == 0; }));

    std::ranges::fill(recognizer.reserved, ~u64{});
    EXPECT_EQ(Behavior::ResetRecognizer(SystemGesture::ValidHandle, &recognizer),
              SystemGesture::Ok);
    EXPECT_TRUE(std::ranges::all_of(recognizer.reserved, [](u64 value) { return value == 0; }));

    EXPECT_EQ(Behavior::CreateRecognizer(SystemGesture::ValidHandle, nullptr),
              SystemGesture::ErrorInvalidArgument);
}

TEST(SystemGestureBehavior, RecognizerInformationValidatesPointersAndInitializesOutput) {
    SystemGesture::TouchRecognizer recognizer{};
    SystemGesture::TouchRecognizerInformation information;
    std::memset(&information, 0xff, sizeof(information));

    EXPECT_EQ(Behavior::GetRecognizerInformation(SystemGesture::ValidHandle, &recognizer,
                                                 &information),
              SystemGesture::Ok);
    const SystemGesture::TouchRecognizerInformation empty_information{};
    EXPECT_EQ(std::memcmp(&information, &empty_information, sizeof(information)), 0);
    EXPECT_EQ(Behavior::GetRecognizerInformation(SystemGesture::ValidHandle, nullptr,
                                                 &information),
              SystemGesture::ErrorInvalidArgument);
}

TEST(SystemGestureBehavior, TouchEventsReportNoGestureAndInitializeOutputs) {
    SystemGesture::TouchRecognizer recognizer{};
    SystemGesture::TouchEvent event;
    std::ranges::fill(event.reserved, u8{0xff});
    u32 count = 99;

    EXPECT_EQ(Behavior::GetTouchEvents(SystemGesture::ValidHandle, &recognizer, &event, 1, &count),
              SystemGesture::Ok);
    EXPECT_EQ(count, 0u);
    EXPECT_TRUE(std::ranges::all_of(event.reserved, [](u8 value) { return value == 0; }));
    EXPECT_EQ(Behavior::GetTouchEventsCount(SystemGesture::ValidHandle, &recognizer), 0);

    EXPECT_EQ(Behavior::GetTouchEventByIndex(SystemGesture::ValidHandle, &recognizer, 0, &event),
              SystemGesture::ErrorEventDataNotFound);
    EXPECT_EQ(Behavior::GetTouchEventsCount(SystemGesture::ValidHandle, nullptr),
              SystemGesture::ErrorInvalidArgument);
}
