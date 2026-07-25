// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "core/libraries/gnmdriver/graphics_event.h"
#include "core/libraries/kernel/equeue.h"
#include "core/libraries/kernel/equeue_wait_policy.h"

using Libraries::Kernel::EqueueEvent;
using Libraries::Kernel::OrbisKernelEvent;

TEST(GnmGraphicsEvent, RegistrationUsesClearSemantics) {
    constexpr u64 event_id = 0x40;
    auto* const user_data = reinterpret_cast<void*>(0x1234);

    auto event = Libraries::GnmDriver::MakeGraphicsEvent(event_id, user_data);

    EXPECT_EQ(event.event.ident, event_id);
    EXPECT_EQ(event.event.filter, OrbisKernelEvent::Filter::GraphicsCore);
    EXPECT_EQ(event.event.flags,
              OrbisKernelEvent::Flags::Add | OrbisKernelEvent::Flags::Clear);
    EXPECT_EQ(event.event.udata, user_data);
}

TEST(EqueueEvent, ClearGraphicsEventsPreserveRepeatedTriggers) {
    EqueueEvent event{};
    event.event.filter = OrbisKernelEvent::Filter::GraphicsCore;

    event.Trigger(reinterpret_cast<void*>(0x40));
    event.Trigger(reinterpret_cast<void*>(0x41));
    event.Trigger(reinterpret_cast<void*>(0x42));

    EXPECT_EQ(event.event.data, 0x40u);
    event.ConsumeTrigger();
    EXPECT_TRUE(event.IsTriggered());
    EXPECT_EQ(event.event.data, 0x41u);
    event.ConsumeTrigger();
    EXPECT_TRUE(event.IsTriggered());
    EXPECT_EQ(event.event.data, 0x42u);
    event.ConsumeTrigger();
    EXPECT_FALSE(event.IsTriggered());
    EXPECT_EQ(event.event.data, 0u);
}

TEST(EqueueEvent, DrainReturnsRepeatedGraphicsTriggersUpToCallerCapacity) {
    EqueueEvent event{};
    event.event.filter = OrbisKernelEvent::Filter::GraphicsCore;
    event.event.flags = OrbisKernelEvent::Flags::Clear;

    event.Trigger(reinterpret_cast<void*>(0x40));
    event.Trigger(reinterpret_cast<void*>(0x41));
    event.Trigger(reinterpret_cast<void*>(0x42));

    std::array<OrbisKernelEvent, 3> delivered{};
    EXPECT_EQ(event.DrainTriggers(delivered.data(), static_cast<int>(delivered.size())), 3);
    EXPECT_EQ(delivered[0].data, 0x40u);
    EXPECT_EQ(delivered[1].data, 0x41u);
    EXPECT_EQ(delivered[2].data, 0x42u);
    EXPECT_FALSE(event.IsTriggered());
}

TEST(EqueueEvent, DrainPreservesRepeatedGraphicsTriggersBeyondCallerCapacity) {
    EqueueEvent event{};
    event.event.filter = OrbisKernelEvent::Filter::GraphicsCore;
    event.event.flags = OrbisKernelEvent::Flags::Clear;

    event.Trigger(reinterpret_cast<void*>(0x40));
    event.Trigger(reinterpret_cast<void*>(0x41));
    event.Trigger(reinterpret_cast<void*>(0x42));

    std::array<OrbisKernelEvent, 2> first_batch{};
    EXPECT_EQ(event.DrainTriggers(first_batch.data(), static_cast<int>(first_batch.size())), 2);
    EXPECT_EQ(first_batch[0].data, 0x40u);
    EXPECT_EQ(first_batch[1].data, 0x41u);
    ASSERT_TRUE(event.IsTriggered());

    OrbisKernelEvent final_event{};
    EXPECT_EQ(event.DrainTriggers(&final_event, 1), 1);
    EXPECT_EQ(final_event.data, 0x42u);
    EXPECT_FALSE(event.IsTriggered());
}

TEST(EqueueEvent, OrdinaryClearEventsStillCoalesceRepeatedTriggers) {
    EqueueEvent event{};
    event.event.filter = OrbisKernelEvent::Filter::VideoOut;

    event.Trigger(nullptr);
    event.Trigger(nullptr);
    event.ConsumeTrigger();

    EXPECT_FALSE(event.IsTriggered());
}

TEST(EqueueWaitPolicy, ReadyRegularEventWinsOverPendingSmallTimer) {
    using Libraries::Kernel::EqueueWaitSource;
    using Libraries::Kernel::SelectEqueueWaitSource;

    EXPECT_EQ(SelectEqueueWaitSource(true, true), EqueueWaitSource::RegularEvent);
    EXPECT_EQ(SelectEqueueWaitSource(false, true), EqueueWaitSource::SmallTimer);
    EXPECT_EQ(SelectEqueueWaitSource(false, false), EqueueWaitSource::ConditionVariable);
}
