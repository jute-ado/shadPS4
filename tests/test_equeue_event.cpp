// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/gnmdriver/graphics_event.h"
#include "core/libraries/kernel/equeue.h"

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

TEST(EqueueEvent, ClearGraphicsEventsCoalesceRepeatedTriggers) {
    EqueueEvent event{};
    event.event.filter = OrbisKernelEvent::Filter::GraphicsCore;

    event.Trigger(reinterpret_cast<void*>(0x40));
    event.Trigger(reinterpret_cast<void*>(0x40));
    event.Trigger(reinterpret_cast<void*>(0x40));

    event.ConsumeTrigger();
    EXPECT_FALSE(event.IsTriggered());
    EXPECT_EQ(event.event.data, 0u);
}

TEST(EqueueEvent, OrdinaryClearEventsStillCoalesceRepeatedTriggers) {
    EqueueEvent event{};
    event.event.filter = OrbisKernelEvent::Filter::VideoOut;

    event.Trigger(nullptr);
    event.Trigger(nullptr);
    event.ConsumeTrigger();

    EXPECT_FALSE(event.IsTriggered());
}
