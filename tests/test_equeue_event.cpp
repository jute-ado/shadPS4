// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/equeue.h"

using Libraries::Kernel::EqueueEvent;
using Libraries::Kernel::OrbisKernelEvent;

TEST(EqueueEvent, RetainsRepeatedGraphicsTriggersUntilAcknowledged) {
    EqueueEvent event{};
    event.event.filter = OrbisKernelEvent::Filter::GraphicsCore;

    event.Trigger(reinterpret_cast<void*>(0x40));
    event.Trigger(reinterpret_cast<void*>(0x40));
    event.Trigger(reinterpret_cast<void*>(0x40));

    event.ConsumeTrigger();
    EXPECT_TRUE(event.IsTriggered());
    EXPECT_EQ(event.event.data, 0x40u);

    event.ConsumeTrigger();
    EXPECT_TRUE(event.IsTriggered());
    EXPECT_EQ(event.event.data, 0x40u);

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
