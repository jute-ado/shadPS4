// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/kernel/equeue.h"

using Libraries::Kernel::EqueueEvent;
using Libraries::Kernel::OrbisKernelEvent;

TEST(EqueueGraphicsOccurrence, RepeatedGraphicsTriggersRemainPendingUntilEachIsConsumed) {
    EqueueEvent event{};
    event.event.filter = OrbisKernelEvent::Filter::GraphicsCore;
    event.event.flags = OrbisKernelEvent::Flags::Clear;

    auto* const event_type = reinterpret_cast<void*>(0x1234);
    event.Trigger(event_type);
    event.Trigger(event_type);
    event.Trigger(event_type);

    event.Clear();
    EXPECT_TRUE(event.IsTriggered());
    EXPECT_EQ(event.event.data, reinterpret_cast<uintptr_t>(event_type));

    event.Clear();
    EXPECT_TRUE(event.IsTriggered());
    EXPECT_EQ(event.event.data, reinterpret_cast<uintptr_t>(event_type));

    event.Clear();
    EXPECT_FALSE(event.IsTriggered());
    EXPECT_EQ(event.event.data, 0u);
}

TEST(EqueueGraphicsOccurrence, OrdinaryClearEventsStillCoalesceRepeatedTriggers) {
    EqueueEvent event{};
    event.event.filter = OrbisKernelEvent::Filter::VideoOut;
    event.event.flags = OrbisKernelEvent::Flags::Clear;

    event.Trigger(nullptr);
    event.Trigger(nullptr);
    event.Trigger(nullptr);
    event.Clear();

    EXPECT_FALSE(event.IsTriggered());
    EXPECT_EQ(event.event.data, 0u);
}
