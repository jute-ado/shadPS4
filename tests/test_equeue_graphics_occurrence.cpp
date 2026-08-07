// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <vector>
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

TEST(EqueueGraphicsOccurrence, ReadyGraphicsIdsShareCapacityBeforeRepeatedOccurrences) {
    EqueueEvent eop{};
    eop.event.ident = 0x40;
    eop.event.filter = OrbisKernelEvent::Filter::GraphicsCore;
    eop.event.flags = OrbisKernelEvent::Flags::Clear;
    eop.Trigger(reinterpret_cast<void*>(0x40));
    eop.Trigger(reinterpret_cast<void*>(0x40));
    eop.Trigger(reinterpret_cast<void*>(0x40));

    EqueueEvent compute{};
    compute.event.ident = 0x1;
    compute.event.filter = OrbisKernelEvent::Filter::GraphicsCore;
    compute.event.flags = OrbisKernelEvent::Flags::Clear;
    compute.Trigger(reinterpret_cast<void*>(0x1));

    std::vector<EqueueEvent> events;
    events.emplace_back(std::move(eop));
    events.emplace_back(std::move(compute));

    std::array<OrbisKernelEvent, 2> delivered{};
    ASSERT_EQ(Libraries::Kernel::DrainReadyEvents(events, delivered.data(), 2), 2);
    EXPECT_EQ(delivered[0].ident, 0x40u);
    EXPECT_EQ(delivered[1].ident, 0x1u);

    ASSERT_EQ(Libraries::Kernel::DrainReadyEvents(events, delivered.data(), 2), 2);
    EXPECT_EQ(delivered[0].ident, 0x40u);
    EXPECT_EQ(delivered[1].ident, 0x40u);
    EXPECT_EQ(Libraries::Kernel::DrainReadyEvents(events, delivered.data(), 2), 0);
}

TEST(EqueueGraphicsOccurrence, RemainingCapacityDrainsRepeatedGraphicsOccurrences) {
    EqueueEvent eop{};
    eop.event.ident = 0x40;
    eop.event.filter = OrbisKernelEvent::Filter::GraphicsCore;
    eop.event.flags = OrbisKernelEvent::Flags::Clear;
    eop.Trigger(reinterpret_cast<void*>(0x40));
    eop.Trigger(reinterpret_cast<void*>(0x40));
    eop.Trigger(reinterpret_cast<void*>(0x40));

    EqueueEvent compute{};
    compute.event.ident = 0x1;
    compute.event.filter = OrbisKernelEvent::Filter::GraphicsCore;
    compute.event.flags = OrbisKernelEvent::Flags::Clear;
    compute.Trigger(reinterpret_cast<void*>(0x1));

    std::vector<EqueueEvent> events;
    events.emplace_back(std::move(eop));
    events.emplace_back(std::move(compute));

    std::array<OrbisKernelEvent, 4> delivered{};
    ASSERT_EQ(Libraries::Kernel::DrainReadyEvents(events, delivered.data(), 4), 4);
    EXPECT_EQ(delivered[0].ident, 0x40u);
    EXPECT_EQ(delivered[1].ident, 0x1u);
    EXPECT_EQ(delivered[2].ident, 0x40u);
    EXPECT_EQ(delivered[3].ident, 0x40u);
    EXPECT_EQ(Libraries::Kernel::DrainReadyEvents(events, delivered.data(), 1), 0);
}
