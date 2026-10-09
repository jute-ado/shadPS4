// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/gnmdriver/graphics_event.h"

TEST(GnmGraphicsEvent, RegistrationUsesClearSemantics) {
    constexpr u64 event_id = 0x40;
    auto* const user_data = reinterpret_cast<void*>(0x1234);

    const auto event = Libraries::GnmDriver::MakeGraphicsEvent(event_id, user_data);

    EXPECT_EQ(event.event.ident, event_id);
    EXPECT_EQ(event.event.filter, Libraries::Kernel::OrbisKernelEvent::Filter::GraphicsCore);
    EXPECT_EQ(event.event.flags, Libraries::Kernel::OrbisKernelEvent::Flags::Add |
                                     Libraries::Kernel::OrbisKernelEvent::Flags::Clear);
    EXPECT_EQ(event.event.udata, user_data);
}

TEST(GnmGraphicsEvent, RepeatedTriggersRemainIndividuallyDrainable) {
    using Libraries::Kernel::DrainReadyEvents;
    using Libraries::Kernel::EqueueEvent;
    using Libraries::Kernel::OrbisKernelEvent;

    auto eop = Libraries::GnmDriver::MakeGraphicsEvent(0x40, nullptr);
    eop.Trigger(reinterpret_cast<void*>(0x40));
    eop.Trigger(reinterpret_cast<void*>(0x40));
    eop.Trigger(reinterpret_cast<void*>(0x40));

    auto compute = Libraries::GnmDriver::MakeGraphicsEvent(0x1, nullptr);
    compute.Trigger(reinterpret_cast<void*>(0x1));

    std::vector<EqueueEvent> events;
    events.emplace_back(std::move(eop));
    events.emplace_back(std::move(compute));

    std::array<OrbisKernelEvent, 4> delivered{};
    ASSERT_EQ(DrainReadyEvents(events, delivered.data(), delivered.size()), 4);
    EXPECT_EQ(delivered[0].ident, 0x40u);
    EXPECT_EQ(delivered[0].fflags, 1u);
    EXPECT_EQ(delivered[1].ident, 0x1u);
    EXPECT_EQ(delivered[1].fflags, 1u);
    EXPECT_EQ(delivered[2].ident, 0x40u);
    EXPECT_EQ(delivered[2].fflags, 2u);
    EXPECT_EQ(delivered[3].ident, 0x40u);
    EXPECT_EQ(delivered[3].fflags, 3u);

    OrbisKernelEvent remaining{};
    EXPECT_EQ(DrainReadyEvents(events, &remaining, 1), 0);
}
