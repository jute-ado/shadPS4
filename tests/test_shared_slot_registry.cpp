// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <memory>

#include <gtest/gtest.h>

#include "common/shared_slot_registry.h"

namespace Common {
namespace {

struct LifetimeProbe {
    explicit LifetimeProbe(std::atomic<int>* destructions_) : destructions{destructions_} {}
    ~LifetimeProbe() {
        ++*destructions;
    }

    std::atomic<int>* destructions;
};

TEST(SharedSlotRegistry, RemovedObjectLivesUntilOutstandingLeaseIsReleased) {
    std::atomic<int> destructions{};
    SharedSlotRegistry<LifetimeProbe> registry;
    const SlotId id = registry.Insert(std::make_shared<LifetimeProbe>(&destructions));
    auto lease = registry.Acquire(id);

    auto removed = registry.Remove(id);

    EXPECT_FALSE(registry.Acquire(id));
    EXPECT_EQ(destructions.load(), 0);
    removed.reset();
    EXPECT_EQ(destructions.load(), 0);
    lease.reset();
    EXPECT_EQ(destructions.load(), 1);
}

TEST(SharedSlotRegistry, InvalidAndOutOfRangeHandlesAreRejected) {
    SharedSlotRegistry<LifetimeProbe> registry;

    EXPECT_FALSE(registry.Acquire(SlotId{}));
    EXPECT_FALSE(registry.Acquire(SlotId{SlotId::INVALID_INDEX - 1}));
    EXPECT_FALSE(registry.Remove(SlotId{}));
    EXPECT_FALSE(registry.Remove(SlotId{SlotId::INVALID_INDEX - 1}));
}

} // namespace
} // namespace Common
