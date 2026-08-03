// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <thread>

#include "core/address_space_operation_gate.h"

namespace {

template <typename TryLock>
bool TryFromAnotherThread(TryLock&& try_lock) {
    bool acquired = false;
    std::jthread thread{[&] { acquired = try_lock().owns_lock(); }};
    thread.join();
    return acquired;
}

} // namespace

TEST(AddressSpaceOperationGate, AllowsIndependentTrackingOperationsToOverlap) {
    Core::AddressSpaceOperationGate gate;

    auto first = gate.LockForTracking();

    EXPECT_TRUE(first.owns_lock());
    EXPECT_TRUE(TryFromAnotherThread([&] { return gate.TryLockForTracking(); }));
    EXPECT_FALSE(TryFromAnotherThread([&] { return gate.TryLockForMutation(); }));
}

TEST(AddressSpaceOperationGate, SerializesMappingAndGuestProtectionMutations) {
    Core::AddressSpaceOperationGate gate;

    auto mutation = gate.LockForMutation();

    EXPECT_TRUE(mutation.owns_lock());
    EXPECT_FALSE(TryFromAnotherThread([&] { return gate.TryLockForTracking(); }));
    EXPECT_FALSE(TryFromAnotherThread([&] { return gate.TryLockForMutation(); }));
}
