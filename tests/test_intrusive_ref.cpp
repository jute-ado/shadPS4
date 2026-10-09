// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "common/intrusive_ref.h"

namespace {

struct TrackedObject {
    int references = 1;
    bool destroyed = false;
};

void Retain(TrackedObject* object) {
    ++object->references;
}

void Release(TrackedObject* object) {
    if (--object->references == 0) {
        object->destroyed = true;
    }
}

TEST(IntrusiveRef, AssignmentRetainsReplacementAndReleasesPreviousObject) {
    TrackedObject previous;
    TrackedObject replacement;
    auto* destination = &previous;
    bool replacement_retained_before_release = false;

    Common::AssignIntrusiveRef(destination, &replacement, Retain, [&](TrackedObject* object) {
        replacement_retained_before_release = replacement.references == 2;
        Release(object);
    });

    EXPECT_EQ(destination, &replacement);
    EXPECT_TRUE(replacement_retained_before_release);
    EXPECT_TRUE(previous.destroyed);
    EXPECT_EQ(replacement.references, 2);
}

TEST(IntrusiveRef, NullAssignmentReleasesPreviousObject) {
    TrackedObject previous;
    auto* destination = &previous;

    Common::AssignIntrusiveRef(destination, nullptr, Retain, Release);

    EXPECT_EQ(destination, nullptr);
    EXPECT_TRUE(previous.destroyed);
}

TEST(IntrusiveRef, AliasingAssignmentDoesNotChangeReferenceCount) {
    TrackedObject object;
    auto* destination = &object;

    Common::AssignIntrusiveRef(destination, &object, Retain, Release);

    EXPECT_EQ(destination, &object);
    EXPECT_EQ(object.references, 1);
    EXPECT_FALSE(object.destroyed);
}

} // namespace
