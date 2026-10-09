// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <stdexcept>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/guest_upload_snapshot.h"

namespace {

struct MutatingGuestMemory {
    void Lock() {
        locked = true;
    }

    void Unlock() {
        locked = false;
        if (mutation_deferred) {
            MutateNow();
        }
    }

    void RequestMutation() {
        if (locked) {
            mutation_deferred = true;
        } else {
            MutateNow();
        }
    }

    void CopySnapshot() {
        std::ranges::copy(source.begin(), source.begin() + 2, destination.begin());
        RequestMutation();
        std::ranges::copy(source.begin() + 2, source.end(), destination.begin() + 2);
    }

    void MutateNow() {
        source.fill(2);
        mutation_deferred = false;
    }

    std::array<int, 4> source{1, 1, 1, 1};
    std::array<int, 4> destination{};
    bool locked{};
    bool mutation_deferred{};
};

} // namespace

TEST(GuestUploadSnapshot, DefersGuestMutationUntilTheSnapshotIsComplete) {
    MutatingGuestMemory memory;

    VideoCore::WithGuestUploadSnapshot(
        [&] { memory.Lock(); }, [&] { memory.CopySnapshot(); }, [&] { memory.Unlock(); });

    EXPECT_EQ(memory.destination, (std::array<int, 4>{1, 1, 1, 1}));
    EXPECT_EQ(memory.source, (std::array<int, 4>{2, 2, 2, 2}));
    EXPECT_FALSE(memory.locked);
}

TEST(GuestUploadSnapshot, ReleasesTheGuestRangeWhenSnapshottingThrows) {
    MutatingGuestMemory memory;

    EXPECT_THROW(
        VideoCore::WithGuestUploadSnapshot(
            [&] { memory.Lock(); }, [] { throw std::runtime_error{"copy failed"}; },
            [&] { memory.Unlock(); }),
        std::runtime_error);
    EXPECT_FALSE(memory.locked);
}
