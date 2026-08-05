// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <stdexcept>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/guest_upload_transaction.h"

namespace {

struct MutatingGuestMemory {
    void Lock() {
        locked = true;
    }

    void Unlock() {
        locked = false;
        if (mutation_deferred) {
            source.fill(2);
            mutation_deferred = false;
        }
    }

    void Snapshot() {
        std::ranges::copy(source, snapshot.begin());
        if (locked) {
            mutation_deferred = true;
        } else {
            source.fill(2);
        }
    }

    std::array<int, 4> source{1, 1, 1, 1};
    std::array<int, 4> snapshot{};
    bool locked{};
    bool mutation_deferred{};
};

} // namespace

TEST(GuestUploadTransaction, ReleasesGuestRangeBeforePublishingSnapshot) {
    MutatingGuestMemory memory;
    bool published_while_locked{};
    std::array<int, 4> published{};

    VideoCore::WithGuestUploadTransaction(
        [&] { memory.Lock(); }, [&] { memory.Snapshot(); }, [&] { memory.Unlock(); },
        [&] {
            published_while_locked = memory.locked;
            published = memory.snapshot;
        });

    EXPECT_FALSE(published_while_locked);
    EXPECT_EQ(published, (std::array<int, 4>{1, 1, 1, 1}));
    EXPECT_EQ(memory.source, (std::array<int, 4>{2, 2, 2, 2}));
}

TEST(GuestUploadTransaction, ReleasesGuestRangeAndSkipsPublishWhenSnapshotThrows) {
    MutatingGuestMemory memory;
    bool published{};

    EXPECT_THROW(
        VideoCore::WithGuestUploadTransaction(
            [&] { memory.Lock(); }, [] { throw std::runtime_error{"copy failed"}; },
            [&] { memory.Unlock(); }, [&] { published = true; }),
        std::runtime_error);
    EXPECT_FALSE(memory.locked);
    EXPECT_FALSE(published);
}
