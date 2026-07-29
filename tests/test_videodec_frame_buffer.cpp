// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <future>
#include <shared_mutex>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/validated_shared_access.h"
#include "core/libraries/videodec/video_utils.h"

TEST(VideodecFrameBuffer, AccountsForAlignedNV12CropSpace) {
    const auto layout = Libraries::Videodec::GetNV12FrameLayout(1920, 1080);

    EXPECT_EQ(layout.pitch, 1920);
    EXPECT_EQ(layout.height, 1088);
    EXPECT_EQ(layout.size, 3'133'440);
    EXPECT_FALSE(Libraries::Videodec::CanCopyNV12Data(3'110'400, 1920, 1080));
    EXPECT_TRUE(Libraries::Videodec::CanCopyNV12Data(layout.size, 1920, 1080));
}

TEST(VideodecFrameBuffer, AcceptsExactAlreadyAlignedNV12Storage) {
    const auto layout = Libraries::Videodec::GetNV12FrameLayout(1280, 720);

    EXPECT_EQ(layout.pitch, 1280);
    EXPECT_EQ(layout.height, 720);
    EXPECT_EQ(layout.size, 1'382'400);
    EXPECT_TRUE(Libraries::Videodec::CanCopyNV12Data(layout.size, 1280, 720));
}

TEST(VideodecFrameBuffer, HoldsMappingLockForEntireValidatedAccess) {
    using namespace std::chrono_literals;

    std::shared_mutex mutex;
    std::promise<void> access_started;
    std::promise<void> allow_access_to_finish;
    auto allow_access = allow_access_to_finish.get_future().share();

    std::thread reader([&] {
        EXPECT_TRUE(Common::WithValidatedSharedAccess(
            mutex, [] { return true; },
            [&] {
                access_started.set_value();
                allow_access.wait();
            }));
    });

    access_started.get_future().wait();
    std::promise<void> writer_acquired;
    std::thread writer([&] {
        std::unique_lock lock{mutex};
        writer_acquired.set_value();
    });

    auto writer_ready = writer_acquired.get_future();
    EXPECT_EQ(writer_ready.wait_for(100ms), std::future_status::timeout);
    allow_access_to_finish.set_value();
    reader.join();
    writer.join();
    EXPECT_EQ(writer_ready.wait_for(1s), std::future_status::ready);
}

TEST(VideodecFrameBuffer, ClearsHostWriteBarrierBeforeAccess) {
    std::shared_mutex mutex;
    std::vector<std::string_view> events;
    bool write_barrier_active = true;

    EXPECT_TRUE(Common::WithPreparedValidatedSharedAccess(
        mutex,
        [&] {
            events.emplace_back("validate");
            return true;
        },
        [&] {
            events.emplace_back("prepare");
            write_barrier_active = false;
        },
        [&] {
            events.emplace_back("access");
            EXPECT_FALSE(write_barrier_active);
        }));
    EXPECT_EQ(events, (std::vector<std::string_view>{"validate", "prepare", "access"}));
}
