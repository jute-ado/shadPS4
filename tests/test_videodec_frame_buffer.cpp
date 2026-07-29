// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
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

TEST(VideodecFrameBuffer, PacksStridedNV12IntoAlignedGuestLayout) {
    constexpr u32 width = 4;
    constexpr u32 height = 2;
    constexpr u32 source_pitch = 6;
    const auto layout = Libraries::Videodec::GetNV12FrameLayout(width, height);

    const std::array<u8, source_pitch * height> luma{
        1, 2, 3, 4, 0xee, 0xee, 5, 6, 7, 8, 0xee, 0xee,
    };
    const std::array<u8, source_pitch> chroma{9, 10, 11, 12, 0xee, 0xee};
    std::vector<u8> packed(layout.size, 0xcd);

    const Libraries::Videodec::NV12SourceView source{
        .width = width,
        .height = height,
        .luma = luma.data(),
        .luma_pitch = source_pitch,
        .chroma = chroma.data(),
        .chroma_pitch = source_pitch,
    };
    ASSERT_TRUE(Libraries::Videodec::PackNV12Data(packed, source));

    EXPECT_TRUE(std::ranges::equal(std::span{packed}.first<4>(), std::span{luma}.first<4>()));
    EXPECT_TRUE(std::ranges::equal(std::span{packed}.subspan(layout.pitch, width),
                                   std::span{luma}.subspan(source_pitch, width)));
    EXPECT_EQ(packed[width], 0xcd);

    for (u32 y = height; y < layout.height; ++y) {
        EXPECT_TRUE(std::ranges::equal(std::span{packed}.subspan(y * layout.pitch, width),
                                       std::span{luma}.subspan(source_pitch, width)));
    }

    const u64 chroma_offset = static_cast<u64>(layout.pitch) * layout.height;
    for (u32 y = 0; y < layout.height / 2; ++y) {
        EXPECT_TRUE(
            std::ranges::equal(std::span{packed}.subspan(chroma_offset + y * layout.pitch, width),
                               std::span{chroma}.first<4>()));
    }
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

TEST(VideodecFrameBuffer, ReconcilesHostWriteBarrierAfterAccess) {
    std::shared_mutex mutex;
    std::vector<std::string_view> events;

    EXPECT_TRUE(Common::WithPreparedValidatedSharedAccess(
        mutex,
        [&] {
            events.emplace_back("validate");
            return true;
        },
        [&] { events.emplace_back("prepare"); }, [&] { events.emplace_back("access"); },
        [&] { events.emplace_back("finalize"); }));
    EXPECT_EQ(events, (std::vector<std::string_view>{"validate", "prepare", "access", "finalize"}));
}
