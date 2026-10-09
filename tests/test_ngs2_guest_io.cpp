// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <span>

#include <gtest/gtest.h>

#include "core/libraries/ngs2/ngs2_guest_io.h"

namespace Libraries::Ngs2::GuestIo {
namespace {

TEST(Ngs2GuestIo, SnapshotsDescriptorArrayOnceAndStagesEveryOutput) {
    std::array<u8, 4> first_output{};
    std::array<u8, 8> second_output{};
    const std::array guest_descriptors{
        OrbisNgs2RenderBufferInfo{first_output.data(), first_output.size(), 1, 2},
        OrbisNgs2RenderBufferInfo{second_output.data(), second_output.size(), 2, 4},
    };
    u32 read_calls = 0;
    u32 validate_calls = 0;
    RenderBufferSnapshot snapshot;

    const auto result = SnapshotRenderBuffers(
        guest_descriptors.data(), guest_descriptors.size(),
        [&](const void* source, void* destination, size_t size) {
            ++read_calls;
            EXPECT_EQ(source, guest_descriptors.data());
            EXPECT_EQ(size, sizeof(guest_descriptors));
            std::memcpy(destination, source, size);
            return true;
        },
        [&](std::span<const Core::GuestMemoryRange> ranges) {
            ++validate_calls;
            EXPECT_EQ(ranges.size(), 2);
            EXPECT_EQ(ranges[0].address, reinterpret_cast<VAddr>(first_output.data()));
            EXPECT_EQ(ranges[1].address, reinterpret_cast<VAddr>(second_output.data()));
            return true;
        },
        &snapshot);

    EXPECT_EQ(result, SnapshotError::None);
    EXPECT_EQ(read_calls, 1);
    EXPECT_EQ(validate_calls, 1);
    ASSERT_EQ(snapshot.descriptors.size(), 2);
    ASSERT_EQ(snapshot.output_writes.size(), 2);
    EXPECT_NE(snapshot.descriptors[0].buffer, first_output.data());
    EXPECT_NE(snapshot.descriptors[1].buffer, second_output.data());
    EXPECT_EQ(snapshot.descriptors[0].bufferSize, first_output.size());
    EXPECT_EQ(snapshot.descriptors[1].bufferSize, second_output.size());
    EXPECT_EQ(snapshot.output_writes[0].address,
              reinterpret_cast<VAddr>(first_output.data()));
    EXPECT_EQ(snapshot.output_writes[1].address,
              reinterpret_cast<VAddr>(second_output.data()));
    auto* const staged_first_output = static_cast<u8*>(snapshot.descriptors[0].buffer);
    staged_first_output[0] = 0x5a;
    EXPECT_EQ(static_cast<const u8*>(snapshot.output_writes[0].data)[0], 0x5a);
}

TEST(Ngs2GuestIo, RejectsDescriptorCountBeforeReadingWhenByteLimitWouldBeExceeded) {
    bool read_called = false;
    bool validate_called = false;
    RenderBufferSnapshot snapshot;
    const u32 excessive_count = static_cast<u32>(MaxRenderDescriptorBytes /
                                                  sizeof(OrbisNgs2RenderBufferInfo) + 1);

    const auto result = SnapshotRenderBuffers(
        reinterpret_cast<const OrbisNgs2RenderBufferInfo*>(0x1000), excessive_count,
        [&](const void*, void*, size_t) {
            read_called = true;
            return true;
        },
        [&](std::span<const Core::GuestMemoryRange>) {
            validate_called = true;
            return true;
        },
        &snapshot);

    EXPECT_EQ(result, SnapshotError::InvalidDescriptorList);
    EXPECT_FALSE(read_called);
    EXPECT_FALSE(validate_called);
}

TEST(Ngs2GuestIo, RejectsAllOutputsTogetherBeforeAllocatingStaging) {
    std::array<u8, 4> first_output{};
    std::array<u8, 8> second_output{};
    const std::array guest_descriptors{
        OrbisNgs2RenderBufferInfo{first_output.data(), first_output.size(), 1, 2},
        OrbisNgs2RenderBufferInfo{second_output.data(), second_output.size(), 2, 4},
    };
    u32 validate_calls = 0;
    RenderBufferSnapshot snapshot;

    const auto result = SnapshotRenderBuffers(
        guest_descriptors.data(), guest_descriptors.size(),
        [&](const void* source, void* destination, size_t size) {
            std::memcpy(destination, source, size);
            return true;
        },
        [&](std::span<const Core::GuestMemoryRange> ranges) {
            ++validate_calls;
            EXPECT_EQ(ranges.size(), 2);
            return false;
        },
        &snapshot);

    EXPECT_EQ(result, SnapshotError::InvalidOutputAddress);
    EXPECT_EQ(validate_calls, 1);
    EXPECT_TRUE(snapshot.staging.empty());
    EXPECT_TRUE(snapshot.output_writes.empty());
}

} // namespace
} // namespace Libraries::Ngs2::GuestIo
