// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "core/libraries/kernel/file_read_staging.h"

using Libraries::Kernel::ReadFileThroughStaging;

TEST(FileReadStaging, SizesStorageBeforeReadingAndCopiesOnlyBytesRead) {
    std::vector<u8> staging;
    std::array<u8, 6> destination{};
    destination.fill(0xcc);

    const auto bytes = ReadFileThroughStaging(
        staging, destination.data(), destination.size(), [](std::span<u8> buffer) {
            EXPECT_EQ(buffer.size(), 6u);
            buffer[0] = 0x11;
            buffer[1] = 0x22;
            buffer[2] = 0x33;
            return 3u;
        });

    EXPECT_EQ(bytes, 3u);
    EXPECT_EQ(staging.size(), destination.size());
    EXPECT_EQ(destination, (std::array<u8, 6>{0x11, 0x22, 0x33, 0xcc, 0xcc, 0xcc}));
}

TEST(FileReadStaging, EmptyReadDoesNotCallReaderOrTouchDestination) {
    std::vector<u8> staging{0xaa};
    u8 destination = 0xcc;
    bool reader_called = false;

    const auto bytes = ReadFileThroughStaging(
        staging, &destination, 0, [&](std::span<u8>) {
            reader_called = true;
            return 0u;
        });

    EXPECT_EQ(bytes, 0u);
    EXPECT_FALSE(reader_called);
    EXPECT_EQ(destination, 0xcc);
    EXPECT_TRUE(staging.empty());
}
