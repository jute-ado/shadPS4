// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/stream_buffer_copy.h"

namespace VideoCore {
namespace {

class FakeMemory {
public:
    bool IsValidMapping(VAddr address) {
        ++mapping_check_count;
        checked_address = address;
        return valid_mapping;
    }

    void CopySparseMemory(VAddr address, u8* destination, u64 size) {
        sparse_address = address;
        sparse_size = size;
        std::memcpy(destination, sparse_data.data(), size);
        ++sparse_copy_count;
    }

    bool valid_mapping{};
    u32 mapping_check_count{};
    VAddr checked_address{};
    VAddr sparse_address{};
    u64 sparse_size{};
    u32 sparse_copy_count{};
    std::array<u8, 4> sparse_data{0x12, 0x34, 0x56, 0x78};
};

TEST(StreamBufferCopy, UsesSparseCopyForMappedGuestAddresses) {
    constexpr VAddr GuestAddress = 0x12345000;
    FakeMemory memory{.valid_mapping = true};
    std::array<u8, 4> destination{};

    CopyStreamBufferSource(memory, GuestAddress, destination.data(), destination.size());

    EXPECT_EQ(memory.checked_address, GuestAddress);
    EXPECT_EQ(memory.mapping_check_count, 1);
    EXPECT_EQ(memory.sparse_address, GuestAddress);
    EXPECT_EQ(memory.sparse_size, destination.size());
    EXPECT_EQ(memory.sparse_copy_count, 1);
    EXPECT_EQ(destination, memory.sparse_data);
}

TEST(StreamBufferCopy, CopiesOrdinaryHostDataDirectly) {
    const std::array<u8, 4> source{0x87, 0x65, 0x43, 0x21};
    FakeMemory memory{.valid_mapping = true};
    std::array<u8, 4> destination{};

    CopyStreamBufferSource(memory, source.data(), destination.data(), destination.size());

    EXPECT_EQ(memory.mapping_check_count, 0);
    EXPECT_EQ(memory.sparse_copy_count, 0);
    EXPECT_EQ(destination, source);
}

} // namespace
} // namespace VideoCore
