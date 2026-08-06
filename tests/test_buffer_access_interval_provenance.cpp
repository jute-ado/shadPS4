// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_access_interval_provenance.h"
#include "video_core/buffer_cache/buffer_access_range_batch.h"
#include "video_core/texture_cache/tiling_work_range.h"

namespace {

using RangeState = VideoCore::BasicBufferAccessRangeState<std::uint32_t>;
using RangeBatch =
    VideoCore::BasicBufferAccessRangeBatch<std::uint32_t, std::uint32_t, std::uint32_t>;

TEST(BufferAccessRangeState, DisjointTransitionsRetainTheirIndependentPriorAccess) {
    RangeState state{128, 0x01, 0x10};

    const auto shader = state.Transition(0, 64, 0x06, 0x20);
    const auto geometry = state.Transition(64, 64, 0x08, 0x40);

    EXPECT_TRUE(shader.requires_barrier);
    EXPECT_EQ(shader.prior_access, 0x01);
    EXPECT_EQ(shader.prior_stages, 0x10);
    EXPECT_TRUE(geometry.requires_barrier);
    EXPECT_EQ(geometry.prior_access, 0x01);
    EXPECT_EQ(geometry.prior_stages, 0x10);
    EXPECT_EQ(state.IntervalCount(), 2);
}

TEST(BufferAccessRangeState, RepeatedMatchingReadNeedsNoBarrierOrStateSplit) {
    RangeState state{128, 0x01, 0x10, 0x02};

    const auto repeated = state.Transition(32, 64, 0x01, 0x10);

    EXPECT_FALSE(repeated.requires_barrier);
    EXPECT_EQ(repeated.prior_access, 0x01);
    EXPECT_EQ(repeated.prior_stages, 0x10);
    EXPECT_EQ(state.IntervalCount(), 1);
}

TEST(BufferAccessRangeState, RepeatedMatchingWriteStillRequiresDependency) {
    RangeState state{128, 0x02, 0x10, 0x02};

    const auto repeated = state.Transition(32, 64, 0x02, 0x10);

    EXPECT_TRUE(repeated.requires_barrier);
    EXPECT_EQ(repeated.prior_access, 0x02);
    EXPECT_EQ(repeated.prior_stages, 0x10);
}

TEST(BufferAccessRangeState, OverlapUnionsOnlyTheCoveredPriorIntervals) {
    RangeState state{192, 0x01, 0x10};
    const auto first = state.Transition(0, 64, 0x02, 0x20);
    const auto last = state.Transition(128, 64, 0x04, 0x40);
    EXPECT_TRUE(first.requires_barrier);
    EXPECT_TRUE(last.requires_barrier);

    const auto overlap = state.Transition(32, 128, 0x08, 0x80);

    EXPECT_TRUE(overlap.requires_barrier);
    EXPECT_EQ(overlap.prior_access, 0x01 | 0x02 | 0x04);
    EXPECT_EQ(overlap.prior_stages, 0x10 | 0x20 | 0x40);
}

TEST(BufferAccessRangeState, CapacityCoarseningNeverDiscardsPriorDependencies) {
    VideoCore::BasicBufferAccessRangeState<std::uint32_t, std::uint32_t, 4> state{64, 0, 0};
    for (std::uint32_t i = 0; i < 8; ++i) {
        const auto transition = state.Transition(i * 8, 8, 1U << i, 1U << (i + 8));
        EXPECT_LE(state.IntervalCount(), 4);
        if (i != 0) {
            EXPECT_TRUE(transition.requires_barrier);
        }
    }

    const auto all = state.Transition(0, 64, 0x100, 0x10000);
    EXPECT_EQ(all.prior_access, 0xFF);
    EXPECT_EQ(all.prior_stages, 0xFF00);
}

TEST(BufferAccessRangeState, ConservativeUnionCannotSuppressALaterBarrier) {
    VideoCore::BasicBufferAccessRangeState<std::uint32_t, std::uint32_t, 1> state{64, 0x01, 0x10};
    const auto first = state.Transition(0, 32, 0x02, 0x20);
    ASSERT_TRUE(first.requires_barrier);
    ASSERT_EQ(state.IntervalCount(), 1);

    const auto union_match = state.Transition(0, 64, 0x03, 0x30);
    EXPECT_TRUE(union_match.requires_barrier);
    EXPECT_EQ(union_match.prior_access, 0x03);
    EXPECT_EQ(union_match.prior_stages, 0x30);
}

TEST(BufferAccessRangeBatch, OverlappingCurrentAccessesAreOrderIndependentAndRangeScoped) {
    RangeBatch write_then_read;
    write_then_read.Add(7, 0, 64, 0x02, 0x20);
    write_then_read.Add(7, 32, 64, 0x01, 0x10);

    RangeBatch read_then_write;
    read_then_write.Add(7, 32, 64, 0x01, 0x10);
    read_then_write.Add(7, 0, 64, 0x02, 0x20);

    ASSERT_EQ(write_then_read.Entries().size(), 3);
    EXPECT_TRUE(std::ranges::equal(write_then_read.Entries(), read_then_write.Entries()));
    EXPECT_EQ(write_then_read.Entries()[0], (RangeBatch::Entry{7, 0, 32, 0x02, 0x20}));
    EXPECT_EQ(write_then_read.Entries()[1], (RangeBatch::Entry{7, 32, 32, 0x03, 0x30}));
    EXPECT_EQ(write_then_read.Entries()[2], (RangeBatch::Entry{7, 64, 32, 0x01, 0x10}));
}

TEST(BufferAccessRangeBatch, DisjointRangesAndDifferentBuffersStayIndependent) {
    RangeBatch batch;
    batch.Add(11, 128, 32, 0x04, 0x40);
    batch.Add(11, 0, 32, 0x01, 0x10);
    batch.Add(12, 0, 32, 0x02, 0x20);

    ASSERT_EQ(batch.Entries().size(), 3);
    EXPECT_EQ(batch.Entries()[0], (RangeBatch::Entry{11, 0, 32, 0x01, 0x10}));
    EXPECT_EQ(batch.Entries()[1], (RangeBatch::Entry{11, 128, 32, 0x04, 0x40}));
    EXPECT_EQ(batch.Entries()[2], (RangeBatch::Entry{12, 0, 32, 0x02, 0x20}));
}

TEST(BufferAccessRangeBatch, ClearStartsANewCommand) {
    RangeBatch batch;
    batch.Add(17, 0, 64, 0x02, 0x20);
    batch.Clear();
    batch.Add(17, 0, 64, 0x01, 0x10);

    ASSERT_EQ(batch.Entries().size(), 1);
    EXPECT_EQ(batch.Entries()[0], (RangeBatch::Entry{17, 0, 64, 0x01, 0x10}));
}

TEST(BufferAccessRangeBatch, DeterministicOverlaysMatchAByteReferenceModel) {
    RangeBatch batch;
    std::array<std::array<std::uint32_t, 64>, 2> reference_access{};
    std::array<std::array<std::uint32_t, 64>, 2> reference_stages{};
    std::uint32_t sequence = 0xC001D00D;
    for (std::uint32_t operation = 0; operation < 96; ++operation) {
        sequence = sequence * 1664525U + 1013904223U;
        const std::uint32_t key_index = sequence & 1U;
        const std::uint32_t offset = (sequence >> 8) % 56;
        const std::uint32_t size = 1 + ((sequence >> 16) % (64 - offset));
        const std::uint32_t access = 1U << (operation % 8);
        const std::uint32_t stages = 1U << (8 + operation % 8);
        batch.Add(100 + key_index, offset, size, access, stages);
        for (std::uint32_t byte = offset; byte < offset + size; ++byte) {
            reference_access[key_index][byte] |= access;
            reference_stages[key_index][byte] |= stages;
        }
    }

    std::array<std::array<std::uint32_t, 64>, 2> actual_access{};
    std::array<std::array<std::uint32_t, 64>, 2> actual_stages{};
    std::array<std::uint64_t, 2> prior_end{};
    for (const auto& entry : batch.Entries()) {
        const auto key_index = entry.key - 100;
        ASSERT_LT(key_index, 2U);
        EXPECT_GE(entry.offset, prior_end[key_index]);
        prior_end[key_index] = entry.offset + entry.size;
        ASSERT_LE(prior_end[key_index], 64U);
        for (auto byte = entry.offset; byte < entry.offset + entry.size; ++byte) {
            actual_access[key_index][byte] = entry.access;
            actual_stages[key_index][byte] = entry.stages;
        }
    }
    EXPECT_EQ(actual_access, reference_access);
    EXPECT_EQ(actual_stages, reference_stages);
}

TEST(TilingWorkRange, StopsBeforeTheFirstMipThatDoesNotFit) {
    const std::array mips = {
        VideoCore::BasicTilingMipRange{.offset = 0, .size = 128},
        VideoCore::BasicTilingMipRange{.offset = 128, .size = 64},
        VideoCore::BasicTilingMipRange{.offset = 192, .size = 32},
    };

    const auto work = VideoCore::ComputeTilingWorkRange(mips, 192);

    EXPECT_EQ(work.num_mips, 2);
    EXPECT_EQ(work.buffer_span, 192);
    EXPECT_EQ(work.dispatch_size, 192);
}

TEST(TilingWorkRange, DistinguishesSparseBufferSpanFromDispatchBytes) {
    const std::array mips = {
        VideoCore::BasicTilingMipRange{.offset = 0, .size = 64},
        VideoCore::BasicTilingMipRange{.offset = 128, .size = 32},
    };

    const auto work = VideoCore::ComputeTilingWorkRange(mips, 160);

    EXPECT_EQ(work.num_mips, 2);
    EXPECT_EQ(work.buffer_span, 160);
    EXPECT_EQ(work.dispatch_size, 96);
}

} // namespace
