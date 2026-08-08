// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <bit>
#include <limits>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/bda_fallback_consumption.h"

namespace VideoCore {
namespace {

constexpr BdaFallbackWindowConfig TestConfig{
    .first_frame = 100,
    .frame_count = 3,
    .max_operations_per_frame = 8,
};

std::vector<std::atomic<u32>> MakeAtomicWords(const BdaFallbackWindowConfig& config) {
    return std::vector<std::atomic<u32>>(BdaFallbackWindowWordCount(config));
}

std::vector<u32> ReadWords(const std::vector<std::atomic<u32>>& words) {
    std::vector<u32> result(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        result[i] = words[i].load(std::memory_order_relaxed);
    }
    return result;
}

TEST(BdaFallbackConsumption, ComputesCheckedFrameOperationStageIndex) {
    const auto first = PlanBdaFallbackMark(TestConfig, true, 100, 0, 0);
    EXPECT_EQ(first.status, BdaFallbackMarkStatus::Valid);
    EXPECT_EQ(first.bit_index, 0);
    EXPECT_EQ(first.word_index, 0);
    EXPECT_EQ(first.bit_mask, 1U);

    const auto next_frame = PlanBdaFallbackMark(TestConfig, true, 101, 2, 3);
    const u64 expected_bit = 8 * NumBdaFallbackLogicalStages +
                             2 * NumBdaFallbackLogicalStages + 3;
    EXPECT_EQ(next_frame.status, BdaFallbackMarkStatus::Valid);
    EXPECT_EQ(next_frame.bit_index, expected_bit);
    EXPECT_EQ(next_frame.word_index, expected_bit / 32);
    EXPECT_EQ(next_frame.bit_mask, 1U << (expected_bit % 32));
}

TEST(BdaFallbackConsumption, DisabledAndOutsideWindowDoNotTouchStorage) {
    auto words = MakeAtomicWords(TestConfig);
    const auto disabled = PlanBdaFallbackMark(TestConfig, false, 100, 0, 0);
    const auto before = PlanBdaFallbackMark(TestConfig, true, 99, 0, 0);
    const auto after = PlanBdaFallbackMark(TestConfig, true, 103, 0, 0);

    EXPECT_EQ(disabled.status, BdaFallbackMarkStatus::Disabled);
    EXPECT_EQ(before.status, BdaFallbackMarkStatus::OutsideWindow);
    EXPECT_EQ(after.status, BdaFallbackMarkStatus::OutsideWindow);
    EXPECT_FALSE(MarkBdaFallback(words, disabled));
    EXPECT_FALSE(MarkBdaFallback(words, before));
    EXPECT_FALSE(MarkBdaFallback(words, after));
    EXPECT_EQ(ReadWords(words), std::vector<u32>(words.size(), 0));
}

TEST(BdaFallbackConsumption, DuplicateLaneMarksAreIdempotent) {
    auto words = MakeAtomicWords(TestConfig);
    const auto mark = PlanBdaFallbackMark(TestConfig, true, 100, 3, 2);

    ASSERT_TRUE(MarkBdaFallback(words, mark));
    ASSERT_TRUE(MarkBdaFallback(words, mark));

    const auto snapshot = ReadWords(words);
    EXPECT_EQ(std::popcount(snapshot[mark.word_index]), 1);
    EXPECT_EQ(snapshot[mark.word_index], mark.bit_mask);
}

TEST(BdaFallbackConsumption, ConcurrentAtomicOrRetainsDifferentMarks) {
    auto words = MakeAtomicWords(TestConfig);
    const auto first = PlanBdaFallbackMark(TestConfig, true, 100, 0, 0);
    const auto second = PlanBdaFallbackMark(TestConfig, true, 100, 0, 1);
    ASSERT_EQ(first.word_index, second.word_index);

    std::atomic<bool> start{};
    std::jthread thread_a([&] {
        while (!start.load(std::memory_order_acquire)) {
        }
        MarkBdaFallback(words, first);
    });
    std::jthread thread_b([&] {
        while (!start.load(std::memory_order_acquire)) {
        }
        MarkBdaFallback(words, second);
    });
    start.store(true, std::memory_order_release);
    thread_a.join();
    thread_b.join();

    EXPECT_EQ(words[first.word_index].load(std::memory_order_relaxed),
              first.bit_mask | second.bit_mask);
}

TEST(BdaFallbackConsumption, ReportsOperationStageAndCapacityLossExplicitly) {
    const auto operation_overflow = PlanBdaFallbackMark(TestConfig, true, 100, 8, 0);
    const auto stage_overflow =
        PlanBdaFallbackMark(TestConfig, true, 100, 0, NumBdaFallbackLogicalStages);
    EXPECT_EQ(operation_overflow.status, BdaFallbackMarkStatus::OperationOverflow);
    EXPECT_EQ(stage_overflow.status, BdaFallbackMarkStatus::StageOverflow);

    const BdaFallbackWindowConfig too_many_frames{
        .first_frame = 0,
        .frame_count = MaxBdaFallbackWindowFrames + 1,
        .max_operations_per_frame = 1,
    };
    const BdaFallbackWindowConfig wrapped{
        .first_frame = std::numeric_limits<u64>::max() - 1,
        .frame_count = 3,
        .max_operations_per_frame = 1,
    };
    EXPECT_EQ(PlanBdaFallbackMark(too_many_frames, true, 0, 0, 0).status,
              BdaFallbackMarkStatus::CapacityExceeded);
    EXPECT_EQ(PlanBdaFallbackMark(wrapped, true, wrapped.first_frame, 0, 0).status,
              BdaFallbackMarkStatus::CapacityExceeded);
}

TEST(BdaFallbackConsumption, ReducerReportsUnavailableIncompleteAndCapacityLoss) {
    BdaFallbackFrameReducer reducer(TestConfig);
    std::vector<u32> words(BdaFallbackFrameWordCount(TestConfig));

    const auto unavailable =
        reducer.Observe(100, {}, BdaFallbackFrameAvailability::Unavailable, false);
    EXPECT_EQ(unavailable.status, BdaFallbackFrameStatus::Unavailable);
    const auto incomplete =
        reducer.Observe(100, words, BdaFallbackFrameAvailability::Incomplete, false);
    EXPECT_EQ(incomplete.status, BdaFallbackFrameStatus::Incomplete);
    const auto overflow =
        reducer.Observe(100, words, BdaFallbackFrameAvailability::Complete, true);
    EXPECT_EQ(overflow.status, BdaFallbackFrameStatus::OperationOverflow);
    const auto short_words = std::span<const u32>{words}.first(words.size() - 1);
    const auto capacity =
        reducer.Observe(100, short_words, BdaFallbackFrameAvailability::Complete, false);
    EXPECT_EQ(capacity.status, BdaFallbackFrameStatus::CapacityLoss);
}

TEST(BdaFallbackConsumption, ReducerDetectsStableChangeAndExactAbaReturn) {
    BdaFallbackFrameReducer reducer(TestConfig);
    std::vector<u32> generation_a(BdaFallbackFrameWordCount(TestConfig));
    std::vector<u32> generation_b(BdaFallbackFrameWordCount(TestConfig));
    generation_a[0] = 1U << 2;
    generation_b[0] = 1U << 9;

    const auto first =
        reducer.Observe(100, generation_a, BdaFallbackFrameAvailability::Complete, false);
    EXPECT_FALSE(first.has_previous);
    EXPECT_FALSE(first.changed_from_previous);

    const auto changed =
        reducer.Observe(101, generation_b, BdaFallbackFrameAvailability::Complete, false);
    EXPECT_TRUE(changed.has_previous);
    EXPECT_TRUE(changed.changed_from_previous);
    EXPECT_FALSE(changed.exact_aba_return);

    const auto returned =
        reducer.Observe(102, generation_a, BdaFallbackFrameAvailability::Complete, false);
    EXPECT_TRUE(returned.changed_from_previous);
    EXPECT_TRUE(returned.exact_aba_return);
    EXPECT_EQ(returned.aba_middle_frame, 101);

    BdaFallbackFrameReducer stable_reducer(TestConfig);
    stable_reducer.Observe(100, generation_a, BdaFallbackFrameAvailability::Complete, false);
    const auto stable =
        stable_reducer.Observe(101, generation_a, BdaFallbackFrameAvailability::Complete, false);
    EXPECT_TRUE(stable.has_previous);
    EXPECT_FALSE(stable.changed_from_previous);
    EXPECT_FALSE(stable.exact_aba_return);
}

TEST(BdaFallbackConsumption, ReducerSuppressesAbaAcrossCoverageGap) {
    BdaFallbackFrameReducer reducer(TestConfig);
    std::vector<u32> generation_a(BdaFallbackFrameWordCount(TestConfig));
    std::vector<u32> generation_b(BdaFallbackFrameWordCount(TestConfig));
    generation_a[0] = 1;
    generation_b[0] = 2;

    reducer.Observe(100, generation_a, BdaFallbackFrameAvailability::Complete, false);
    reducer.Observe(101, generation_b, BdaFallbackFrameAvailability::Unavailable, false);
    const auto returned =
        reducer.Observe(102, generation_a, BdaFallbackFrameAvailability::Complete, false);

    EXPECT_FALSE(returned.has_previous);
    EXPECT_FALSE(returned.exact_aba_return);
}

} // namespace
} // namespace VideoCore
