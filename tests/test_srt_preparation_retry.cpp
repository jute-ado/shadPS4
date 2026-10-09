// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <optional>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/srt_preparation_retry.h"
#include "video_core/srt_snapshot_epoch_coordinator.h"

namespace VideoCore {
namespace {

using Result = std::optional<u32>;
constexpr auto IsReady = [](const Result& result) { return result.has_value(); };

struct EpochMemory {
    SrtSnapshotEpochCoordinator coordinator;
    u32 watchers{};
    u32 begins{};
    u8 value = 0x11;

    struct Epoch {
        EpochMemory& memory;
        u64 id;
        bool result{};
        ~Epoch() {
            (void)Seal();
        }
        bool BeginPageCopy(VAddr address, u64 size) {
            return memory.coordinator.BeginPageCopy(id, address, size,
                                                    [&](VAddr, u64) { ++memory.watchers; });
        }
        bool EndPageCopy(VAddr address, u64 size, bool copied) {
            return memory.coordinator.EndPageCopy(id, address, size, copied);
        }
        bool Valid() const {
            return id ? memory.coordinator.IsEpochValid(id) : result;
        }
        bool Seal() {
            if (id) {
                result = memory.coordinator.SealEpoch(id, [&](VAddr, u64) { --memory.watchers; });
                id = 0;
            }
            return result;
        }
    };

    Epoch BeginSnapshotEpoch() {
        EXPECT_EQ(watchers, 0u);
        ++begins;
        return {*this, coordinator.BeginEpoch()};
    }
    bool IsGpuMapped(VAddr, u64) {
        return true;
    }
    bool ReadMemorySnapshot(VAddr, void* output, u64 size) {
        std::memset(output, value, size);
        return true;
    }
    void Write() {
        coordinator.InvalidateRangeAndWait(0x4010, 1, [&](VAddr, u64) { --watchers; });
        value = 0x42;
    }
};

TEST(SrtPreparationRetry, FreshAttemptReleasesActualEpochAndRecapturesDescriptorBytes) {
    EpochMemory memory;
    u32 binding = 3;
    u32 attempts = 0;
    std::vector<u32> observed_descriptors;
    const auto result = RetrySrtPreparation(
        binding,
        [&](u32& next_binding, SrtMemoryReadFailure& failure, bool&) -> Result {
            SrtMemoryReadSession session{memory, memory};
            std::array<u8, 4> descriptor{};
            EXPECT_TRUE(session.Read(0x4010, descriptor.data(), descriptor.size()));
            observed_descriptors.push_back(descriptor[0]);
            next_binding += 2;
            if (++attempts == 1) {
                memory.Write();
            }
            if (!session.Seal()) {
                failure = session.LastFailure();
                return std::nullopt;
            }
            return descriptor[0];
        },
        IsReady);
    EXPECT_EQ(result, 0x42u);
    EXPECT_EQ(memory.begins, 2u);
    EXPECT_EQ(memory.watchers, 0u);
    EXPECT_EQ(binding, 5u);
    EXPECT_EQ(observed_descriptors, (std::vector<u32>{0x11, 0x42}));
}

TEST(SrtPreparationRetry, RetriesInvalidatedAttemptWithoutConsumingTheDispatch) {
    u32 binding = 4;
    u32 attempts = 0;
    u32 dispatched = 0;
    std::vector<u32> binding_starts;
    const auto result = RetrySrtPreparation(
        binding,
        [&](u32& next_binding, SrtMemoryReadFailure& failure, bool&) -> Result {
            binding_starts.push_back(next_binding);
            next_binding += 7;
            if (++attempts == 1) {
                failure = SrtMemoryReadFailure::EpochInvalidated;
                return std::nullopt;
            }
            return 0x51u;
        },
        IsReady);
    if (result) {
        ++dispatched;
    }
    ASSERT_EQ(result, 0x51u);
    EXPECT_EQ(dispatched, 1u);
    EXPECT_EQ(attempts, 2u);
    EXPECT_EQ(binding_starts, (std::vector<u32>{4, 4}));
    EXPECT_EQ(binding, 11u);
}

TEST(SrtPreparationRetry, SustainedInvalidationIsBoundedAndDoesNotPublishBindings) {
    u32 binding = 9;
    u32 attempts = 0;
    const auto result = RetrySrtPreparation(
        binding,
        [&](u32& next_binding, SrtMemoryReadFailure& failure, bool&) -> Result {
            EXPECT_EQ(next_binding, 9u);
            next_binding = 99;
            ++attempts;
            failure = SrtMemoryReadFailure::EpochInvalidated;
            return std::nullopt;
        },
        IsReady);
    EXPECT_FALSE(result);
    EXPECT_EQ(attempts, MaxSrtPreparationAttempts);
    EXPECT_EQ(binding, 9u);
}

TEST(SrtPreparationRetry, StopsImmediatelyForEveryNonTransientFailure) {
    constexpr std::array reasons{SrtMemoryReadFailure::None,
                                 SrtMemoryReadFailure::InvalidRequest,
                                 SrtMemoryReadFailure::UnmappedRange,
                                 SrtMemoryReadFailure::ReadbackUnavailable,
                                 SrtMemoryReadFailure::PageCopyRejected,
                                 SrtMemoryReadFailure::NewPageAfterSeal,
                                 SrtMemoryReadFailure::MissingCapturedPage};
    for (const auto reason : reasons) {
        u32 binding = 2;
        u32 attempts = 0;
        const auto result = RetrySrtPreparation(
            binding,
            [&](u32& next_binding, SrtMemoryReadFailure& failure, bool&) -> Result {
                ++attempts;
                ++next_binding;
                failure = reason;
                return std::nullopt;
            },
            IsReady);
        EXPECT_FALSE(result);
        EXPECT_EQ(attempts, 1u);
        EXPECT_EQ(binding, 2u);
    }
}

TEST(SrtPreparationRetry, DoesNotRetryTranslationRejectionEvenWithInvalidation) {
    u32 binding = 2;
    u32 attempts = 0;
    const auto result = RetrySrtPreparation(
        binding,
        [&](u32&, SrtMemoryReadFailure& failure, bool& translation_rejected) -> Result {
            ++attempts;
            failure = SrtMemoryReadFailure::EpochInvalidated;
            translation_rejected = true;
            return std::nullopt;
        },
        IsReady);
    EXPECT_FALSE(result);
    EXPECT_EQ(attempts, 1u);
}

TEST(SrtPreparationRetry, RetryCanEndWithPermanentFailureWithoutPublishing) {
    u32 binding = 8;
    u32 attempts = 0;
    const auto result = RetrySrtPreparation(
        binding,
        [&](u32& next_binding, SrtMemoryReadFailure& failure, bool&) -> Result {
            next_binding = 100;
            failure = ++attempts == 1 ? SrtMemoryReadFailure::EpochInvalidated
                                      : SrtMemoryReadFailure::UnmappedRange;
            return std::nullopt;
        },
        IsReady);
    EXPECT_FALSE(result);
    EXPECT_EQ(attempts, 2u);
    EXPECT_EQ(binding, 8u);
}

TEST(SrtPreparationRetry, FirstSuccessPublishesOnceWithoutRetry) {
    u32 binding = 1;
    u32 attempts = 0;
    const auto result = RetrySrtPreparation(
        binding,
        [&](u32& next_binding, SrtMemoryReadFailure&, bool&) -> Result {
            ++attempts;
            next_binding += 2;
            return 4;
        },
        IsReady);
    EXPECT_EQ(result, 4u);
    EXPECT_EQ(attempts, 1u);
    EXPECT_EQ(binding, 3u);
}

TEST(SrtPreparationRetry, ActualPipelineCreatesReaderAndRuntimeInsideEachAttempt) {
    std::ifstream file{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << file.rdbuf();
    const auto source = contents.str();
    const auto start = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto end = source.find("PipelineCache::ReplaceShader(", start);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(start, end - start);
    const auto retry = body.find("RetrySrtPreparation(");
    ASSERT_NE(retry, std::string::npos);
    const auto session = body.find("SrtReadSession srt_read_session{");
    const auto runtime = body.find("BuildRuntimeInfo(stage, l_stage, reader, draw_user_data)");
    ASSERT_NE(session, std::string::npos);
    ASSERT_NE(runtime, std::string::npos);
    EXPECT_GT(session, retry);
    EXPECT_GT(runtime, session);
    EXPECT_NE(body.find("attempt_failure = srt_read_session.LastFailure()"), std::string::npos);
    EXPECT_NE(body.find("attempt_translation_rejected = translation_rejected"), std::string::npos);
}

} // namespace
} // namespace VideoCore
