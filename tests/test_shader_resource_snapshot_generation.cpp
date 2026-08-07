// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "shader_recompiler/resource_snapshot_generation.h"

namespace Shader {
namespace {

using Snapshot = std::vector<u32>;

constexpr std::array<u32, 4> UserDataA{0xa0, 0xa1, 0xa2, 0xa3};
constexpr std::array<u32, 4> UserDataB{0xb0, 0xb1, 0xb2, 0xb3};
constexpr std::array<u32, 4> ResourcesA{0xaa10, 0xaa11, 0xaa12, 0xaa13};
constexpr std::array<u32, 4> ResourcesB{0xbb10, 0xbb11, 0xbb12, 0xbb13};

Snapshot MakeSnapshot(const std::array<u32, 4>& user_data,
                      const std::array<u32, 4>& resources) {
    Snapshot result;
    result.insert(result.end(), user_data.begin(), user_data.end());
    result.insert(result.end(), resources.begin(), resources.end());
    return result;
}

TEST(ShaderResourceSnapshotGeneration, AcceptsStableFirstGeneration) {
    const Snapshot generation_a = MakeSnapshot(UserDataA, ResourcesA);
    u32 captures{};

    const auto observation = ObserveResourceSnapshotGeneration(
        generation_a, UserDataA.size(), true, 2, 64,
        [&]() -> std::optional<Snapshot> {
            ++captures;
            return generation_a;
        });

    EXPECT_EQ(observation.status, ResourceSnapshotGenerationStatus::Stable);
    EXPECT_TRUE(observation.rendered_generation_stable);
    EXPECT_EQ(observation.validation_captures, 1);
    EXPECT_EQ(captures, 1);
}

TEST(ShaderResourceSnapshotGeneration, AcceptsStableReplacementGeneration) {
    const Snapshot generation_b = MakeSnapshot(UserDataB, ResourcesB);

    const auto observation = ObserveResourceSnapshotGeneration(
        generation_b, UserDataB.size(), true, 2, 64,
        [&]() -> std::optional<Snapshot> { return generation_b; });

    EXPECT_EQ(observation.status, ResourceSnapshotGenerationStatus::Stable);
    EXPECT_TRUE(observation.rendered_generation_stable);
    EXPECT_FALSE(observation.user_data_changed);
    EXPECT_FALSE(observation.resource_data_changed);
}

TEST(ShaderResourceSnapshotGeneration, RejectsHybridMutationEvenWhenLaterGenerationStabilizes) {
    const Snapshot hybrid = MakeSnapshot(UserDataA, ResourcesB);
    const Snapshot generation_b = MakeSnapshot(UserDataB, ResourcesB);
    std::array snapshots{generation_b, generation_b};
    size_t next{};

    const auto observation = ObserveResourceSnapshotGeneration(
        hybrid, UserDataA.size(), true, 2, 64,
        [&]() -> std::optional<Snapshot> { return snapshots[next++]; });

    EXPECT_EQ(observation.status, ResourceSnapshotGenerationStatus::ChangedThenStable);
    EXPECT_FALSE(observation.rendered_generation_stable);
    EXPECT_TRUE(observation.user_data_changed);
    EXPECT_FALSE(observation.resource_data_changed);
    EXPECT_EQ(observation.validation_captures, 2);
}

TEST(ShaderResourceSnapshotGeneration, RejectsNestedParentChildMutation) {
    // The first two resource words represent a parent table entry and the latter two its child.
    const Snapshot nested_hybrid{UserDataA[0], UserDataA[1], UserDataA[2], UserDataA[3],
                                 ResourcesA[0], ResourcesA[1], ResourcesB[2], ResourcesB[3]};
    const Snapshot nested_b = MakeSnapshot(UserDataA, ResourcesB);
    std::array snapshots{nested_b, nested_b};
    size_t next{};

    const auto observation = ObserveResourceSnapshotGeneration(
        nested_hybrid, UserDataA.size(), true, 2, 64,
        [&]() -> std::optional<Snapshot> { return snapshots[next++]; });

    EXPECT_EQ(observation.status, ResourceSnapshotGenerationStatus::ChangedThenStable);
    EXPECT_FALSE(observation.rendered_generation_stable);
    EXPECT_FALSE(observation.user_data_changed);
    EXPECT_TRUE(observation.resource_data_changed);
}

TEST(ShaderResourceSnapshotGeneration, ReportsBoundedRetryExhaustionAsUnavailable) {
    const Snapshot generation_a = MakeSnapshot(UserDataA, ResourcesA);
    const Snapshot generation_b = MakeSnapshot(UserDataB, ResourcesB);
    std::array snapshots{generation_b, generation_a, generation_b};
    size_t next{};

    const auto observation = ObserveResourceSnapshotGeneration(
        generation_a, UserDataA.size(), true, snapshots.size(), 64,
        [&]() -> std::optional<Snapshot> { return snapshots[next++]; });

    EXPECT_EQ(observation.status, ResourceSnapshotGenerationStatus::RetryExhausted);
    EXPECT_FALSE(observation.rendered_generation_stable);
    EXPECT_FALSE(observation.available);
    EXPECT_EQ(observation.validation_captures, snapshots.size());
}

TEST(ShaderResourceSnapshotGeneration, ReportsCaptureFailureAsUnavailable) {
    const Snapshot generation_a = MakeSnapshot(UserDataA, ResourcesA);

    const auto observation = ObserveResourceSnapshotGeneration(
        generation_a, UserDataA.size(), true, 2, 64,
        []() -> std::optional<Snapshot> { return std::nullopt; });

    EXPECT_EQ(observation.status, ResourceSnapshotGenerationStatus::CaptureUnavailable);
    EXPECT_FALSE(observation.available);
    EXPECT_EQ(observation.validation_captures, 1);
}

TEST(ShaderResourceSnapshotGeneration, DisabledAndCapacityPathsPerformNoValidationWork) {
    const Snapshot generation_a = MakeSnapshot(UserDataA, ResourcesA);
    u32 captures{};
    const auto capture = [&]() -> std::optional<Snapshot> {
        ++captures;
        return generation_a;
    };

    const auto disabled = ObserveResourceSnapshotGeneration(
        generation_a, UserDataA.size(), false, 2, 64, capture);
    EXPECT_EQ(disabled.status, ResourceSnapshotGenerationStatus::Disabled);
    EXPECT_EQ(captures, 0);

    const auto oversized = ObserveResourceSnapshotGeneration(
        generation_a, UserDataA.size(), true, 2, generation_a.size() - 1, capture);
    EXPECT_EQ(oversized.status, ResourceSnapshotGenerationStatus::CapacityExceeded);
    EXPECT_FALSE(oversized.available);
    EXPECT_EQ(captures, 0);

    const auto excessive_retries = ObserveResourceSnapshotGeneration(
        generation_a, UserDataA.size(), true, MaxResourceSnapshotValidationCaptures + 1, 64,
        capture);
    EXPECT_EQ(excessive_retries.status, ResourceSnapshotGenerationStatus::CapacityExceeded);
    EXPECT_FALSE(excessive_retries.available);
    EXPECT_EQ(captures, 0);
}

} // namespace
} // namespace Shader
