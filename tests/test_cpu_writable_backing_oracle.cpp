// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>

#include <gtest/gtest.h>

#include "core/cpu_writable_backing_oracle.h"

namespace Core {
namespace {

constexpr CpuWritableBackingOracleCandidate DirectCandidate(u64 page_count = 4) {
    return {
        .mapping_class = PhysicalBackingMappingClass::Direct,
        .cpu_read = true,
        .cpu_write = true,
        .physical_backing_eligible = true,
        .complete_provenance = true,
        .owned_allocation = true,
        .page_count = page_count,
    };
}

TEST(CpuWritableBackingOracle, DisabledAndInvalidConfigurationsFailClosed) {
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration(nullptr, nullptr).enabled);
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration("0", "1").enabled);
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration("1", nullptr).enabled);
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration("1", "0").enabled);
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration("1", "-1").enabled);
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration("1", "+1").enabled);
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration("1", "1,2").enabled);
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration("1", " 1").enabled);
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration("yes", "1").enabled);

    const auto too_large = std::to_string(CpuWritableBackingOracle::HardMaxCandidates + 1ULL);
    EXPECT_FALSE(ParseCpuWritableBackingOracleConfiguration("1", too_large.c_str()).enabled);

    CpuWritableBackingOracle disabled{{.enabled = false, .selector = 1}};
    EXPECT_FALSE(disabled.Consider(DirectCandidate()).selected);
    EXPECT_EQ(disabled.GetCoverage(), CpuWritableBackingOracleCoverage{});
    EXPECT_FALSE(disabled.TakeSelectionEvent());
}

TEST(CpuWritableBackingOracle, SelectsOneBasedMatchAndNeverBypassesAnotherMapping) {
    CpuWritableBackingOracle oracle{{.enabled = true, .selector = 2}};
    EXPECT_FALSE(oracle.Consider(DirectCandidate(2)).selected);

    const auto selected = oracle.Consider(DirectCandidate(7));
    EXPECT_TRUE(selected.selected);
    EXPECT_EQ(selected.candidate_ordinal, 2);
    EXPECT_EQ(selected.selected_count, 1);

    EXPECT_FALSE(oracle.Consider(DirectCandidate(9)).selected);
    const auto coverage = oracle.GetCoverage();
    EXPECT_EQ(coverage.valid_candidates, 3);
    EXPECT_EQ(coverage.selected_count, 1);
    EXPECT_EQ(coverage.rejected_after_selection, 1);
    EXPECT_EQ(coverage.loss, CpuWritableBackingOracleLoss::None);
}

TEST(CpuWritableBackingOracle, InvalidMappingsDoNotConsumeSequentialOrdinals) {
    CpuWritableBackingOracle oracle{{.enabled = true, .selector = 1}};
    auto invalid = DirectCandidate();

    invalid.mapping_class = PhysicalBackingMappingClass::Unsupported;
    EXPECT_FALSE(oracle.Consider(invalid).selected);
    invalid = DirectCandidate();
    invalid.mapping_class = PhysicalBackingMappingClass::Flexible;
    EXPECT_FALSE(oracle.Consider(invalid).selected);
    invalid = DirectCandidate();
    invalid.cpu_read = false;
    EXPECT_FALSE(oracle.Consider(invalid).selected);
    invalid = DirectCandidate();
    invalid.cpu_write = false;
    EXPECT_FALSE(oracle.Consider(invalid).selected);
    invalid = DirectCandidate();
    invalid.physical_backing_eligible = false;
    EXPECT_FALSE(oracle.Consider(invalid).selected);
    invalid = DirectCandidate();
    invalid.complete_provenance = false;
    EXPECT_FALSE(oracle.Consider(invalid).selected);
    invalid = DirectCandidate();
    invalid.owned_allocation = false;
    EXPECT_FALSE(oracle.Consider(invalid).selected);
    EXPECT_FALSE(oracle.Consider(DirectCandidate(0)).selected);

    const auto selected = oracle.Consider(DirectCandidate());
    EXPECT_TRUE(selected.selected);
    EXPECT_EQ(selected.candidate_ordinal, 1);
    EXPECT_EQ(oracle.GetCoverage().invalid_candidates, 8);
}

TEST(CpuWritableBackingOracle, DirectAndPooledClassesHaveDeterministicCounts) {
    CpuWritableBackingOracle oracle{{.enabled = true, .selector = 2}};
    EXPECT_FALSE(oracle.Consider(DirectCandidate()).selected);
    auto pooled = DirectCandidate(11);
    pooled.mapping_class = PhysicalBackingMappingClass::Pooled;
    const auto selected = oracle.Consider(pooled);
    EXPECT_TRUE(selected.selected);

    const auto event = oracle.TakeSelectionEvent();
    ASSERT_TRUE(event);
    EXPECT_EQ(event->selector, 2);
    EXPECT_EQ(event->mapping_class, PhysicalBackingMappingClass::Pooled);
    EXPECT_EQ(event->page_count, 11);
    EXPECT_EQ(event->candidate_ordinal, 2);
    EXPECT_EQ(event->selected_count, 1);
    EXPECT_FALSE(oracle.TakeSelectionEvent());
}

TEST(CpuWritableBackingOracle, CandidateCapacityFailsClosedWithExplicitLoss) {
    CpuWritableBackingOracle oracle{{.enabled = true, .selector = 3, .candidate_cap = 2}};
    EXPECT_FALSE(oracle.Consider(DirectCandidate()).selected);
    EXPECT_FALSE(oracle.Consider(DirectCandidate()).selected);
    EXPECT_FALSE(oracle.Consider(DirectCandidate()).selected);

    const auto coverage = oracle.GetCoverage();
    EXPECT_EQ(coverage.valid_candidates, 2);
    EXPECT_EQ(coverage.dropped_candidates, 1);
    EXPECT_EQ(coverage.selected_count, 0);
    EXPECT_NE(coverage.loss & CpuWritableBackingOracleLoss::CandidateCapacity,
              CpuWritableBackingOracleLoss::None);
}

} // namespace
} // namespace Core
