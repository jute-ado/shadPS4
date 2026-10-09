// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/srt_failure_flat_provenance.h"

namespace VideoCore {
namespace {

constexpr u64 ShadowShaderHash = 0x9422f66b;
constexpr VAddr ShadowMalformedAddress = 0xaab485027cd0;
constexpr u32 ShadowPointerLow = 0x85027cd0;
constexpr u32 ShadowPointerHigh = 0x414caab4;

SrtFailureFlatSnapshot MakeShadowSnapshot() {
    SrtFailureFlatSnapshot snapshot{};
    snapshot.total_dwords = SrtFailureFlatSnapshot::PrefixDwords;
    snapshot.copied_dwords = SrtFailureFlatSnapshot::PrefixDwords;
    snapshot.has_first_walker_prefix = true;
    snapshot.words[18] = ShadowPointerLow;
    snapshot.words[19] = ShadowPointerHigh;
    return snapshot;
}

TEST(SrtFailureFlatProvenance, CapturesShadowMalformedNestedPointer) {
    const auto snapshot = MakeShadowSnapshot();
    const SrtReadFailureRequest request{
        .known = true,
        .address = ShadowMalformedAddress,
        .size = 4,
        .output_null = false,
        .invalid_reasons = 16,
    };

    const auto provenance =
        CaptureSrtFailureFlatProvenance(ShadowShaderHash, "aggregate_refresh", snapshot, request,
                                        SrtMemoryReadFailure::InvalidRequest);

    ASSERT_TRUE(provenance);
    EXPECT_EQ(provenance->shader_hash, ShadowShaderHash);
    EXPECT_EQ(provenance->phase, "aggregate_refresh");
    EXPECT_EQ(provenance->low_flat_index, 18U);
    EXPECT_EQ(provenance->high_flat_index, 19U);
    EXPECT_EQ(provenance->raw_low, ShadowPointerLow);
    EXPECT_EQ(provenance->raw_high, ShadowPointerHigh);
    EXPECT_EQ(provenance->raw_address, 0x414caab485027cd0ULL);
    EXPECT_EQ(provenance->masked_address, ShadowMalformedAddress);
    EXPECT_EQ(provenance->pointer_size, 4U);
    EXPECT_EQ(provenance->invalid_reasons, 16U);
    EXPECT_EQ(provenance->pointer_failure, SrtMemoryReadFailure::InvalidRequest);
}

TEST(SrtFailureFlatProvenance, RequiresKnownRequestAndMatchingRetainedPointer) {
    const auto snapshot = MakeShadowSnapshot();
    const SrtReadFailureRequest unknown{};
    EXPECT_FALSE(CaptureSrtFailureFlatProvenance(ShadowShaderHash, "aggregate_refresh", snapshot,
                                                 unknown, SrtMemoryReadFailure::InvalidRequest));

    const SrtReadFailureRequest unrelated{
        .known = true,
        .address = 0x12345000,
        .size = 4,
        .output_null = false,
        .invalid_reasons = 0,
    };
    EXPECT_FALSE(CaptureSrtFailureFlatProvenance(ShadowShaderHash, "aggregate_refresh", snapshot,
                                                 unrelated, SrtMemoryReadFailure::InvalidRequest));
}

TEST(SrtFailureFlatProvenanceSource, ExactTargetLogEmitsPointerFields) {
    std::ifstream file{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(file.is_open());
    std::ostringstream contents;
    contents << file.rdbuf();
    const auto source = contents.str();

    EXPECT_NE(source.find("CaptureSrtFailureFlatProvenance("), std::string::npos);
    EXPECT_NE(source.find("SrtFailureFlatProvenanceV1"), std::string::npos);
    EXPECT_NE(source.find("raw_low={:08x}"), std::string::npos);
    EXPECT_NE(source.find("raw_high={:08x}"), std::string::npos);
    EXPECT_NE(source.find("raw_address={:#x}"), std::string::npos);
    EXPECT_NE(source.find("masked_address={:#x}"), std::string::npos);
    EXPECT_NE(source.find("pointer_size={}"), std::string::npos);
    EXPECT_NE(source.find("invalid_reasons={}"), std::string::npos);
    EXPECT_NE(source.find("pointer_failure={}"), std::string::npos);
}

} // namespace
} // namespace VideoCore
