// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#if __has_include("video_core/compute_write_provenance.h")
#include "video_core/compute_write_provenance.h"

namespace {
using VideoCore::ComputeWriteProvenancePolicy;
using VideoCore::ComputeWriteProvenanceReceipt;
using VideoCore::ComputeWriteProvenanceSample;

ComputeWriteProvenanceSample Sample() {
    return {.shader_hash = 7,
            .resource_index = 2,
            .push_binding = 3,
            .raw = {0x1010, 0, 0x40, 0},
            .base_address = 0x1010,
            .bound_size = 0x40};
}

TEST(ComputeWriteProvenance, DefaultOffIgnoresMalformedDependentFields) {
    for (const auto enable :
         {std::optional<std::string_view>{}, std::optional<std::string_view>{"0"}}) {
        const auto policy = ComputeWriteProvenancePolicy::Parse(enable, "bad", "bad", "bad");
        EXPECT_TRUE(policy.valid);
        EXPECT_FALSE(policy.enabled);
        ComputeWriteProvenanceReceipt receipt{policy};
        EXPECT_FALSE(receipt.Enabled());
        EXPECT_FALSE(receipt.Observe(470, true, true, true, Sample()));
    }
}

TEST(ComputeWriteProvenance, ExactOptInRequiresAValidExplicitGuestRange) {
    for (const auto begin : {"1000", "0x1000", "0X1000"}) {
        const auto policy = ComputeWriteProvenancePolicy::Parse("1", {}, begin, "1100");
        EXPECT_TRUE(policy.valid);
        EXPECT_TRUE(policy.enabled);
        EXPECT_EQ(policy.start_frame, 470U);
        EXPECT_EQ(policy.range_begin, 0x1000U);
        EXPECT_EQ(policy.range_end, 0x1100U);
    }
    EXPECT_FALSE(ComputeWriteProvenancePolicy::Parse("1", {}, {}, "1100").valid);
    EXPECT_FALSE(ComputeWriteProvenancePolicy::Parse("1", {}, "1000", {}).valid);
    for (const auto value : {"", "true", "01", " 1", "1 ", "+1"}) {
        EXPECT_FALSE(ComputeWriteProvenancePolicy::Parse(value, {}, "1000", "1100").valid);
    }
    for (const auto value : {"", "0x", "-1000", "+1000", " 1000", "1000 ", "gg", "0"}) {
        EXPECT_FALSE(ComputeWriteProvenancePolicy::Parse("1", {}, value, "1100").valid);
    }
    for (const auto end : {"1000", "fff", "10000000001", "10000000000000000"}) {
        EXPECT_FALSE(ComputeWriteProvenancePolicy::Parse("1", {}, "1000", end).valid);
    }
    EXPECT_TRUE(ComputeWriteProvenancePolicy::Parse("1", {}, "ffffffffff", "10000000000").valid);
}

TEST(ComputeWriteProvenance, WindowAndWritableComputeStorageFilterBeforeCapacity) {
    ComputeWriteProvenanceReceipt receipt{
        ComputeWriteProvenancePolicy::Parse("1", "5", "1000", "1100")};
    for (unsigned repeat = 0; repeat < 100; ++repeat) {
        EXPECT_FALSE(receipt.Observe(-1, true, true, true, Sample()));
        EXPECT_FALSE(receipt.Observe(4, true, true, true, Sample()));
        EXPECT_FALSE(receipt.Observe(5, false, true, true, Sample()));
        EXPECT_FALSE(receipt.Observe(5, true, false, true, Sample()));
        EXPECT_FALSE(receipt.Observe(5, true, true, false, Sample()));
    }
    EXPECT_EQ(receipt.Observe(5, true, true, true, Sample()), 0U);
    for (const auto frame : {"", "-1", "+1", " 1", "1 ", "470f", "1000001", "4294967296"}) {
        EXPECT_FALSE(ComputeWriteProvenancePolicy::Parse("1", frame, "1000", "1100").valid);
    }
}

TEST(ComputeWriteProvenance, HalfOpenBoundedOverlapRejectsEmptyAndOverflow) {
    const auto matches = [](std::uint64_t base, std::uint64_t size) {
        ComputeWriteProvenanceReceipt receipt{
            ComputeWriteProvenancePolicy::Parse("1", "0", "1000", "1100")};
        auto sample = Sample();
        sample.base_address = base;
        sample.bound_size = size;
        return receipt.Observe(0, true, true, true, sample).has_value();
    };
    EXPECT_FALSE(matches(0xf00, 0x100));
    EXPECT_TRUE(matches(0xfff, 2));
    EXPECT_TRUE(matches(0x1000, 0x100));
    EXPECT_TRUE(matches(0x10ff, 1));
    EXPECT_FALSE(matches(0x1100, 1));
    EXPECT_FALSE(matches(0x1000, 0));
    EXPECT_FALSE(matches(0x1000, UINT64_MAX));
    EXPECT_FALSE(matches(0, 0x2000));
    EXPECT_FALSE(matches(0xffffffffff, 2));
}

TEST(ComputeWriteProvenance, RepeatedDescriptorsDoNotConsumeUniqueRows) {
    ComputeWriteProvenanceReceipt receipt{
        ComputeWriteProvenancePolicy::Parse("1", "0", "1000", "1100")};
    EXPECT_EQ(receipt.Observe(0, true, true, true, Sample()), 0U);
    for (unsigned repeat = 0; repeat < 100; ++repeat) {
        EXPECT_FALSE(receipt.Observe(1, true, true, true, Sample()));
    }
    auto sample = Sample();
    ++sample.raw[3];
    EXPECT_EQ(receipt.Observe(1, true, true, true, sample), 1U);
    ++sample.shader_hash;
    EXPECT_EQ(receipt.Observe(1, true, true, true, sample), 2U);
    ++sample.resource_index;
    EXPECT_EQ(receipt.Observe(1, true, true, true, sample), 3U);
    ++sample.bound_size;
    EXPECT_EQ(receipt.Observe(1, true, true, true, sample), 4U);
}

TEST(ComputeWriteProvenance, FixedCapacityAndDisableStopAllFurtherRows) {
    ComputeWriteProvenanceReceipt receipt{
        ComputeWriteProvenancePolicy::Parse("1", "0", "1000", "1100")};
    for (std::uint32_t row = 0; row < 32; ++row) {
        auto sample = Sample();
        sample.shader_hash = row;
        EXPECT_EQ(receipt.Observe(0, true, true, true, sample), row);
    }
    EXPECT_FALSE(receipt.Enabled());
    auto sample = Sample();
    sample.shader_hash = 100;
    EXPECT_FALSE(receipt.Observe(0, true, true, true, sample));
    ComputeWriteProvenanceReceipt disabled{
        ComputeWriteProvenancePolicy::Parse("1", "0", "1000", "1100")};
    disabled.Disable();
    EXPECT_FALSE(disabled.Observe(0, true, true, true, sample));
}
} // namespace
#else
TEST(ComputeWriteProvenance, ProductionBoundedPolicyExists) {
    FAIL() << "The production compute-write metadata receipt is not implemented";
}
#endif

#ifdef SHADPS4_INDEX_PROVENANCE_RASTERIZER_SOURCE
namespace {
std::string ReadComputeProvenanceRasterizer() {
    std::ifstream input{SHADPS4_INDEX_PROVENANCE_RASTERIZER_SOURCE};
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

TEST(ComputeWriteProvenanceWiring, LaunchPolicyIsOptInAndUsesExplicitRange) {
    const auto source = ReadComputeProvenanceRasterizer();
    EXPECT_NE(source.find("SHADPS4_COMPUTE_WRITE_PROVENANCE\""), std::string::npos);
    EXPECT_NE(source.find("SHADPS4_COMPUTE_WRITE_PROVENANCE_START_FRAME"), std::string::npos);
    EXPECT_NE(source.find("SHADPS4_COMPUTE_WRITE_PROVENANCE_RANGE_BEGIN"), std::string::npos);
    EXPECT_NE(source.find("SHADPS4_COMPUTE_WRITE_PROVENANCE_RANGE_END"), std::string::npos);
}

TEST(ComputeWriteProvenanceWiring, ReceiptPrecedesExistingNormalBufferPreparation) {
    const auto source = ReadComputeProvenanceRasterizer();
    const auto begin = source.find("void Rasterizer::BindBuffers(");
    const auto end = source.find("bool Rasterizer::FinalizeUtilityBufferBindings(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto observe = body.find("RecordComputeWriteProvenance(stage, i, binding.buffer");
    const auto prepare = body.find("buffer_cache.PrepareBufferBinding(");
    ASSERT_NE(observe, std::string::npos);
    EXPECT_LT(observe, prepare);
    EXPECT_NE(body.find("vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id"),
              std::string::npos);
}

TEST(ComputeWriteProvenanceWiring, DisabledHookDoesNotReadClockMemoryOrScheduler) {
    const auto source = ReadComputeProvenanceRasterizer();
    const auto begin = source.find("void Rasterizer::RecordComputeWriteProvenance(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("\nvoid Rasterizer::BindBuffers(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto disabled = body.find("!compute_write_provenance.Enabled()");
    const auto frame = body.find("GetFrameNum()");
    ASSERT_NE(disabled, std::string::npos);
    ASSERT_NE(frame, std::string::npos);
    EXPECT_LT(disabled, frame);
    EXPECT_NE(body.find("Shader::Stage::Compute"), std::string::npos);
    EXPECT_NE(body.find("ComputeWriteProvenanceV1"), std::string::npos);
    for (const auto forbidden : {"scheduler.", "ReadMemory", "Download", "GetSharp(",
                                 "steady_clock", "system_clock", "Map(", "FindBuffer("}) {
        EXPECT_EQ(body.find(forbidden), std::string::npos) << forbidden;
    }
}
} // namespace
#endif
