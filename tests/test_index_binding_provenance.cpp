// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "video_core/index_binding_provenance.h"

namespace {
using VideoCore::IndexBindingProvenancePolicy;
using VideoCore::IndexBindingProvenanceReceipt;

TEST(IndexBindingProvenance, DefaultOffIgnoresOtherEnvironmentValues) {
    for (const auto enable :
         {std::optional<std::string_view>{}, std::optional<std::string_view>{"0"}}) {
        const auto policy = IndexBindingProvenancePolicy::Parse(enable, "bad", "bad hash");
        EXPECT_TRUE(policy.valid);
        EXPECT_FALSE(policy.enabled);
        IndexBindingProvenanceReceipt receipt{policy};
        EXPECT_FALSE(receipt.Enabled());
        EXPECT_FALSE(receipt.Observe(1000000, 1));
    }
}

TEST(IndexBindingProvenance, ExactOptInDefaultsToFrame470AndAnyHash) {
    const auto policy = IndexBindingProvenancePolicy::Parse("1", std::nullopt, std::nullopt);
    EXPECT_TRUE(policy.valid);
    EXPECT_TRUE(policy.enabled);
    EXPECT_EQ(policy.start_frame, 470U);
    EXPECT_FALSE(policy.vertex_hash);
    IndexBindingProvenanceReceipt receipt{policy};
    EXPECT_FALSE(receipt.Observe(-1, 1));
    EXPECT_FALSE(receipt.Observe(469, 1));
    EXPECT_EQ(receipt.Observe(470, 1), 0U);
    EXPECT_EQ(receipt.Observe(471, 2), 1U);
}

TEST(IndexBindingProvenance, ExactOptionalHashFiltersBeforeConsumingCapacity) {
    for (const auto hash : {"a2a164eb", "0xa2a164eb", "0XA2A164EB"}) {
        const auto policy = IndexBindingProvenancePolicy::Parse("1", "12", hash);
        ASSERT_TRUE(policy.valid);
        IndexBindingProvenanceReceipt receipt{policy};
        for (unsigned index = 0; index < 100; ++index) {
            EXPECT_FALSE(receipt.Observe(12, 0xa2a164ec));
        }
        EXPECT_EQ(receipt.Observe(12, 0xa2a164eb), 0U);
    }
}

TEST(IndexBindingProvenance, MalformedEnableAndWindowFailClosed) {
    for (const auto value : {"", "true", "01", " 1", "1 ", "+1"}) {
        EXPECT_FALSE(IndexBindingProvenancePolicy::Parse(value, {}, {}).valid) << value;
    }
    for (const auto value : {"", "-1", "+1", " 1", "1 ", "470f", "1000001", "4294967296"}) {
        const auto policy = IndexBindingProvenancePolicy::Parse("1", value, {});
        EXPECT_FALSE(policy.valid) << value;
        EXPECT_FALSE(IndexBindingProvenanceReceipt{policy}.Enabled());
    }
    EXPECT_TRUE(IndexBindingProvenancePolicy::Parse("1", "1000000", {}).valid);
}

TEST(IndexBindingProvenance, MalformedHashFailsClosed) {
    for (const auto value : {"", "0x", "-1", "+1", " 1", "1 ", "g", "1,2", "10000000000000000"}) {
        const auto policy = IndexBindingProvenancePolicy::Parse("1", {}, value);
        EXPECT_FALSE(policy.valid) << value;
        EXPECT_FALSE(IndexBindingProvenanceReceipt{policy}.Enabled());
    }
}

TEST(IndexBindingProvenance, HardCapIsSixteenTotalRowsAndDisableIsFinal) {
    IndexBindingProvenanceReceipt receipt{IndexBindingProvenancePolicy::Parse("1", "0", {})};
    for (std::uint32_t row = 0; row < 16; ++row) {
        EXPECT_EQ(receipt.Observe(0, 1), row);
    }
    EXPECT_FALSE(receipt.Enabled());
    EXPECT_FALSE(receipt.Observe(1, 1));
    IndexBindingProvenanceReceipt disabled{IndexBindingProvenancePolicy::Parse("1", "0", {})};
    disabled.Disable();
    EXPECT_FALSE(disabled.Observe(0, 1));
}

TEST(IndexBindingProvenance, DescribesCurrentSpanWithoutTreatingDrawCountAsCapacity) {
    const auto short_draw = VideoCore::DescribeIndexBindingPreparation(3, 4);
    EXPECT_EQ(short_draw.wide_bytes, 12U);
    EXPECT_EQ(short_draw.prepared_bytes, 12U);
    EXPECT_FALSE(short_draw.truncated);
    const auto capacity = VideoCore::DescribeIndexBindingPreparation(65536, 4);
    EXPECT_EQ(capacity.wide_bytes, 262144U);
    EXPECT_EQ(capacity.prepared_bytes, 262144U);
}

TEST(IndexBindingProvenance, WideArithmeticMakesExistingU32TruncationExplicit) {
    const auto value =
        VideoCore::DescribeIndexBindingPreparation(std::numeric_limits<std::uint32_t>::max(), 4);
    EXPECT_EQ(value.wide_bytes, 17179869180ULL);
    EXPECT_EQ(value.prepared_bytes, 4294967292U);
    EXPECT_TRUE(value.truncated);
    EXPECT_EQ(VideoCore::DescribeIndexBindingPreparation(0, 2).wide_bytes, 0U);
}

std::string RendererSource() {
    std::ifstream file{SHADPS4_INDEX_PROVENANCE_RASTERIZER_SOURCE, std::ios::binary};
    EXPECT_TRUE(file.is_open());
    std::string text{std::istreambuf_iterator<char>{file}, {}};
    std::erase(text, '\r');
    return text;
}

TEST(IndexBindingProvenanceWiring, BothIndexedPathsObserveImmediatelyBeforePreparation) {
    const auto text = RendererSource();
    for (const auto marker : {"void Rasterizer::Draw(", "void Rasterizer::DrawIndirect("}) {
        const auto begin = text.find(marker);
        ASSERT_NE(begin, std::string::npos);
        // Direct draws assign the optional preparation to a local while indirect draws
        // return/consume it in the same function.  The contract is the call itself, not
        // a particular spelling of the result handling.
        const auto prepare = text.find("buffer_cache.PrepareIndexBuffer(", begin);
        ASSERT_NE(prepare, std::string::npos);
        const auto record = text.find("RecordIndexBindingProvenance(*pipeline,", begin);
        ASSERT_NE(record, std::string::npos) << marker;
        EXPECT_LT(record, prepare) << marker;
    }
}

TEST(IndexBindingProvenanceWiring, DefaultOffReceiptReadsOnlyExistingRegisterState) {
    const auto text = RendererSource();
    const auto begin = text.find("void Rasterizer::RecordIndexBindingProvenance(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = text.find("\nvoid Rasterizer::", begin + 1);
    ASSERT_NE(end, std::string::npos);
    const auto body = text.substr(begin, end - begin);
    EXPECT_LT(body.find("if (!index_binding_provenance.Enabled())"),
              body.find("DebugState.GetFrameNum("));
    for (const auto field : {"regs.max_index_size", "regs.num_indices", "regs.index_base_address",
                             "IndexBindingProvenanceV1", "indirect_first_index_known=false"}) {
        EXPECT_NE(body.find(field), std::string::npos) << field;
    }
    for (const auto forbidden :
         {"ObtainBuffer(", "ReadMemory(", "Wait(", "Finish(", "DownloadBuffer", "GetBufferData("}) {
        EXPECT_EQ(body.find(forbidden), std::string::npos) << forbidden;
    }
}

} // namespace
