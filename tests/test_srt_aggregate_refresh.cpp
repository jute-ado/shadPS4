// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>
#include <sstream>
#include <utility>

#include <gtest/gtest.h>

namespace VideoCore {
namespace {

// This small model represents the cached aggregate used while selecting a shader module.  The
// first two words are an already-published root/child pair and the remaining words are another
// child range plus unrelated state.  A reader can invalidate the epoch after the first child is
// copied, which is the failure shape seen in aggregate_refresh diagnostics.
struct AggregateRefreshModel {
    static constexpr std::array<unsigned, 4> Initial{
        0x1000U, // root
        0x1111U, // child range A
        0x2222U, // child range B
        0x3333U, // unrelated cached state
    };

    std::array<unsigned, 4> published = Initial;
    std::array<unsigned, 3> refreshed{
        0xAAAAU,
        0xBBBBU,
        0xCCCCU,
    };
    unsigned published_binding = 7U;
    unsigned refreshed_binding = 9U;
    unsigned reads{};
    bool epoch_valid = true;

    bool ReadChild(unsigned index, unsigned& destination) {
        if (!epoch_valid || index >= refreshed.size()) {
            return false;
        }
        destination = refreshed[index];
        ++reads;
        // Invalidation occurs between child-range reads: range A was copied, range B is refused.
        if (reads == 1) {
            epoch_valid = false;
        }
        return true;
    }

    // RED witness: this is the unsafe operation that publishes a mixed epoch when the second
    // child read is refused.  The test below deliberately records that mixed state so the hazard
    // remains executable and reviewable without leaving a permanently failing target.
    bool RefreshInPlace() {
        published_binding = refreshed_binding;
        if (!ReadChild(0, published[1])) {
            return false;
        }
        if (!ReadChild(1, published[2])) {
            return false;
        }
        if (!ReadChild(2, published[3])) {
            return false;
        }
        return epoch_valid;
    }

    // GREEN candidate: stage every child range, then publish one complete aggregate only after
    // the reader's epoch remains valid.  A failed attempt leaves the previously published state
    // and its binding identity untouched.
    bool RefreshTransactionally() {
        auto candidate = published;
        const auto candidate_binding = refreshed_binding;
        for (unsigned index = 0; index < refreshed.size(); ++index) {
            if (!ReadChild(index, candidate[index + 1])) {
                return false;
            }
        }
        if (!epoch_valid) {
            return false;
        }
        published = std::move(candidate);
        published_binding = candidate_binding;
        return true;
    }
};

TEST(SrtAggregateRefreshModel, InPlaceRedWitnessExposesMixedEpochAfterChildInvalidation) {
    AggregateRefreshModel model;

    EXPECT_FALSE(model.RefreshInPlace());
    EXPECT_EQ(model.reads, 1U);
    EXPECT_FALSE(model.epoch_valid);
    EXPECT_EQ(model.published_binding, 9U);
    // This is the bug shape: child A is from the new epoch while child B remains stale.
    EXPECT_EQ(model.published, (std::array<unsigned, 4>{0x1000U, 0xAAAAU, 0x2222U, 0x3333U}));
}

TEST(SrtAggregateRefreshModel, TransactionalRefreshRejectsPartialChildSnapshot) {
    AggregateRefreshModel model;
    const auto old_published = model.published;

    EXPECT_FALSE(model.RefreshTransactionally());
    EXPECT_EQ(model.reads, 1U);
    EXPECT_FALSE(model.epoch_valid);
    EXPECT_EQ(model.published_binding, 7U);
    EXPECT_EQ(model.published, old_published);
}

TEST(SrtAggregateRefreshSource, CachedAggregateIsStagedAndSealedBeforePublication) {
    std::ifstream file{SHADPS4_PIPELINE_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(file.is_open());
    std::ostringstream contents;
    contents << file.rdbuf();
    const auto source = contents.str();

    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    const auto function_end = source.find("PipelineCache::ReplaceShader(", get_program);
    ASSERT_NE(get_program, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(get_program, function_end - get_program);

    const auto aggregate = body.find("auto aggregate_snapshot = program->info");
    const auto refresh_guard =
        body.find("if (!aggregate_snapshot.RefreshFlatBuf(reader))", aggregate);
    const auto refresh_failure =
        body.find("return fail_snapshot(\"aggregate_refresh\"", refresh_guard);
    const auto cached_publish = body.find("*probe->info = std::move(*selected_info)", aggregate);
    ASSERT_NE(aggregate, std::string::npos);
    ASSERT_NE(refresh_guard, std::string::npos);
    ASSERT_NE(refresh_failure, std::string::npos);
    ASSERT_NE(cached_publish, std::string::npos);

    // The refresh must operate on a value copy. An invalid child range can therefore not mutate
    // the published module while the caller is still deciding whether to reuse it.
    EXPECT_LT(aggregate, refresh_guard);
    EXPECT_LT(refresh_failure, cached_publish);

    // A successful cached selection still needs to finish the epoch before replacing module
    // metadata. This guards against a future refactor that moves publication above the seal.
    const auto cached_seal = body.substr(0, cached_publish).rfind("if (!seal_snapshot())");
    ASSERT_NE(cached_seal, std::string::npos);
    EXPECT_LT(cached_seal, cached_publish);
    EXPECT_NE(body.substr(cached_seal, cached_publish - cached_seal).find("cached_seal"),
              std::string::npos);

    // The retry helper must own the attempt-local binding. A failed epoch may not consume the
    // caller's current binding before a later attempt succeeds.
    std::ifstream retry_file{SHADPS4_SRT_PREPARATION_RETRY_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(retry_file.is_open());
    std::ostringstream retry_contents;
    retry_contents << retry_file.rdbuf();
    const auto retry_source = retry_contents.str();
    EXPECT_NE(retry_source.find("auto attempted_binding = binding"), std::string::npos);
    EXPECT_NE(retry_source.find("binding = attempted_binding"), std::string::npos);
}

} // namespace
} // namespace VideoCore
