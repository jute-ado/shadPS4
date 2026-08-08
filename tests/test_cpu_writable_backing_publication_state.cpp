// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/cpu_writable_backing_publication_state.h"

namespace VideoCore {
namespace {

using State = CpuWritableBackingPublicationState;

TEST(CpuWritableBackingPublicationState,
     PublishesOnlyAfterSynchronousHostVisibilityBeforeSubmission) {
    State state;
    const auto mapping = state.Map(1, 10);
    ASSERT_TRUE(mapping);
    EXPECT_FALSE(state.IsImportedBackingPublished());

    const auto plan = state.PrepareReadSubmission(*mapping, 1);
    ASSERT_TRUE(plan);
    EXPECT_TRUE(plan->needs_host_visibility);

    std::vector<std::string_view> events;
    const auto submission = state.CommitReadSubmission(*plan, [&] {
        EXPECT_FALSE(state.IsImportedBackingPublished());
        events.emplace_back("host-visible");
        return true;
    });
    ASSERT_TRUE(submission);
    events.emplace_back("submitted");
    EXPECT_EQ(events, (std::vector<std::string_view>{"host-visible", "submitted"}));
    EXPECT_TRUE(state.IsImportedBackingPublished());

    bool repeated_visibility = false;
    const auto clean_plan = state.PrepareReadSubmission(*mapping, 2);
    ASSERT_TRUE(clean_plan);
    EXPECT_FALSE(clean_plan->needs_host_visibility);
    const auto clean_submission = state.CommitReadSubmission(*clean_plan, [&] {
        repeated_visibility = true;
        return true;
    });
    ASSERT_TRUE(clean_submission);
    EXPECT_FALSE(repeated_visibility);
}

TEST(CpuWritableBackingPublicationState,
     InFlightReadsRetireBeforeAHostWriteLeaseAndBlockNewSubmissions) {
    State state;
    const auto mapping = state.Map(1, 10);
    ASSERT_TRUE(mapping);
    const auto initial_plan = state.PrepareReadSubmission(*mapping, 7);
    ASSERT_TRUE(initial_plan);
    const auto in_flight = state.CommitReadSubmission(*initial_plan, [] { return true; });
    ASSERT_TRUE(in_flight);

    const auto request = state.RequestHostWrite(*mapping);
    ASSERT_TRUE(request);
    EXPECT_EQ(request->required_submission_generation, 7);
    EXPECT_FALSE(state.IsImportedBackingPublished());
    EXPECT_FALSE(state.AcquireHostWrite(*request));
    EXPECT_FALSE(state.PrepareReadSubmission(*mapping, 8));

    EXPECT_TRUE(state.CompleteSubmission(*in_flight));
    const auto lease = state.AcquireHostWrite(*request);
    ASSERT_TRUE(lease);
    EXPECT_TRUE(state.CommitHostWrite(*lease));
    EXPECT_FALSE(state.IsImportedBackingPublished());

    const auto next_plan = state.PrepareReadSubmission(*mapping, 8);
    ASSERT_TRUE(next_plan);
    EXPECT_TRUE(next_plan->needs_host_visibility);
}

TEST(CpuWritableBackingPublicationState,
     FailedOrThrowingVisibilityCannotPublishAndRemainsRetryable) {
    State state;
    const auto mapping = state.Map(1, 10);
    ASSERT_TRUE(mapping);
    const auto plan = state.PrepareReadSubmission(*mapping, 1);
    ASSERT_TRUE(plan);

    EXPECT_FALSE(state.CommitReadSubmission(*plan, [] { return false; }));
    EXPECT_FALSE(state.IsImportedBackingPublished());

    EXPECT_FALSE(state.CommitReadSubmission(
        *plan, []() -> bool { throw std::runtime_error{"visibility failed"}; }));
    EXPECT_FALSE(state.IsImportedBackingPublished());

    EXPECT_TRUE(state.CommitReadSubmission(*plan, [] { return true; }));
    EXPECT_TRUE(state.IsImportedBackingPublished());
}

TEST(CpuWritableBackingPublicationState,
     GpuWritesAndCacheOwnersSuppressImportedBackingAndInvalidatePreparedReads) {
    State gpu_write;
    const auto gpu_mapping = gpu_write.Map(1, 10);
    ASSERT_TRUE(gpu_mapping);
    const auto gpu_plan = gpu_write.PrepareReadSubmission(*gpu_mapping, 1);
    ASSERT_TRUE(gpu_plan);
    const auto gpu_read = gpu_write.CommitReadSubmission(*gpu_plan, [] { return true; });
    ASSERT_TRUE(gpu_read);
    EXPECT_TRUE(gpu_write.CompleteSubmission(*gpu_read));

    const auto stale_read = gpu_write.PrepareReadSubmission(*gpu_mapping, 2);
    ASSERT_TRUE(stale_read);
    const auto gpu_submission = gpu_write.BeginGpuWriteSubmission(*gpu_mapping, 2);
    ASSERT_TRUE(gpu_submission);
    EXPECT_FALSE(gpu_write.IsImportedBackingPublished());
    EXPECT_FALSE(gpu_write.CommitReadSubmission(*stale_read, [] { return true; }));
    EXPECT_FALSE(gpu_write.RequestHostWrite(*gpu_mapping));

    State cache_owner;
    const auto cache_mapping = cache_owner.Map(1, 10);
    ASSERT_TRUE(cache_mapping);
    const auto cache_plan = cache_owner.PrepareReadSubmission(*cache_mapping, 1);
    ASSERT_TRUE(cache_plan);
    const auto cache_read = cache_owner.CommitReadSubmission(*cache_plan, [] { return true; });
    ASSERT_TRUE(cache_read);
    EXPECT_TRUE(cache_owner.CompleteSubmission(*cache_read));

    const auto stale_cache_read = cache_owner.PrepareReadSubmission(*cache_mapping, 2);
    ASSERT_TRUE(stale_cache_read);
    const auto owner = cache_owner.AcquireCacheOwner(*cache_mapping, 1);
    ASSERT_TRUE(owner);
    EXPECT_FALSE(cache_owner.IsImportedBackingPublished());
    EXPECT_FALSE(cache_owner.CommitReadSubmission(*stale_cache_read, [] { return true; }));
    EXPECT_FALSE(cache_owner.RequestHostWrite(*cache_mapping));
}

TEST(CpuWritableBackingPublicationState,
     RetiredMappingAndStateGenerationsCannotRepublishNewOwnership) {
    State state;
    const auto old_mapping = state.Map(1, 10);
    ASSERT_TRUE(old_mapping);
    const auto old_read = state.PrepareReadSubmission(*old_mapping, 1);
    ASSERT_TRUE(old_read);
    const auto old_write = state.RequestHostWrite(*old_mapping);
    ASSERT_TRUE(old_write);

    EXPECT_TRUE(state.RetireMapping(*old_mapping));
    const auto new_mapping = state.Map(2, 11);
    ASSERT_TRUE(new_mapping);
    EXPECT_FALSE(state.AcquireHostWrite(*old_write));
    EXPECT_FALSE(state.CommitReadSubmission(*old_read, [] { return true; }));
    EXPECT_FALSE(state.PrepareReadSubmission(*old_mapping, 2));

    const auto new_read = state.PrepareReadSubmission(*new_mapping, 2);
    ASSERT_TRUE(new_read);
    EXPECT_TRUE(state.CommitReadSubmission(*new_read, [] { return true; }));
}

TEST(CpuWritableBackingPublicationState, RejectsRetirementWhileGpuOrHostWorkOwnsThePage) {
    State gpu;
    const auto gpu_mapping = gpu.Map(1, 10);
    ASSERT_TRUE(gpu_mapping);
    const auto plan = gpu.PrepareReadSubmission(*gpu_mapping, 1);
    ASSERT_TRUE(plan);
    const auto submission = gpu.CommitReadSubmission(*plan, [] { return true; });
    ASSERT_TRUE(submission);
    EXPECT_FALSE(gpu.RetireMapping(*gpu_mapping));
    EXPECT_TRUE(gpu.CompleteSubmission(*submission));
    EXPECT_TRUE(gpu.RetireMapping(*gpu_mapping));

    State host;
    const auto host_mapping = host.Map(1, 10);
    ASSERT_TRUE(host_mapping);
    const auto request = host.RequestHostWrite(*host_mapping);
    ASSERT_TRUE(request);
    const auto lease = host.AcquireHostWrite(*request);
    ASSERT_TRUE(lease);
    EXPECT_FALSE(host.RetireMapping(*host_mapping));
}

TEST(CpuWritableBackingPublicationState, FailsClosedWhenMonotonicGenerationsExhaust) {
    constexpr u64 Max = std::numeric_limits<u64>::max();
    State state{Max};
    EXPECT_FALSE(state.Map(1, 1));
}

} // namespace
} // namespace VideoCore
