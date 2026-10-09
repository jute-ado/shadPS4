// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

using Byte = std::uint8_t;

struct OwnerIdentity {
    std::uint32_t slot{};
    std::uint64_t generation{};

    bool operator==(const OwnerIdentity&) const = default;
};

struct Owner {
    OwnerIdentity identity;
    std::vector<Byte> bytes;
    bool alive = true;
};

struct CpuWrite {
    std::size_t offset{};
    std::vector<Byte> bytes;
};

struct SnapshotRequest {
    std::size_t owner_index{};
    OwnerIdentity expected_owner;
    std::size_t source_offset{};
    std::size_t size{};
    std::vector<Byte> cpu_before_image;
    std::vector<CpuWrite> pending_cpu_writes;
    std::vector<Byte> output;
    bool published = false;
    bool owner_dirty_cleared = false;
};

// Keep the model deliberately small: the future production batch may choose a different cap,
// but a CPU-only characterization must make its memory and work envelope explicit. Existing
// two-request fixtures consume seven bytes, leaving room to exercise a successful batch while
// still making oversized plans fail before a scheduler submission.
struct SrtSnapshotBatchLimits {
    static constexpr std::size_t MaximumRequests = 4;
    static constexpr std::size_t MaximumStagingBytes = 8;
};

class SchedulerModel {
public:
    bool SubmitStagingBatch() {
        ++submission_count;
        return submit_succeeds;
    }

    bool WaitForStagingBatch(const std::function<void()>& after_wait) {
        ++completion_wait_count;
        if (after_wait) {
            after_wait();
        }
        return wait_succeeds;
    }

    std::size_t submission_count{};
    std::size_t completion_wait_count{};
    bool submit_succeeds = true;
    bool wait_succeeds = true;
};

class BatchedSrtReadbackModel {
public:
    explicit BatchedSrtReadbackModel(std::vector<Owner> source_owners)
        : owners{std::move(source_owners)} {}

    using WaitHook = std::function<void(BatchedSrtReadbackModel&)>;

    bool Execute(std::vector<SnapshotRequest>& requests, SchedulerModel& scheduler,
                 std::optional<std::size_t> copy_failure_index = std::nullopt,
                 WaitHook after_wait = {}) {
        if (requests.size() > SrtSnapshotBatchLimits::MaximumRequests) {
            return false;
        }
        const auto planned_epoch = epoch;
        std::vector<Plan> plans;
        plans.reserve(requests.size());

        // The entire plan is validated before SubmitStagingBatch. This prevents a malformed
        // range or owner from publishing an earlier request in a multi-range operation.
        std::size_t total_size = 0;
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const auto& request = requests[index];
            if (!ValidateRequest(request)) {
                return false;
            }
            if (request.size > SrtSnapshotBatchLimits::MaximumStagingBytes ||
                total_size > SrtSnapshotBatchLimits::MaximumStagingBytes - request.size) {
                return false;
            }
            plans.push_back(Plan{index, request.owner_index, request.expected_owner,
                                 request.source_offset, request.size, total_size});
            total_size += request.size;
        }

        if (!scheduler.SubmitStagingBatch()) {
            return false;
        }

        std::vector<Byte> staging(total_size);
        for (std::size_t plan_index = 0; plan_index < plans.size(); ++plan_index) {
            const auto& plan = plans[plan_index];
            if (copy_failure_index && *copy_failure_index == plan_index) {
                // Submission happened, but an individual copy could not be recorded. Nothing
                // may be published and there is no completion wait for an incomplete batch.
                return false;
            }
            const auto& owner = owners[plan.owner_index];
            std::copy_n(owner.bytes.begin() + static_cast<std::ptrdiff_t>(plan.source_offset),
                        static_cast<std::ptrdiff_t>(plan.size),
                        staging.begin() + static_cast<std::ptrdiff_t>(plan.staging_offset));
        }

        if (!scheduler.WaitForStagingBatch([&] {
                if (after_wait) {
                    after_wait(*this);
                }
            })) {
            return false;
        }

        // A completed staging copy is still unusable if the guest mapping epoch or any source
        // owner changed while the batch was in flight.
        if (epoch != planned_epoch ||
            !std::all_of(plans.begin(), plans.end(), [&](const Plan& plan) {
                const auto& owner = owners[plan.owner_index];
                return owner.alive && owner.identity == plan.expected_owner &&
                       Contains(owner.bytes.size(), plan.source_offset, plan.size);
            })) {
            return false;
        }

        std::vector<Publication> publications;
        publications.reserve(plans.size());
        for (const auto& plan : plans) {
            const auto& request = requests[plan.request_index];
            std::vector<Byte> candidate = request.cpu_before_image;
            std::copy_n(staging.begin() + static_cast<std::ptrdiff_t>(plan.staging_offset),
                        static_cast<std::ptrdiff_t>(plan.size), candidate.begin());

            // CPU writes are reconciled after the GPU bytes are composed. The newest CPU write
            // therefore wins on overlap, without changing neighbouring bytes in this request.
            for (const auto& write : request.pending_cpu_writes) {
                if (!Contains(candidate.size(), write.offset, write.bytes.size())) {
                    return false;
                }
                std::copy(write.bytes.begin(), write.bytes.end(),
                          candidate.begin() + static_cast<std::ptrdiff_t>(write.offset));
            }
            publications.push_back(Publication{plan.request_index, std::move(candidate)});
        }

        // Commit only after every request has passed validation and composition. This is the
        // transaction boundary that prevents partial publication on a later-range failure.
        for (auto& publication : publications) {
            auto& request = requests[publication.request_index];
            request.output = std::move(publication.bytes);
            request.published = true;
            request.owner_dirty_cleared = true;
        }
        return true;
    }

    void InvalidateEpoch() {
        ++epoch;
    }

    void ReplaceOwner(std::size_t owner_index) {
        ASSERT_LT(owner_index, owners.size());
        auto& owner = owners[owner_index];
        owner.alive = true;
        ++owner.identity.generation;
    }

    std::vector<Owner> owners;
    std::uint64_t epoch = 17;

private:
    struct Plan {
        std::size_t request_index;
        std::size_t owner_index;
        OwnerIdentity expected_owner;
        std::size_t source_offset;
        std::size_t size;
        std::size_t staging_offset;
    };

    struct Publication {
        std::size_t request_index;
        std::vector<Byte> bytes;
    };

    static bool Contains(std::size_t total, std::size_t offset, std::size_t size) {
        return offset <= total && size <= total - offset;
    }

    bool ValidateRequest(const SnapshotRequest& request) const {
        if (request.owner_index >= owners.size() || request.size == 0 ||
            request.cpu_before_image.size() != request.size) {
            return false;
        }
        const auto& owner = owners[request.owner_index];
        if (!owner.alive || owner.identity != request.expected_owner ||
            !Contains(owner.bytes.size(), request.source_offset, request.size)) {
            return false;
        }
        return std::all_of(request.pending_cpu_writes.begin(), request.pending_cpu_writes.end(),
                           [&](const CpuWrite& write) {
                               return !write.bytes.empty() &&
                                      Contains(request.size, write.offset, write.bytes.size());
                           });
    }
};

struct SubmissionControl {
    std::size_t submissions{};
    std::size_t waits{};
};

SubmissionControl RunSequentialControl(std::size_t request_count) {
    SubmissionControl control;
    for (std::size_t index = 0; index < request_count; ++index) {
        // This mirrors the current per-request shape: each GPU-composed snapshot owns its own
        // staging submission and completion wait.
        ++control.submissions;
        ++control.waits;
    }
    return control;
}

std::vector<Owner> MakeOwners() {
    return {
        Owner{OwnerIdentity{3, 10}, {0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7}},
        Owner{OwnerIdentity{7, 20}, {0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7}},
    };
}

std::vector<SnapshotRequest> MakeRequests(const std::vector<Owner>& owners) {
    return {
        SnapshotRequest{0,
                        owners[0].identity,
                        1,
                        4,
                        {0x10, 0x11, 0x12, 0x13},
                        {CpuWrite{1, {0xf1, 0xf2}}},
                        {0xcc, 0xcc, 0xcc, 0xcc}},
        SnapshotRequest{1,
                        owners[1].identity,
                        2,
                        3,
                        {0x20, 0x21, 0x22},
                        {},
                        {0xcc, 0xcc, 0xcc}},
    };
}

TEST(SrtSnapshotBatchingModel, SequentialControlNeedsOneSubmissionAndWaitPerRequest) {
    const auto control = RunSequentialControl(2);

    // TDD RED control: the existing implementation has this cost. Once a production batch
    // plan is wired, the integration expectation must become one/one for these two requests.
    EXPECT_EQ(control.submissions, 2U);
    EXPECT_EQ(control.waits, 2U);
}

TEST(SrtSnapshotBatchingModel, BatchSharesSubmissionAndWaitAndKeepsRangeOwnership) {
    auto owners = MakeOwners();
    auto requests = MakeRequests(owners);
    BatchedSrtReadbackModel model{std::move(owners)};
    SchedulerModel scheduler;

    ASSERT_TRUE(model.Execute(requests, scheduler));
    EXPECT_EQ(scheduler.submission_count, 1U);
    EXPECT_EQ(scheduler.completion_wait_count, 1U);
    EXPECT_EQ(requests[0].output, (std::vector<Byte>{0xa1, 0xf1, 0xf2, 0xa4}));
    EXPECT_EQ(requests[1].output, (std::vector<Byte>{0xb2, 0xb3, 0xb4}));
    EXPECT_TRUE(requests[0].published);
    EXPECT_TRUE(requests[1].published);
    EXPECT_TRUE(requests[0].owner_dirty_cleared);
    EXPECT_TRUE(requests[1].owner_dirty_cleared);
}

TEST(SrtSnapshotBatchingModel, CpuWriteReconciliationWinsOnlyWithinItsRange) {
    auto owners = MakeOwners();
    auto requests = MakeRequests(owners);
    requests[0].pending_cpu_writes = {CpuWrite{2, {0xf3}}};
    BatchedSrtReadbackModel model{std::move(owners)};
    SchedulerModel scheduler;

    ASSERT_TRUE(model.Execute(requests, scheduler));
    EXPECT_EQ(requests[0].output, (std::vector<Byte>{0xa1, 0xa2, 0xf3, 0xa4}));
    EXPECT_EQ(requests[1].output, (std::vector<Byte>{0xb2, 0xb3, 0xb4}));
}

TEST(SrtSnapshotBatchingModel, EpochInvalidationAfterWaitFailsTransactionally) {
    auto owners = MakeOwners();
    auto requests = MakeRequests(owners);
    const auto sentinels = std::vector<std::vector<Byte>>{requests[0].output, requests[1].output};
    BatchedSrtReadbackModel model{std::move(owners)};
    SchedulerModel scheduler;

    EXPECT_FALSE(model.Execute(requests, scheduler, std::nullopt,
                               [](BatchedSrtReadbackModel& model) { model.InvalidateEpoch(); }));
    EXPECT_EQ(scheduler.submission_count, 1U);
    EXPECT_EQ(scheduler.completion_wait_count, 1U);
    EXPECT_EQ(requests[0].output, sentinels[0]);
    EXPECT_EQ(requests[1].output, sentinels[1]);
    EXPECT_FALSE(requests[0].published);
    EXPECT_FALSE(requests[1].published);
    EXPECT_FALSE(requests[0].owner_dirty_cleared);
    EXPECT_FALSE(requests[1].owner_dirty_cleared);
}

TEST(SrtSnapshotBatchingModel, ReplacedOwnerFailsAllRangesWithoutPartialPublication) {
    auto owners = MakeOwners();
    auto requests = MakeRequests(owners);
    const auto sentinels = std::vector<std::vector<Byte>>{requests[0].output, requests[1].output};
    BatchedSrtReadbackModel model{std::move(owners)};
    SchedulerModel scheduler;

    EXPECT_FALSE(model.Execute(requests, scheduler, std::nullopt,
                               [](BatchedSrtReadbackModel& model) { model.ReplaceOwner(1); }));
    EXPECT_EQ(requests[0].output, sentinels[0]);
    EXPECT_EQ(requests[1].output, sentinels[1]);
    EXPECT_FALSE(requests[0].published);
    EXPECT_FALSE(requests[1].published);
    EXPECT_FALSE(requests[0].owner_dirty_cleared);
    EXPECT_FALSE(requests[1].owner_dirty_cleared);
}

TEST(SrtSnapshotBatchingModel, InvalidLaterRangeFailsBeforeSubmission) {
    auto owners = MakeOwners();
    auto requests = MakeRequests(owners);
    requests[1].expected_owner.generation++;
    const auto sentinels = std::vector<std::vector<Byte>>{requests[0].output, requests[1].output};
    BatchedSrtReadbackModel model{std::move(owners)};
    SchedulerModel scheduler;

    EXPECT_FALSE(model.Execute(requests, scheduler));
    EXPECT_EQ(scheduler.submission_count, 0U);
    EXPECT_EQ(scheduler.completion_wait_count, 0U);
    EXPECT_EQ(requests[0].output, sentinels[0]);
    EXPECT_EQ(requests[1].output, sentinels[1]);
    EXPECT_FALSE(requests[0].published);
    EXPECT_FALSE(requests[1].published);
}

TEST(SrtSnapshotBatchingModel, CopyFailureDoesNotWaitOrPublishEarlierRanges) {
    auto owners = MakeOwners();
    auto requests = MakeRequests(owners);
    const auto sentinels = std::vector<std::vector<Byte>>{requests[0].output, requests[1].output};
    BatchedSrtReadbackModel model{std::move(owners)};
    SchedulerModel scheduler;

    EXPECT_FALSE(model.Execute(requests, scheduler, 1));
    EXPECT_EQ(scheduler.submission_count, 1U);
    EXPECT_EQ(scheduler.completion_wait_count, 0U);
    EXPECT_EQ(requests[0].output, sentinels[0]);
    EXPECT_EQ(requests[1].output, sentinels[1]);
    EXPECT_FALSE(requests[0].published);
    EXPECT_FALSE(requests[1].published);
    EXPECT_FALSE(requests[0].owner_dirty_cleared);
    EXPECT_FALSE(requests[1].owner_dirty_cleared);
}

TEST(SrtSnapshotBatchingModel, BoundedBatchRejectsTooManyRequestsBeforeSubmission) {
    auto owners = MakeOwners();
    auto requests = MakeRequests(owners);
    // Five tiny ranges are intentionally larger than the candidate bounded batch. The model
    // must reject the plan before creating a staging submission; this keeps the CPU-only test
    // honest about the allocation bound without coupling it to Vulkan.
    requests.push_back(requests[0]);
    requests.push_back(requests[1]);
    requests.push_back(requests[0]);
    const auto sentinels =
        std::vector<std::vector<Byte>>{requests[0].output, requests[1].output, requests[2].output,
                                       requests[3].output, requests[4].output};
    BatchedSrtReadbackModel model{std::move(owners)};
    SchedulerModel scheduler;

    EXPECT_FALSE(model.Execute(requests, scheduler));
    EXPECT_EQ(scheduler.submission_count, 0U);
    EXPECT_EQ(scheduler.completion_wait_count, 0U);
    for (std::size_t index = 0; index < requests.size(); ++index) {
        EXPECT_EQ(requests[index].output, sentinels[index]);
        EXPECT_FALSE(requests[index].published);
        EXPECT_FALSE(requests[index].owner_dirty_cleared);
    }
}

TEST(SrtSnapshotBatchingModel, BoundedBatchRejectsOversizedStagingBeforeSubmission) {
    auto owners = MakeOwners();
    owners[0].bytes.resize(64, 0xee);
    auto requests = MakeRequests(owners);
    requests[0].source_offset = 8;
    requests[0].size = 9;
    requests[0].cpu_before_image.assign(requests[0].size, 0x11);
    requests[0].pending_cpu_writes.clear();
    const auto sentinels =
        std::vector<std::vector<Byte>>{requests[0].output, requests[1].output};
    BatchedSrtReadbackModel model{std::move(owners)};
    SchedulerModel scheduler;

    // The two requests need 12 bytes of CPU-only staging (9 + 3), beyond the model's bounded
    // staging budget. Rejecting before SubmitStagingBatch prevents an unbounded allocation.
    EXPECT_FALSE(model.Execute(requests, scheduler));
    EXPECT_EQ(scheduler.submission_count, 0U);
    EXPECT_EQ(scheduler.completion_wait_count, 0U);
    EXPECT_EQ(requests[0].output, sentinels[0]);
    EXPECT_EQ(requests[1].output, sentinels[1]);
    EXPECT_FALSE(requests[0].published);
    EXPECT_FALSE(requests[1].published);
    EXPECT_FALSE(requests[0].owner_dirty_cleared);
    EXPECT_FALSE(requests[1].owner_dirty_cleared);
}

// This model isolates the remaining ownership question from the batched readback tests above.
// A page snapshot owns bytes captured from one allocation/version, while publication can happen
// after that allocation has been unmapped and its slot rebound.  The production readback path
// already validates owners at its asynchronous completion boundary; this model records the
// stronger invariant needed by the SRT aggregate-refresh path: publication itself must be
// transactional and must reject both an invalidated epoch and a changed owner generation.
struct SnapshotPublicationModel {
    Owner original{OwnerIdentity{7, 41}, {0x11, 0x12, 0x13, 0x14}};
    Owner replacement{OwnerIdentity{7, 42}, {0x21, 0x22, 0x23, 0x24}};
    Owner* current_owner = &original;
    OwnerIdentity captured_owner{};
    std::vector<Byte> captured_bytes;
    std::vector<Byte> published{0xcc, 0xcc, 0xcc, 0xcc};
    bool epoch_valid = true;
    bool owner_dirty_cleared = false;

    void Capture() {
        captured_owner = current_owner->identity;
        captured_bytes = current_owner->bytes;
    }

    void UnmapAndReplace() {
        epoch_valid = false;
        original.alive = false;
        current_owner = &replacement;
    }

    void ReplaceOwnerWithoutEpochNotification() {
        // Keep this separate from UnmapAndReplace: an owner identity check must still reject a
        // same-address slot reuse even if a hypothetical caller failed to propagate the epoch
        // invalidation to the publication boundary.
        original.alive = false;
        current_owner = &replacement;
    }

    // RED witness: this is the unsafe publication shape. It copies the old snapshot and clears
    // ownership without re-checking either mapping epoch or slot generation.
    bool PublishWithoutOwnershipCheck() {
        published = captured_bytes;
        owner_dirty_cleared = true;
        return true;
    }

    // GREEN contract for a future integration seam: all staged bytes and ownership side effects
    // become visible together, only if the exact source identity is still current.
    bool PublishTransactionally() {
        if (!epoch_valid || current_owner == nullptr || !current_owner->alive ||
            current_owner->identity != captured_owner) {
            return false;
        }
        published = captured_bytes;
        owner_dirty_cleared = true;
        return true;
    }
};

TEST(SrtSnapshotOwnershipModel, RedWitnessPublishesStaleBytesAfterUnmapAndRebind) {
    SnapshotPublicationModel model;
    model.Capture();
    model.UnmapAndReplace();

    ASSERT_TRUE(model.PublishWithoutOwnershipCheck());
    EXPECT_EQ(model.published, (std::vector<Byte>{0x11, 0x12, 0x13, 0x14}));
    EXPECT_EQ(model.current_owner, &model.replacement);
    EXPECT_TRUE(model.owner_dirty_cleared);
    EXPECT_NE(model.current_owner->identity, model.captured_owner);
}

TEST(SrtSnapshotOwnershipModel, TransactionalPublicationRejectsStaleGenerationAndPreservesOutput) {
    SnapshotPublicationModel model;
    model.Capture();
    model.UnmapAndReplace();

    const auto sentinel = model.published;
    EXPECT_FALSE(model.PublishTransactionally());
    EXPECT_EQ(model.published, sentinel);
    EXPECT_FALSE(model.owner_dirty_cleared);
    EXPECT_FALSE(model.epoch_valid);
    EXPECT_EQ(model.current_owner->identity, (OwnerIdentity{7, 42}));
}

TEST(SrtSnapshotOwnershipModel, TransactionalPublicationRejectsGenerationChangeWithoutEpochSignal) {
    SnapshotPublicationModel model;
    model.Capture();
    model.ReplaceOwnerWithoutEpochNotification();

    const auto sentinel = model.published;
    EXPECT_FALSE(model.PublishTransactionally());
    EXPECT_EQ(model.published, sentinel);
    EXPECT_FALSE(model.owner_dirty_cleared);
    EXPECT_TRUE(model.epoch_valid);
    EXPECT_NE(model.current_owner->identity, model.captured_owner);
}

TEST(SrtSnapshotOwnershipModel, TransactionalPublicationAcceptsUnchangedOwner) {
    SnapshotPublicationModel model;
    model.Capture();

    ASSERT_TRUE(model.PublishTransactionally());
    EXPECT_EQ(model.published, (std::vector<Byte>{0x11, 0x12, 0x13, 0x14}));
    EXPECT_TRUE(model.owner_dirty_cleared);
}

#if defined(SHADPS4_SRT_BATCHING_SOURCE_PATH)

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.good());
    std::string source{std::istreambuf_iterator<char>{input}, {}};
    std::erase(source, '\r');
    return source;
}

std::string FunctionBody(const std::string& source, std::string_view signature) {
    const auto begin = source.find(signature);
    EXPECT_NE(begin, std::string::npos);
    if (begin == std::string::npos) {
        return {};
    }
    const auto open = source.find('{', begin);
    EXPECT_NE(open, std::string::npos);
    if (open == std::string::npos) {
        return {};
    }
    unsigned depth{};
    for (std::size_t index = open; index < source.size(); ++index) {
        if (source[index] == '{') {
            ++depth;
        } else if (source[index] == '}' && --depth == 0) {
            return source.substr(begin, index - begin + 1);
        }
    }
    ADD_FAILURE() << "Unbalanced function body for " << signature;
    return {};
}

TEST(SrtSnapshotBatchingSource, ExistingSingleReadbackPublishesOnlyAfterFinishAndInvalidation) {
    const auto source = ReadSource(SHADPS4_SRT_BATCHING_SOURCE_PATH);
    const auto body = FunctionBody(source, "bool BufferCache::DownloadBufferMemorySnapshot(");
    ASSERT_FALSE(body.empty());

    const auto publish = body.find("CompletePrivateSnapshotReadback");
    const auto finish = body.find("scheduler.Finish()", publish);
    const auto invalidate = body.find("download_buffer.InvalidateMappedRange", finish);
    ASSERT_NE(finish, std::string::npos);
    ASSERT_NE(invalidate, std::string::npos);
    ASSERT_NE(publish, std::string::npos);
    EXPECT_LT(publish, finish);
    EXPECT_LT(finish, invalidate);
}

#if defined(SHADPS4_SRT_BATCHING_INTEGRATION_WIRING)

TEST(SrtSnapshotBatchingSource, ProductionBatchIntegrationMustMoveFinishOutsidePerRequestBody) {
    const auto source = ReadSource(SHADPS4_SRT_BATCHING_SOURCE_PATH);
    const auto body = FunctionBody(source, "bool BufferCache::DownloadBufferMemorySnapshot(");
    ASSERT_FALSE(body.empty());

    // This is the intentional TDD RED assertion. Enable the integration macro only when a
    // production SrtSnapshotBatchPlan exists and its one-submit/one-wait boundary is ready.
    EXPECT_NE(source.find("SrtSnapshotBatchPlan"), std::string::npos);
    EXPECT_EQ(body.find("scheduler.Finish()"), std::string::npos);
}

#endif
#endif

} // namespace
