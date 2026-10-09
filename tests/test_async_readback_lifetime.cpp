// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

class SchedulerModel {
public:
    using Callback = std::function<void()>;

    class ScopedPendingOperationDrainSuppression {
    public:
        explicit ScopedPendingOperationDrainSuppression(SchedulerModel& scheduler)
            : scheduler{scheduler} {
            ++this->scheduler.suppression_depth;
        }

        ScopedPendingOperationDrainSuppression(const ScopedPendingOperationDrainSuppression&) =
            delete;
        ScopedPendingOperationDrainSuppression& operator=(
            const ScopedPendingOperationDrainSuppression&) = delete;

        ~ScopedPendingOperationDrainSuppression() {
            --scheduler.suppression_depth;
        }

    private:
        SchedulerModel& scheduler;
    };

    void Defer(Callback callback) {
        pending.push_back(std::move(callback));
    }

    void WaitDuringStreamMap() {
        // StreamBuffer::Map can wait for an old watch. Scheduler::Wait flushes the
        // current command buffer, whose owner-side completion path drains callbacks.
        Flush();
    }

    void Flush() {
        PopPendingOperations();
    }

    void PopPendingOperations() {
        if (suppression_depth != 0) {
            return;
        }
        auto callbacks = std::move(pending);
        pending.clear();
        for (auto& callback : callbacks) {
            callback();
        }
    }

    void DrainAfterCopyRecording() {
        PopPendingOperations();
    }

private:
    std::vector<Callback> pending;
    unsigned suppression_depth{};
};

struct BufferModel {
    bool alive = true;
};

struct AsyncReadbackModel {
    SchedulerModel scheduler;
    BufferModel source;
    BufferModel* planned_source{};
    bool copy_recorded_from_live_owner{};

    void ResolveOwnerAndQueueRetirement() {
        planned_source = &source;
        scheduler.Defer([this] { source.alive = false; });
    }

    void MapStagingBuffer() {
        scheduler.WaitDuringStreamMap();
    }

    void RecordCopyCommands() {
        copy_recorded_from_live_owner = planned_source != nullptr && planned_source->alive;
    }

    void RunWithoutProtection() {
        ResolveOwnerAndQueueRetirement();
        MapStagingBuffer();
        RecordCopyCommands();
        scheduler.DrainAfterCopyRecording();
    }

    void RunWithProtection() {
        SchedulerModel::ScopedPendingOperationDrainSuppression guard{scheduler};
        ResolveOwnerAndQueueRetirement();
        MapStagingBuffer();
        RecordCopyCommands();
        scheduler.DrainAfterCopyRecording();
    }
};

struct SrtSnapshotBufferModel {
    struct OwnerIdentity {
        unsigned slot;
        unsigned long long generation;

        bool operator==(const OwnerIdentity&) const = default;
    } owner;
    bool alive = true;
};

struct SrtSnapshotModel {
    SchedulerModel scheduler;
    SrtSnapshotBufferModel original{{7, 11}};
    SrtSnapshotBufferModel replacement{{7, 12}};
    SrtSnapshotBufferModel* current_owner = &original;
    SrtSnapshotBufferModel* planned_source{};
    SrtSnapshotBufferModel::OwnerIdentity planned_owner{};
    bool copy_recorded_from_planned_owner{};
    bool replacement_visible_during_copy{};

    void PlanOverlayAndQueueReplacement() {
        planned_source = current_owner;
        planned_owner = current_owner->owner;
        scheduler.Defer([this] {
            original.alive = false;
            current_owner = &replacement;
        });
    }

    void MapStagingBuffer() {
        scheduler.WaitDuringStreamMap();
    }

    void RecordCopyCommands() {
        replacement_visible_during_copy = current_owner == &replacement;
        copy_recorded_from_planned_owner = planned_source != nullptr && planned_source->alive &&
                                           planned_source->owner == planned_owner;
    }

    void RunWithoutProtection() {
        PlanOverlayAndQueueReplacement();
        MapStagingBuffer();
        RecordCopyCommands();
        scheduler.DrainAfterCopyRecording();
    }

    void RunWithProtection() {
        {
            SchedulerModel::ScopedPendingOperationDrainSuppression guard{scheduler};
            PlanOverlayAndQueueReplacement();
            MapStagingBuffer();
            RecordCopyCommands();
        }
        // The real GPU-owner boundary drains callbacks only after the snapshot has returned.
        scheduler.DrainAfterCopyRecording();
    }
};

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
    for (std::size_t i = open; i < source.size(); ++i) {
        if (source[i] == '{') {
            ++depth;
        } else if (source[i] == '}' && --depth == 0) {
            return source.substr(begin, i - begin + 1);
        }
    }
    ADD_FAILURE() << "Unbalanced function body for " << signature;
    return {};
}

TEST(AsyncReadbackLifetimeModel, UnprotectedMapCharacterizesRawOwnerHazard) {
    AsyncReadbackModel operation;
    operation.RunWithoutProtection();

    // This is the concrete failure: owner resolution leaves a raw Buffer* in the
    // copy plan, then Map can flush and run a deferred slot erase before recording.
    EXPECT_FALSE(operation.copy_recorded_from_live_owner);
}

TEST(AsyncReadbackLifetimeModel, ScopedProtectionDefersRetirementUntilAfterCopy) {
    AsyncReadbackModel operation;
    operation.RunWithProtection();

    EXPECT_TRUE(operation.copy_recorded_from_live_owner);
}

TEST(SrtSnapshotLifetimeModel, ReplacementCannotRetirePlannedSourceBeforeCopy) {
    SrtSnapshotModel unprotected;
    unprotected.RunWithoutProtection();
    EXPECT_FALSE(unprotected.copy_recorded_from_planned_owner);
    EXPECT_TRUE(unprotected.replacement_visible_during_copy);

    SrtSnapshotModel protected_snapshot;
    protected_snapshot.RunWithProtection();
    EXPECT_TRUE(protected_snapshot.copy_recorded_from_planned_owner);
    EXPECT_FALSE(protected_snapshot.replacement_visible_during_copy);
    EXPECT_EQ(protected_snapshot.current_owner, &protected_snapshot.replacement);
}

TEST(AsyncReadbackLifetimeWiring, DownloadKeepsOwnersProtectedAcrossMapAndCopyRecording) {
    const auto source = ReadSource(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto body = FunctionBody(source, "void BufferCache::DownloadBufferMemory(");
    ASSERT_FALSE(body.empty());

    const auto guard = body.find("ScopedPendingOperationDrainSuppression");
    const auto owner_plan = body.find("PlanOwnedDownloadRanges");
    const auto map = body.find("download_buffer.Map");
    const auto copy = body.find("cmdbuf.copyBuffer");
    ASSERT_NE(owner_plan, std::string::npos);
    ASSERT_NE(map, std::string::npos);
    ASSERT_NE(copy, std::string::npos);
    EXPECT_NE(guard, std::string::npos);
    EXPECT_LT(guard, owner_plan);
    EXPECT_LT(owner_plan, map);
    EXPECT_LT(map, copy);
}

TEST(AsyncReadbackLifetimeWiring, SrtSnapshotKeepsPlannedOwnersProtectedAcrossMapAndCopy) {
    const auto source = ReadSource(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto body = FunctionBody(source, "bool BufferCache::DownloadBufferMemorySnapshot(");
    ASSERT_FALSE(body.empty());

    const auto guard = body.find("ScopedPendingOperationDrainSuppression");
    const auto owner_plan = body.find("gpu_modified_ranges.ForEachInRange");
    const auto map = body.find("download_buffer.Map");
    const auto copy = body.find("cmdbuf.copyBuffer");
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(owner_plan, std::string::npos);
    ASSERT_NE(map, std::string::npos);
    ASSERT_NE(copy, std::string::npos);
    EXPECT_LT(guard, owner_plan);
    EXPECT_LT(owner_plan, map);
    EXPECT_LT(map, copy);
}

TEST(AsyncReadbackLifetimeWiring, SchedulerDrainHonorsProtectionScope) {
    const auto source = ReadSource(SHADPS4_SCHEDULER_SOURCE_PATH);
    const auto pop = FunctionBody(source, "void Scheduler::PopPendingOperations()");
    ASSERT_FALSE(pop.empty());
    const auto guard = pop.find("IsPendingOperationDrainSuppressed()");
    ASSERT_NE(guard, std::string::npos);
    EXPECT_LT(guard, pop.find("std::unique_lock"));
}

} // namespace
