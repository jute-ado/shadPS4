// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <unordered_map>

#include "common/types.h"

namespace VideoCore {

struct CpuWritableBackingMapping {
    u64 mapping_generation{};
    u64 allocation_generation{};

    auto operator<=>(const CpuWritableBackingMapping&) const = default;
};

struct CpuWritableBackingReadPlan {
    CpuWritableBackingMapping mapping{};
    u64 submission_generation{};
    u64 state_generation{};
    bool needs_host_visibility{};

    auto operator<=>(const CpuWritableBackingReadPlan&) const = default;
};

struct CpuWritableBackingSubmission {
    CpuWritableBackingMapping mapping{};
    u64 submission_generation{};
    u64 visibility_generation{};
    bool writes_backing{};

    auto operator<=>(const CpuWritableBackingSubmission&) const = default;
};

struct CpuWritableBackingHostWriteRequest {
    CpuWritableBackingMapping mapping{};
    u64 request_generation{};
    u64 state_generation{};
    u64 required_submission_generation{};

    auto operator<=>(const CpuWritableBackingHostWriteRequest&) const = default;
};

struct CpuWritableBackingHostWriteLease {
    CpuWritableBackingMapping mapping{};
    u64 request_generation{};
    u64 state_generation{};

    auto operator<=>(const CpuWritableBackingHostWriteLease&) const = default;
};

struct CpuWritableBackingCacheOwner {
    CpuWritableBackingMapping mapping{};
    u64 owner_generation{};
    u64 state_generation{};

    auto operator<=>(const CpuWritableBackingCacheOwner&) const = default;
};

/// Pure policy for admitting a CPU-writable imported page to BDA reads.
///
/// The policy deliberately does not perform Vulkan or CPU-fault synchronization. A caller must
/// serialize every method, prevent the guest write until AcquireHostWrite succeeds, perform the
/// write only while holding that lease, and make the host bytes visible synchronously inside the
/// CommitReadSubmission callback. Merely queueing asynchronous visibility work is insufficient.
///
/// A host-write request immediately withdraws imported publication for future submissions. It
/// cannot acquire a write lease until every older submission has retired. GPU writes and cache
/// ownership fail closed and cannot be undone by an older host-write or read-plan token.
class CpuWritableBackingPublicationState {
public:
    explicit CpuWritableBackingPublicationState(u64 last_state_generation_ = 0) noexcept
        : last_state_generation{last_state_generation_} {}

    [[nodiscard]] std::optional<CpuWritableBackingMapping> Map(u64 mapping_generation,
                                                               u64 allocation_generation) {
        if (phase != Phase::Unmapped || mapping_generation == 0 || allocation_generation == 0 ||
            mapping_generation <= last_mapping_generation ||
            allocation_generation <= last_allocation_generation) {
            return std::nullopt;
        }
        const auto next_state = NextGeneration(last_state_generation);
        if (!next_state) {
            return std::nullopt;
        }
        mapping = {
            .mapping_generation = mapping_generation,
            .allocation_generation = allocation_generation,
        };
        last_mapping_generation = mapping_generation;
        last_allocation_generation = allocation_generation;
        last_state_generation = *next_state;
        phase = Phase::HostDirty;
        return mapping;
    }

    /// Invalidates pending plans and requests. Work that may already access the page, an acquired
    /// host-write lease, and cache ownership must be retired by their owners first.
    [[nodiscard]] bool RetireMapping(const CpuWritableBackingMapping& expected) {
        if (!Matches(expected) || !active_submissions.empty() || phase == Phase::HostWriteLeased ||
            phase == Phase::VisibilityCommitting || phase == Phase::CacheOwned) {
            return false;
        }
        const auto next_state = NextGeneration(last_state_generation);
        if (!next_state) {
            return false;
        }
        last_state_generation = *next_state;
        phase = Phase::Unmapped;
        mapping = {};
        pending_host_write.reset();
        return true;
    }

    [[nodiscard]] std::optional<CpuWritableBackingReadPlan> PrepareReadSubmission(
        const CpuWritableBackingMapping& expected, u64 submission_generation) const {
        if (!Matches(expected) || (phase != Phase::HostDirty && phase != Phase::ImportedVisible) ||
            submission_generation == 0 || submission_generation <= last_submission_generation) {
            return std::nullopt;
        }
        return CpuWritableBackingReadPlan{
            .mapping = mapping,
            .submission_generation = submission_generation,
            .state_generation = last_state_generation,
            .needs_host_visibility = phase == Phase::HostDirty,
        };
    }

    /// Commits a prepared read only after the caller's synchronous host-visibility action. The
    /// callback runs while same-page state changes are rejected. False or an exception leaves the
    /// page unpublished and the identical plan retryable.
    template <typename MakeHostVisible>
        requires std::invocable<MakeHostVisible&> &&
                 std::convertible_to<std::invoke_result_t<MakeHostVisible&>, bool>
    [[nodiscard]] std::optional<CpuWritableBackingSubmission> CommitReadSubmission(
        const CpuWritableBackingReadPlan& plan, MakeHostVisible&& make_host_visible) {
        if (!Matches(plan.mapping) || plan.state_generation != last_state_generation ||
            plan.submission_generation == 0 ||
            plan.submission_generation <= last_submission_generation ||
            (phase != Phase::HostDirty && phase != Phase::ImportedVisible) ||
            plan.needs_host_visibility != (phase == Phase::HostDirty)) {
            return std::nullopt;
        }

        const auto next_state = NextGeneration(last_state_generation);
        const auto next_visibility = plan.needs_host_visibility
                                         ? NextGeneration(last_visibility_generation)
                                         : std::optional<u64>{last_visibility_generation};
        if (!next_state || !next_visibility || *next_visibility == 0) {
            return std::nullopt;
        }

        if (plan.needs_host_visibility) {
            phase = Phase::VisibilityCommitting;
            bool visible = false;
            try {
                visible = static_cast<bool>(std::invoke(make_host_visible));
            } catch (...) {
                phase = Phase::HostDirty;
                return std::nullopt;
            }
            if (!visible) {
                phase = Phase::HostDirty;
                return std::nullopt;
            }
            last_visibility_generation = *next_visibility;
            phase = Phase::ImportedVisible;
        }

        const CpuWritableBackingSubmission submission{
            .mapping = mapping,
            .submission_generation = plan.submission_generation,
            .visibility_generation = last_visibility_generation,
            .writes_backing = false,
        };
        if (!active_submissions.emplace(submission.submission_generation, submission).second) {
            return std::nullopt;
        }
        last_submission_generation = submission.submission_generation;
        last_state_generation = *next_state;
        return submission;
    }

    /// Starts an ordered GPU write after imported input has been made visible. Publication is
    /// withdrawn before the caller records the write, and remains suppressed for this mapping.
    [[nodiscard]] std::optional<CpuWritableBackingSubmission> BeginGpuWriteSubmission(
        const CpuWritableBackingMapping& expected, u64 submission_generation) {
        if (!Matches(expected) || phase != Phase::ImportedVisible || submission_generation == 0 ||
            submission_generation <= last_submission_generation) {
            return std::nullopt;
        }
        const auto next_state = NextGeneration(last_state_generation);
        if (!next_state) {
            return std::nullopt;
        }
        const CpuWritableBackingSubmission submission{
            .mapping = mapping,
            .submission_generation = submission_generation,
            .visibility_generation = last_visibility_generation,
            .writes_backing = true,
        };
        if (!active_submissions.emplace(submission_generation, submission).second) {
            return std::nullopt;
        }
        last_submission_generation = submission_generation;
        last_state_generation = *next_state;
        phase = Phase::GpuWriteSuppressed;
        return submission;
    }

    [[nodiscard]] bool CompleteSubmission(const CpuWritableBackingSubmission& submission) {
        if (!Matches(submission.mapping)) {
            return false;
        }
        const auto it = active_submissions.find(submission.submission_generation);
        if (it == active_submissions.end() || it->second != submission) {
            return false;
        }
        active_submissions.erase(it);
        return true;
    }

    /// Withdraws imported publication and blocks CPU writes while a device-local cache owner is
    /// authoritative. Restoration requires a separate ordered writeback policy.
    [[nodiscard]] std::optional<CpuWritableBackingCacheOwner> AcquireCacheOwner(
        const CpuWritableBackingMapping& expected, u64 owner_generation) {
        if (!Matches(expected) || phase != Phase::ImportedVisible || owner_generation == 0 ||
            owner_generation <= last_owner_generation) {
            return std::nullopt;
        }
        const auto next_state = NextGeneration(last_state_generation);
        if (!next_state) {
            return std::nullopt;
        }
        last_owner_generation = owner_generation;
        last_state_generation = *next_state;
        phase = Phase::CacheOwned;
        return CpuWritableBackingCacheOwner{
            .mapping = mapping,
            .owner_generation = owner_generation,
            .state_generation = last_state_generation,
        };
    }

    /// Requests permission for a future CPU write. The guest write must not execute yet.
    [[nodiscard]] std::optional<CpuWritableBackingHostWriteRequest> RequestHostWrite(
        const CpuWritableBackingMapping& expected) {
        if (!Matches(expected) || (phase != Phase::HostDirty && phase != Phase::ImportedVisible)) {
            return std::nullopt;
        }
        const auto next_request = NextGeneration(last_request_generation);
        const auto next_state = NextGeneration(last_state_generation);
        if (!next_request || !next_state) {
            return std::nullopt;
        }
        last_request_generation = *next_request;
        last_state_generation = *next_state;
        phase = Phase::HostWriteBlocked;
        pending_host_write = CpuWritableBackingHostWriteRequest{
            .mapping = mapping,
            .request_generation = last_request_generation,
            .state_generation = last_state_generation,
            .required_submission_generation = MaxActiveSubmissionGeneration(),
        };
        return pending_host_write;
    }

    /// Grants a write lease only after all submissions that could observe the old bytes retired.
    [[nodiscard]] std::optional<CpuWritableBackingHostWriteLease> AcquireHostWrite(
        const CpuWritableBackingHostWriteRequest& request) {
        if (phase != Phase::HostWriteBlocked || !pending_host_write ||
            *pending_host_write != request || !Matches(request.mapping) ||
            !active_submissions.empty()) {
            return std::nullopt;
        }
        const auto next_state = NextGeneration(last_state_generation);
        if (!next_state) {
            return std::nullopt;
        }
        last_state_generation = *next_state;
        phase = Phase::HostWriteLeased;
        return CpuWritableBackingHostWriteLease{
            .mapping = mapping,
            .request_generation = request.request_generation,
            .state_generation = last_state_generation,
        };
    }

    /// Records completion of the actual host write. The page remains unpublished until a later
    /// read submission synchronously makes those bytes visible.
    [[nodiscard]] bool CommitHostWrite(const CpuWritableBackingHostWriteLease& lease) {
        if (phase != Phase::HostWriteLeased || !pending_host_write || !Matches(lease.mapping) ||
            lease.request_generation != pending_host_write->request_generation ||
            lease.state_generation != last_state_generation) {
            return false;
        }
        const auto next_host_write = NextGeneration(last_host_write_generation);
        const auto next_state = NextGeneration(last_state_generation);
        if (!next_host_write || !next_state) {
            return false;
        }
        last_host_write_generation = *next_host_write;
        last_state_generation = *next_state;
        phase = Phase::HostDirty;
        pending_host_write.reset();
        return true;
    }

    [[nodiscard]] bool IsImportedBackingPublished() const noexcept {
        return phase == Phase::ImportedVisible;
    }

private:
    enum class Phase {
        Unmapped,
        HostDirty,
        ImportedVisible,
        HostWriteBlocked,
        HostWriteLeased,
        VisibilityCommitting,
        GpuWriteSuppressed,
        CacheOwned,
    };

    [[nodiscard]] static constexpr std::optional<u64> NextGeneration(u64 current) noexcept {
        if (current == std::numeric_limits<u64>::max()) {
            return std::nullopt;
        }
        return current + 1;
    }

    [[nodiscard]] bool Matches(const CpuWritableBackingMapping& expected) const noexcept {
        return phase != Phase::Unmapped && mapping == expected;
    }

    [[nodiscard]] u64 MaxActiveSubmissionGeneration() const noexcept {
        u64 maximum = 0;
        for (const auto& [generation, submission] : active_submissions) {
            static_cast<void>(submission);
            maximum = std::max(maximum, generation);
        }
        return maximum;
    }

    Phase phase{Phase::Unmapped};
    CpuWritableBackingMapping mapping{};
    std::optional<CpuWritableBackingHostWriteRequest> pending_host_write;
    std::unordered_map<u64, CpuWritableBackingSubmission> active_submissions;
    u64 last_mapping_generation{};
    u64 last_allocation_generation{};
    u64 last_state_generation{};
    u64 last_visibility_generation{};
    u64 last_submission_generation{};
    u64 last_request_generation{};
    u64 last_host_write_generation{};
    u64 last_owner_generation{};
};

} // namespace VideoCore
