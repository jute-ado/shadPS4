// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>

#include "common/types.h"

namespace AmdGpu {

enum class IndirectArgsKind : u8 {
    Draw,
    DrawIndexed,
    Dispatch,
};

struct IndirectArgsFingerprintSnapshot {
    bool should_report{};
    bool matches_previous_frame{};
    u64 sequence{};
    u64 combined_hash{};
    u64 changed_invocation_mask{};
    u32 invocations{};
    u32 argument_records{};
    u32 bytes_hashed{};
    u32 truncated_invocations{};
    u32 truncated_argument_records{};
};

class IndirectArgsFingerprintDiagnostic {
public:
    static constexpr u32 MaxInvocations = 64;
    static constexpr u32 MaxBytesPerFrame = 64_KB;

    explicit IndirectArgsFingerprintDiagnostic(u64 report_limit_) : report_limit{report_limit_} {}

    void Record(IndirectArgsKind kind, const void* base, u32 stride, u32 count,
                u32 command_size) noexcept {
        if (current_invocations >= MaxInvocations) {
            ++truncated_invocations;
            return;
        }

        u64 invocation_hash = EmptyHash();
        MixValue(invocation_hash, kind);
        MixValue(invocation_hash, stride);
        MixValue(invocation_hash, count);
        MixValue(invocation_hash, command_size);

        const u32 remaining_bytes = MaxBytesPerFrame - bytes_hashed;
        const u32 records_to_hash =
            command_size == 0 ? 0 : std::min(count, remaining_bytes / command_size);
        const auto* bytes = static_cast<const u8*>(base);
        for (u32 record = 0; record < records_to_hash; ++record) {
            MixBytes(invocation_hash, bytes + static_cast<size_t>(record) * stride, command_size);
        }

        invocation_hashes[current_invocations++] = invocation_hash;
        MixValue(combined_hash, invocation_hash);
        argument_records += records_to_hash;
        bytes_hashed += records_to_hash * command_size;
        truncated_argument_records += count - records_to_hash;
    }

    [[nodiscard]] IndirectArgsFingerprintSnapshot TakeSnapshot() noexcept {
        if (reports_emitted >= report_limit) {
            return {};
        }

        u64 changed_mask{};
        const u32 compared_count = std::max(current_invocations, previous_invocations);
        for (u32 invocation = 0; invocation < compared_count; ++invocation) {
            const bool present_now = invocation < current_invocations;
            const bool present_before = has_previous && invocation < previous_invocations;
            if (!present_now || !present_before ||
                invocation_hashes[invocation] != previous_hashes[invocation]) {
                changed_mask |= 1ULL << invocation;
            }
        }

        const bool matches_previous =
            has_previous && current_invocations == previous_invocations && changed_mask == 0;
        const IndirectArgsFingerprintSnapshot snapshot{
            .should_report = true,
            .matches_previous_frame = matches_previous,
            .sequence = ++reports_emitted,
            .combined_hash = combined_hash,
            .changed_invocation_mask = changed_mask,
            .invocations = current_invocations,
            .argument_records = argument_records,
            .bytes_hashed = bytes_hashed,
            .truncated_invocations = truncated_invocations,
            .truncated_argument_records = truncated_argument_records,
        };

        previous_hashes = invocation_hashes;
        previous_invocations = current_invocations;
        has_previous = true;
        ResetCurrent();
        return snapshot;
    }

    [[nodiscard]] static constexpr u64 EmptyHash() noexcept {
        return 14695981039346656037ULL;
    }

private:
    static void MixBytes(u64& hash, const void* data, size_t size) noexcept {
        const auto* bytes = static_cast<const u8*>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    }

    template <typename T>
    static void MixValue(u64& hash, const T& value) noexcept {
        MixBytes(hash, &value, sizeof(value));
    }

    void ResetCurrent() noexcept {
        invocation_hashes = {};
        combined_hash = EmptyHash();
        current_invocations = 0;
        argument_records = 0;
        bytes_hashed = 0;
        truncated_invocations = 0;
        truncated_argument_records = 0;
    }

    const u64 report_limit;
    u64 reports_emitted{};
    std::array<u64, MaxInvocations> invocation_hashes{};
    std::array<u64, MaxInvocations> previous_hashes{};
    u64 combined_hash{EmptyHash()};
    u32 current_invocations{};
    u32 previous_invocations{};
    u32 argument_records{};
    u32 bytes_hashed{};
    u32 truncated_invocations{};
    u32 truncated_argument_records{};
    bool has_previous{};
};

} // namespace AmdGpu
