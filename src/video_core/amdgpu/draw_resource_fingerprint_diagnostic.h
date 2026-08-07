// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "common/types.h"

namespace AmdGpu {

enum class DrawResourceDescriptorKind : u8 {
    Buffer,
    Image,
    Sampler,
    FMask,
    Vertex,
};

struct DrawResourceFingerprintSnapshot {
    static constexpr u32 MaxReportedChangedDraws = 16;

    bool should_report{};
    bool matches_previous_frame{};
    u64 sequence{};
    u64 combined_hash{};
    u32 draws{};
    u32 descriptors{};
    u32 bytes_hashed{};
    u32 changed_draws{};
    u32 reported_changed_draws{};
    std::array<u32, MaxReportedChangedDraws> first_changed_draw_ordinals{};
    u32 truncated_draws{};
    u32 truncated_descriptors{};
    u64 truncated_bytes{};
};

/**
 * Bounded diagnostic fingerprint of the immutable descriptor values consumed by ordered draws.
 * It intentionally excludes resource contents and raw guest addresses from its reports.
 */
class DrawResourceFingerprintDiagnostic {
public:
    static constexpr u32 MaxDrawsPerFrame = 2048;
    static constexpr u32 MaxDescriptorsPerDraw = 128;
    static constexpr u32 MaxBytesPerFrame = 2_MB;

    explicit DrawResourceFingerprintDiagnostic(u64 report_limit_) : report_limit{report_limit_} {}

    void BeginDraw() noexcept {
        if (draw_active) {
            EndDraw();
        }
        draw_active = true;
        accept_draw = current_draws < MaxDrawsPerFrame;
        current_draw_hash = EmptyHash();
        current_draw_descriptors = 0;
        if (!accept_draw) {
            ++truncated_draws;
        }
    }

    void RecordDescriptor(DrawResourceDescriptorKind kind, const void* descriptor,
                          size_t size) noexcept {
        if (!draw_active || !accept_draw) {
            return;
        }
        if (current_draw_descriptors >= MaxDescriptorsPerDraw) {
            ++truncated_descriptors;
            truncated_bytes += size;
            return;
        }
        if (size > MaxBytesPerFrame - bytes_hashed) {
            ++truncated_descriptors;
            truncated_bytes += size;
            return;
        }

        MixValue(current_draw_hash, kind);
        MixValue(current_draw_hash, size);
        MixBytes(current_draw_hash, descriptor, size);
        ++current_draw_descriptors;
        ++descriptors;
        bytes_hashed += static_cast<u32>(size);
    }

    void EndDraw() noexcept {
        if (!draw_active) {
            return;
        }
        if (accept_draw) {
            MixValue(current_draw_hash, current_draw_descriptors);
            current_draw_hashes[current_draws++] = current_draw_hash;
            MixValue(combined_hash, current_draw_hash);
        }
        draw_active = false;
        accept_draw = false;
    }

    [[nodiscard]] DrawResourceFingerprintSnapshot TakeSnapshot() noexcept {
        EndDraw();
        if (reports_emitted >= report_limit) {
            ResetCurrent();
            return {};
        }

        DrawResourceFingerprintSnapshot snapshot{
            .should_report = true,
            .sequence = ++reports_emitted,
            .combined_hash = combined_hash,
            .draws = current_draws,
            .descriptors = descriptors,
            .bytes_hashed = bytes_hashed,
            .truncated_draws = truncated_draws,
            .truncated_descriptors = truncated_descriptors,
            .truncated_bytes = truncated_bytes,
        };

        const u32 compared_draws = std::max(current_draws, previous_draws);
        for (u32 draw = 0; draw < compared_draws; ++draw) {
            const bool present_now = draw < current_draws;
            const bool present_before = has_previous && draw < previous_draws;
            if (present_now && present_before &&
                current_draw_hashes[draw] == previous_draw_hashes[draw]) {
                continue;
            }
            ++snapshot.changed_draws;
            if (snapshot.reported_changed_draws <
                DrawResourceFingerprintSnapshot::MaxReportedChangedDraws) {
                snapshot.first_changed_draw_ordinals[snapshot.reported_changed_draws++] = draw;
            }
        }
        snapshot.matches_previous_frame =
            has_previous && current_draws == previous_draws && snapshot.changed_draws == 0;

        previous_draw_hashes = current_draw_hashes;
        previous_draws = current_draws;
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
        current_draw_hashes = {};
        combined_hash = EmptyHash();
        current_draw_hash = EmptyHash();
        current_draws = 0;
        descriptors = 0;
        bytes_hashed = 0;
        current_draw_descriptors = 0;
        truncated_draws = 0;
        truncated_descriptors = 0;
        truncated_bytes = 0;
        draw_active = false;
        accept_draw = false;
    }

    const u64 report_limit;
    u64 reports_emitted{};
    std::array<u64, MaxDrawsPerFrame> current_draw_hashes{};
    std::array<u64, MaxDrawsPerFrame> previous_draw_hashes{};
    u64 combined_hash{EmptyHash()};
    u64 current_draw_hash{EmptyHash()};
    u64 truncated_bytes{};
    u32 current_draws{};
    u32 previous_draws{};
    u32 descriptors{};
    u32 bytes_hashed{};
    u32 current_draw_descriptors{};
    u32 truncated_draws{};
    u32 truncated_descriptors{};
    bool draw_active{};
    bool accept_draw{};
    bool has_previous{};
};

} // namespace AmdGpu
