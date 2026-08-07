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
    bool shape_matches_previous_frame{};
    bool location_matches_previous_frame{};
    u64 sequence{};
    u64 combined_hash{};
    u64 shape_combined_hash{};
    u64 location_combined_hash{};
    u32 draws{};
    u32 descriptors{};
    u32 bytes_hashed{};
    u32 changed_draws{};
    u32 reported_changed_draws{};
    std::array<u32, MaxReportedChangedDraws> first_changed_draw_ordinals{};
    u32 changed_shape_draws{};
    u32 reported_changed_shape_draws{};
    std::array<u32, MaxReportedChangedDraws> first_changed_shape_draw_ordinals{};
    u32 changed_location_draws{};
    u32 reported_changed_location_draws{};
    std::array<u32, MaxReportedChangedDraws> first_changed_location_draw_ordinals{};
    u32 location_aba_return_draws{};
    u32 reported_location_aba_return_draws{};
    std::array<u32, MaxReportedChangedDraws> first_location_aba_return_draw_ordinals{};
    u32 changed_host_identity_draws{};
    u32 reported_changed_host_identity_draws{};
    std::array<u32, MaxReportedChangedDraws> first_changed_host_identity_draw_ordinals{};
    u32 host_identity_aba_return_draws{};
    u32 reported_host_identity_aba_return_draws{};
    std::array<u32, MaxReportedChangedDraws> first_host_identity_aba_return_draw_ordinals{};
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
        current_draw_shape_hash = EmptyHash();
        current_draw_location_hash = EmptyHash();
        current_draw_host_identity_hash = EmptyHash();
        current_draw_descriptors = 0;
        current_draw_host_identities = 0;
        if (!accept_draw) {
            ++truncated_draws;
        }
    }

    void RecordDescriptor(DrawResourceDescriptorKind kind, const void* descriptor,
                          size_t size) noexcept {
        RecordDescriptor(kind, descriptor, size, descriptor, size, descriptor, size);
    }

    void RecordDescriptor(DrawResourceDescriptorKind kind, const void* descriptor, size_t size,
                          const void* shape, size_t shape_size) noexcept {
        RecordDescriptor(kind, descriptor, size, shape, shape_size, descriptor, size);
    }

    void RecordDescriptor(DrawResourceDescriptorKind kind, const void* descriptor, size_t size,
                          const void* shape, size_t shape_size, const void* location,
                          size_t location_size) noexcept {
        if (!draw_active || !accept_draw) {
            return;
        }
        if (current_draw_descriptors >= MaxDescriptorsPerDraw) {
            ++truncated_descriptors;
            truncated_bytes += size;
            return;
        }
        if (size > MaxBytesPerFrame - bytes_hashed ||
            shape_size > MaxBytesPerFrame - shape_bytes_hashed) {
            ++truncated_descriptors;
            truncated_bytes += size + shape_size;
            return;
        }

        MixValue(current_draw_hash, kind);
        MixValue(current_draw_hash, size);
        MixBytes(current_draw_hash, descriptor, size);
        MixValue(current_draw_shape_hash, kind);
        MixValue(current_draw_shape_hash, shape_size);
        MixBytes(current_draw_shape_hash, shape, shape_size);
        MixValue(current_draw_location_hash, kind);
        MixValue(current_draw_location_hash, location_size);
        MixBytes(current_draw_location_hash, location, location_size);
        ++current_draw_descriptors;
        ++descriptors;
        bytes_hashed += static_cast<u32>(size);
        shape_bytes_hashed += static_cast<u32>(shape_size);
    }

    void RecordHostIdentity(u32 binding_ordinal, u32 slot, u64 uid, u64 backing) noexcept {
        if (!draw_active || !accept_draw) {
            return;
        }
        MixValue(current_draw_host_identity_hash, binding_ordinal);
        MixValue(current_draw_host_identity_hash, slot);
        MixValue(current_draw_host_identity_hash, uid);
        MixValue(current_draw_host_identity_hash, backing);
        ++current_draw_host_identities;
    }

    void EndDraw() noexcept {
        if (!draw_active) {
            return;
        }
        if (accept_draw) {
            MixValue(current_draw_hash, current_draw_descriptors);
            MixValue(current_draw_shape_hash, current_draw_descriptors);
            MixValue(current_draw_location_hash, current_draw_descriptors);
            current_draw_hashes[current_draws++] = current_draw_hash;
            current_draw_shape_hashes[current_draws - 1] = current_draw_shape_hash;
            current_draw_location_hashes[current_draws - 1] = current_draw_location_hash;
            MixValue(current_draw_host_identity_hash, current_draw_host_identities);
            current_draw_host_identity_hashes[current_draws - 1] =
                current_draw_host_identity_hash;
            MixValue(combined_hash, current_draw_hash);
            MixValue(shape_combined_hash, current_draw_shape_hash);
            MixValue(location_combined_hash, current_draw_location_hash);
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
            .shape_combined_hash = shape_combined_hash,
            .location_combined_hash = location_combined_hash,
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
        for (u32 draw = 0; draw < compared_draws; ++draw) {
            const bool present_now = draw < current_draws;
            const bool present_before = has_previous && draw < previous_draws;
            if (present_now && present_before &&
                current_draw_host_identity_hashes[draw] ==
                    previous_draw_host_identity_hashes[draw]) {
                continue;
            }
            ++snapshot.changed_host_identity_draws;
            if (snapshot.reported_changed_host_identity_draws <
                DrawResourceFingerprintSnapshot::MaxReportedChangedDraws) {
                snapshot.first_changed_host_identity_draw_ordinals
                    [snapshot.reported_changed_host_identity_draws++] = draw;
            }
        }
        for (u32 draw = 0; draw < compared_draws; ++draw) {
            const bool present_now = draw < current_draws;
            const bool present_before = has_previous && draw < previous_draws;
            if (present_now && present_before &&
                current_draw_location_hashes[draw] == previous_draw_location_hashes[draw]) {
                continue;
            }
            ++snapshot.changed_location_draws;
            if (snapshot.reported_changed_location_draws <
                DrawResourceFingerprintSnapshot::MaxReportedChangedDraws) {
                snapshot.first_changed_location_draw_ordinals
                    [snapshot.reported_changed_location_draws++] = draw;
            }
        }
        const u32 aba_compared_draws = std::max(current_draws, previous_previous_draws);
        for (u32 draw = 0; draw < aba_compared_draws; ++draw) {
            const bool present_in_all = draw < current_draws && has_previous &&
                                        draw < previous_draws && has_previous_previous &&
                                        draw < previous_previous_draws;
            if (!present_in_all ||
                current_draw_location_hashes[draw] !=
                    previous_previous_draw_location_hashes[draw] ||
                current_draw_location_hashes[draw] == previous_draw_location_hashes[draw]) {
                continue;
            }
            ++snapshot.location_aba_return_draws;
            if (snapshot.reported_location_aba_return_draws <
                DrawResourceFingerprintSnapshot::MaxReportedChangedDraws) {
                snapshot.first_location_aba_return_draw_ordinals
                    [snapshot.reported_location_aba_return_draws++] = draw;
            }
        }
        for (u32 draw = 0; draw < aba_compared_draws; ++draw) {
            const bool present_in_all = draw < current_draws && has_previous &&
                                        draw < previous_draws && has_previous_previous &&
                                        draw < previous_previous_draws;
            if (!present_in_all ||
                current_draw_host_identity_hashes[draw] !=
                    previous_previous_draw_host_identity_hashes[draw] ||
                current_draw_host_identity_hashes[draw] ==
                    previous_draw_host_identity_hashes[draw]) {
                continue;
            }
            ++snapshot.host_identity_aba_return_draws;
            if (snapshot.reported_host_identity_aba_return_draws <
                DrawResourceFingerprintSnapshot::MaxReportedChangedDraws) {
                snapshot.first_host_identity_aba_return_draw_ordinals
                    [snapshot.reported_host_identity_aba_return_draws++] = draw;
            }
        }
        for (u32 draw = 0; draw < compared_draws; ++draw) {
            const bool present_now = draw < current_draws;
            const bool present_before = has_previous && draw < previous_draws;
            if (present_now && present_before &&
                current_draw_shape_hashes[draw] == previous_draw_shape_hashes[draw]) {
                continue;
            }
            ++snapshot.changed_shape_draws;
            if (snapshot.reported_changed_shape_draws <
                DrawResourceFingerprintSnapshot::MaxReportedChangedDraws) {
                snapshot.first_changed_shape_draw_ordinals
                    [snapshot.reported_changed_shape_draws++] = draw;
            }
        }
        snapshot.matches_previous_frame =
            has_previous && current_draws == previous_draws && snapshot.changed_draws == 0;
        snapshot.shape_matches_previous_frame = has_previous && current_draws == previous_draws &&
                                                snapshot.changed_shape_draws == 0;
        snapshot.location_matches_previous_frame = has_previous && current_draws == previous_draws &&
                                                   snapshot.changed_location_draws == 0;

        previous_previous_draw_location_hashes = previous_draw_location_hashes;
        previous_previous_draw_host_identity_hashes = previous_draw_host_identity_hashes;
        previous_previous_draws = previous_draws;
        has_previous_previous = has_previous;
        previous_draw_hashes = current_draw_hashes;
        previous_draw_shape_hashes = current_draw_shape_hashes;
        previous_draw_location_hashes = current_draw_location_hashes;
        previous_draw_host_identity_hashes = current_draw_host_identity_hashes;
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
        current_draw_shape_hashes = {};
        current_draw_location_hashes = {};
        current_draw_host_identity_hashes = {};
        combined_hash = EmptyHash();
        shape_combined_hash = EmptyHash();
        location_combined_hash = EmptyHash();
        current_draw_hash = EmptyHash();
        current_draw_shape_hash = EmptyHash();
        current_draw_location_hash = EmptyHash();
        current_draw_host_identity_hash = EmptyHash();
        current_draws = 0;
        descriptors = 0;
        bytes_hashed = 0;
        shape_bytes_hashed = 0;
        current_draw_descriptors = 0;
        current_draw_host_identities = 0;
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
    std::array<u64, MaxDrawsPerFrame> current_draw_shape_hashes{};
    std::array<u64, MaxDrawsPerFrame> previous_draw_shape_hashes{};
    std::array<u64, MaxDrawsPerFrame> current_draw_location_hashes{};
    std::array<u64, MaxDrawsPerFrame> previous_draw_location_hashes{};
    std::array<u64, MaxDrawsPerFrame> previous_previous_draw_location_hashes{};
    std::array<u64, MaxDrawsPerFrame> current_draw_host_identity_hashes{};
    std::array<u64, MaxDrawsPerFrame> previous_draw_host_identity_hashes{};
    std::array<u64, MaxDrawsPerFrame> previous_previous_draw_host_identity_hashes{};
    u64 combined_hash{EmptyHash()};
    u64 shape_combined_hash{EmptyHash()};
    u64 location_combined_hash{EmptyHash()};
    u64 current_draw_hash{EmptyHash()};
    u64 current_draw_shape_hash{EmptyHash()};
    u64 current_draw_location_hash{EmptyHash()};
    u64 current_draw_host_identity_hash{EmptyHash()};
    u64 truncated_bytes{};
    u32 current_draws{};
    u32 previous_draws{};
    u32 previous_previous_draws{};
    u32 descriptors{};
    u32 bytes_hashed{};
    u32 shape_bytes_hashed{};
    u32 current_draw_descriptors{};
    u32 current_draw_host_identities{};
    u32 truncated_draws{};
    u32 truncated_descriptors{};
    bool draw_active{};
    bool accept_draw{};
    bool has_previous{};
    bool has_previous_previous{};
};

} // namespace AmdGpu
