// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <optional>
#include <vector>
#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtRangeRead = bool PS4_SYSV_ABI (*)(void* /*context*/, VAddr /*address*/,
                                               void* /*output*/, u64 /*size*/);

struct SrtMemoryReader {
    PFN_SrtRangeRead callback{};
    void* context{};
    const bool* read_failed{};

    [[nodiscard]] bool Valid() const noexcept {
        return read_failed == nullptr || !*read_failed;
    }
};

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/,
                                            PFN_SrtRangeRead /*read_callback*/,
                                            void* /*read_context*/);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    struct SrtRangeReservation {
        u32 pointer_lo_flat_index{};
        u32 source_offset_dw{};
        u32 destination_flat_offset_dw{};
        u32 span_dw{};

        bool operator==(const SrtRangeReservation&) const noexcept = default;
    };

    static constexpr u32 MaxRangeReservations = 64;
    static constexpr u32 MaxRangeSnapshotDwords = 1U << 20;

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs
    std::vector<SrtRangeReservation> range_reservations;

    [[nodiscard]] std::optional<u32> FindReservedRange(u32 pointer_lo_flat_index,
                                                       u32 source_offset_dw,
                                                       u32 span_dw) const noexcept {
        if (span_dw == 0 || source_offset_dw > std::numeric_limits<u32>::max() - span_dw) {
            return std::nullopt;
        }
        const u32 source_end_dw = source_offset_dw + span_dw;
        for (const auto& reservation : range_reservations) {
            if (reservation.pointer_lo_flat_index != pointer_lo_flat_index ||
                source_offset_dw < reservation.source_offset_dw) {
                continue;
            }
            const u32 reservation_end_dw = reservation.source_offset_dw + reservation.span_dw;
            if (source_end_dw <= reservation_end_dw) {
                return reservation.destination_flat_offset_dw +
                       (source_offset_dw - reservation.source_offset_dw);
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<u32> TryReserveRange(u32 pointer_lo_flat_index,
                                                     u32 source_offset_dw, u32 span_dw) noexcept {
        const u32 pointer_limit = range_reservations.empty()
                                      ? flattened_bufsize_dw
                                      : range_reservations.front().destination_flat_offset_dw;
        if (span_dw == 0 || pointer_lo_flat_index >= pointer_limit ||
            pointer_limit - pointer_lo_flat_index < 2 ||
            source_offset_dw > std::numeric_limits<u32>::max() - span_dw) {
            return std::nullopt;
        }

        const u32 source_end_dw = source_offset_dw + span_dw;
        if (const auto destination =
                FindReservedRange(pointer_lo_flat_index, source_offset_dw, span_dw)) {
            return destination;
        }
        for (auto& existing : range_reservations) {
            if (existing.pointer_lo_flat_index != pointer_lo_flat_index) {
                continue;
            }
            const u32 existing_end_dw = existing.source_offset_dw + existing.span_dw;
            // Sorted forward ranges can share one contiguous read. Never prepend, because that
            // would invalidate destinations already written into IR.
            if (source_offset_dw >= existing.source_offset_dw &&
                source_offset_dw <= existing_end_dw &&
                existing.destination_flat_offset_dw + existing.span_dw == flattened_bufsize_dw) {
                const u32 extension_dw = source_end_dw - existing_end_dw;
                if (flattened_bufsize_dw > std::numeric_limits<u32>::max() - extension_dw ||
                    extension_dw > MaxRangeSnapshotDwords ||
                    flattened_bufsize_dw > MaxRangeSnapshotDwords - extension_dw) {
                    return std::nullopt;
                }
                existing.span_dw += extension_dw;
                flattened_bufsize_dw += extension_dw;
                return existing.destination_flat_offset_dw +
                       (source_offset_dw - existing.source_offset_dw);
            }
            if (source_offset_dw < existing_end_dw && existing.source_offset_dw < source_end_dw) {
                return std::nullopt;
            }
        }

        if (!range_reservations.empty()) {
            const auto& previous = range_reservations.back();
            if (pointer_lo_flat_index < previous.pointer_lo_flat_index ||
                (pointer_lo_flat_index == previous.pointer_lo_flat_index &&
                 source_offset_dw < previous.source_offset_dw)) {
                return std::nullopt;
            }
        }
        if (range_reservations.size() >= MaxRangeReservations ||
            flattened_bufsize_dw > std::numeric_limits<u32>::max() - span_dw ||
            flattened_bufsize_dw + span_dw > MaxRangeSnapshotDwords) {
            return std::nullopt;
        }
        const u32 destination_flat_offset_dw = flattened_bufsize_dw;
        range_reservations.push_back({
            .pointer_lo_flat_index = pointer_lo_flat_index,
            .source_offset_dw = source_offset_dw,
            .destination_flat_offset_dw = destination_flat_offset_dw,
            .span_dw = span_dw,
        });
        flattened_bufsize_dw += span_dw;
        return destination_flat_offset_dw;
    }

    [[nodiscard]] bool ValidateRangeReservations() const noexcept {
        if (flattened_bufsize_dw < 16 || flattened_bufsize_dw > MaxRangeSnapshotDwords ||
            range_reservations.size() > MaxRangeReservations) {
            return false;
        }
        const u32 pointer_limit = range_reservations.empty()
                                      ? flattened_bufsize_dw
                                      : range_reservations.front().destination_flat_offset_dw;
        for (size_t i = 0; i < range_reservations.size(); ++i) {
            const auto& reservation = range_reservations[i];
            if (reservation.span_dw == 0 || reservation.span_dw > flattened_bufsize_dw ||
                reservation.pointer_lo_flat_index >= pointer_limit ||
                pointer_limit - reservation.pointer_lo_flat_index < 2 ||
                reservation.source_offset_dw >
                    std::numeric_limits<u32>::max() - reservation.span_dw ||
                reservation.destination_flat_offset_dw >
                    flattened_bufsize_dw - reservation.span_dw) {
                return false;
            }
            const u32 destination_end =
                reservation.destination_flat_offset_dw + reservation.span_dw;
            for (size_t j = 0; j < i; ++j) {
                const auto& previous = range_reservations[j];
                const u32 previous_destination_end =
                    previous.destination_flat_offset_dw + previous.span_dw;
                if (reservation.destination_flat_offset_dw < previous_destination_end &&
                    previous.destination_flat_offset_dw < destination_end) {
                    return false;
                }
                if (reservation.pointer_lo_flat_index == previous.pointer_lo_flat_index) {
                    const u32 source_end = reservation.source_offset_dw + reservation.span_dw;
                    const u32 previous_source_end = previous.source_offset_dw + previous.span_dw;
                    if (reservation.source_offset_dw <= previous_source_end &&
                        previous.source_offset_dw <= source_end) {
                        return false;
                    }
                }
            }
            if (i != 0) {
                const auto& previous = range_reservations[i - 1];
                if (reservation.destination_flat_offset_dw !=
                        previous.destination_flat_offset_dw + previous.span_dw ||
                    reservation.pointer_lo_flat_index < previous.pointer_lo_flat_index ||
                    (reservation.pointer_lo_flat_index == previous.pointer_lo_flat_index &&
                     reservation.source_offset_dw < previous.source_offset_dw)) {
                    return false;
                }
            }
        }
        return range_reservations.empty() || range_reservations.back().destination_flat_offset_dw +
                                                     range_reservations.back().span_dw ==
                                                 flattened_bufsize_dw;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
