// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace VideoCore {

// Bounded dependency state for one Vulkan buffer. Intervals are disjoint and cover the complete
// buffer. If access patterns exceed the fixed capacity, neighboring states are conservatively
// unioned; this may add synchronization but cannot discard a prior producer.
template <typename Access, typename Stages = Access, std::size_t MaxIntervals = 16>
class BasicBufferAccessRangeState {
    static_assert(MaxIntervals > 0);

public:
    struct TransitionResult {
        Access prior_access{};
        Stages prior_stages{};
        bool requires_barrier{};
    };

    BasicBufferAccessRangeState() = default;

    BasicBufferAccessRangeState(std::uint64_t size, Access initial_access, Stages initial_stages)
        : total_size{size} {
        if (size != 0) {
            intervals[0] = State{
                .offset = 0,
                .size = size,
                .access = initial_access,
                .stages = initial_stages,
            };
            interval_count = 1;
        }
    }

    [[nodiscard]] TransitionResult Transition(std::uint64_t offset, std::uint64_t size,
                                              Access destination_access,
                                              Stages destination_stages) {
        if (size == 0 || offset > total_size || size > total_size - offset) {
            return {};
        }

        TransitionResult result{};
        bool saw_prior = false;
        bool destination_matches_prior = true;
        for (std::size_t i = 0; i < interval_count; ++i) {
            const auto& state = intervals[i];
            if (!Overlaps(offset, size, state.offset, state.size)) {
                continue;
            }
            saw_prior = true;
            result.prior_access |= state.access;
            result.prior_stages |= state.stages;
            destination_matches_prior &= !state.conservative &&
                                         state.access == destination_access &&
                                         state.stages == destination_stages;
        }
        result.requires_barrier = saw_prior && !destination_matches_prior;
        if (!result.requires_barrier) {
            return result;
        }

        std::array<State, MaxIntervals + 2> replaced{};
        std::size_t replaced_count{};
        const auto end = offset + size;
        const auto append = [&](State state) { replaced[replaced_count++] = state; };
        for (std::size_t i = 0; i < interval_count; ++i) {
            const auto& state = intervals[i];
            const auto state_end = state.offset + state.size;
            if (!Overlaps(offset, size, state.offset, state.size)) {
                append(state);
                continue;
            }
            if (state.offset < offset) {
                append(State{
                    .offset = state.offset,
                    .size = offset - state.offset,
                    .access = state.access,
                    .stages = state.stages,
                    .conservative = state.conservative,
                });
            }
            if (state_end > end) {
                append(State{
                    .offset = end,
                    .size = state_end - end,
                    .access = state.access,
                    .stages = state.stages,
                    .conservative = state.conservative,
                });
            }
        }
        append(State{
            .offset = offset,
            .size = size,
            .access = destination_access,
            .stages = destination_stages,
        });
        std::ranges::sort(replaced.begin(), replaced.begin() + replaced_count, {}, &State::offset);

        std::array<State, MaxIntervals + 2> coalesced{};
        std::size_t coalesced_count{};
        for (std::size_t i = 0; i < replaced_count; ++i) {
            const auto& state = replaced[i];
            if (coalesced_count != 0) {
                auto& previous = coalesced[coalesced_count - 1];
                if (previous.offset + previous.size == state.offset &&
                    previous.access == state.access && previous.stages == state.stages &&
                    previous.conservative == state.conservative) {
                    previous.size += state.size;
                    continue;
                }
            }
            coalesced[coalesced_count++] = state;
        }

        while (coalesced_count > MaxIntervals) {
            std::size_t merge_index{};
            std::uint64_t smallest_span = std::numeric_limits<std::uint64_t>::max();
            for (std::size_t i = 1; i < coalesced_count; ++i) {
                const auto span = coalesced[i - 1].size + coalesced[i].size;
                if (span < smallest_span) {
                    smallest_span = span;
                    merge_index = i - 1;
                }
            }
            auto& left = coalesced[merge_index];
            const auto& right = coalesced[merge_index + 1];
            left.conservative = left.conservative || right.conservative ||
                                left.access != right.access || left.stages != right.stages;
            left.size += right.size;
            left.access |= right.access;
            left.stages |= right.stages;
            for (std::size_t i = merge_index + 1; i + 1 < coalesced_count; ++i) {
                coalesced[i] = coalesced[i + 1];
            }
            --coalesced_count;
        }

        interval_count = coalesced_count;
        std::ranges::copy_n(coalesced.begin(), interval_count, intervals.begin());
        return result;
    }

    [[nodiscard]] std::size_t IntervalCount() const noexcept {
        return interval_count;
    }

private:
    struct State {
        std::uint64_t offset{};
        std::uint64_t size{};
        Access access{};
        Stages stages{};
        bool conservative{};
    };

    static bool Overlaps(std::uint64_t left_offset, std::uint64_t left_size,
                         std::uint64_t right_offset, std::uint64_t right_size) noexcept {
        return left_offset < right_offset + right_size && right_offset < left_offset + left_size;
    }

    std::array<State, MaxIntervals> intervals{};
    std::uint64_t total_size{};
    std::size_t interval_count{};
};

} // namespace VideoCore
