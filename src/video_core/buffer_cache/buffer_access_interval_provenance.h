// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace VideoCore {

enum class BufferAccessRole : std::uint32_t {
    None = 0,
    ShaderRead = 1U << 0,
    ShaderReadWrite = 1U << 1,
    VertexRead = 1U << 2,
    IndexRead = 1U << 3,
    IndirectRead = 1U << 4,
    TransferRead = 1U << 5,
    TransferWrite = 1U << 6,
    UploadWrite = 1U << 7,
    FillWrite = 1U << 8,
    JoinRead = 1U << 9,
    JoinWrite = 1U << 10,
};

constexpr BufferAccessRole operator|(BufferAccessRole lhs, BufferAccessRole rhs) noexcept {
    return static_cast<BufferAccessRole>(static_cast<std::uint32_t>(lhs) |
                                         static_cast<std::uint32_t>(rhs));
}

constexpr BufferAccessRole& operator|=(BufferAccessRole& lhs, BufferAccessRole rhs) noexcept {
    lhs = lhs | rhs;
    return lhs;
}

constexpr BufferAccessRole operator&(BufferAccessRole lhs, BufferAccessRole rhs) noexcept {
    return static_cast<BufferAccessRole>(static_cast<std::uint32_t>(lhs) &
                                         static_cast<std::uint32_t>(rhs));
}

constexpr bool HasBufferAccessRole(BufferAccessRole roles, BufferAccessRole role) noexcept {
    return (roles & role) != BufferAccessRole::None;
}

struct BufferBarrierObservation {
    bool emitted{};
    std::uint64_t offset{};
    std::uint64_t size{};
    std::uint64_t source_access{};
    std::uint64_t source_stages{};
    std::uint64_t destination_access{};
    std::uint64_t destination_stages{};
};

template <typename Key>
class BasicBufferAccessIntervalProvenance {
public:
    struct Observation {
        std::uint64_t offset{};
        std::uint64_t size{};
        BufferAccessRole roles{};
        std::uint64_t access{};
        std::uint64_t stages{};
        bool writes{};
        BufferBarrierObservation barrier{};
    };

    struct Producer {
        std::uint64_t command_id{};
        std::uint64_t tick{};
        BufferAccessRole roles{};
        std::uint64_t access{};
        std::uint64_t stages{};
        bool writes{};

        bool operator==(const Producer&) const = default;
    };

    struct Transition {
        Key key{};
        std::uint64_t offset{};
        std::uint64_t size{};
        std::optional<Producer> prior;
        std::vector<Observation> observations;
        BufferAccessRole current_roles{};
        std::uint64_t current_access{};
        std::uint64_t current_stages{};
        bool current_writes{};
    };

    void BeginCommand(std::uint64_t command_id, std::uint64_t tick) {
        current_command_id = command_id;
        current_tick = tick;
        current.clear();
        command_active = true;
    }

    bool Observe(Key key, std::uint64_t offset, std::uint64_t size, BufferAccessRole roles,
                 std::uint64_t access, std::uint64_t stages, bool writes,
                 BufferBarrierObservation barrier = {}) {
        if (!command_active || size == 0 ||
            offset > std::numeric_limits<std::uint64_t>::max() - size) {
            return false;
        }
        auto& entry = FindOrCreate(current, key);
        entry.values.push_back(Observation{
            .offset = offset,
            .size = size,
            .roles = roles,
            .access = access,
            .stages = stages,
            .writes = writes,
            .barrier = barrier,
        });
        return true;
    }

    std::vector<Transition> CommitCommand() {
        std::vector<Transition> transitions;
        if (!command_active) {
            return transitions;
        }

        for (const auto& current_entry : current) {
            const auto* prior_entry = Find(states, current_entry.key);
            std::vector<std::uint64_t> boundaries;
            boundaries.reserve(current_entry.values.size() * 2 +
                               (prior_entry == nullptr ? 0 : prior_entry->values.size() * 2));
            for (const auto& observation : current_entry.values) {
                boundaries.push_back(observation.offset);
                boundaries.push_back(observation.offset + observation.size);
            }
            if (prior_entry != nullptr) {
                for (const auto& state : prior_entry->values) {
                    if (std::ranges::any_of(current_entry.values, [&](const Observation& value) {
                            return Overlaps(value.offset, value.size, state.offset, state.size);
                        })) {
                        boundaries.push_back(state.offset);
                        boundaries.push_back(state.offset + state.size);
                    }
                }
            }
            std::ranges::sort(boundaries);
            boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

            for (std::size_t i = 1; i < boundaries.size(); ++i) {
                const auto begin = boundaries[i - 1];
                const auto end = boundaries[i];
                if (begin == end) {
                    continue;
                }

                Transition transition{
                    .key = current_entry.key,
                    .offset = begin,
                    .size = end - begin,
                };
                for (const auto& observation : current_entry.values) {
                    if (!Overlaps(begin, end - begin, observation.offset, observation.size)) {
                        continue;
                    }
                    transition.observations.push_back(observation);
                    transition.current_roles |= observation.roles;
                    transition.current_access |= observation.access;
                    transition.current_stages |= observation.stages;
                    transition.current_writes |= observation.writes;
                }
                if (transition.observations.empty()) {
                    continue;
                }
                if (prior_entry != nullptr) {
                    const auto prior =
                        std::ranges::find_if(prior_entry->values, [&](const StateInterval& state) {
                            return Overlaps(begin, end - begin, state.offset, state.size);
                        });
                    if (prior != prior_entry->values.end()) {
                        transition.prior = prior->producer;
                    }
                }
                transitions.push_back(std::move(transition));
            }
        }

        for (const auto& transition : transitions) {
            ReplaceState(transition.key, transition.offset, transition.size,
                         Producer{
                             .command_id = current_command_id,
                             .tick = current_tick,
                             .roles = transition.current_roles,
                             .access = transition.current_access,
                             .stages = transition.current_stages,
                             .writes = transition.current_writes,
                         });
        }
        current.clear();
        command_active = false;
        return transitions;
    }

private:
    struct StateInterval {
        std::uint64_t offset{};
        std::uint64_t size{};
        Producer producer{};
    };

    template <typename Value>
    struct Entry {
        Key key{};
        std::vector<Value> values;
    };

    static bool Overlaps(std::uint64_t left_offset, std::uint64_t left_size,
                         std::uint64_t right_offset, std::uint64_t right_size) noexcept {
        return left_offset < right_offset + right_size && right_offset < left_offset + left_size;
    }

    template <typename Value>
    static Entry<Value>* Find(std::vector<Entry<Value>>& entries, const Key& key) {
        const auto entry = std::ranges::find(entries, key, &Entry<Value>::key);
        return entry == entries.end() ? nullptr : &*entry;
    }

    template <typename Value>
    static const Entry<Value>* Find(const std::vector<Entry<Value>>& entries, const Key& key) {
        const auto entry = std::ranges::find(entries, key, &Entry<Value>::key);
        return entry == entries.end() ? nullptr : &*entry;
    }

    template <typename Value>
    static Entry<Value>& FindOrCreate(std::vector<Entry<Value>>& entries, const Key& key) {
        if (auto* entry = Find(entries, key)) {
            return *entry;
        }
        return entries.emplace_back(Entry<Value>{.key = key});
    }

    void ReplaceState(Key key, std::uint64_t offset, std::uint64_t size, Producer producer) {
        auto& entry = FindOrCreate(states, key);
        std::vector<StateInterval> replaced;
        replaced.reserve(entry.values.size() + 1);
        const auto end = offset + size;
        for (const auto& state : entry.values) {
            const auto state_end = state.offset + state.size;
            if (!Overlaps(offset, size, state.offset, state.size)) {
                replaced.push_back(state);
                continue;
            }
            if (state.offset < offset) {
                replaced.push_back(StateInterval{
                    .offset = state.offset,
                    .size = offset - state.offset,
                    .producer = state.producer,
                });
            }
            if (state_end > end) {
                replaced.push_back(StateInterval{
                    .offset = end,
                    .size = state_end - end,
                    .producer = state.producer,
                });
            }
        }
        replaced.push_back(StateInterval{
            .offset = offset,
            .size = size,
            .producer = producer,
        });
        std::ranges::sort(replaced, {}, &StateInterval::offset);

        std::vector<StateInterval> coalesced;
        coalesced.reserve(replaced.size());
        for (const auto& state : replaced) {
            if (!coalesced.empty() &&
                coalesced.back().offset + coalesced.back().size == state.offset &&
                coalesced.back().producer == state.producer) {
                coalesced.back().size += state.size;
            } else {
                coalesced.push_back(state);
            }
        }
        entry.values = std::move(coalesced);
    }

    std::vector<Entry<Observation>> current;
    std::vector<Entry<StateInterval>> states;
    std::uint64_t current_command_id{};
    std::uint64_t current_tick{};
    bool command_active{};
};

} // namespace VideoCore
