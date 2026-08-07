// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string_view>

#include "common/types.h"

namespace AmdGpu {

enum class CommandBufferKind : u32 {
    TopLevelDcb,
    TopLevelCcb,
    IndirectDcb,
    IndirectCcb,
};

enum class CommandBufferMutationPhase : u32 {
    None,
    Initial,
    LaterResume,
    Final,
};

struct CommandBufferLifetimeSnapshot {
    u64 sequence_before{};
    u64 observed_buffers{};
    u64 initial_mutations{};
    u64 later_resume_mutations{};
    u64 final_mutations{};
    u64 resume_checks{};
    u64 resume_check_capacity_loss{};
    u64 oversized_buffers{};
    u64 invalid_remaining_spans{};
    u64 last_buffer_ordinal{};
    CommandBufferKind last_buffer_kind{};
    CommandBufferMutationPhase last_mutation_phase{};
    u32 last_resume_ordinal{};
    u32 last_logical_word_offset{};
    u32 last_remaining_words{};
    u64 sequence_after{};
};

/**
 * Fixed-memory diagnostic for command words retained by the asynchronous command processor.
 * Signatures remain internal. Snapshots expose only counters and semantic ordinals.
 *
 * A probe can observe only mutations that persist across one of its sampling boundaries. An exact
 * A-B-A rewrite entirely between two samples is intentionally outside its coverage.
 */
class CommandBufferLifetimeDiagnostic {
public:
    struct Config {
        bool enabled{};
        u32 max_words_per_signature{1u << 20};
        u32 max_resume_checks{64};
    };

private:
    struct State {
        std::atomic<u64> sequence{};
        std::atomic<u64> next_buffer_ordinal{};
        std::atomic<u64> observed_buffers{};
        std::atomic<u64> initial_mutations{};
        std::atomic<u64> later_resume_mutations{};
        std::atomic<u64> final_mutations{};
        std::atomic<u64> resume_checks{};
        std::atomic<u64> resume_check_capacity_loss{};
        std::atomic<u64> oversized_buffers{};
        std::atomic<u64> invalid_remaining_spans{};
        std::atomic<u64> last_buffer_ordinal{};
        std::atomic<u32> last_buffer_kind{};
        std::atomic<u32> last_mutation_phase{};
        std::atomic<u32> last_resume_ordinal{};
        std::atomic<u32> last_logical_word_offset{};
        std::atomic<u32> last_remaining_words{};
    };

public:
    class Probe {
    public:
        Probe() = default;

        Probe(const Probe&) = delete;
        Probe& operator=(const Probe&) = delete;
        Probe(Probe&&) noexcept = default;
        Probe& operator=(Probe&&) noexcept = default;

        [[nodiscard]] bool Active() const noexcept {
            return diagnostic != nullptr;
        }

        void ObserveInitial(std::span<const u32> current) noexcept {
            if (!Active() || initial_observed) {
                return;
            }
            initial_observed = true;
            if (!SameWholeSpan(current) || HashWords(current) != submitted_signature) {
                if (!initial_mutation_recorded) {
                    initial_mutation_recorded = true;
                    diagnostic->RecordMutation(*this, CommandBufferMutationPhase::Initial, 0, 0,
                                               original.size());
                }
            }
        }

        void Suspend(std::span<const u32> remaining) noexcept {
            resume_armed = false;
            if (!Active()) {
                return;
            }
            if (resume_checks_started >= diagnostic->config.max_resume_checks) {
                if (!resume_capacity_recorded) {
                    resume_capacity_recorded = true;
                    diagnostic->state.resume_check_capacity_loss.fetch_add(1);
                }
                return;
            }

            u32 logical_offset{};
            if (!LocateRemaining(remaining, logical_offset)) {
                diagnostic->state.invalid_remaining_spans.fetch_add(1);
                return;
            }

            suspended_logical_offset = logical_offset;
            suspended_words = static_cast<u32>(remaining.size());
            ++resume_checks_started;
            resume_armed = true;
        }

        void Resume(std::span<const u32> remaining) noexcept {
            if (!Active() || !resume_armed) {
                return;
            }
            resume_armed = false;
            const u32 resume_ordinal = resume_checks_started;
            diagnostic->state.resume_checks.fetch_add(1);

            u32 logical_offset{};
            const bool located = LocateRemaining(remaining, logical_offset);
            const bool shape_changed = !located || logical_offset != suspended_logical_offset ||
                                       remaining.size() != suspended_words;
            const bool content_changed = HashWords(original) != submitted_signature;
            if (shape_changed) {
                diagnostic->state.invalid_remaining_spans.fetch_add(1);
            }
            if ((shape_changed || content_changed) && !later_mutation_recorded) {
                later_mutation_recorded = true;
                diagnostic->RecordMutation(*this, CommandBufferMutationPhase::LaterResume,
                                           resume_ordinal, suspended_logical_offset,
                                           suspended_words);
            }
        }

        void ObserveFinal() noexcept {
            if (!Active() || final_observed) {
                return;
            }
            final_observed = true;
            if (HashWords(original) != submitted_signature && !final_mutation_recorded) {
                final_mutation_recorded = true;
                diagnostic->RecordMutation(*this, CommandBufferMutationPhase::Final,
                                           resume_checks_started, 0, original.size());
            }
        }

    private:
        friend class CommandBufferLifetimeDiagnostic;

        Probe(CommandBufferLifetimeDiagnostic& diagnostic_, CommandBufferKind kind_, u64 ordinal_,
              std::span<const u32> original_, u64 submitted_signature_) noexcept
            : diagnostic{&diagnostic_}, kind{kind_}, ordinal{ordinal_}, original{original_},
              submitted_signature{submitted_signature_} {}

        [[nodiscard]] static u64 HashWords(std::span<const u32> words) noexcept {
            u64 hash = 14695981039346656037ULL;
            hash ^= words.size();
            hash *= 1099511628211ULL;
            for (const u32 word : words) {
                hash ^= word;
                hash *= 1099511628211ULL;
            }
            return hash;
        }

        [[nodiscard]] bool SameWholeSpan(std::span<const u32> current) const noexcept {
            return current.data() == original.data() && current.size() == original.size();
        }

        [[nodiscard]] bool LocateRemaining(std::span<const u32> remaining,
                                           u32& logical_offset) const noexcept {
            const auto original_begin = reinterpret_cast<std::uintptr_t>(original.data());
            const auto remaining_begin = reinterpret_cast<std::uintptr_t>(remaining.data());
            const auto original_bytes = original.size_bytes();
            const auto remaining_bytes = remaining.size_bytes();
            if (remaining_begin < original_begin || remaining_bytes > original_bytes) {
                return false;
            }
            const auto byte_offset = remaining_begin - original_begin;
            if (byte_offset % sizeof(u32) != 0 || byte_offset > original_bytes - remaining_bytes) {
                return false;
            }
            logical_offset = static_cast<u32>(byte_offset / sizeof(u32));
            return true;
        }

        CommandBufferLifetimeDiagnostic* diagnostic{};
        CommandBufferKind kind{};
        u64 ordinal{};
        std::span<const u32> original{};
        u64 submitted_signature{};
        u32 suspended_logical_offset{};
        u32 suspended_words{};
        u32 resume_checks_started{};
        bool initial_observed{};
        bool final_observed{};
        bool resume_armed{};
        bool resume_capacity_recorded{};
        bool initial_mutation_recorded{};
        bool later_mutation_recorded{};
        bool final_mutation_recorded{};
    };

    explicit CommandBufferLifetimeDiagnostic(Config config_) noexcept : config{config_} {}

    [[nodiscard]] bool Enabled() const noexcept {
        return config.enabled;
    }

    [[nodiscard]] Probe Begin(CommandBufferKind kind, std::span<const u32> words) noexcept {
        if (!config.enabled || words.empty()) {
            return {};
        }
        if (words.size() > config.max_words_per_signature) {
            state.oversized_buffers.fetch_add(1);
            return {};
        }
        const u64 ordinal = state.next_buffer_ordinal.fetch_add(1) + 1;
        state.observed_buffers.fetch_add(1);
        return Probe{*this, kind, ordinal, words, Probe::HashWords(words)};
    }

    [[nodiscard]] CommandBufferLifetimeSnapshot Read() const noexcept {
        CommandBufferLifetimeSnapshot snapshot{};
        snapshot.sequence_before = state.sequence.load();
        snapshot.observed_buffers = state.observed_buffers.load();
        snapshot.initial_mutations = state.initial_mutations.load();
        snapshot.later_resume_mutations = state.later_resume_mutations.load();
        snapshot.final_mutations = state.final_mutations.load();
        snapshot.resume_checks = state.resume_checks.load();
        snapshot.resume_check_capacity_loss = state.resume_check_capacity_loss.load();
        snapshot.oversized_buffers = state.oversized_buffers.load();
        snapshot.invalid_remaining_spans = state.invalid_remaining_spans.load();
        snapshot.last_buffer_ordinal = state.last_buffer_ordinal.load();
        snapshot.last_buffer_kind = static_cast<CommandBufferKind>(state.last_buffer_kind.load());
        snapshot.last_mutation_phase =
            static_cast<CommandBufferMutationPhase>(state.last_mutation_phase.load());
        snapshot.last_resume_ordinal = state.last_resume_ordinal.load();
        snapshot.last_logical_word_offset = state.last_logical_word_offset.load();
        snapshot.last_remaining_words = state.last_remaining_words.load();
        snapshot.sequence_after = state.sequence.load();
        return snapshot;
    }

private:
    void RecordMutation(const Probe& probe, CommandBufferMutationPhase phase, u32 resume_ordinal,
                        u32 logical_word_offset, size_t remaining_words) noexcept {
        switch (phase) {
        case CommandBufferMutationPhase::Initial:
            state.initial_mutations.fetch_add(1);
            break;
        case CommandBufferMutationPhase::LaterResume:
            state.later_resume_mutations.fetch_add(1);
            break;
        case CommandBufferMutationPhase::Final:
            state.final_mutations.fetch_add(1);
            break;
        case CommandBufferMutationPhase::None:
            return;
        }

        state.last_buffer_ordinal.store(probe.ordinal);
        state.last_buffer_kind.store(static_cast<u32>(probe.kind));
        state.last_mutation_phase.store(static_cast<u32>(phase));
        state.last_resume_ordinal.store(resume_ordinal);
        state.last_logical_word_offset.store(logical_word_offset);
        state.last_remaining_words.store(static_cast<u32>(remaining_words));
        state.sequence.fetch_add(1);
    }

    Config config;
    State state{};
};

[[nodiscard]] inline bool CommandBufferLifetimeDiagnosticEnabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_DIAGNOSTIC_COMMAND_BUFFER_LIFETIME");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

[[nodiscard]] inline CommandBufferLifetimeDiagnostic&
GetCommandBufferLifetimeDiagnostic() noexcept {
    static CommandBufferLifetimeDiagnostic diagnostic{
        {.enabled = CommandBufferLifetimeDiagnosticEnabled()}};
    return diagnostic;
}

} // namespace AmdGpu
