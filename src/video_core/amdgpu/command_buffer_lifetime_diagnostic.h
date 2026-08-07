// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

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
    bool frozen{};
    bool stable{};
    bool count_only_mode{};
    u32 in_flight_operations{};
    u64 minimum_buffer_ordinal{};
    u32 selected_buffer_limit{};
    u64 total_hash_byte_budget{};
    u64 sequence_before{};
    u64 observed_buffers{};
    u64 last_observed_buffer_ordinal{};
    CommandBufferKind last_observed_buffer_kind{};
    u64 count_only_buffers{};
    u64 pre_window_buffers{};
    u64 selected_buffers{};
    u64 selection_capacity_loss{};
    u64 hashed_bytes{};
    u64 submit_hashes{};
    u64 initial_hashes{};
    u64 later_resume_hashes{};
    u64 final_hashes{};
    u64 hash_budget_exhaustions{};
    u64 submit_budget_exhaustions{};
    u64 initial_budget_exhaustions{};
    u64 later_resume_budget_exhaustions{};
    u64 final_budget_exhaustions{};
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
    static constexpr u32 HardMaxSelectedBuffers = 64;
    static constexpr u64 HardMaxHashByteBudget = 256ULL * 1024 * 1024;

    struct Config {
        bool enabled{};
        bool count_only{};
        u64 minimum_buffer_ordinal{1};
        u32 selected_buffer_count{16};
        u64 total_hash_byte_budget{64ULL * 1024 * 1024};
        u32 max_words_per_signature{1u << 20};
        u32 max_resume_checks{64};
    };

private:
    enum class HashPhase {
        Submit,
        Initial,
        LaterResume,
        Final,
    };

    struct State {
        std::atomic<bool> frozen{};
        std::atomic<u32> in_flight_operations{};
        std::atomic<u64> sequence{};
        std::atomic<u64> next_buffer_ordinal{};
        std::atomic<u64> observed_buffers{};
        std::atomic<u64> last_observed_buffer_ordinal{};
        std::atomic<u32> last_observed_buffer_kind{};
        std::atomic<u64> count_only_buffers{};
        std::atomic<u64> pre_window_buffers{};
        std::atomic<u64> selected_buffers{};
        std::atomic<u64> selection_capacity_loss{};
        std::atomic<u64> hashed_bytes{};
        std::atomic<u64> submit_hashes{};
        std::atomic<u64> initial_hashes{};
        std::atomic<u64> later_resume_hashes{};
        std::atomic<u64> final_hashes{};
        std::atomic<u64> hash_budget_exhaustions{};
        std::atomic<u64> submit_budget_exhaustions{};
        std::atomic<u64> initial_budget_exhaustions{};
        std::atomic<u64> later_resume_budget_exhaustions{};
        std::atomic<u64> final_budget_exhaustions{};
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

    class OperationGuard {
    public:
        explicit OperationGuard(CommandBufferLifetimeDiagnostic* diagnostic_) noexcept {
            if (diagnostic_ == nullptr ||
                diagnostic_->state.frozen.load(std::memory_order_acquire)) {
                return;
            }
            diagnostic_->state.in_flight_operations.fetch_add(1, std::memory_order_acq_rel);
            if (diagnostic_->state.frozen.load(std::memory_order_acquire)) {
                diagnostic_->state.in_flight_operations.fetch_sub(1, std::memory_order_release);
                return;
            }
            diagnostic = diagnostic_;
        }

        OperationGuard(const OperationGuard&) = delete;
        OperationGuard& operator=(const OperationGuard&) = delete;

        ~OperationGuard() {
            if (diagnostic != nullptr) {
                diagnostic->state.in_flight_operations.fetch_sub(1, std::memory_order_release);
            }
        }

        [[nodiscard]] explicit operator bool() const noexcept {
            return diagnostic != nullptr;
        }

    private:
        CommandBufferLifetimeDiagnostic* diagnostic{};
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
            OperationGuard guard{diagnostic};
            if (!guard) {
                return;
            }
            initial_observed = true;
            const auto signature = diagnostic->TryHash(current, HashPhase::Initial);
            if (!signature.has_value()) {
                return;
            }
            if (!SameWholeSpan(current) || *signature != submitted_signature) {
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
            OperationGuard guard{diagnostic};
            if (!guard) {
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
            OperationGuard guard{diagnostic};
            if (!guard) {
                return;
            }
            resume_armed = false;
            const u32 resume_ordinal = resume_checks_started;
            diagnostic->state.resume_checks.fetch_add(1);

            u32 logical_offset{};
            const bool located = LocateRemaining(remaining, logical_offset);
            const bool shape_changed = !located || logical_offset != suspended_logical_offset ||
                                       remaining.size() != suspended_words;
            const auto signature = diagnostic->TryHash(original, HashPhase::LaterResume);
            const bool content_changed = signature.has_value() && *signature != submitted_signature;
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
            OperationGuard guard{diagnostic};
            if (!guard) {
                return;
            }
            final_observed = true;
            const auto signature = diagnostic->TryHash(original, HashPhase::Final);
            if (signature.has_value() && *signature != submitted_signature &&
                !final_mutation_recorded) {
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

    explicit CommandBufferLifetimeDiagnostic(Config config_) noexcept
        : config{NormalizeConfig(config_)} {}

    [[nodiscard]] bool Enabled() const noexcept {
        return config.enabled;
    }

    [[nodiscard]] Probe Begin(CommandBufferKind kind, std::span<const u32> words) noexcept {
        if (!config.enabled || words.empty()) {
            return {};
        }
        OperationGuard guard{this};
        if (!guard) {
            return {};
        }

        const u64 ordinal = state.next_buffer_ordinal.fetch_add(1) + 1;
        state.observed_buffers.fetch_add(1);
        state.last_observed_buffer_ordinal.store(ordinal);
        state.last_observed_buffer_kind.store(static_cast<u32>(kind));
        if (config.count_only) {
            state.count_only_buffers.fetch_add(1);
            return {};
        }
        if (ordinal < config.minimum_buffer_ordinal) {
            state.pre_window_buffers.fetch_add(1);
            return {};
        }
        if (ordinal - config.minimum_buffer_ordinal >= config.selected_buffer_count) {
            state.selection_capacity_loss.fetch_add(1);
            return {};
        }

        state.selected_buffers.fetch_add(1);
        if (words.size() > config.max_words_per_signature) {
            state.oversized_buffers.fetch_add(1);
            return {};
        }
        const auto signature = TryHash(words, HashPhase::Submit);
        if (!signature.has_value()) {
            return {};
        }
        return Probe{*this, kind, ordinal, words, *signature};
    }

    [[nodiscard]] CommandBufferLifetimeSnapshot Read() const noexcept {
        return ReadSnapshot();
    }

    [[nodiscard]] CommandBufferLifetimeSnapshot FreezeAndRead(
        u32 maximum_spin_count = 100'000) noexcept {
        state.frozen.store(true, std::memory_order_release);
        for (u32 spin = 0; spin < maximum_spin_count &&
                           state.in_flight_operations.load(std::memory_order_acquire) != 0;
             ++spin) {
            std::atomic_signal_fence(std::memory_order_seq_cst);
        }
        return ReadSnapshot();
    }

private:
    [[nodiscard]] CommandBufferLifetimeSnapshot ReadSnapshot() const noexcept {
        CommandBufferLifetimeSnapshot snapshot{};
        snapshot.frozen = state.frozen.load(std::memory_order_acquire);
        snapshot.in_flight_operations = state.in_flight_operations.load(std::memory_order_acquire);
        snapshot.stable = snapshot.frozen && snapshot.in_flight_operations == 0;
        snapshot.count_only_mode = config.count_only;
        snapshot.minimum_buffer_ordinal = config.minimum_buffer_ordinal;
        snapshot.selected_buffer_limit = config.selected_buffer_count;
        snapshot.total_hash_byte_budget = config.total_hash_byte_budget;
        snapshot.sequence_before = state.sequence.load();
        snapshot.observed_buffers = state.observed_buffers.load();
        snapshot.last_observed_buffer_ordinal = state.last_observed_buffer_ordinal.load();
        snapshot.last_observed_buffer_kind =
            static_cast<CommandBufferKind>(state.last_observed_buffer_kind.load());
        snapshot.count_only_buffers = state.count_only_buffers.load();
        snapshot.pre_window_buffers = state.pre_window_buffers.load();
        snapshot.selected_buffers = state.selected_buffers.load();
        snapshot.selection_capacity_loss = state.selection_capacity_loss.load();
        snapshot.hashed_bytes = state.hashed_bytes.load();
        snapshot.submit_hashes = state.submit_hashes.load();
        snapshot.initial_hashes = state.initial_hashes.load();
        snapshot.later_resume_hashes = state.later_resume_hashes.load();
        snapshot.final_hashes = state.final_hashes.load();
        snapshot.hash_budget_exhaustions = state.hash_budget_exhaustions.load();
        snapshot.submit_budget_exhaustions = state.submit_budget_exhaustions.load();
        snapshot.initial_budget_exhaustions = state.initial_budget_exhaustions.load();
        snapshot.later_resume_budget_exhaustions = state.later_resume_budget_exhaustions.load();
        snapshot.final_budget_exhaustions = state.final_budget_exhaustions.load();
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

    [[nodiscard]] static Config NormalizeConfig(Config source) noexcept {
        source.minimum_buffer_ordinal = std::max<u64>(source.minimum_buffer_ordinal, 1);
        source.selected_buffer_count =
            std::min(source.selected_buffer_count, HardMaxSelectedBuffers);
        source.total_hash_byte_budget =
            std::min(source.total_hash_byte_budget, HardMaxHashByteBudget);
        return source;
    }

    [[nodiscard]] std::optional<u64> TryHash(std::span<const u32> words, HashPhase phase) noexcept {
        const u64 bytes = words.size_bytes();
        u64 consumed = state.hashed_bytes.load(std::memory_order_relaxed);
        while (bytes <= config.total_hash_byte_budget &&
               consumed <= config.total_hash_byte_budget - bytes) {
            if (state.hashed_bytes.compare_exchange_weak(consumed, consumed + bytes,
                                                         std::memory_order_relaxed)) {
                HashCount(phase).fetch_add(1);
                return Probe::HashWords(words);
            }
        }
        state.hash_budget_exhaustions.fetch_add(1);
        BudgetExhaustionCount(phase).fetch_add(1);
        return std::nullopt;
    }

    [[nodiscard]] std::atomic<u64>& HashCount(HashPhase phase) noexcept {
        switch (phase) {
        case HashPhase::Submit:
            return state.submit_hashes;
        case HashPhase::Initial:
            return state.initial_hashes;
        case HashPhase::LaterResume:
            return state.later_resume_hashes;
        case HashPhase::Final:
            return state.final_hashes;
        }
        std::unreachable();
    }

    [[nodiscard]] std::atomic<u64>& BudgetExhaustionCount(HashPhase phase) noexcept {
        switch (phase) {
        case HashPhase::Submit:
            return state.submit_budget_exhaustions;
        case HashPhase::Initial:
            return state.initial_budget_exhaustions;
        case HashPhase::LaterResume:
            return state.later_resume_budget_exhaustions;
        case HashPhase::Final:
            return state.final_budget_exhaustions;
        }
        std::unreachable();
    }

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

[[nodiscard]] inline bool CommandBufferLifetimeCountOnlyEnabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_DIAGNOSTIC_COMMAND_BUFFER_LIFETIME_COUNT_ONLY");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

[[nodiscard]] inline u64 CommandBufferLifetimeUnsignedEnvironment(const char* name,
                                                                  u64 default_value,
                                                                  u64 maximum_value) noexcept {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    const std::string_view text{value};
    u64 parsed{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return default_value;
    }
    return std::min(parsed, maximum_value);
}

[[nodiscard]] inline CommandBufferLifetimeDiagnostic&
GetCommandBufferLifetimeDiagnostic() noexcept {
    static CommandBufferLifetimeDiagnostic diagnostic{{
        .enabled = CommandBufferLifetimeDiagnosticEnabled(),
        .count_only = CommandBufferLifetimeCountOnlyEnabled(),
        .minimum_buffer_ordinal = CommandBufferLifetimeUnsignedEnvironment(
            "SHADPS4_DIAGNOSTIC_COMMAND_BUFFER_LIFETIME_MIN_ORDINAL", 1,
            std::numeric_limits<u64>::max()),
        .selected_buffer_count = static_cast<u32>(CommandBufferLifetimeUnsignedEnvironment(
            "SHADPS4_DIAGNOSTIC_COMMAND_BUFFER_LIFETIME_BUFFER_COUNT", 16,
            CommandBufferLifetimeDiagnostic::HardMaxSelectedBuffers)),
        .total_hash_byte_budget = CommandBufferLifetimeUnsignedEnvironment(
            "SHADPS4_DIAGNOSTIC_COMMAND_BUFFER_LIFETIME_HASH_BYTE_BUDGET", 64ULL * 1024 * 1024,
            CommandBufferLifetimeDiagnostic::HardMaxHashByteBudget),
    }};
    return diagnostic;
}

} // namespace AmdGpu
