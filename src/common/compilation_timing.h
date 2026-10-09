// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

namespace Common {

enum class CompilationPhase {
    GraphicsWorker,
    GraphicsWait,
    OwnerTranslate,
    OwnerSpirv,
    OwnerModule,
    OwnerCompute
};
enum class CompilationOutcome { Exception, Complete, Ready, Failed, Stopped, Rejected };
enum class CompilationRecordKind { Start, Finish, Overflow, Terminal };

struct CompilationContext {
    CompilationPhase phase{};
    std::uint64_t key{};
    int stage{-1};
    int logical{-1};
    std::int64_t perm{-1};
};

struct CompilationTraceRecord {
    CompilationRecordKind kind{};
    CompilationContext context{};
    std::uint64_t id{};
    std::int64_t at_ns{};
    std::int64_t elapsed_ns{};
    CompilationOutcome outcome{CompilationOutcome::Exception};
    std::uint64_t limit{};
    std::uint64_t admitted{};
    std::uint64_t dropped{};
    std::uint32_t code{};
};

class CompilationTraceBudget {
public:
    static constexpr std::uint64_t DefaultLimit = 16'384;
    struct Claim {
        std::uint64_t id{};
        bool overflow{};
    };
    struct Counts {
        std::uint64_t admitted{};
        std::uint64_t dropped{};
    };
    explicit CompilationTraceBudget(std::uint64_t limit = DefaultLimit) : limit{limit} {}
    Claim Admit() noexcept {
        auto count = admitted.load(std::memory_order_relaxed);
        while (count < limit) {
            if (admitted.compare_exchange_weak(count, count + 1, std::memory_order_relaxed)) {
                return {.id = count + 1};
            }
        }
        auto lost = dropped.load(std::memory_order_relaxed);
        while (lost != std::numeric_limits<std::uint64_t>::max() &&
               !dropped.compare_exchange_weak(lost, lost + 1, std::memory_order_relaxed)) {
        }
        return {.overflow = !overflow_reported.exchange(true, std::memory_order_relaxed)};
    }
    Counts Snapshot() const noexcept {
        // Concurrent diagnostic counters, not a transaction or proof of trace completeness.
        return {admitted.load(std::memory_order_relaxed), dropped.load(std::memory_order_relaxed)};
    }
    bool ClaimTerminal() noexcept {
        return !terminal_reported.exchange(true, std::memory_order_relaxed);
    }
    std::uint64_t Limit() const noexcept {
        return limit;
    }

private:
    std::uint64_t limit;
    std::atomic<std::uint64_t> admitted{};
    std::atomic<std::uint64_t> dropped{};
    std::atomic<bool> overflow_reported{};
    std::atomic<bool> terminal_reported{};
};

inline CompilationTraceBudget& ProcessCompilationTraceBudget() {
    static CompilationTraceBudget budget;
    return budget;
}

template <typename Clock = std::chrono::steady_clock>
std::int64_t CompilationTimestampNs() {
    static_assert(Clock::is_steady);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
        .count();
}

template <typename Sink, typename Clock = std::chrono::steady_clock>
class ScopedCompilationSpan {
public:
    template <typename ContextFactory>
    ScopedCompilationSpan(Sink sink, bool enabled, CompilationTraceBudget& budget,
                          ContextFactory&& context_factory)
        : sink{std::move(sink)} {
        if (!enabled) {
            return;
        }
        const auto claim = budget.Admit();
        if (!claim.id) {
            if (claim.overflow) {
                const auto at = CompilationTimestampNs<Clock>();
                const auto counts = budget.Snapshot();
                this->sink(CompilationTraceRecord{.kind = CompilationRecordKind::Overflow,
                                                  .at_ns = at,
                                                  .limit = budget.Limit(),
                                                  .admitted = counts.admitted,
                                                  .dropped = counts.dropped});
            }
            return;
        }
        // Diagnostic-only metadata (including pipeline hashes) is lazy, even after exhaustion.
        context = std::forward<ContextFactory>(context_factory)();
        id = claim.id;
        started = CompilationTimestampNs<Clock>();
        this->sink(CompilationTraceRecord{
            .kind = CompilationRecordKind::Start, .context = context, .id = id, .at_ns = started});
    }
    ScopedCompilationSpan(const ScopedCompilationSpan&) = delete;
    ScopedCompilationSpan& operator=(const ScopedCompilationSpan&) = delete;
    ~ScopedCompilationSpan() {
        if (id) {
            const auto ended = CompilationTimestampNs<Clock>();
            sink(CompilationTraceRecord{.kind = CompilationRecordKind::Finish,
                                        .context = context,
                                        .id = id,
                                        .at_ns = ended,
                                        .elapsed_ns = ended - started,
                                        .outcome = outcome});
        }
    }
    void SetOutcome(CompilationOutcome value) noexcept {
        outcome = value;
    }

private:
    Sink sink;
    CompilationContext context{};
    std::uint64_t id{};
    std::int64_t started{};
    CompilationOutcome outcome{CompilationOutcome::Exception};
};

template <typename Clock = std::chrono::steady_clock, typename Sink>
void EmitCompilationTerminal(bool enabled, CompilationTraceBudget& budget, Sink&& sink,
                             std::uint32_t code) {
    if (!enabled || !budget.ClaimTerminal()) {
        return;
    }
    const auto at = CompilationTimestampNs<Clock>();
    const auto counts = budget.Snapshot();
    std::forward<Sink>(sink)(CompilationTraceRecord{.kind = CompilationRecordKind::Terminal,
                                                    .at_ns = at,
                                                    .admitted = counts.admitted,
                                                    .dropped = counts.dropped,
                                                    .code = code});
}

} // namespace Common
