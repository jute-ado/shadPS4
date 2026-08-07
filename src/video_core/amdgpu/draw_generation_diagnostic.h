// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <span>
#include <type_traits>

#include "common/types.h"

namespace AmdGpu {

enum class DrawIssueKind : u8 {
    DirectNonIndexed,
    DirectIndexed,
    IndirectNonIndexed,
    IndirectIndexed,
};

/** Order-sensitive internal signature. Signature values are never exposed by diagnostic reports. */
class DrawGenerationSignature {
public:
    template <std::integral T>
    void Add(T value) noexcept {
        Mix(static_cast<u64>(value));
    }

    template <typename T>
        requires std::is_enum_v<T>
    void Add(T value) noexcept {
        Add(std::to_underlying(value));
    }

    void AddFloat(float value) noexcept {
        Add(std::bit_cast<u32>(value));
    }

    [[nodiscard]] u64 Value() const noexcept {
        return value;
    }

private:
    void Mix(u64 input) noexcept {
        // SplitMix64 finalizer, folded into an order-sensitive FNV-style stream.
        input ^= input >> 30;
        input *= 0xbf58476d1ce4e5b9ULL;
        input ^= input >> 27;
        input *= 0x94d049bb133111ebULL;
        input ^= input >> 31;
        value ^= input;
        value *= 1099511628211ULL;
    }

    u64 value{14695981039346656037ULL};
};

[[nodiscard]] inline u64 HashCommandWords(std::span<const u32> words) noexcept {
    DrawGenerationSignature signature;
    signature.Add(words.size());
    for (const u32 word : words) {
        signature.Add(word);
    }
    return signature.Value();
}

struct DrawGenerationSnapshot {
    static constexpr u32 MaxReportedOrdinals = 16;

    u64 sequence{};
    u64 aba_middle_sequence{};
    u32 draws{};
    u32 direct_draws{};
    u32 indirect_draws{};
    u32 changed_from_previous{};
    u32 reported_changed_from_previous{};
    std::array<u32, MaxReportedOrdinals> first_changed_from_previous_ordinals{};
    u32 exact_aba_return_draws{};
    u32 reported_exact_aba_return_draws{};
    std::array<u32, MaxReportedOrdinals> first_exact_aba_return_ordinals{};
    u32 submissions{};
    u32 mutated_submissions{};
    u32 truncated_draws{};
    u32 truncated_submissions{};
    bool has_previous{};
    bool count_changed{};
};

/**
 * Bounded reducer for command-buffer integrity and the exact ordered graphics draws issued by the
 * renderer. It reports only counts and sequential draw ordinals, never command words, addresses,
 * state values, or signatures.
 */
class DrawGenerationDiagnostic {
public:
    static constexpr u32 MaxDrawsPerFrame = 2048;
    static constexpr u32 MaxSubmissionsPerFrame = 512;

    void ObserveDraw(DrawIssueKind kind, u64 signature) noexcept {
        if (current_draws >= MaxDrawsPerFrame) {
            ++truncated_draws;
            return;
        }
        current_draw_signatures[current_draws++] = signature;
        if (kind == DrawIssueKind::DirectNonIndexed || kind == DrawIssueKind::DirectIndexed) {
            ++direct_draws;
        } else {
            ++indirect_draws;
        }
    }

    void ObserveSubmission(u64 submitted_dcb, u64 parsed_dcb, u64 submitted_ccb,
                           u64 parsed_ccb) noexcept {
        if (submissions >= MaxSubmissionsPerFrame) {
            ++truncated_submissions;
            return;
        }
        ++submissions;
        mutated_submissions += submitted_dcb != parsed_dcb || submitted_ccb != parsed_ccb;
    }

    [[nodiscard]] DrawGenerationSnapshot TakeSnapshot() noexcept {
        DrawGenerationSnapshot snapshot{
            .sequence = ++sequence,
            .aba_middle_sequence = sequence > 1 ? sequence - 1 : 0,
            .draws = current_draws,
            .direct_draws = direct_draws,
            .indirect_draws = indirect_draws,
            .submissions = submissions,
            .mutated_submissions = mutated_submissions,
            .truncated_draws = truncated_draws,
            .truncated_submissions = truncated_submissions,
            .has_previous = has_previous,
            .count_changed = has_previous && current_draws != previous_draws,
        };

        if (has_previous) {
            const u32 compared_draws = std::max(current_draws, previous_draws);
            for (u32 draw = 0; draw < compared_draws; ++draw) {
                const bool same = draw < current_draws && draw < previous_draws &&
                                  current_draw_signatures[draw] == previous_draw_signatures[draw];
                if (same) {
                    continue;
                }
                ++snapshot.changed_from_previous;
                if (snapshot.reported_changed_from_previous <
                    DrawGenerationSnapshot::MaxReportedOrdinals) {
                    snapshot.first_changed_from_previous_ordinals
                        [snapshot.reported_changed_from_previous++] = draw;
                }
            }
        }

        if (has_previous_previous) {
            const u32 compared_draws =
                std::min({current_draws, previous_draws, previous_previous_draws});
            for (u32 draw = 0; draw < compared_draws; ++draw) {
                const bool exact_aba =
                    current_draw_signatures[draw] == previous_previous_draw_signatures[draw] &&
                    current_draw_signatures[draw] != previous_draw_signatures[draw];
                if (!exact_aba) {
                    continue;
                }
                ++snapshot.exact_aba_return_draws;
                if (snapshot.reported_exact_aba_return_draws <
                    DrawGenerationSnapshot::MaxReportedOrdinals) {
                    snapshot.first_exact_aba_return_ordinals
                        [snapshot.reported_exact_aba_return_draws++] = draw;
                }
            }
        }

        previous_previous_draw_signatures = previous_draw_signatures;
        previous_previous_draws = previous_draws;
        has_previous_previous = has_previous;
        previous_draw_signatures = current_draw_signatures;
        previous_draws = current_draws;
        has_previous = true;
        ResetCurrent();
        return snapshot;
    }

private:
    void ResetCurrent() noexcept {
        current_draw_signatures = {};
        current_draws = 0;
        direct_draws = 0;
        indirect_draws = 0;
        submissions = 0;
        mutated_submissions = 0;
        truncated_draws = 0;
        truncated_submissions = 0;
    }

    std::array<u64, MaxDrawsPerFrame> current_draw_signatures{};
    std::array<u64, MaxDrawsPerFrame> previous_draw_signatures{};
    std::array<u64, MaxDrawsPerFrame> previous_previous_draw_signatures{};
    u64 sequence{};
    u32 current_draws{};
    u32 previous_draws{};
    u32 previous_previous_draws{};
    u32 direct_draws{};
    u32 indirect_draws{};
    u32 submissions{};
    u32 mutated_submissions{};
    u32 truncated_draws{};
    u32 truncated_submissions{};
    bool has_previous{};
    bool has_previous_previous{};
};

} // namespace AmdGpu
