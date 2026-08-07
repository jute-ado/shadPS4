// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/logging/log.h"
#include "video_core/amdgpu/command_buffer_lifetime_diagnostic.h"

namespace AmdGpu {

/**
 * Emits one privacy-safe frozen report for either crash or normal shutdown. When the diagnostic
 * gate is disabled this does not instantiate, freeze, or log the diagnostic.
 */
inline bool ReportCommandBufferLifetimeDiagnosticOnce() noexcept {
    if (!CommandBufferLifetimeDiagnosticEnabled()) {
        return false;
    }
    const auto snapshot = GetCommandBufferLifetimeDiagnostic().FreezeAndReadOnce();
    if (!snapshot.has_value()) {
        return false;
    }

    LOG_CRITICAL(Debug,
                 "Command buffer lifetime diagnostic window: frozen {} stable {} in-flight {}, "
                 "count-only {}, minimum {}, limit {}, observed {} last observed {} kind {}, "
                 "count-only buffers {}, pre-window {}, selected {}, selection loss {}",
                 snapshot->frozen, snapshot->stable, snapshot->in_flight_operations,
                 snapshot->count_only_mode, snapshot->minimum_buffer_ordinal,
                 snapshot->selected_buffer_limit, snapshot->observed_buffers,
                 snapshot->last_observed_buffer_ordinal,
                 static_cast<u32>(snapshot->last_observed_buffer_kind),
                 snapshot->count_only_buffers, snapshot->pre_window_buffers,
                 snapshot->selected_buffers, snapshot->selection_capacity_loss);
    LOG_CRITICAL(Debug,
                 "Command buffer lifetime diagnostic budget: bytes {}/{}, hashes "
                 "submit/initial/later/final {}/{}/{}/{}, budget loss total {} "
                 "submit/initial/later/final {}/{}/{}/{}",
                 snapshot->hashed_bytes, snapshot->total_hash_byte_budget, snapshot->submit_hashes,
                 snapshot->initial_hashes, snapshot->later_resume_hashes, snapshot->final_hashes,
                 snapshot->hash_budget_exhaustions, snapshot->submit_budget_exhaustions,
                 snapshot->initial_budget_exhaustions, snapshot->later_resume_budget_exhaustions,
                 snapshot->final_budget_exhaustions);
    LOG_CRITICAL(Debug,
                 "Command buffer lifetime diagnostic pre-fetch: baseline bytes {}/{} buffers {}, "
                 "budget loss {} allocation loss {}, checks {} capacity loss {} invalid ranges {}, "
                 "mutations {}, last buffer {} kind {} packet {} offset {} words {}",
                 snapshot->baseline_bytes, snapshot->baseline_byte_budget,
                 snapshot->baseline_buffers, snapshot->baseline_budget_exhaustions,
                 snapshot->baseline_allocation_failures, snapshot->prefetch_checks,
                 snapshot->prefetch_check_capacity_loss, snapshot->prefetch_invalid_ranges,
                 snapshot->prefetch_mutations, snapshot->last_prefetch_buffer_ordinal,
                 static_cast<u32>(snapshot->last_prefetch_buffer_kind),
                 snapshot->last_prefetch_packet_index, snapshot->last_prefetch_word_offset,
                 snapshot->last_prefetch_word_count);
    LOG_CRITICAL(Debug,
                 "Command buffer lifetime diagnostic: sequence {}..{}, observed {}, mutations "
                 "initial/later/final {}/{}/{}, resume checks {} capacity loss {}, oversized {}, "
                 "invalid spans {}, last buffer {} kind {} phase {} resume {} offset {} words {}",
                 snapshot->sequence_before, snapshot->sequence_after, snapshot->observed_buffers,
                 snapshot->initial_mutations, snapshot->later_resume_mutations,
                 snapshot->final_mutations, snapshot->resume_checks,
                 snapshot->resume_check_capacity_loss, snapshot->oversized_buffers,
                 snapshot->invalid_remaining_spans, snapshot->last_buffer_ordinal,
                 static_cast<u32>(snapshot->last_buffer_kind),
                 static_cast<u32>(snapshot->last_mutation_phase), snapshot->last_resume_ordinal,
                 snapshot->last_logical_word_offset, snapshot->last_remaining_words);
    return true;
}

} // namespace AmdGpu
