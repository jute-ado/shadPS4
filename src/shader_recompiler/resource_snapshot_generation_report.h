// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/logging/log.h"
#include "shader_recompiler/resource_snapshot_generation.h"

namespace Shader {

inline bool ReportResourceSnapshotGenerationDiagnosticOnce(bool crash) noexcept {
    if (!ResourceSnapshotGenerationDiagnosticEnabled()) {
        return false;
    }

    auto& diagnostic = GetResourceSnapshotGenerationDiagnostic();
    if (!diagnostic.TryMarkReported()) {
        return false;
    }
    const auto snapshot = diagnostic.Read();

    if (crash) {
        LOG_CRITICAL(Debug,
                     "Shader resource generation diagnostic: crash stable={} writers={}..{} "
                     "sequence={}..{} "
                     "frames={} draws={} dispatches={} observations={} words={} max_words={}",
                     snapshot.stable, snapshot.writers_before, snapshot.writers_after,
                     snapshot.sequence_before, snapshot.sequence_after, snapshot.frames,
                     snapshot.draws, snapshot.dispatches, snapshot.observations,
                     snapshot.observed_words, snapshot.maximum_snapshot_words);
    } else {
        LOG_INFO(Render_Recompiler,
                 "Shader resource generation diagnostic: shutdown stable={} writers={}..{} "
                 "sequence={}..{} "
                 "frames={} draws={} dispatches={} observations={} words={} max_words={}",
                 snapshot.stable, snapshot.writers_before, snapshot.writers_after,
                 snapshot.sequence_before, snapshot.sequence_after, snapshot.frames, snapshot.draws,
                 snapshot.dispatches, snapshot.observations, snapshot.observed_words,
                 snapshot.maximum_snapshot_words);
    }
    LOG_WARNING(Render_Recompiler,
                "Shader resource generation coverage: stable={} changed_then_stable={} "
                "capture_unavailable={} retry_exhausted={} capacity_exceeded={} "
                "validation_captures={} user_data_changes={} resource_data_changes={}",
                snapshot.stable_generations, snapshot.changed_then_stable,
                snapshot.capture_unavailable, snapshot.retry_exhausted, snapshot.capacity_exceeded,
                snapshot.validation_captures, snapshot.user_data_changes,
                snapshot.resource_data_changes);
    LOG_WARNING(Render_Recompiler,
                "Shader resource generation stages: fragment={} tess_control={} tess_eval={} "
                "vertex={} geometry={} compute={} abnormal={} last_frame={} last_draw={} "
                "last_dispatch={} last_stage={} last_status={} last_words={}",
                snapshot.stage_observations[0], snapshot.stage_observations[1],
                snapshot.stage_observations[2], snapshot.stage_observations[3],
                snapshot.stage_observations[4], snapshot.stage_observations[5],
                snapshot.abnormal_occurrences, snapshot.last_frame, snapshot.last_draw,
                snapshot.last_dispatch, snapshot.last_stage, static_cast<u32>(snapshot.last_status),
                snapshot.last_snapshot_words);
    return true;
}

} // namespace Shader
