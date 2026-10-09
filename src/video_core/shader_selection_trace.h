// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <mutex>

#include "common/types.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/shader_capture_provenance.h"

namespace VideoCore {

inline constexpr char ShaderSelectionTraceEnvironment[] =
    "SHADPS4_SHADER_SELECTION_TRACE_PATH";

enum class ShaderSelectionCacheAction : u8 {
    New,
    Reuse,
    Reject,
};

struct ShaderSelectionTraceRecord {
    u64 dispatch_sequence{};
    Shader::Stage physical_stage{};
    Shader::LogicalStage logical_stage{};
    u64 program_hash{};
    u64 srt_source_digest{};
    u64 srt_content_digest{};
    u64 semantic_identity{};
    ShaderSelectionCacheAction cache_action{ShaderSelectionCacheAction::Reject};
    u32 module_count{};
    u64 selected_permutation_id{};
    u64 compute_pipeline_id{};
};

/**
 * Returns an address-free aggregate of which authoritative sources supplied one SRT capture.
 * Guest addresses, mapping tokens, writer handles and raw contents deliberately do not
 * participate.
 */
[[nodiscard]] u64 DigestShaderSelectionSrtSources(
    const SrtCaptureProvenance& provenance) noexcept;

/** Returns an address-free aggregate of the already-hashed SRT page contents. */
[[nodiscard]] u64 DigestShaderSelectionSrtContents(
    const SrtCaptureProvenance& provenance) noexcept;

/**
 * Opt-in, bounded JSONL evidence for deterministic shader-permutation selection diagnostics.
 *
 * The environment-gated default is disabled without opening a file. Evidence files are created
 * exclusively and never overwrite or append to existing data. Every failure closes the file and
 * permanently disables the recorder. Record() is noexcept and returns before locking while the
 * recorder is disabled.
 */
class ShaderSelectionTrace {
public:
    struct Limits {
        size_t max_records;
        size_t max_bytes;
    };
    static constexpr Limits DefaultLimits{4096, 1024 * 1024};

    ShaderSelectionTrace() noexcept;
    explicit ShaderSelectionTrace(const std::filesystem::path& path) noexcept;
    ShaderSelectionTrace(const std::filesystem::path& path, Limits limits) noexcept;
    ~ShaderSelectionTrace();

    ShaderSelectionTrace(const ShaderSelectionTrace&) = delete;
    ShaderSelectionTrace& operator=(const ShaderSelectionTrace&) = delete;

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled.load(std::memory_order_relaxed);
    }

    void Record(const ShaderSelectionTraceRecord& record) noexcept;

private:
    void Open(const std::filesystem::path& path) noexcept;
    void DisableLocked() noexcept;

    Limits limits{DefaultLimits};
    std::atomic_bool enabled{};
    std::mutex mutex;
    std::FILE* file{};
    size_t record_count{};
    size_t byte_count{};
};

} // namespace VideoCore
