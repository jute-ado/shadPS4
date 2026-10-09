// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/shader_selection_trace.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <string_view>

#ifdef _WIN32
#include <share.h>
#endif

namespace VideoCore {
namespace {

constexpr u64 FnvOffset = 14695981039346656037ULL;
constexpr u64 FnvPrime = 1099511628211ULL;

void DigestByte(u64& digest, u8 value) noexcept {
    digest ^= value;
    digest *= FnvPrime;
}

void DigestU64(u64& digest, u64 value) noexcept {
    for (u32 shift = 0; shift < 64; shift += 8) {
        DigestByte(digest, static_cast<u8>(value >> shift));
    }
}

void DigestDomain(u64& digest, std::string_view domain) noexcept {
    for (const char value : domain) {
        DigestByte(digest, static_cast<u8>(value));
    }
}

[[nodiscard]] constexpr std::string_view CacheActionName(
    ShaderSelectionCacheAction action) noexcept {
    switch (action) {
    case ShaderSelectionCacheAction::New:
        return "new";
    case ShaderSelectionCacheAction::Reuse:
        return "reuse";
    case ShaderSelectionCacheAction::Reject:
        return "reject";
    }
    return "reject";
}

} // namespace

u64 DigestShaderSelectionSrtSources(const SrtCaptureProvenance& provenance) noexcept {
    u64 digest = FnvOffset;
    DigestDomain(digest, "shadPS4.shader-selection-srt-sources.v1");
    DigestU64(digest, provenance.Ranges().size());
    DigestByte(digest, provenance.Truncated());
    for (const auto& range : provenance.Ranges()) {
        DigestByte(digest, static_cast<u8>(range.source));
        DigestByte(digest, range.exact_gpu_modified);
        DigestByte(digest, range.coarse_gpu_owned);
    }
    return digest;
}

u64 DigestShaderSelectionSrtContents(const SrtCaptureProvenance& provenance) noexcept {
    u64 digest = FnvOffset;
    DigestDomain(digest, "shadPS4.shader-selection-srt-contents.v1");
    DigestU64(digest, provenance.Ranges().size());
    DigestByte(digest, provenance.Truncated());
    for (const auto& range : provenance.Ranges()) {
        DigestU64(digest, range.content_digest);
    }
    return digest;
}

ShaderSelectionTrace::ShaderSelectionTrace() noexcept {
    const char* path = std::getenv(ShaderSelectionTraceEnvironment);
    if (path != nullptr && path[0] != '\0') {
        Open(path);
    }
}

ShaderSelectionTrace::ShaderSelectionTrace(const std::filesystem::path& path) noexcept {
    Open(path);
}

ShaderSelectionTrace::ShaderSelectionTrace(const std::filesystem::path& path,
                                           Limits limits_) noexcept
    : limits{limits_} {
    Open(path);
}

ShaderSelectionTrace::~ShaderSelectionTrace() {
    std::scoped_lock lock{mutex};
    DisableLocked();
}

void ShaderSelectionTrace::Open(const std::filesystem::path& path) noexcept {
    try {
        if (path.empty() || limits.max_records == 0 || limits.max_bytes == 0) {
            return;
        }
        const auto absolute = std::filesystem::absolute(path);
        if (absolute.has_parent_path()) {
            std::filesystem::create_directories(absolute.parent_path());
        }
#ifdef _WIN32
        file = _wfsopen(absolute.c_str(), L"wbx", _SH_DENYWR);
        if (file == nullptr) {
            return;
        }
#else
        file = std::fopen(absolute.c_str(), "wbx");
        if (file == nullptr) {
            return;
        }
#endif
        enabled.store(true, std::memory_order_relaxed);
    } catch (...) {
        file = nullptr;
        enabled.store(false, std::memory_order_relaxed);
    }
}

void ShaderSelectionTrace::DisableLocked() noexcept {
    enabled.store(false, std::memory_order_relaxed);
    if (file != nullptr) {
        std::fclose(file);
        file = nullptr;
    }
}

void ShaderSelectionTrace::Record(const ShaderSelectionTraceRecord& record) noexcept {
    if (!Enabled()) {
        return;
    }

    try {
        std::scoped_lock lock{mutex};
        if (!Enabled() || file == nullptr) {
            return;
        }
        if (record_count >= limits.max_records) {
            DisableLocked();
            return;
        }

        const auto action = CacheActionName(record.cache_action);
        std::array<char, 768> line{};
        const auto written = std::snprintf(
            line.data(), line.size(),
            "{\"schema\":1,\"kind\":\"shader_selection\",\"selection_sequence\":%" PRIu64
            ",\"dispatch_sequence\":%" PRIu64
            ",\"physical_stage\":%u,\"logical_stage\":%u,"
            "\"program_hash\":\"%016" PRIx64 "\","
            "\"srt_source_digest\":\"%016" PRIx64 "\","
            "\"srt_content_digest\":\"%016" PRIx64 "\","
            "\"semantic_identity\":\"%016" PRIx64 "\","
            "\"cache_action\":\"%.*s\",\"module_count\":%u,"
            "\"selected_permutation_id\":\"%016" PRIx64 "\","
            "\"compute_pipeline_id\":\"%016" PRIx64 "\"}\n",
            static_cast<u64>(record_count + 1), record.dispatch_sequence,
            static_cast<u32>(record.physical_stage), static_cast<u32>(record.logical_stage),
            record.program_hash, record.srt_source_digest, record.srt_content_digest,
            record.semantic_identity, static_cast<int>(action.size()), action.data(),
            record.module_count, record.selected_permutation_id, record.compute_pipeline_id);
        if (written <= 0 || static_cast<size_t>(written) >= line.size()) {
            DisableLocked();
            return;
        }
        const auto line_size = static_cast<size_t>(written);
        if (line_size > limits.max_bytes - std::min(byte_count, limits.max_bytes)) {
            DisableLocked();
            return;
        }
        if (std::fwrite(line.data(), 1, line_size, file) != line_size ||
            std::fflush(file) != 0) {
            DisableLocked();
            return;
        }
        ++record_count;
        byte_count += line_size;
    } catch (...) {
        // Lock acquisition itself is permitted to throw. The atomic gate is the only operation
        // needed to fail closed; the destructor will close an already-open file.
        enabled.store(false, std::memory_order_relaxed);
    }
}

} // namespace VideoCore
