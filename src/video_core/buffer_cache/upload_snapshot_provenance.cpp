// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/buffer_cache/upload_snapshot_provenance.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <vector>
#include <xxhash.h>

#include "common/logging/log.h"
#include "core/memory.h"

namespace VideoCore {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<std::string_view, 4> ClassificationNames{
    "stable", "copied_old", "copied_new", "copy_mismatch"};

constexpr std::string_view PathName(UploadSnapshotPath path) noexcept {
    switch (path) {
    case UploadSnapshotPath::DirectStream:
        return "direct_stream";
    case UploadSnapshotPath::CachedStaging:
        return "cached_staging";
    }
    return "unknown";
}

u64 ReadUnsignedEnvironment(const char* name, u64 fallback) noexcept {
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return fallback;
    }
    char* end{};
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end != value && *end == '\0' ? static_cast<u64>(parsed) : fallback;
}

bool DiagnosticEnabled() noexcept {
    const char* value = std::getenv("SHADPS4_UPLOAD_SNAPSHOT_DIAGNOSTIC");
    return value && std::string_view{value} != "0" && !std::string_view{value}.empty();
}

struct UploadSnapshotCounters {
    std::atomic<u64> samples{};
    std::atomic<u64> stable{};
    std::atomic<u64> copied_old{};
    std::atomic<u64> copied_new{};
    std::atomic<u64> copy_mismatch{};
    std::atomic<u64> oversize{};
    std::atomic<u64> anomaly_logs{};
    std::atomic<bool> announced{};
    std::atomic<bool> summarized{};
};

struct UploadSnapshotConfiguration {
    bool enabled{DiagnosticEnabled()};
    u64 start_ms{ReadUnsignedEnvironment("SHADPS4_UPLOAD_SNAPSHOT_START_MS", 55000)};
    u64 duration_ms{ReadUnsignedEnvironment("SHADPS4_UPLOAD_SNAPSHOT_DURATION_MS", 45000)};
    u64 maximum_samples{ReadUnsignedEnvironment("SHADPS4_UPLOAD_SNAPSHOT_MAX_SAMPLES", 500000)};
    u64 maximum_bytes{ReadUnsignedEnvironment("SHADPS4_UPLOAD_SNAPSHOT_MAX_BYTES", 65536)};
    u64 maximum_anomaly_logs{
        ReadUnsignedEnvironment("SHADPS4_UPLOAD_SNAPSHOT_MAX_ANOMALY_LOGS", 512)};
    Clock::time_point process_start{Clock::now()};
    UploadSnapshotCounters counters;
};

UploadSnapshotConfiguration& Configuration() {
    static UploadSnapshotConfiguration configuration;
    return configuration;
}

void LogSummary(UploadSnapshotConfiguration& config, u64 elapsed_ms) {
    if (config.counters.summarized.exchange(true)) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "UploadSnapshotSummary elapsed_ms={} samples={} stable={} copied_old={} "
             "copied_new={} copy_mismatch={} oversize={}",
             elapsed_ms, config.counters.samples.load(), config.counters.stable.load(),
             config.counters.copied_old.load(), config.counters.copied_new.load(),
             config.counters.copy_mismatch.load(), config.counters.oversize.load());
}

} // namespace

void CopyGuestMemoryWithUploadProvenance(VAddr source, u8* destination, u64 size,
                                         UploadSnapshotPath path, u64 scheduler_tick) {
    auto* memory = Core::Memory::Instance();
    auto& config = Configuration();
    if (!config.enabled) {
        memory->CopySparseMemory(source, destination, size);
        return;
    }

    const u64 elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               Clock::now() - config.process_start)
                               .count();
    const u64 end_ms = config.start_ms + config.duration_ms;
    if (elapsed_ms < config.start_ms || elapsed_ms >= end_ms) {
        memory->CopySparseMemory(source, destination, size);
        if (elapsed_ms >= end_ms) {
            LogSummary(config, elapsed_ms);
        }
        return;
    }

    if (!config.counters.announced.exchange(true)) {
        LOG_INFO(Render_Vulkan,
                 "UploadSnapshotEnabled start_ms={} duration_ms={} max_samples={} max_bytes={} "
                 "max_anomaly_logs={}",
                 config.start_ms, config.duration_ms, config.maximum_samples,
                 config.maximum_bytes, config.maximum_anomaly_logs);
    }

    if (size > config.maximum_bytes ||
        config.counters.samples.load(std::memory_order_relaxed) >= config.maximum_samples) {
        if (size > config.maximum_bytes) {
            ++config.counters.oversize;
        }
        memory->CopySparseMemory(source, destination, size);
        return;
    }

    ++config.counters.samples;
    thread_local std::vector<u8> before;
    thread_local std::vector<u8> after;
    before.resize(static_cast<size_t>(size));
    after.resize(static_cast<size_t>(size));
    memory->CopySparseMemory(source, before.data(), size);
    memory->CopySparseMemory(source, destination, size);
    memory->CopySparseMemory(source, after.data(), size);

    const u64 before_hash = XXH3_64bits(before.data(), before.size());
    const u64 copied_hash = XXH3_64bits(destination, static_cast<size_t>(size));
    const u64 after_hash = XXH3_64bits(after.data(), after.size());
    const auto classification = ClassifyUploadSnapshot(before_hash, copied_hash, after_hash);

    std::atomic<u64>* counter{};
    switch (classification) {
    case UploadSnapshotClassification::Stable:
        counter = &config.counters.stable;
        break;
    case UploadSnapshotClassification::CopiedOldState:
        counter = &config.counters.copied_old;
        break;
    case UploadSnapshotClassification::CopiedNewState:
        counter = &config.counters.copied_new;
        break;
    case UploadSnapshotClassification::CopyMismatch:
        counter = &config.counters.copy_mismatch;
        break;
    }
    ++*counter;

    if (classification != UploadSnapshotClassification::Stable &&
        config.counters.anomaly_logs.fetch_add(1) < config.maximum_anomaly_logs) {
        const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
        LOG_INFO(Render_Vulkan,
                 "UploadSnapshotAnomaly wall_ms={} elapsed_ms={} path={} size={} tick={} class={} "
                 "before_hash={:016x} copied_hash={:016x} after_hash={:016x}",
                 wall_ms, elapsed_ms, PathName(path), size, scheduler_tick,
                 ClassificationNames[static_cast<size_t>(classification)], before_hash,
                 copied_hash, after_hash);
    }
}

} // namespace VideoCore
