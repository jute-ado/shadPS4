// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/types.h"

namespace VideoCore {

enum class ShaderCaptureSource : u8 {
    Refused,
    CpuBacking,
    GpuBuffer,
};

struct PotentialShaderWriteProvenance {
    u64 program_hash{};
    u32 stage{};
    u32 descriptor_index{};
    u64 mapping_token{};
    VAddr address{};
    u64 size{};
    u64 sequence{};

    bool operator==(const PotentialShaderWriteProvenance&) const = default;
};

class PotentialShaderWriteTracker {
public:
    static constexpr size_t MaxWrites = 256;

    void Record(bool enabled, u64 program_hash, u32 stage, u32 descriptor_index,
                u64 mapping_token, VAddr address, u64 size) noexcept {
        if (!enabled || mapping_token == 0 || size == 0 ||
            address > std::numeric_limits<VAddr>::max() - size) {
            return;
        }
        if (++sequence == 0) {
            ++sequence;
        }
        writes[next] = PotentialShaderWriteProvenance{
            .program_hash = program_hash,
            .stage = stage,
            .descriptor_index = descriptor_index,
            .mapping_token = mapping_token,
            .address = address,
            .size = size,
            .sequence = sequence,
        };
        next = (next + 1) % MaxWrites;
        count = std::min(count + 1, MaxWrites);
    }

    [[nodiscard]] std::optional<PotentialShaderWriteProvenance> LatestOverlap(
        VAddr address, u64 size, u64 mapping_token) const noexcept {
        if (mapping_token == 0 || size == 0 ||
            address > std::numeric_limits<VAddr>::max() - size) {
            return std::nullopt;
        }
        const VAddr end = address + size;
        // Walk the ring from newest to oldest. This remains correct even if the diagnostic
        // sequence counter wraps after an effectively unreachable number of records.
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t index = (next + MaxWrites - 1 - offset) % MaxWrites;
            const auto& write = writes[index];
            const VAddr write_end = write.address + write.size;
            if (write.mapping_token == mapping_token && address < write_end &&
                write.address < end) {
                return write;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] size_t RecordCount() const noexcept {
        return count;
    }

private:
    std::array<PotentialShaderWriteProvenance, MaxWrites> writes{};
    size_t count{};
    size_t next{};
    u64 sequence{};
};

struct SnapshotReadProvenance {
    bool completed{};
    bool exact_gpu_modified{};
    bool coarse_gpu_owned{};
    ShaderCaptureSource source{ShaderCaptureSource::Refused};
    std::optional<PotentialShaderWriteProvenance> potential_writer;
};

struct ShaderCaptureRangeProvenance {
    VAddr address{};
    u64 size{};
    bool exact_gpu_modified{};
    bool coarse_gpu_owned{};
    ShaderCaptureSource source{ShaderCaptureSource::Refused};
    u64 mapping_token{};
    u64 content_digest{};
    std::optional<PotentialShaderWriteProvenance> potential_writer;

    bool operator==(const ShaderCaptureRangeProvenance&) const = default;
};

[[nodiscard]] constexpr u64 DigestShaderCaptureContent(std::span<const u8> content) noexcept {
    // Stable FNV-1a is sufficient for comparing debug captures and keeps this optional path
    // independent of the persistent shader-cache hash ABI.
    u64 digest = 14695981039346656037ULL;
    for (const u8 byte : content) {
        digest ^= byte;
        digest *= 1099511628211ULL;
    }
    return digest;
}

class SrtCaptureProvenance {
public:
    static constexpr size_t MaxRanges = 64;

    void Record(const ShaderCaptureRangeProvenance& range) noexcept {
        if (range_count == MaxRanges) {
            truncated = true;
            return;
        }
        ranges[range_count++] = range;
    }

    [[nodiscard]] bool CanRecord() const noexcept {
        return range_count < MaxRanges;
    }

    void MarkTruncated() noexcept {
        truncated = true;
    }

    [[nodiscard]] std::span<const ShaderCaptureRangeProvenance> Ranges() const noexcept {
        return std::span{ranges}.first(range_count);
    }

    [[nodiscard]] bool Truncated() const noexcept {
        return truncated;
    }

private:
    std::array<ShaderCaptureRangeProvenance, MaxRanges> ranges{};
    size_t range_count{};
    bool truncated{};
};

struct ShaderCaptureProgramProvenance {
    u64 program_hash{};
    u32 stage{};
    bool succeeded{};
    bool ranges_truncated{};
    std::vector<ShaderCaptureRangeProvenance> ranges;
};

[[nodiscard]] constexpr std::string_view ShaderCaptureSourceName(
    ShaderCaptureSource source) noexcept {
    switch (source) {
    case ShaderCaptureSource::CpuBacking:
        return "cpu";
    case ShaderCaptureSource::GpuBuffer:
        return "gpu";
    case ShaderCaptureSource::Refused:
        return "refused";
    }
    return "refused";
}

[[nodiscard]] inline std::string FormatShaderCaptureProvenance(
    const ShaderCaptureProgramProvenance& program) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "shader_capture_provenance={\"program\":\"0x" << std::hex << std::setfill('0')
           << std::setw(16) << program.program_hash << "\",\"stage\":" << std::dec
           << program.stage << ",\"outcome\":\""
           << (program.succeeded ? "accepted" : "refused") << "\",\"truncated\":"
           << std::boolalpha << program.ranges_truncated << ",\"ranges\":[";
    for (size_t index = 0; index < program.ranges.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        const auto& range = program.ranges[index];
        output << "{\"address\":\"0x" << std::hex << std::setw(16) << range.address
               << "\",\"size\":" << std::dec << range.size << ",\"exact\":"
               << range.exact_gpu_modified << ",\"coarse\":" << range.coarse_gpu_owned
               << ",\"source\":\"" << ShaderCaptureSourceName(range.source)
               << "\",\"token\":" << range.mapping_token << ",\"digest\":\"0x" << std::hex
               << std::setw(16) << range.content_digest << "\",\"writer\":";
        if (range.potential_writer) {
            const auto& writer = *range.potential_writer;
            output << "{\"kind\":\"admitted-potential-write\",\"program\":\"0x"
                   << std::hex << std::setw(16)
                   << writer.program_hash << "\",\"stage\":" << std::dec << writer.stage
                   << ",\"descriptor\":" << writer.descriptor_index
                   << ",\"mapping_token\":" << writer.mapping_token
                   << ",\"sequence\":" << writer.sequence << ",\"address\":\"0x" << std::hex
                   << std::setw(16) << writer.address << "\",\"size\":" << std::dec
                   << writer.size << '}';
        } else {
            output << "null";
        }
        output << '}';
    }
    output << "]}";
    return std::move(output).str();
}

class ShaderCaptureProvenanceStore {
public:
    static constexpr size_t MaxPrograms = 256;

    bool Record(u64 program_hash, u32 stage, bool succeeded,
                const SrtCaptureProvenance& capture) {
        const auto duplicate = std::ranges::find_if(records, [&](const auto& record) {
            return record.program_hash == program_hash && record.stage == stage &&
                   record.succeeded == succeeded &&
                   record.ranges_truncated == capture.Truncated() &&
                   std::ranges::equal(record.ranges, capture.Ranges());
        });
        if (duplicate != records.end()) {
            return false;
        }
        if (records.size() == MaxPrograms) {
            ++dropped_programs;
            return false;
        }
        records.emplace_back(ShaderCaptureProgramProvenance{
            .program_hash = program_hash,
            .stage = stage,
            .succeeded = succeeded,
            .ranges_truncated = capture.Truncated(),
            .ranges = {capture.Ranges().begin(), capture.Ranges().end()},
        });
        return true;
    }

    [[nodiscard]] std::span<const ShaderCaptureProgramProvenance> Records() const noexcept {
        return records;
    }

    [[nodiscard]] u64 DroppedPrograms() const noexcept {
        return dropped_programs;
    }

private:
    std::vector<ShaderCaptureProgramProvenance> records;
    u64 dropped_programs{};
};

class ShaderCaptureProvenanceRecorder {
public:
    [[nodiscard]] std::optional<std::string> Record(
        bool enabled, u64 program_hash, u32 stage, bool succeeded,
        const SrtCaptureProvenance& capture) {
        if (!enabled || !store.Record(program_hash, stage, succeeded, capture)) {
            return std::nullopt;
        }
        return FormatShaderCaptureProvenance(store.Records().back());
    }

    [[nodiscard]] const ShaderCaptureProvenanceStore& Store() const noexcept {
        return store;
    }

private:
    ShaderCaptureProvenanceStore store;
};

} // namespace VideoCore
