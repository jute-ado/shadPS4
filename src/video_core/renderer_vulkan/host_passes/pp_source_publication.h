// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string>

#include "common/types.h"
#include "video_core/texture_cache/image_refresh_result.h"

namespace Vulkan::HostPasses {

enum class PpSourcePublicationClass : u8 {
    CleanGpuTracked,
    CleanResident,
    DirtyNoUpload,
    CpuUpload,
    Unsupported,
};

struct PpSourcePublicationDescriptor {
    VideoCore::ImageRefreshResult refresh{};
    bool gpu_modified_before{};
};

[[nodiscard]] constexpr PpSourcePublicationClass ClassifyPpSourcePublication(
    PpSourcePublicationDescriptor descriptor) noexcept {
    switch (descriptor.refresh) {
    case VideoCore::ImageRefreshResult::Clean:
        return descriptor.gpu_modified_before ? PpSourcePublicationClass::CleanGpuTracked
                                              : PpSourcePublicationClass::CleanResident;
    case VideoCore::ImageRefreshResult::MaybeCpuDirtyUnchanged:
    case VideoCore::ImageRefreshResult::GpuModifiedUnchanged:
        return PpSourcePublicationClass::DirtyNoUpload;
    case VideoCore::ImageRefreshResult::Uploaded:
        return PpSourcePublicationClass::CpuUpload;
    case VideoCore::ImageRefreshResult::MultisampledDirty:
        return PpSourcePublicationClass::Unsupported;
    }
    return PpSourcePublicationClass::Unsupported;
}

[[nodiscard]] constexpr PpSourcePublicationClass ClassifyPpSourcePublication(
    VideoCore::ImageRefreshObservation observation) noexcept {
    return ClassifyPpSourcePublication({
        .refresh = observation.result,
        .gpu_modified_before = observation.gpu_modified_before,
    });
}

struct PpSourcePublicationWindow {
    u64 start{};
    u32 count{};

    [[nodiscard]] constexpr bool Contains(u64 sequence) const noexcept {
        return count != 0 && sequence >= start && sequence - start < count;
    }

    [[nodiscard]] constexpr bool IsFinal(u64 sequence) const noexcept {
        return count != 0 && sequence >= start && sequence - start == count - 1;
    }
};

struct PpSourcePublicationObservation {
    u64 sequence{};
    PpSourcePublicationClass classification{PpSourcePublicationClass::Unsupported};
    u32 selected{};
    u32 emitted{};
    u32 expected{};
    u32 clean_gpu_tracked{};
    u32 clean_resident{};
    u32 dirty_no_upload{};
    u32 cpu_upload{};
    u32 unsupported{};
    u32 loss{};
    bool final{};
};

class PpSourcePublicationCoverage {
public:
    explicit constexpr PpSourcePublicationCoverage(PpSourcePublicationWindow window_) noexcept
        : window{window_} {}

    [[nodiscard]] std::optional<PpSourcePublicationObservation> Observe(
        u64 sequence, PpSourcePublicationClass classification) noexcept {
        if (!window.Contains(sequence)) {
            return std::nullopt;
        }
        ++selected;
        ++emitted;
        if (has_last && sequence != last_sequence + 1) {
            ++loss;
        }
        has_last = true;
        last_sequence = sequence;
        switch (classification) {
        case PpSourcePublicationClass::CleanGpuTracked:
            ++clean_gpu_tracked;
            break;
        case PpSourcePublicationClass::CleanResident:
            ++clean_resident;
            break;
        case PpSourcePublicationClass::DirtyNoUpload:
            ++dirty_no_upload;
            break;
        case PpSourcePublicationClass::CpuUpload:
            ++cpu_upload;
            break;
        case PpSourcePublicationClass::Unsupported:
            ++unsupported;
            break;
        }
        const bool final = window.IsFinal(sequence);
        if (final && emitted != window.count) {
            ++loss;
        }
        return PpSourcePublicationObservation{
            .sequence = sequence,
            .classification = classification,
            .selected = selected,
            .emitted = emitted,
            .expected = window.count,
            .clean_gpu_tracked = clean_gpu_tracked,
            .clean_resident = clean_resident,
            .dirty_no_upload = dirty_no_upload,
            .cpu_upload = cpu_upload,
            .unsupported = unsupported,
            .loss = loss,
            .final = final,
        };
    }

private:
    PpSourcePublicationWindow window{};
    u64 last_sequence{};
    u32 selected{};
    u32 emitted{};
    u32 clean_gpu_tracked{};
    u32 clean_resident{};
    u32 dirty_no_upload{};
    u32 cpu_upload{};
    u32 unsupported{};
    u32 loss{};
    bool has_last{};
};

[[nodiscard]] inline std::string FormatPpSourcePublicationObservation(
    const PpSourcePublicationObservation& observation) {
    return "FGSCP s=" + std::to_string(observation.sequence) +
           " r=" + std::to_string(static_cast<u32>(observation.classification));
}

[[nodiscard]] inline std::string FormatPpSourcePublicationCoverage(
    const PpSourcePublicationObservation& observation) {
    return "FGSCPC s=" + std::to_string(observation.sequence) +
           " n=" + std::to_string(observation.emitted) + '/' +
           std::to_string(observation.selected) + '/' + std::to_string(observation.expected) +
           " g=" + std::to_string(observation.clean_gpu_tracked) +
           " r=" + std::to_string(observation.clean_resident) +
           " d=" + std::to_string(observation.dirty_no_upload) +
           " u=" + std::to_string(observation.cpu_upload) +
           " x=" + std::to_string(observation.unsupported) +
           " l=" + std::to_string(observation.loss);
}

} // namespace Vulkan::HostPasses
