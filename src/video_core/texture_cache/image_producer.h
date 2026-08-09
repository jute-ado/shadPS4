// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>
#include <string_view>

#include "common/types.h"

namespace VideoCore {

enum class ImageProducerClass : u8 {
    Unknown,
    ColorAttachment,
    StorageImage,
    Transfer,
    CpuUpload,
};

struct ImageProducerObservation {
    ImageProducerClass classification{ImageProducerClass::Unknown};
    bool produced_since_last_observation{};

    auto operator<=>(const ImageProducerObservation&) const = default;
};

class ImageProducerState {
public:
    void Mark(const ImageProducerClass classification) noexcept {
        last_class = classification;
        ++producer_epoch;
    }

    [[nodiscard]] ImageProducerObservation Observe() noexcept {
        const bool produced = producer_epoch != observed_epoch;
        observed_epoch = producer_epoch;
        return {.classification = last_class, .produced_since_last_observation = produced};
    }

    [[nodiscard]] ImageProducerObservation ObserveSampledInput() noexcept {
        const bool produced = producer_epoch != sampled_input_observed_epoch;
        sampled_input_observed_epoch = producer_epoch;
        return {.classification = last_class, .produced_since_last_observation = produced};
    }

    [[nodiscard]] ImageProducerObservation Peek() const noexcept {
        return {
            .classification = last_class,
            .produced_since_last_observation = producer_epoch != observed_epoch,
        };
    }

    void Reset() noexcept {
        last_class = ImageProducerClass::Unknown;
        producer_epoch = 0;
        observed_epoch = 0;
        sampled_input_observed_epoch = 0;
    }

private:
    ImageProducerClass last_class{ImageProducerClass::Unknown};
    u64 producer_epoch{};
    u64 observed_epoch{};
    u64 sampled_input_observed_epoch{};
};

[[nodiscard]] inline bool IsPpSourceProducerTrackingEnabled() noexcept {
    static const bool enabled = [] {
#ifdef _WIN32
        char* stage{};
        size_t length{};
        if (_dupenv_s(&stage, &length, "SHADPS4_FINAL_GUEST_SURFACE_STAGE") != 0 || !stage) {
            return false;
        }
        const bool matches = std::string_view{stage} == "pp_source_publication_reconstruction";
        std::free(stage);
        return matches;
#else
        const char* stage = std::getenv("SHADPS4_FINAL_GUEST_SURFACE_STAGE");
        return stage && std::string_view{stage} == "pp_source_publication_reconstruction";
#endif
    }();
    return enabled;
}

} // namespace VideoCore
