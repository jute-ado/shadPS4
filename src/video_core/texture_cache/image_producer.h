// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>
#include <limits>
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
        AdvanceMutationEpoch();
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

    [[nodiscard]] u64 Epoch() const noexcept {
        return producer_epoch;
    }

    [[nodiscard]] u64 MutationEpoch() const noexcept {
        return mutation_overflow ? 0 : mutation_epoch;
    }

    void Reset() noexcept {
        AdvanceMutationEpoch();
        last_class = ImageProducerClass::Unknown;
        producer_epoch = 0;
        observed_epoch = 0;
        sampled_input_observed_epoch = 0;
    }

private:
    void AdvanceMutationEpoch() noexcept {
        if (mutation_overflow) {
            return;
        }
        if (mutation_epoch == std::numeric_limits<u64>::max()) {
            mutation_epoch = 0;
            mutation_overflow = true;
            return;
        }
        ++mutation_epoch;
    }

    ImageProducerClass last_class{ImageProducerClass::Unknown};
    u64 producer_epoch{};
    u64 observed_epoch{};
    u64 sampled_input_observed_epoch{};
    u64 mutation_epoch{};
    bool mutation_overflow{};
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
