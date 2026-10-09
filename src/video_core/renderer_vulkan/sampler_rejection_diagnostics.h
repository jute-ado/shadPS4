// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace Vulkan {

[[nodiscard]] constexpr std::uint32_t SamplerRejectionReasons(std::uint64_t raw0,
                                                              std::uint64_t raw1) noexcept {
    // Values outside the current decoder's supported enums, not inferred hardware behavior.
    return (((raw0 >> 9) & 7U) > 4U ? 1U : 0U) | (((raw0 >> 29) & 3U) > 2U ? 2U : 0U) |
           (((raw1 >> 26) & 3U) > 2U ? 4U : 0U);
}

struct SamplerRejectionObservation {
    std::uint64_t shader_hash{};
    std::uint32_t stage{};
    std::uint32_t logical_stage{};
    std::uint32_t binding{};
    std::uint32_t sharp_index{};
    std::uint32_t reasons{};
    bool is_inline{};
    bool disable_aniso{};
    bool host_anisotropy_supported{};
    std::uint32_t associated_image{};
    std::uint32_t associated_sharp_index{};
    std::uint32_t associated_descriptor_dwords{};
    std::array<std::uint32_t, 4> words{};
    std::array<std::uint32_t, 4> effective_words{};
    std::array<std::uint32_t, 8> associated_words{};

    bool operator==(const SamplerRejectionObservation&) const = default;
};

// GPU-owner-only diagnostics. The tuple owns its data; it never retains a guest pointer.
class SamplerRejectionDiagnostics {
public:
    static constexpr std::size_t MaxObservations = 32;

    explicit SamplerRejectionDiagnostics(bool enabled) : enabled{enabled} {}

    template <typename BuildObservation>
    [[nodiscard]] std::optional<SamplerRejectionObservation> Observe(BuildObservation&& build) {
        if (!enabled || count == MaxObservations) {
            return std::nullopt;
        }
        const SamplerRejectionObservation observation = build();
        for (std::size_t index = 0; index < count; ++index) {
            if (observations[index] == observation) {
                return std::nullopt;
            }
        }
        observations[count++] = observation;
        return observation;
    }

    [[nodiscard]] std::size_t RecordedCount() const noexcept {
        return count;
    }

private:
    bool enabled{};
    std::size_t count{};
    std::array<SamplerRejectionObservation, MaxObservations> observations{};
};

} // namespace Vulkan
