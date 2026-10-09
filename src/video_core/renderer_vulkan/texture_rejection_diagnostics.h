// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string_view>

namespace Vulkan {

[[nodiscard]] constexpr bool ParseTextureRejectionDiagnostics(std::string_view value) noexcept {
    return value == "1";
}

[[nodiscard]] inline bool TextureRejectionDiagnosticsEnabled() noexcept {
    // Launch-time opt-in, shared by binding and failure-only allocation diagnostics.
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_TEXTURE_REJECTION_DIAGNOSTICS");
        return value && ParseTextureRejectionDiagnostics(value);
    }();
    return enabled;
}

[[nodiscard]] constexpr std::uint32_t TextureRejectionReasons(bool address_40bit,
                                                              bool valid_data_format,
                                                              bool valid_number_format) noexcept {
    // Raw pitch is not a validity predicate: block/tile/power-of-two padding follows.
    // Preserve historical reason identities; mask 0x2 belonged to the removed pitch veto.
    return (!address_40bit ? 1U : 0U) | (!valid_data_format ? 4U : 0U) |
           (!valid_number_format ? 8U : 0U);
}

struct TextureRejectionObservation {
    std::uint64_t shader_hash{};
    std::uint32_t stage{};
    std::uint32_t logical_stage{};
    std::uint32_t sharp_index{};
    std::uint32_t array_index{};
    std::uint32_t descriptor_count{};
    std::uint32_t descriptor_stride_dw{};
    std::uint32_t reasons{};
    bool is_r128{};
    bool base_byte_logical_gpu_mapped{};
    std::array<std::uint32_t, 8> words{};

    bool operator==(const TextureRejectionObservation&) const = default;
};

// Used only on the serialized GPU-owner binding path. Own copies, never guest pointers.
class TextureRejectionDiagnostics {
public:
    static constexpr std::size_t MaxObservations = 32;

    explicit TextureRejectionDiagnostics(bool enabled) : enabled{enabled} {}

    template <typename BuildObservation>
    [[nodiscard]] std::optional<TextureRejectionObservation> Observe(BuildObservation&& build) {
        if (!enabled || count == MaxObservations) {
            return std::nullopt;
        }
        const TextureRejectionObservation observation = build();
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
    std::array<TextureRejectionObservation, MaxObservations> observations{};
};

// Synchronous FindImage context, also owned: never a pointer into guest or SRT memory.
class ScopedTextureDiagnosticContext {
public:
    template <typename BuildObservation>
    ScopedTextureDiagnosticContext(bool enabled, BuildObservation&& build) {
        if (enabled) {
            observation = build();
            previous = current;
            current = &*observation;
        }
    }

    ~ScopedTextureDiagnosticContext() {
        if (observation) {
            current = previous;
        }
    }

    ScopedTextureDiagnosticContext(const ScopedTextureDiagnosticContext&) = delete;
    ScopedTextureDiagnosticContext& operator=(const ScopedTextureDiagnosticContext&) = delete;

    [[nodiscard]] static const TextureRejectionObservation* Current() noexcept {
        return current;
    }

private:
    inline static thread_local const TextureRejectionObservation* current{};
    const TextureRejectionObservation* previous{};
    std::optional<TextureRejectionObservation> observation;
};

} // namespace Vulkan
