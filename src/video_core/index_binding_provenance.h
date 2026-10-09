// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>

namespace VideoCore {

struct IndexBindingProvenancePolicy {
    bool enabled{};
    bool valid{true};
    std::uint32_t start_frame{470};
    std::optional<std::uint64_t> vertex_hash;

    [[nodiscard]] static IndexBindingProvenancePolicy Parse(
        std::optional<std::string_view> enable, std::optional<std::string_view> start,
        std::optional<std::string_view> hash) noexcept {
        if (!enable || *enable == "0") {
            return {};
        }
        IndexBindingProvenancePolicy result{.valid = false};
        if (*enable != "1") {
            return result;
        }
        const auto parse = [](std::string_view text, auto& value, int base) {
            if (text.empty() || text.front() == '+' || text.front() == '-') {
                return false;
            }
            const auto [end, error] =
                std::from_chars(text.data(), text.data() + text.size(), value, base);
            return error == std::errc{} && end == text.data() + text.size();
        };
        if (start && (!parse(*start, result.start_frame, 10) || result.start_frame > 1000000)) {
            return result;
        }
        if (hash) {
            auto text = *hash;
            if (text.starts_with("0x") || text.starts_with("0X")) {
                text.remove_prefix(2);
            }
            std::uint64_t value{};
            if (!parse(text, value, 16)) {
                return result;
            }
            result.vertex_hash = value;
        }
        result.enabled = result.valid = true;
        return result;
    }
};

// Observation of the current preparation arithmetic, not a capacity policy. The wide
// value exposes truncation without changing the existing u32 preparation behavior.
struct IndexBindingPreparationSpan {
    std::uint64_t wide_bytes{};
    std::uint32_t prepared_bytes{};
    bool truncated{};
};

[[nodiscard]] constexpr IndexBindingPreparationSpan DescribeIndexBindingPreparation(
    std::uint32_t num_indices, std::uint32_t index_size) noexcept {
    const std::uint64_t wide = std::uint64_t{num_indices} * index_size;
    const auto prepared = static_cast<std::uint32_t>(wide);
    return {wide, prepared, wide != prepared};
}

// Renderer-owner-thread only; never reads guest memory or GPU indirect arguments.
class IndexBindingProvenanceReceipt {
public:
    static constexpr std::uint32_t Capacity = 16;

    explicit IndexBindingProvenanceReceipt(IndexBindingProvenancePolicy policy_ = {})
        : policy{policy_} {}

    [[nodiscard]] bool Enabled() const noexcept {
        return policy.enabled && policy.valid && recorded < Capacity;
    }

    [[nodiscard]] std::optional<std::uint32_t> Observe(std::int32_t frame,
                                                       std::uint64_t hash) noexcept {
        if (!Enabled() || frame < 0 || static_cast<std::uint32_t>(frame) < policy.start_frame ||
            (policy.vertex_hash && *policy.vertex_hash != hash)) {
            return std::nullopt;
        }
        return recorded++;
    }

    void Disable() noexcept {
        policy.enabled = false;
    }

private:
    IndexBindingProvenancePolicy policy;
    std::uint32_t recorded{};
};

} // namespace VideoCore
