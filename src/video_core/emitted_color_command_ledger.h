// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <charconv>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace VideoCore {

struct ColorCommandLedgerPolicy {
    bool enabled{};
    bool valid{true};
    std::uint64_t start_ms{};
    std::uint64_t duration_ms{};

    [[nodiscard]] static ColorCommandLedgerPolicy Parse(
        std::optional<std::string_view> enable, std::optional<std::string_view> start,
        std::optional<std::string_view> duration) noexcept {
        if (!enable || *enable == "0") {
            return {};
        }
        ColorCommandLedgerPolicy result{.valid = false};
        const auto number = [](std::optional<std::string_view> text, std::uint64_t& value) {
            if (!text || text->empty() || text->front() < '0' || text->front() > '9') {
                return false;
            }
            const auto [end, error] =
                std::from_chars(text->data(), text->data() + text->size(), value);
            return error == std::errc{} && end == text->data() + text->size();
        };
        if (*enable != "1" || !number(start, result.start_ms) ||
            !number(duration, result.duration_ms) || result.start_ms > 600000 ||
            result.duration_ms == 0 || result.duration_ms > 5000) {
            return result;
        }
        result.enabled = result.valid = true;
        return result;
    }
};

enum class ColorCommandKind : std::uint32_t {
    Direct,
    Indexed,
    Indirect,
    IndexedIndirect,
    IndirectCount,
    IndexedIndirectCount,
};

struct ColorCommandTarget {
    bool bound{};
    std::uint32_t format{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t layers{};
    // Runtime image identity is intentionally not a guest or host address. It lets the bounded
    // diagnostic distinguish two target images with identical view metadata without leaking
    // private game memory layout into logs.
    std::uint64_t resource_id{};
    auto operator<=>(const ColorCommandTarget&) const = default;
};

struct ColorCommandDepthTarget {
    bool bound{};
    std::uint64_t resource_id{};
    std::uint32_t format{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t layers{};
    bool has_depth{};
    bool has_stencil{};
    bool depth_clear{};
    bool stencil_clear{};
    auto operator<=>(const ColorCommandDepthTarget&) const = default;
};

struct ColorCommandKey {
    std::uint64_t vertex_hash{};
    std::uint64_t fragment_hash{};
    std::uint64_t pipeline_key_hash{};
    bool has_fragment{};
    ColorCommandKind kind{};
    std::array<ColorCommandTarget, 8> colors{};
    bool has_presentable_color{};
    ColorCommandDepthTarget depth{};
    auto operator<=>(const ColorCommandKey&) const = default;
};

// Only direct commands have known element/instance counts here. The indirect
// API count or limit is not an index count, instance count, or executed-draw count.
struct EmittedColorCommand {
    ColorCommandKey key;
    std::uint32_t elements{};
    std::uint32_t instances{};
    std::uint32_t command_count{};
    std::uint32_t max_command_count{};
    std::uint32_t stride{};
};

enum class ColorLedgerEnd { Window, Capacity, Teardown };

struct ColorLedgerSummary {
    ColorLedgerEnd reason{};
    std::uint64_t elapsed_ms{};
    std::uint64_t observed{};
    std::uint64_t duplicates{};
    std::size_t unique{};
};

class EmittedColorCommandLedger {
public:
    static constexpr std::size_t Capacity = 32;

    explicit EmittedColorCommandLedger(ColorCommandLedgerPolicy policy_) : policy{policy_} {}

    [[nodiscard]] bool Enabled() const noexcept {
        return policy.enabled && policy.valid && !closed;
    }

    template <typename Build, typename Row, typename End>
    void Observe(std::uint64_t elapsed_ms, Build&& build, Row&& row, End&& end) {
        if (!Enabled() || elapsed_ms < policy.start_ms) {
            return;
        }
        if (elapsed_ms >= policy.start_ms + policy.duration_ms) {
            Finish(ColorLedgerEnd::Window, elapsed_ms, end);
            return;
        }
        const auto command = build();
        if (!command) {
            return;
        }
        ++observed;
        for (std::size_t index = 0; index < unique; ++index) {
            if (keys[index] == command->key) {
                ++duplicates;
                return;
            }
        }
        if (unique == Capacity) {
            Finish(ColorLedgerEnd::Capacity, elapsed_ms, end);
            return;
        }
        const auto ordinal = unique++;
        keys[ordinal] = command->key;
        row(ordinal, elapsed_ms, *command);
    }

    template <typename End>
    void Close(std::uint64_t elapsed_ms, End&& end) {
        if (Enabled()) {
            Finish(ColorLedgerEnd::Teardown, elapsed_ms, end);
        }
    }

    // Diagnostic reporting failure must not add work or exceptions to rendering.
    void Disable() noexcept {
        closed = true;
    }

private:
    template <typename End>
    void Finish(ColorLedgerEnd reason, std::uint64_t elapsed_ms, End&& end) {
        closed = true;
        end(ColorLedgerSummary{reason, elapsed_ms, observed, duplicates, unique});
    }

    ColorCommandLedgerPolicy policy;
    std::array<ColorCommandKey, Capacity> keys{};
    std::size_t unique{};
    std::uint64_t observed{};
    std::uint64_t duplicates{};
    bool closed{};
};

} // namespace VideoCore
