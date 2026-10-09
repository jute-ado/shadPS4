// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <limits>
#include <optional>

namespace AmdGpu {

// The INDEX_BUFFER_SIZE packet targets VGT_DMA_SIZE. It is retained independently
// from VGT_NUM_INDICES, which direct and auto draws update for the draw itself.
template <typename Registers>
void ApplyIndexBufferSize(Registers& regs, std::uint32_t size) noexcept {
    regs.index_size = size;
}

template <typename Registers>
void ApplyDirectIndexDrawSize(Registers& regs, std::uint32_t max_size,
                              std::uint32_t index_count) noexcept {
    regs.max_index_size = max_size;
    regs.num_indices = index_count;
}

template <typename Registers>
void ApplyAutoIndexDrawSize(Registers& regs, std::uint32_t index_count) noexcept {
    regs.num_indices = index_count;
}

// The indirect index-buffer preparation consumer uses VGT_DMA_SIZE, not the last
// direct/auto draw count in VGT_NUM_INDICES.
template <typename Registers>
[[nodiscard]] constexpr std::uint32_t CurrentIndexPreparationCount(const Registers& regs) noexcept {
    return regs.index_size;
}

template <typename Registers>
[[nodiscard]] constexpr std::uint64_t CurrentIndexPreparationBytes(
    const Registers& regs, std::uint32_t index_size) noexcept {
    return std::uint64_t{CurrentIndexPreparationCount(regs)} * index_size;
}

template <typename Registers>
[[nodiscard]] constexpr std::uint64_t CurrentDirectIndexPreparationBytes(
    const Registers& regs, std::uint32_t index_size) noexcept {
    return std::uint64_t{regs.num_indices} * index_size;
}

[[nodiscard]] constexpr std::optional<std::uint64_t> TryGetIndexAddress(
    std::uint64_t base_address, std::uint32_t index_offset,
    std::uint32_t index_size) noexcept {
    const auto byte_offset = std::uint64_t{index_offset} * index_size;
    if (base_address > std::numeric_limits<std::uint64_t>::max() - byte_offset) {
        return std::nullopt;
    }
    return base_address + byte_offset;
}

struct IndirectBufferRange {
    std::uint64_t address{};
    std::uint32_t size{};
};

[[nodiscard]] constexpr std::optional<IndirectBufferRange> TryGetIndirectBufferRange(
    std::uint64_t base_address, std::uint64_t offset, std::uint64_t size) noexcept {
    if (base_address > std::numeric_limits<std::uint64_t>::max() - offset) {
        return std::nullopt;
    }
    const auto address = base_address + offset;
    if (size > std::numeric_limits<std::uint32_t>::max() ||
        size > std::numeric_limits<std::uint64_t>::max() - address) {
        return std::nullopt;
    }
    return IndirectBufferRange{address, static_cast<std::uint32_t>(size)};
}

template <typename Registers>
[[nodiscard]] constexpr std::optional<std::uint32_t> TryGetIndexPreparationBytes(
    const Registers& regs, std::uint32_t index_size, bool indirect) noexcept {
    const auto bytes = indirect ? CurrentIndexPreparationBytes(regs, index_size)
                                : CurrentDirectIndexPreparationBytes(regs, index_size);
    if (bytes > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(bytes);
}

} // namespace AmdGpu
