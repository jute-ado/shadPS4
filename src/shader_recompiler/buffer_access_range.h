// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>

#include "common/types.h"

namespace Shader {

struct BufferAddressLayout {
    u32 stride{};
    u8 data_format{};
    u8 element_size{};
    u8 index_stride{};
    bool swizzle_enable{};
    bool add_tid_enable{};

    template <typename Buffer>
    [[nodiscard]] static constexpr BufferAddressLayout From(const Buffer& buffer) noexcept {
        return BufferAddressLayout{
            .stride = static_cast<u32>(buffer.stride),
            .data_format = static_cast<u8>(buffer.data_format),
            .element_size = static_cast<u8>(buffer.element_size),
            .index_stride = static_cast<u8>(buffer.index_stride),
            .swizzle_enable = static_cast<bool>(buffer.swizzle_enable),
            .add_tid_enable = static_cast<bool>(buffer.add_tid_enable),
        };
    }

    bool operator==(const BufferAddressLayout&) const = default;
};

class BufferAccessRange {
public:
    constexpr void RecordAddressLayout(BufferAddressLayout value) noexcept {
        if (!has_address_layout) {
            address_layout = value;
            has_address_layout = true;
        } else if (address_layout != value) {
            MarkDynamic();
        }
    }

    constexpr void Add(u64 byte_offset, u64 byte_size) noexcept {
        if (byte_offset > std::numeric_limits<u64>::max() - byte_size) {
            MarkDynamic();
            return;
        }
        upper_bound = std::max(upper_bound, byte_offset + byte_size);
    }

    constexpr void MarkDynamic() noexcept {
        is_bounded = false;
    }

    [[nodiscard]] constexpr bool IsBounded() const noexcept {
        return is_bounded;
    }

    [[nodiscard]] constexpr u64 UpperBound() const noexcept {
        return upper_bound;
    }

    [[nodiscard]] constexpr u64 Fit(u64 declared_size) const noexcept {
        if (!is_bounded || upper_bound == 0) {
            return declared_size;
        }
        return std::min(declared_size, upper_bound);
    }

    [[nodiscard]] constexpr u64 Fit(u64 declared_size,
                                    BufferAddressLayout runtime_layout) const noexcept {
        if (!has_address_layout || address_layout != runtime_layout) {
            return declared_size;
        }
        return Fit(declared_size);
    }

private:
    u64 upper_bound{};
    BufferAddressLayout address_layout{};
    bool is_bounded{true};
    bool has_address_layout{};
};

static_assert(std::is_trivially_copyable_v<BufferAddressLayout>);
static_assert(std::is_trivially_copyable_v<BufferAccessRange>);

} // namespace Shader

