// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtRangeRead = bool PS4_SYSV_ABI (*)(void* /*context*/, VAddr /*address*/,
                                               void* /*output*/, u64 /*size*/);

struct SrtMemoryReader {
    PFN_SrtRangeRead callback{};
    void* context{};
    const bool* read_failed{};

    [[nodiscard]] bool Valid() const noexcept {
        return read_failed == nullptr || !*read_failed;
    }
};

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/,
                                            PFN_SrtRangeRead /*read_callback*/,
                                            void* /*read_context*/);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    struct SrtRangeReservation {
        u32 pointer_lo_flat_index{};
        u32 source_offset_dw{};
        u32 destination_flat_offset_dw{};
        u32 span_dw{};

        bool operator==(const SrtRangeReservation&) const noexcept = default;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs
    std::vector<SrtRangeReservation> range_reservations;

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
