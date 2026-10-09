#pragma once

#include <limits>

#include "common/types.h"

namespace Vulkan {

inline constexpr u64 MaxGpuAddress = 1ull << 40;
static_assert(MaxGpuAddress == 0x10000000000ull);

enum class GpuSpanAdmissionReason { Accepted, AddressOverflow, SizeOverflow, Unmapped };

inline GpuSpanAdmissionReason CheckGpuSpanAdmission(VAddr base, u64 offset, u64 size,
                                                    bool mapped) {
    const u64 base_address = static_cast<u64>(base);
    if (base_address >= MaxGpuAddress || offset > MaxGpuAddress - base_address) {
        return GpuSpanAdmissionReason::AddressOverflow;
    }
    const u64 start = base_address + offset;
    if (size > MaxGpuAddress - start || size > std::numeric_limits<u32>::max()) {
        return GpuSpanAdmissionReason::SizeOverflow;
    }
    return mapped ? GpuSpanAdmissionReason::Accepted : GpuSpanAdmissionReason::Unmapped;
}

inline bool IsGpuSpanAdmissible(VAddr base, u64 offset, u64 size, bool mapped) {
    return CheckGpuSpanAdmission(base, offset, size, mapped) ==
           GpuSpanAdmissionReason::Accepted;
}

} // namespace Vulkan
