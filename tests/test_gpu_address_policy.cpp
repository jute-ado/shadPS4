// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>

#include "core/gpu_address_policy.h"

namespace Core {

TEST(GpuAddressPolicy, AcceptsOnlyRangesInsideTheFortyBitGpuAddressSpace) {
    EXPECT_TRUE(IsGpuAddressRangeValid(0x240000000, 64_MB));
    EXPECT_TRUE(IsGpuAddressRangeValid(MaxGpuVirtualAddress - 1, 1));
    EXPECT_FALSE(IsGpuAddressRangeValid(MaxGpuVirtualAddress, 0));
    EXPECT_TRUE(IsGpuAddressRangeValid(0xf63f94dba5, 1));
    EXPECT_FALSE(IsGpuAddressRangeValid(MaxGpuVirtualAddress - 4, 8));
    EXPECT_FALSE(IsGpuAddressRangeValid(std::numeric_limits<VAddr>::max() - 4, 16));
}

} // namespace Core
