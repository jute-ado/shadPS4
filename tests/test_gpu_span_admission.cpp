// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include <optional>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/gpu_span_admission.h"

namespace {

using Vulkan::CheckGpuSpanAdmission;
using Vulkan::CheckedGpuProduct;
using Vulkan::GpuSpanAdmissionReason;
using Vulkan::TryMakeGpuSpan;

TEST(GpuSpanAdmission, PreservesOrdinaryRange) {
    const auto span = TryMakeGpuSpan(0x1000, 0x20, 0x40);

    ASSERT_TRUE(span.has_value());
    EXPECT_EQ(span->address, 0x1020u);
    EXPECT_EQ(span->size, 0x40u);
}

TEST(GpuSpanAdmission, RejectsAddressOverflowBeforeMappingProbe) {
    constexpr VAddr max_gpu_address = Vulkan::MaxGpuAddress;

    EXPECT_FALSE(TryMakeGpuSpan(max_gpu_address - 4, 5, 1).has_value());
    EXPECT_FALSE(TryMakeGpuSpan(max_gpu_address - 4, 0, 5).has_value());
    EXPECT_FALSE(TryMakeGpuSpan(max_gpu_address - 4, 4, 0).has_value());
    EXPECT_FALSE(TryMakeGpuSpan(max_gpu_address, 0, 0).has_value());
}

TEST(GpuSpanAdmission, RejectsBufferSizeThatWouldNarrowToU32) {
    EXPECT_FALSE(
        TryMakeGpuSpan(0x1000, 0, static_cast<u64>(std::numeric_limits<u32>::max()) + 1)
            .has_value());
}

TEST(GpuSpanAdmission, AllowsEmptyRangeWithoutWrapping) {
    const auto span = TryMakeGpuSpan(0x7000, 0x20, 0);

    ASSERT_TRUE(span.has_value());
    EXPECT_EQ(span->address, 0x7020u);
    EXPECT_EQ(span->size, 0u);
}

TEST(GpuSpanAdmission, DistinguishesUnmappedFromArithmeticFailure) {
    EXPECT_EQ(CheckGpuSpanAdmission(0x4000, 0x10, 0x20, true),
              GpuSpanAdmissionReason::Accepted);
    EXPECT_EQ(CheckGpuSpanAdmission(0x4000, 0x10, 0x20, false),
              GpuSpanAdmissionReason::Unmapped);
    EXPECT_EQ(CheckGpuSpanAdmission(Vulkan::MaxGpuAddress - 1, 2, 1, true),
              GpuSpanAdmissionReason::AddressOverflow);
}

TEST(GpuSpanAdmission, ChecksProductsBeforeAddressAdmission) {
    ASSERT_EQ(CheckedGpuProduct(0x10, 0x20), 0x200u);
    EXPECT_FALSE(CheckedGpuProduct(std::numeric_limits<u64>::max(), 2).has_value());

    const auto indirect_bytes =
        CheckedGpuProduct(std::numeric_limits<u32>::max(), 2u);
    ASSERT_TRUE(indirect_bytes.has_value());
    EXPECT_EQ(*indirect_bytes, static_cast<u64>(std::numeric_limits<u32>::max()) * 2);
    EXPECT_FALSE(TryMakeGpuSpan(0x1000, 0, *indirect_bytes).has_value());
}

} // namespace

TEST(GpuSpanAdmission, RejectsZeroLengthAtExclusiveAddressEndpoint) {
    EXPECT_EQ(CheckGpuSpanAdmission(Vulkan::MaxGpuAddress - 1, 1, 0, true),
              GpuSpanAdmissionReason::SizeOverflow);
}


