// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "common/types.h"

namespace {

struct AdmissionRequest {
    bool special{};
    VAddr address{};
    u64 binding_size{};
    bool has_cpu_vma{};
    u64 cpu_vma_bound{};
    bool gpu_mapped{};
    bool gpu_mapping_at_base{};
    bool cache_registered{};
    u64 gpu_mapped_extent{};
};

struct AdmissionResult {
    u64 size{};
    bool used_cpu_vma_bound{};
};

// This is the policy contract for BindBuffers, independent of the Vulkan objects. CPU-backed
// ranges retain their VMA bound; a range outside the CPU VMA can only use a live GPU mapping and
// an already registered cache allocation. It must never create a new cache allocation from an
// address the host memory manager cannot read.
std::optional<AdmissionResult> ResolveAdmission(const AdmissionRequest& request) {
    if (request.special || request.address == 0 || request.binding_size == 0) {
        return std::nullopt;
    }

    if (!request.has_cpu_vma &&
        (!request.gpu_mapped || !request.gpu_mapping_at_base || !request.cache_registered)) {
        return std::nullopt;
    }

    const u64 bounded_size = request.has_cpu_vma
                                 ? std::min(request.binding_size, request.cpu_vma_bound)
                                 : request.binding_size;
    if (bounded_size == 0 || request.gpu_mapped_extent == 0) {
        return std::nullopt;
    }

    return AdmissionResult{std::min(bounded_size, request.gpu_mapped_extent), request.has_cpu_vma};
}

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string FunctionBody(const std::string& source, std::string_view begin_marker,
                         std::string_view end_marker) {
    const auto begin = source.find(begin_marker);
    EXPECT_NE(begin, std::string::npos);
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = source.find(end_marker, begin);
    EXPECT_NE(end, std::string::npos);
    if (end == std::string::npos) {
        return source.substr(begin);
    }
    return source.substr(begin, end - begin);
}

TEST(GpuLogicalBufferAdmission, AdmitsLiveGpuCacheRangeAfterCpuVmaRetirement) {
    const AdmissionRequest request{
        .address = 0x4000000000ULL,
        .binding_size = 0x2000,
        .has_cpu_vma = false,
        .cpu_vma_bound = 0,
        .gpu_mapped = true,
        .gpu_mapping_at_base = true,
        .cache_registered = true,
        .gpu_mapped_extent = 0x2000,
    };

    // A retired CPU VMA does not invalidate an independently registered GPU/cache owner.
    EXPECT_TRUE(ResolveAdmission(request).has_value());
}

TEST(GpuLogicalBufferAdmission, NonCpuVmaRequiresBothGpuMappingAndCacheRegistration) {
    const AdmissionRequest base{
        .address = 0x4000000000ULL,
        .binding_size = 0x2000,
        .has_cpu_vma = false,
        .cpu_vma_bound = 0,
        .gpu_mapped = true,
        .gpu_mapping_at_base = true,
        .cache_registered = true,
        .gpu_mapped_extent = 0x2000,
    };

    auto no_mapping = base;
    no_mapping.gpu_mapped = false;
    no_mapping.gpu_mapping_at_base = false;
    EXPECT_FALSE(ResolveAdmission(no_mapping));
    auto no_registration = base;
    no_registration.cache_registered = false;
    EXPECT_FALSE(ResolveAdmission(no_registration));
    ASSERT_TRUE(ResolveAdmission(base).has_value());
    EXPECT_EQ(ResolveAdmission(base)->size, 0x2000U);
}

TEST(GpuLogicalBufferAdmission, CpuVmaPathStillClampsBeforeGpuExtent) {
    const AdmissionRequest request{
        .address = 0x200000,
        .binding_size = 0x8000,
        .has_cpu_vma = true,
        .cpu_vma_bound = 0x3000,
        .gpu_mapped = true,
        .gpu_mapping_at_base = true,
        .cache_registered = false,
        .gpu_mapped_extent = 0x3000,
    };

    const auto result = ResolveAdmission(request);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->used_cpu_vma_bound);
    EXPECT_EQ(result->size, 0x3000U);
}

TEST(GpuLogicalBufferAdmission, CpuVmaPathPreservesLeadingHoleOffsetAndBound) {
    const AdmissionRequest request{
        .address = 0x300000,
        .binding_size = 0x10000,
        .has_cpu_vma = true,
        .cpu_vma_bound = 0x10000,
        .gpu_mapped = true,
        .gpu_mapping_at_base = false,
        .cache_registered = true,
        .gpu_mapped_extent = 0x9000,
    };

    const auto result = ResolveAdmission(request);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size, 0x9000U);
}

TEST(GpuLogicalBufferAdmission, NonCpuVmaRejectsLeadingHoleBeforeCacheLookup) {
    const AdmissionRequest request{
        .address = 0x300000,
        .binding_size = 0x10000,
        .has_cpu_vma = false,
        .gpu_mapped = true,
        .gpu_mapping_at_base = false,
        .cache_registered = true,
        .gpu_mapped_extent = 0x9000,
    };

    EXPECT_FALSE(ResolveAdmission(request));
}

TEST(GpuLogicalBufferAdmission, ProductionKeepsConservativeCpuVmaGate) {
    // Until a complete host-readable owner can be proven for GPU-only ranges, production must
    // continue to clamp and resolve ordinary descriptors through the CPU VMA path. The proposed
    // split policy above remains a characterization model, not an accepted renderer contract.
    const auto source = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto body =
        FunctionBody(source, "void Rasterizer::BindBuffers(", "void Rasterizer::BindTextures(");
    ASSERT_FALSE(body.empty());

    const auto clamp = body.find("memory->ClampRangeSize(vsharp.base_address, binding_size)");
    const auto find_buffer = body.find("buffer_cache.FindBuffer(vsharp.base_address, size)");
    ASSERT_NE(clamp, std::string::npos);
    ASSERT_NE(find_buffer, std::string::npos);
    EXPECT_LT(clamp, find_buffer);
}

} // namespace
