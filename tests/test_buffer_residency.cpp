// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "shader_recompiler/addr64_pointer_residency.h"
#include "shader_recompiler/specialization.h"
#include "video_core/buffer_cache/buffer_binding_policy.h"
#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/buffer_cache/fault_range.h"

namespace {

enum class ResidencyCall {
    Synchronize,
    Publish,
};

struct RecordedResidencyCall {
    ResidencyCall operation;
    VAddr address;
    u32 size;
};

class ExpandedDmaBuffer {
public:
    [[nodiscard]] VAddr CpuAddr() const {
        return 0x101E600000;
    }

    [[nodiscard]] size_t SizeBytes() const {
        return 0x200000;
    }
};

class RecordedBufferCache {
public:
    void FindBuffer(VAddr address, u32 size) {
        calls.push_back({ResidencyCall::Publish, address, size});
    }

    void SynchronizeBuffersInRange(VAddr address, u32 size) {
        calls.push_back({ResidencyCall::Synchronize, address, size});
    }

    std::vector<RecordedResidencyCall> calls;
};

} // namespace

TEST(GpuCacheUnmap, SynchronizesBuffersBeforeQueuingTextureEvictionOnTheGpuThread) {
    std::ifstream input{SHADPS4_RASTERIZER_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string source = contents.str();

    const auto function_begin = source.find("void Rasterizer::UnmapMemory(");
    const auto function_end = source.find("void Rasterizer::UpdateDynamicState(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto buffer_unmap = body.find("buffer_cache.InvalidateMemory(addr, size)");
    const auto queued = body.find("liverpool->SendCommand<false>([this, addr, size]");
    const auto texture_unmap = body.find("texture_cache.UnmapMemory(addr, size)", queued);
    const auto queued_end = body.find("});", queued);
    const auto page_unmap = body.find("page_manager.OnGpuUnmap(addr, size)");
    ASSERT_NE(queued, std::string::npos);
    ASSERT_NE(buffer_unmap, std::string::npos);
    ASSERT_NE(texture_unmap, std::string::npos);
    ASSERT_NE(queued_end, std::string::npos);
    ASSERT_NE(page_unmap, std::string::npos);
    EXPECT_LT(buffer_unmap, queued);
    EXPECT_LT(queued, texture_unmap);
    EXPECT_LT(texture_unmap, queued_end);
    EXPECT_LT(queued_end, page_unmap);
    EXPECT_EQ(body.find("SendCommand<true>"), std::string::npos);
}

TEST(BufferResidency, PublishesExpandedDmaMappingOnlyAfterFullSpanIsResident) {
    ExpandedDmaBuffer buffer;
    std::vector<RecordedResidencyCall> calls;

    VideoCore::PublishDmaBufferAfterSynchronization(
        buffer,
        [&](ExpandedDmaBuffer&, VAddr address, u32 size) {
            calls.push_back({ResidencyCall::Synchronize, address, size});
        },
        [&] { calls.push_back({ResidencyCall::Publish, 0, 0}); });

    ASSERT_EQ(calls.size(), 2);
    EXPECT_EQ(calls[0].operation, ResidencyCall::Synchronize);
    EXPECT_EQ(calls[0].address, buffer.CpuAddr());
    EXPECT_EQ(calls[0].size, buffer.SizeBytes());
    EXPECT_EQ(calls[1].operation, ResidencyCall::Publish);
}

TEST(BufferResidency, DoesNotTouchUnpublishedBufferAfterResidencyUpload) {
    u32 touch_count = 0;

    VideoCore::TouchBufferAfterUploadIfRegistered(false, [&] { ++touch_count; });
    EXPECT_EQ(touch_count, 0);

    VideoCore::TouchBufferAfterUploadIfRegistered(true, [&] { ++touch_count; });
    EXPECT_EQ(touch_count, 1);
}

TEST(BufferResidency, RecordsDrawStateAfterEveryPotentiallyFlushingBufferAcquisition) {
    enum class Operation {
        PrepareIndirect,
        PrepareIndex,
        PrepareVertex,
        BindVertex,
        BindIndex,
    };
    constexpr std::array expected_operations{
        Operation::PrepareVertex, Operation::PrepareIndex, Operation::PrepareIndirect,
        Operation::BindVertex,    Operation::BindIndex,
    };

    for (u32 flushing_acquisition = 0; flushing_acquisition < 3; ++flushing_acquisition) {
        u32 command_buffer_generation = 0;
        u32 vertex_state_generation = std::numeric_limits<u32>::max();
        u32 index_state_generation = std::numeric_limits<u32>::max();
        std::vector<Operation> operations;

        const auto prepare = [&](Operation operation, u32 acquisition) {
            operations.push_back(operation);
            if (acquisition == flushing_acquisition) {
                ++command_buffer_generation;
            }
        };

        VideoCore::PrepareDrawBuffersThenBindCommandState(
            [&] {
                prepare(Operation::PrepareVertex, 0);
                return Operation::PrepareVertex;
            },
            [&] {
                prepare(Operation::PrepareIndex, 1);
                return Operation::PrepareIndex;
            },
            [&] { prepare(Operation::PrepareIndirect, 2); },
            [&](Operation prepared_state) {
                EXPECT_EQ(prepared_state, Operation::PrepareVertex);
                operations.push_back(Operation::BindVertex);
                vertex_state_generation = command_buffer_generation;
            },
            [&](Operation prepared_state) {
                EXPECT_EQ(prepared_state, Operation::PrepareIndex);
                operations.push_back(Operation::BindIndex);
                index_state_generation = command_buffer_generation;
            });

        EXPECT_EQ(operations, std::vector(expected_operations.begin(), expected_operations.end()));
        EXPECT_EQ(vertex_state_generation, command_buffer_generation);
        EXPECT_EQ(index_state_generation, command_buffer_generation);
    }
}

TEST(BufferResidency, SelectsOnlyAdjacentFlattenedPointerWordsFromAddr64Dependencies) {
    constexpr std::array flattened_dependencies{116U, 29U, 28U, 29U, 74U};

    const auto roots = Shader::SelectAddr64PointerRoots(3, flattened_dependencies);

    ASSERT_FALSE(roots.overflow);
    ASSERT_EQ(roots.count, 1U);
    EXPECT_EQ(roots.values[0].buffer_resource_index, 3U);
    EXPECT_EQ(roots.values[0].pointer_lo_flat_index, 28U);
}

TEST(BufferResidency, Addr64PointerRootSelectionIsBoundedDeduplicatedAndFailClosed) {
    constexpr std::array isolated_dependencies{1U, 3U, 5U, 7U};
    const auto isolated = Shader::SelectAddr64PointerRoots(0, isolated_dependencies);
    EXPECT_FALSE(isolated.overflow);
    EXPECT_EQ(isolated.count, 0U);

    constexpr std::array duplicated_pair{20U, 21U, 20U, 21U};
    const auto duplicate = Shader::SelectAddr64PointerRoots(2, duplicated_pair);
    ASSERT_FALSE(duplicate.overflow);
    ASSERT_EQ(duplicate.count, 1U);
    EXPECT_EQ(duplicate.values[0].pointer_lo_flat_index, 20U);

    constexpr std::array too_many_pairs{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U,
                                        10U, 11U, 12U, 13U, 14U, 15U, 16U, 17U};
    const auto overflow = Shader::SelectAddr64PointerRoots(1, too_many_pairs);
    EXPECT_TRUE(overflow.overflow);
    EXPECT_EQ(overflow.count, 0U);
}

TEST(BufferResidency, Addr64PointerTraversalSetBoundsQueueGrowthAndDeduplicatesNodes) {
    Shader::Addr64PointerTraversalSet traversal;
    for (u32 node = 1; node <= Shader::MaxAddr64PointerTraversalNodes; ++node) {
        EXPECT_EQ(traversal.TryInsert(node), Shader::Addr64PointerTraversalInsert::Inserted);
    }
    EXPECT_EQ(traversal.TryInsert(1), Shader::Addr64PointerTraversalInsert::Duplicate);
    EXPECT_EQ(traversal.TryInsert(Shader::MaxAddr64PointerTraversalNodes + 1U),
              Shader::Addr64PointerTraversalInsert::Overflow);
    EXPECT_TRUE(traversal.overflow());
    EXPECT_EQ(traversal.size(), Shader::MaxAddr64PointerTraversalNodes);
}

TEST(BufferResidency, Addr64PointerResidencyPlansOneExactBoundedGuestTable) {
    constexpr u64 address_space_size = 1ULL << 40U;

    const auto exact = Shader::PlanAddr64PointerResidency(0x06622E30U, 0x10U,
                                                          address_space_size);
    EXPECT_TRUE(exact.valid);
    EXPECT_EQ(exact.address, 0x1006622E30ULL);
    EXPECT_EQ(exact.size, Shader::Addr64PointerTableBytes);

    EXPECT_FALSE(Shader::PlanAddr64PointerResidency(0U, 0U, address_space_size).valid);
    EXPECT_FALSE(Shader::PlanAddr64PointerResidency(
                     0xFFFFFF80U, 0xFFU, address_space_size)
                     .valid);
}

TEST(BufferResidency, Addr64PointerResidencyUsesExistingFindThenSynchronizePathExactlyOnce) {
    RecordedBufferCache cache;
    constexpr VAddr address = 0x1006622E30ULL;

    VideoCore::MakeDmaFaultRangeResident(cache, address, Shader::Addr64PointerTableBytes);

    ASSERT_EQ(cache.calls.size(), 2U);
    EXPECT_EQ(cache.calls[0].operation, ResidencyCall::Publish);
    EXPECT_EQ(cache.calls[0].address, address);
    EXPECT_EQ(cache.calls[0].size, Shader::Addr64PointerTableBytes);
    EXPECT_EQ(cache.calls[1].operation, ResidencyCall::Synchronize);
    EXPECT_EQ(cache.calls[1].address, address);
    EXPECT_EQ(cache.calls[1].size, Shader::Addr64PointerTableBytes);
}

TEST(BufferResidency, RejectsUnmappedOrOverflowingGuestBufferBindingsBeforeCacheLookup) {
    constexpr VAddr mapped_begin = 0x0B920000ULL;
    constexpr VAddr mapped_end = 0x10000000ULL;
    const auto is_mapped = [](VAddr address, u64 size) {
        return address >= mapped_begin && address < mapped_end && size <= mapped_end - address;
    };

    EXPECT_FALSE(VideoCore::IsGuestBufferBindingUsable(0x7C000ULL, 0x384000ULL, is_mapped));
    EXPECT_FALSE(VideoCore::IsGuestBufferBindingUsable(0, 4, is_mapped));
    EXPECT_FALSE(VideoCore::IsGuestBufferBindingUsable(mapped_begin, 0, is_mapped));
    EXPECT_FALSE(VideoCore::IsGuestBufferBindingUsable(std::numeric_limits<VAddr>::max() - 3,
                                                       8, is_mapped));
}

TEST(BufferResidency, AdmitsMappedBindingBaseWithoutRejectingSparseDeclaredTail) {
    constexpr VAddr mapped_begin = 0x0B920000ULL;
    constexpr VAddr mapped_end = 0x0B930000ULL;
    const auto is_mapped = [](VAddr address, u64 size) {
        return address >= mapped_begin && address < mapped_end && size <= mapped_end - address;
    };

    EXPECT_TRUE(VideoCore::IsGuestBufferBindingUsable(mapped_begin + 0x80, 2_GB, is_mapped));
}

TEST(BufferBindingPolicy, UnmappedGuestBuffersAreNotActiveSpecializationInputs) {
    Shader::Info info{};
    AmdGpu::Buffer first{};
    first.base_address = 0x10000;
    first.stride = 16;
    first.num_records = 64;
    AmdGpu::Buffer second = first;
    second.base_address = 0x20000;
    second.stride = 32;
    info.buffers.push_back({
        .inline_cbuf = first,
        .buffer_type = Shader::BufferType::Guest,
    });

    Shader::StageSpecialization left{};
    left.info = &info;
    left.buffers.resize(1);
    left.buffers[0].stride = first.GetStride();
    left.bitset.set(0);
    Shader::StageSpecialization right{};
    right.info = &info;
    right.buffers.resize(1);
    right.buffers[0].stride = second.GetStride();
    right.bitset.set(0);

    left.CanonicalizeUnusableGuestBuffers([](u32, const auto&) { return false; });
    right.CanonicalizeUnusableGuestBuffers([](u32, const auto&) { return false; });

    EXPECT_FALSE(left.bitset[0]);
    EXPECT_FALSE(right.bitset[0]);
    EXPECT_EQ(left.buffers[0], Shader::BufferSpecialization{});
    EXPECT_EQ(right.buffers[0], Shader::BufferSpecialization{});
    EXPECT_EQ(left, right);
    EXPECT_EQ(right, left);
}

TEST(BufferBindingPolicy, MappedGuestBufferSpecializationsRemainActiveAndDistinct) {
    Shader::Info info{};
    AmdGpu::Buffer sharp{};
    sharp.base_address = 0x10000;
    sharp.stride = 16;
    sharp.num_records = 64;
    info.buffers.push_back({
        .inline_cbuf = sharp,
        .buffer_type = Shader::BufferType::Guest,
    });

    Shader::StageSpecialization left{};
    left.info = &info;
    left.buffers.resize(1);
    left.buffers[0].stride = 16;
    left.bitset.set(0);
    Shader::StageSpecialization right{};
    right.info = &info;
    right.buffers.resize(1);
    right.buffers[0].stride = 32;
    right.bitset.set(0);

    left.CanonicalizeUnusableGuestBuffers([](u32, const auto&) { return true; });
    right.CanonicalizeUnusableGuestBuffers([](u32, const auto&) { return true; });

    EXPECT_TRUE(left.bitset[0]);
    EXPECT_TRUE(right.bitset[0]);
    EXPECT_NE(left, right);
    EXPECT_NE(right, left);
}

TEST(BufferBindingPolicy, InternalBuffersAreNeverMaskedByGuestMappingAdmission) {
    Shader::Info info{};
    info.buffers.push_back({
        .inline_cbuf = AmdGpu::Buffer::Placeholder(64),
        .buffer_type = Shader::BufferType::Flatbuf,
    });

    Shader::StageSpecialization spec{};
    spec.info = &info;
    spec.buffers.resize(1);
    spec.buffers[0].stride = 4;
    spec.bitset.set(0);
    spec.CanonicalizeUnusableGuestBuffers([](u32, const auto&) { return false; });

    EXPECT_TRUE(spec.bitset[0]);
    EXPECT_EQ(spec.buffers[0].stride, 4U);
}

TEST(ImageBindingPolicy, RejectedGuestImagesDoNotCreateShaderPermutations) {
    Shader::Info info{};
    info.buffers.push_back({
        .inline_cbuf = AmdGpu::Buffer::Placeholder(64),
        .buffer_type = Shader::BufferType::Flatbuf,
    });
    info.images.push_back({});

    Shader::StageSpecialization left{};
    left.info = &info;
    left.images.resize(1);
    left.images[0].type = AmdGpu::ImageType::Color2D;
    left.bitset.set(1);
    Shader::StageSpecialization right{};
    right.info = &info;
    right.images.resize(1);
    right.images[0].type = AmdGpu::ImageType::Color3D;
    right.bitset.set(1);

    left.CanonicalizeUnusableGuestImages([](u32, const auto&) { return false; });
    right.CanonicalizeUnusableGuestImages([](u32, const auto&) { return false; });

    EXPECT_FALSE(left.bitset[1]);
    EXPECT_FALSE(right.bitset[1]);
    EXPECT_EQ(left.images[0], Shader::ImageSpecialization{});
    EXPECT_EQ(right.images[0], Shader::ImageSpecialization{});
    EXPECT_EQ(left, right);
    EXPECT_EQ(right, left);
}

TEST(ImageBindingPolicy, AdmittedGuestImageSpecializationsRemainDistinct) {
    Shader::Info info{};
    info.images.push_back({});

    Shader::StageSpecialization left{};
    left.info = &info;
    left.images.resize(1);
    left.images[0].type = AmdGpu::ImageType::Color2D;
    left.bitset.set(0);
    Shader::StageSpecialization right{};
    right.info = &info;
    right.images.resize(1);
    right.images[0].type = AmdGpu::ImageType::Color3D;
    right.bitset.set(0);

    left.CanonicalizeUnusableGuestImages([](u32, const auto&) { return true; });
    right.CanonicalizeUnusableGuestImages([](u32, const auto&) { return true; });

    EXPECT_TRUE(left.bitset[0]);
    EXPECT_TRUE(right.bitset[0]);
    EXPECT_NE(left, right);
    EXPECT_NE(right, left);
}

TEST(ResourceSpecializationPolicy, SamplerOnlyStagesRetainTheirStartingBinding) {
    Shader::Info info{};
    info.samplers.push_back({});

    Shader::StageSpecialization left{};
    left.info = &info;
    left.samplers.resize(1);
    left.start.unified = 4;
    Shader::StageSpecialization right = left;
    right.start.unified = 9;

    EXPECT_NE(left, right);
    EXPECT_NE(right, left);
}
