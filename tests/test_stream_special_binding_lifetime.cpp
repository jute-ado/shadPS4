// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Characterization coverage for utility descriptors that use the recyclable stream buffer.
// A descriptor copied before vertex/index preparation can be overwritten when a later reservation
// wraps the ring. The production path must therefore defer those copies until the final binding
// phase, after all wait-capable preparation has completed.

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/buffer_residency.h"
#include "video_core/renderer_vulkan/utility_buffer_reservation.h"

namespace {

class FakeStream final {
public:
    explicit FakeStream(std::size_t capacity) : storage(capacity) {}

    struct Reservation {
        std::size_t offset{};
        std::size_t size{};
        unsigned generation{};
    };

    [[nodiscard]] Reservation Copy(std::span<const std::byte> bytes) {
        if (offset + bytes.size() > storage.size()) {
            offset = 0;
            ++generation;
        }
        std::memcpy(storage.data() + offset, bytes.data(), bytes.size());
        const Reservation reservation{offset, bytes.size(), generation};
        offset += bytes.size();
        return reservation;
    }

    [[nodiscard]] std::span<const std::byte> Read(const Reservation reservation) const {
        return std::span<const std::byte>{storage}.subspan(reservation.offset, reservation.size);
    }

private:
    std::vector<std::byte> storage;
    std::size_t offset{};
    unsigned generation{};
};

struct UtilityPayload {
    std::size_t descriptor_index{};
    std::size_t alignment{};
    std::vector<std::byte> bytes;
};

struct SyntheticByteCount {
    u64 value{};

    [[nodiscard]] constexpr u64 size() const noexcept {
        return value;
    }
};

struct SyntheticUtilityPayload {
    u64 alignment{};
    SyntheticByteCount bytes;
};

struct UtilityDescriptor {
    std::size_t offset{};
    std::size_t size{};
};

constexpr std::size_t AlignUpSize(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

class FakeContiguousStream final {
public:
    explicit FakeContiguousStream(std::size_t capacity) : storage(capacity, std::byte{0xa5}) {}

    struct Mapping {
        std::byte* data{};
        std::size_t offset{};
    };

    Mapping Map(std::size_t size, std::size_t alignment) {
        ++map_calls;
        if (size > storage.size() || alignment == 0) {
            return {};
        }
        const auto offset = AlignUpSize(next_offset, alignment);
        if (offset > storage.size() || size > storage.size() - offset) {
            return {};
        }
        mapped_offset = offset;
        mapped_size = size;
        return {storage.data() + offset, offset};
    }

    void Commit() {
        ++commit_calls;
        next_offset = mapped_offset + mapped_size;
    }

    std::vector<std::byte> storage;
    std::size_t map_calls{};
    std::size_t commit_calls{};

private:
    std::size_t next_offset{};
    std::size_t mapped_offset{};
    std::size_t mapped_size{};
};

bool FinalizeUtilityPayloads(const std::vector<UtilityPayload>& payloads,
                             FakeContiguousStream& stream,
                             std::vector<std::optional<UtilityDescriptor>>& descriptors) {
    const auto reservation = Vulkan::UtilityBuffer::PlanReservation(
        payloads, static_cast<u64>(stream.storage.size()));
    if (!reservation) {
        return false;
    }

    const auto mapping = stream.Map(reservation->total_size, reservation->base_alignment);
    if (reservation->total_size != 0 && mapping.data == nullptr) {
        return false;
    }

    u64 cursor{};
    for (const auto& payload : payloads) {
        const auto aligned_cursor = Vulkan::UtilityBuffer::CheckedAlignUp(
            cursor, static_cast<u64>(payload.alignment));
        if (!aligned_cursor) {
            return false;
        }
        cursor = *aligned_cursor;
        std::copy(payload.bytes.begin(), payload.bytes.end(), mapping.data + cursor);
        descriptors[payload.descriptor_index] = UtilityDescriptor{
            .offset = mapping.offset + cursor,
            .size = payload.bytes.size(),
        };
        const auto next_cursor = Vulkan::UtilityBuffer::CheckedAdd(
            cursor, static_cast<u64>(payload.bytes.size()));
        if (!next_cursor) {
            return false;
        }
        cursor = *next_cursor;
    }
    stream.Commit();
    return true;
}

TEST(StreamSpecialBindingRed, EagerUtilityCopyIsOverwrittenByLaterVertexWrap) {
    FakeStream stream{16};
    constexpr std::array flat_data{std::byte{'F'}, std::byte{'L'}, std::byte{'A'}, std::byte{'T'},
                                   std::byte{'D'}, std::byte{'E'}, std::byte{'S'}, std::byte{'C'}};
    constexpr std::array vertex_data{std::byte{'V'}, std::byte{'E'}, std::byte{'R'},
                                     std::byte{'T'}, std::byte{'E'}, std::byte{'X'},
                                     std::byte{'P'}, std::byte{'R'}, std::byte{'E'},
                                     std::byte{'P'}, std::byte{'A'}, std::byte{'R'}};

    const auto flat = stream.Copy(flat_data);
    static_cast<void>(stream.Copy(vertex_data));

    // This is the bug model: both reservations alias the same ring allocation after the wrap.
    EXPECT_FALSE(std::ranges::equal(stream.Read(flat), flat_data));
}

TEST(StreamSpecialBinding, DeferringUtilityCopyUntilFinalizationSurvivesPreparationWrap) {
    FakeStream stream{16};
    constexpr std::array flat_data{std::byte{'F'}, std::byte{'L'}, std::byte{'A'}, std::byte{'T'},
                                   std::byte{'D'}, std::byte{'E'}, std::byte{'S'}, std::byte{'C'}};
    constexpr std::array vertex_data{std::byte{'V'}, std::byte{'E'}, std::byte{'R'},
                                     std::byte{'T'}, std::byte{'E'}, std::byte{'X'},
                                     std::byte{'P'}, std::byte{'R'}, std::byte{'E'},
                                     std::byte{'P'}, std::byte{'A'}, std::byte{'R'}};

    std::optional<FakeStream::Reservation> flat;
    const bool published = VideoCore::PrepareDrawBuffersThenBindCommandState(
        [&] { return stream.Copy(vertex_data); }, [] { return std::nullopt; }, [] {}, [] {},
        [&](const auto&, const auto&) {
            flat = stream.Copy(flat_data);
            return true;
        },
        [](const auto&) {}, [](const auto&) {});

    ASSERT_TRUE(published);
    ASSERT_TRUE(flat);
    EXPECT_TRUE(std::ranges::equal(stream.Read(*flat), flat_data));
}

TEST(StreamSpecialBinding, OneContiguousReservationPreservesEveryUtilityPayload) {
    const std::vector<UtilityPayload> payloads{
        {.descriptor_index = 0,
         .alignment = 16,
         .bytes = {std::byte{'F'}, std::byte{'L'}, std::byte{'A'}, std::byte{'T'}}},
        {.descriptor_index = 1,
         .alignment = 64,
         .bytes = {std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}}},
        {.descriptor_index = 2,
         .alignment = 256,
         .bytes = {std::byte{0xa1}, std::byte{0xb2}, std::byte{0xc3}, std::byte{0xd4},
                   std::byte{0xe5}, std::byte{0xf6}}},
    };
    FakeContiguousStream stream{512};
    std::vector<std::optional<UtilityDescriptor>> descriptors(3);

    ASSERT_TRUE(FinalizeUtilityPayloads(payloads, stream, descriptors));
    ASSERT_EQ(stream.map_calls, 1U);
    ASSERT_EQ(stream.commit_calls, 1U);

    ASSERT_TRUE(descriptors[0]);
    EXPECT_EQ(descriptors[0]->offset, 0U);
    EXPECT_EQ(descriptors[0]->size, 4U);
    ASSERT_TRUE(descriptors[1]);
    EXPECT_EQ(descriptors[1]->offset, 64U);
    EXPECT_EQ(descriptors[1]->size, 4U);
    ASSERT_TRUE(descriptors[2]);
    EXPECT_EQ(descriptors[2]->offset, 256U);
    EXPECT_EQ(descriptors[2]->size, 6U);

    EXPECT_TRUE(std::ranges::equal(std::span<const std::byte>{stream.storage}.subspan(
                                       descriptors[0]->offset, descriptors[0]->size),
                                   payloads[0].bytes));
    EXPECT_TRUE(std::ranges::equal(std::span<const std::byte>{stream.storage}.subspan(
                                       descriptors[1]->offset, descriptors[1]->size),
                                   payloads[1].bytes));
    EXPECT_TRUE(std::ranges::equal(std::span<const std::byte>{stream.storage}.subspan(
                                       descriptors[2]->offset, descriptors[2]->size),
                                   payloads[2].bytes));
}

TEST(StreamSpecialBinding, OversizedAggregateFailsBeforePublishingOrCommitting) {
    const std::vector<UtilityPayload> payloads{
        {.descriptor_index = 0, .alignment = 16, .bytes = std::vector<std::byte>(24, std::byte{1})},
        {.descriptor_index = 1, .alignment = 16, .bytes = std::vector<std::byte>(24, std::byte{2})},
    };
    FakeContiguousStream stream{32};
    std::vector<std::optional<UtilityDescriptor>> descriptors(2);

    EXPECT_FALSE(FinalizeUtilityPayloads(payloads, stream, descriptors));
    EXPECT_EQ(stream.map_calls, 0U);
    EXPECT_EQ(stream.commit_calls, 0U);
    EXPECT_FALSE(descriptors[0]);
    EXPECT_FALSE(descriptors[1]);
}

TEST(StreamSpecialBinding, CheckedReservationRejectsOverflowingAlignment) {
    constexpr u64 maximum = std::numeric_limits<u64>::max();

    EXPECT_FALSE(Vulkan::UtilityBuffer::CheckedAlignUp(maximum - 2, 4));
    EXPECT_EQ(Vulkan::UtilityBuffer::CheckedAlignUp(maximum - 4, 4), maximum - 3);

    const std::vector<UtilityPayload> oversized_alignment{
        {.descriptor_index = 0, .alignment = maximum, .bytes = {std::byte{0x5a}}},
    };
    EXPECT_FALSE(Vulkan::UtilityBuffer::PlanReservation(oversized_alignment, 8));
}

TEST(StreamSpecialBinding, CheckedReservationRejectsU64AdditionOverflow) {
    constexpr u64 maximum = std::numeric_limits<u64>::max();

    EXPECT_FALSE(Vulkan::UtilityBuffer::CheckedAdd(maximum - 1, 2));
    EXPECT_EQ(Vulkan::UtilityBuffer::CheckedAdd(maximum - 1, 1), maximum);

    const std::array<SyntheticUtilityPayload, 2> payloads{{
        {.alignment = 1, .bytes = {.value = maximum}},
        {.alignment = 1, .bytes = {.value = 1}},
    }};
    EXPECT_FALSE(Vulkan::UtilityBuffer::PlanReservation(payloads, maximum));
}

TEST(StreamSpecialBinding, CheckedReservationRejectsAlignedTotalOverflow) {
    constexpr u64 maximum = std::numeric_limits<u64>::max();

    const std::array<SyntheticUtilityPayload, 2> payloads{{
        {.alignment = 1, .bytes = {.value = maximum - 1}},
        {.alignment = 4, .bytes = {.value = 1}},
    }};
    EXPECT_FALSE(Vulkan::UtilityBuffer::PlanReservation(payloads, maximum));
}

TEST(StreamSpecialBinding, CheckedReservationRejectsSharedMemoryMultiplicationOverflow) {
    constexpr u64 maximum = std::numeric_limits<u64>::max();

    EXPECT_EQ(Vulkan::UtilityBuffer::CheckedSharedMemorySize(512, 2, 3, 4), 12'288U);
    EXPECT_FALSE(Vulkan::UtilityBuffer::CheckedSharedMemorySize(512, maximum, 2, 1));
    EXPECT_FALSE(Vulkan::UtilityBuffer::CheckedSharedMemorySize(maximum, 2, 1, 1));
}

std::string ReadFile(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::size_t CountOccurrences(std::string_view source, std::string_view needle) {
    std::size_t count{};
    std::size_t offset{};
    while (true) {
        const auto found = source.find(needle, offset);
        if (found == std::string_view::npos) {
            break;
        }
        ++count;
        offset = found + needle.size();
    }
    return count;
}

TEST(StreamSpecialBinding, ProductionDefersAllRecyclableUtilityDescriptors) {
    const auto source = ReadFile(SHADPS4_RASTERIZER_SOURCE_PATH);
    ASSERT_FALSE(source.empty());
    const auto bind = source.find("void Rasterizer::BindBuffers(");
    const auto bind_end = source.find("void Rasterizer::BindTextures(", bind);
    const auto finalize = source.find("bool Rasterizer::FinalizeUtilityBufferBindings(");
    const auto draw = source.find("const bool draw_buffers_ready =");
    ASSERT_NE(bind, std::string::npos);
    ASSERT_NE(bind_end, std::string::npos);
    ASSERT_NE(finalize, std::string::npos);
    ASSERT_NE(draw, std::string::npos);
    const auto bind_body = source.substr(bind, bind_end - bind);
    EXPECT_NE(source.find("pending_utility_buffer_bindings"), std::string::npos);
    EXPECT_NE(source.find("FinalizeUtilityBufferBindings"), std::string::npos);
    EXPECT_EQ(bind_body.find("vk_buffer.Copy(stage.flattened_ud_buf.data()"), std::string::npos);
    EXPECT_EQ(bind_body.find("vk_buffer.Copy(planes.data()"), std::string::npos);
    EXPECT_EQ(bind_body.find("lds_buffer.Map(lds_size"), std::string::npos);
    EXPECT_LT(source.find("FinalizeUtilityBufferBindings()", draw),
              source.find("buffer_cache.FinalizeVertexBuffers", draw));
}

TEST(StreamSpecialBinding, ProductionUsesOneCheckedContiguousUtilityReservation) {
    const auto source = ReadFile(SHADPS4_RASTERIZER_SOURCE_PATH);
    ASSERT_FALSE(source.empty());
    const auto bind = source.find("void Rasterizer::BindBuffers(");
    const auto bind_end = source.find("void Rasterizer::BindTextures(", bind);
    const auto finalize = source.find("bool Rasterizer::FinalizeUtilityBufferBindings(");
    const auto finalize_end = source.find("bool Rasterizer::FinalizeBufferBindings(", finalize);
    ASSERT_NE(bind, std::string::npos);
    ASSERT_NE(bind_end, std::string::npos);
    ASSERT_NE(finalize, std::string::npos);
    ASSERT_NE(finalize_end, std::string::npos);
    const auto bind_body = source.substr(bind, bind_end - bind);
    const auto finalize_body = source.substr(finalize, finalize_end - finalize);

    // Every recyclable utility payload is snapshotted before finalization.
    EXPECT_NE(bind_body.find("stage.flattened_ud_buf"), std::string::npos);
    EXPECT_NE(bind_body.find("planes.data()"), std::string::npos);
    EXPECT_NE(bind_body.find("SharedMemSize()"), std::string::npos);
    EXPECT_NE(bind_body.find("pending.bytes"), std::string::npos);

    // Finalization must map once, publish only after a successful map, and commit once.
    EXPECT_EQ(CountOccurrences(finalize_body, "stream_buffer.Map("), 1U);
    EXPECT_EQ(CountOccurrences(finalize_body, "stream_buffer.Commit("), 1U);
    EXPECT_NE(finalize_body.find("UtilityBuffer::PlanReservation"), std::string::npos);
    EXPECT_NE(finalize_body.find("UtilityBuffer::CheckedAlignUp"), std::string::npos);
    EXPECT_NE(finalize_body.find("UtilityBuffer::CheckedAdd"), std::string::npos);
    const auto plan = finalize_body.find("UtilityBuffer::PlanReservation");
    const auto plan_failure = finalize_body.find("if (!reservation)");
    const auto map = finalize_body.find(
        "stream_buffer.Map(reservation->total_size, reservation->base_alignment)");
    const auto map_failure = finalize_body.find("data == nullptr");
    const auto descriptor = finalize_body.find("buffer_infos[pending.buffer_info_index]");
    const auto commit = finalize_body.find("stream_buffer.Commit()");
    ASSERT_NE(plan, std::string::npos);
    ASSERT_NE(plan_failure, std::string::npos);
    ASSERT_NE(map, std::string::npos);
    ASSERT_NE(map_failure, std::string::npos);
    ASSERT_NE(descriptor, std::string::npos);
    ASSERT_NE(commit, std::string::npos);
    EXPECT_LT(plan, plan_failure);
    EXPECT_LT(plan_failure, map);
    EXPECT_LT(map, map_failure);
    EXPECT_LT(map_failure, descriptor);
    EXPECT_LT(descriptor, commit);
}

TEST(StreamSpecialBinding, ProductionChecksSharedMemoryDispatchSizeBeforeVectorAllocation) {
    const auto source = ReadFile(SHADPS4_RASTERIZER_SOURCE_PATH);
    ASSERT_FALSE(source.empty());
    const auto bind = source.find("void Rasterizer::BindBuffers(");
    const auto bind_end = source.find("void Rasterizer::BindTextures(", bind);
    ASSERT_NE(bind, std::string::npos);
    ASSERT_NE(bind_end, std::string::npos);
    const auto bind_body = source.substr(bind, bind_end - bind);

    const auto checked_size = bind_body.find("UtilityBuffer::CheckedSharedMemorySize(");
    const auto capacity = bind_body.find("stream_buffer.SizeBytes()", checked_size);
    const auto failure = bind_body.find("utility_buffer_bindings_failed = true", checked_size);
    const auto vector = bind_body.find("std::vector<u8>(static_cast<std::size_t>(*lds_size))",
                                      checked_size);
    ASSERT_NE(checked_size, std::string::npos);
    ASSERT_NE(capacity, std::string::npos);
    ASSERT_NE(failure, std::string::npos);
    ASSERT_NE(vector, std::string::npos);
    EXPECT_LT(checked_size, failure);
    EXPECT_LT(capacity, vector);
    EXPECT_LT(failure, vector);
}

} // namespace
