// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Regression coverage for the two-phase draw binding path. A stream-buffer endpoint is only stable
// until the next stream wrap; a later preparation can recycle the same allocation before the
// earlier endpoint is published. Finalization must detect that reservation transition and
// reacquire the logical range from a non-recyclable backing store.

#include <gtest/gtest.h>

#include <compare>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include "video_core/buffer_cache/buffer_residency.h"

namespace {

struct Endpoint {
    unsigned generation{};
    unsigned offset{};
    bool stream_backed{};

    auto operator<=>(const Endpoint&) const = default;
};

class FakeStreamBuffer {
public:
    explicit FakeStreamBuffer(unsigned capacity) : capacity{capacity} {}

    [[nodiscard]] VideoCore::PreparedBufferBinding<Endpoint> Reserve(unsigned size) {
        if (offset + size > capacity) {
            offset = 0;
            ++generation;
        }
        const Endpoint endpoint{generation, offset, true};
        offset += size;
        return VideoCore::PreparedBufferBinding<Endpoint>{
            .address = next_address,
            .size = size,
            .requires_final_resolution = false,
            .stable_endpoint = endpoint,
            .stable_generation = generation,
        };
    }

    [[nodiscard]] unsigned ReservationGeneration() const {
        return generation;
    }

    [[nodiscard]] bool IsCurrent(const std::optional<Endpoint>& endpoint) const {
        return endpoint.has_value() && endpoint->generation == generation;
    }

private:
    unsigned capacity;
    unsigned generation{};
    unsigned offset{};
    static constexpr VAddr next_address = 0x4000;
};

TEST(BufferResidencyRed, ShaderResourceStreamBindingSurvivesLaterVertexPreparationWrap) {
    FakeStreamBuffer stream{16};
    // BindResources runs before PrepareDrawBuffersThenBindCommandState and can therefore reserve
    // a small read-only shader resource in the stream buffer first.
    const auto shader_resource = stream.Reserve(12);

    unsigned reacquire_calls = 0;
    const bool published = VideoCore::PrepareDrawBuffersThenBindCommandState(
        [&] { return stream.Reserve(8); }, [&] { return stream.Reserve(1); }, [] {}, [] {},
        [&](const auto&, const auto&) {
            const auto resource_endpoint = VideoCore::FinalizePreparedBufferBinding(
                shader_resource, [](VAddr, u32) { return std::optional<Endpoint>{}; },
                [&](const auto& prepared) {
                    return prepared.stable_generation &&
                           *prepared.stable_generation == stream.ReservationGeneration();
                },
                [&](VAddr address, u32 size) -> std::optional<Endpoint> {
                    EXPECT_EQ(address, 0x4000U);
                    EXPECT_EQ(size, 12U);
                    ++reacquire_calls;
                    // A production reacquisition is cache-backed. The fake uses the current
                    // generation only to make the lifetime property observable.
                    return Endpoint{stream.ReservationGeneration(), 0x8000U, false};
                });
            EXPECT_TRUE(resource_endpoint.has_value());
            if (!resource_endpoint) {
                return false;
            }
            EXPECT_FALSE(resource_endpoint->stream_backed);
            EXPECT_TRUE(stream.IsCurrent(resource_endpoint));
            return stream.IsCurrent(resource_endpoint);
        },
        [](const auto&) {}, [](const auto&) {});

    EXPECT_TRUE(published);
    EXPECT_EQ(reacquire_calls, 1U);
}

TEST(BufferResidency, StableStreamBindingIsNotReacquiredWithoutAWrap) {
    FakeStreamBuffer stream{16};
    const auto shader_resource = stream.Reserve(4);
    const auto vertex = stream.Reserve(8);
    ASSERT_EQ(shader_resource.stable_generation, vertex.stable_generation);

    unsigned reacquire_calls = 0;
    const auto finalized = VideoCore::FinalizePreparedBufferBinding(
        shader_resource, [](VAddr, u32) { return std::optional<Endpoint>{}; },
        [&](const auto& prepared) {
            return prepared.stable_generation &&
                   *prepared.stable_generation == stream.ReservationGeneration();
        },
        [&](VAddr, u32) -> std::optional<Endpoint> {
            ++reacquire_calls;
            return std::nullopt;
        });

    ASSERT_TRUE(finalized.has_value());
    EXPECT_EQ(*finalized, *shader_resource.stable_endpoint);
    EXPECT_EQ(reacquire_calls, 0U);
}

std::string ReadFile(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(BufferResidency, ProductionStreamBindingCarriesAndConsumesReservationGeneration) {
    const auto buffer_source = ReadFile(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto stream_source = ReadFile(SHADPS4_STREAM_BUFFER_SOURCE_PATH);

    ASSERT_FALSE(buffer_source.empty());
    ASSERT_FALSE(stream_source.empty());
    EXPECT_NE(buffer_source.find("stable_generation"), std::string::npos);
    EXPECT_NE(buffer_source.find("ReservationGeneration()"), std::string::npos);
    EXPECT_NE(buffer_source.find("false /* allow_stream_buffer */"), std::string::npos);
    EXPECT_NE(stream_source.find("++reservation_generation"), std::string::npos);
}

} // namespace
