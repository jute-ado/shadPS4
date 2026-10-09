// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Characterization tests for the late image-descriptor handoff. The finalizer must remain
// side-effect free; stale generations are repaired by the enclosing draw boundary immediately
// before descriptor publication.

#include <cstdint>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#ifndef SHADPS4_LATE_FINALIZATION_RASTERIZER_PATH
#define SHADPS4_LATE_FINALIZATION_RASTERIZER_PATH "src/video_core/renderer_vulkan/vk_rasterizer.cpp"
#endif

namespace {

using ImageId = std::uint32_t;
using ImageUid = std::uint64_t;

struct BindingKey {
    std::uint64_t guest_address{};
    std::uint32_t guest_size{};

    bool operator==(const BindingKey&) const = default;
};

struct Image {
    ImageId id{};
    ImageUid uid{};
    std::uint32_t view{};
    BindingKey key{};
};

struct PendingImageBinding {
    ImageId id{};
    ImageUid uid{};
    BindingKey key{};
};

enum class Event {
    TryGet,
    FindCurrent,
    Wait,
    Mutate,
};

class FakeImageCache {
public:
    void SetCurrent(Image image) {
        current = image;
    }

    void ReplaceCurrent(Image image) {
        events.push_back(Event::Mutate);
        ++mutation_count;
        current = image;
    }

    [[nodiscard]] const Image* TryGet(ImageId id, ImageUid uid) {
        events.push_back(Event::TryGet);
        if (!current || current->id != id || current->uid != uid) {
            return nullptr;
        }
        return &*current;
    }

    [[nodiscard]] const Image* FindCurrent(const BindingKey& key) {
        events.push_back(Event::FindCurrent);
        ++mutating_lookup_count;
        if (!current || current->key != key) {
            return nullptr;
        }
        return &*current;
    }

    void Wait() {
        events.push_back(Event::Wait);
        ++wait_count;
    }

    [[nodiscard]] const std::vector<Event>& Events() const {
        return events;
    }

    std::uint32_t mutation_count{};
    std::uint32_t mutating_lookup_count{};
    std::uint32_t wait_count{};

private:
    std::optional<Image> current;
    std::vector<Event> events;
};

enum class FinalizationDecision {
    Published,
    RetryRequired,
};

// This models the required production boundary: the finalizer may validate the captured token,
// but it must not repair the cache. A stale token becomes a retry signal for the enclosing draw.
[[nodiscard]] FinalizationDecision FinalizeWithoutMutation(FakeImageCache& cache,
                                                           const PendingImageBinding& pending) {
    return cache.TryGet(pending.id, pending.uid) ? FinalizationDecision::Published
                                                 : FinalizationDecision::RetryRequired;
}

[[nodiscard]] std::string ReadRasterizerSource() {
    std::ifstream input{SHADPS4_LATE_FINALIZATION_RASTERIZER_PATH, std::ios::binary};
    EXPECT_TRUE(input.is_open()) << SHADPS4_LATE_FINALIZATION_RASTERIZER_PATH;
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::string FunctionBody(const std::string& source, std::string_view begin_marker,
                                       std::string_view end_marker) {
    const auto begin = source.find(begin_marker);
    const auto end = source.find(end_marker, begin);
    EXPECT_NE(begin, std::string::npos);
    EXPECT_NE(end, std::string::npos);
    return begin == std::string::npos || end == std::string::npos
               ? std::string{}
               : source.substr(begin, end - begin);
}

TEST(LateImageFinalization, ReplacedGenerationRequestsOuterBoundaryRetry) {
    constexpr BindingKey key{.guest_address = 0x180000, .guest_size = 0x4000};
    FakeImageCache cache;
    cache.SetCurrent(Image{.id = 7, .uid = 11, .view = 101, .key = key});
    const PendingImageBinding pending{.id = 7, .uid = 11, .key = key};

    // A cache replacement occurs after BindTextures captured the pending token.
    cache.ReplaceCurrent(Image{.id = 7, .uid = 12, .view = 202, .key = key});
    EXPECT_EQ(FinalizeWithoutMutation(cache, pending), FinalizationDecision::RetryRequired);

    // Rebinding belongs to the enclosing draw boundary, where mutation is allowed.
    const auto* current = cache.FindCurrent(pending.key);
    ASSERT_NE(current, nullptr);
    EXPECT_EQ(current->uid, 12U);
    EXPECT_EQ(current->view, 202U);
    EXPECT_EQ(cache.Events(),
              (std::vector<Event>{Event::Mutate, Event::TryGet, Event::FindCurrent}));
}

TEST(LateImageFinalization, StaleFinalizerDoesNotRepairOrWait) {
    constexpr BindingKey key{.guest_address = 0x240000, .guest_size = 0x2000};
    FakeImageCache cache;
    cache.SetCurrent(Image{.id = 9, .uid = 21, .view = 301, .key = key});
    const PendingImageBinding pending{.id = 9, .uid = 21, .key = key};
    cache.ReplaceCurrent(Image{.id = 9, .uid = 22, .view = 302, .key = key});

    EXPECT_EQ(FinalizeWithoutMutation(cache, pending), FinalizationDecision::RetryRequired);
    EXPECT_EQ(cache.mutating_lookup_count, 0U);
    EXPECT_EQ(cache.wait_count, 0U);
    EXPECT_EQ(cache.mutation_count, 1U); // only the setup replacement ran
    EXPECT_EQ(cache.Events(), (std::vector<Event>{Event::Mutate, Event::TryGet}));
}

TEST(LateImageFinalization, ProductionProvidesLateRebindBoundary) {
    const auto source = ReadRasterizerSource();
    const auto rebind = FunctionBody(source, "bool Rasterizer::RebindPendingImageBindings()",
                                     "bool Rasterizer::FinalizeImageDescriptors(");
    ASSERT_FALSE(rebind.empty());
    for (const auto required : {"pending.desc", "FindImage(", "FindTexture(",
                                "TrackBoundImage("}) {
        EXPECT_NE(rebind.find(required), std::string::npos) << required;
    }

    const auto body = FunctionBody(source, "void Rasterizer::Draw(bool",
                                   "void Rasterizer::DrawIndirect(");
    const auto rebind_call = body.find("RebindPendingImageBindings()");
    const auto finalize_call = body.find("FinalizeImageDescriptors(", rebind_call);
    ASSERT_NE(rebind_call, std::string::npos);
    ASSERT_NE(finalize_call, std::string::npos);
    EXPECT_LT(rebind_call, finalize_call);
}

TEST(LateImageFinalization, ProductionFinalizerRemainsSideEffectFree) {
    const auto source = ReadRasterizerSource();
    const auto body = FunctionBody(source, "bool Rasterizer::FinalizeImageDescriptors(",
                                   "void Rasterizer::TrackBoundImage(");

    EXPECT_NE(body.find("TryGetImage("), std::string::npos);
    for (const auto forbidden : {"FindImage(", "FindTexture(", "UpdateImage(", "scheduler.",
                                 "DrainReadyFaults", "Wait", "Finish("}) {
        EXPECT_EQ(body.find(forbidden), std::string::npos) << forbidden;
    }
}

} // namespace
