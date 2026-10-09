// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// CPU-only executable specification for the owner-side draw admission boundary. A pipeline,
// descriptor, or residency miss can be transient; the guest draw must remain pending until the
// owner retries it. This model deliberately has no Vulkan or emulator dependencies so it can be
// used to pin down the handoff contract before changing Rasterizer::Draw/DrawIndirect.

#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace {

#ifndef SHADPS4_DRAW_ADMISSION_LIVERPOOL_SOURCE_PATH
#define SHADPS4_DRAW_ADMISSION_LIVERPOOL_SOURCE_PATH \
    "src/video_core/amdgpu/liverpool.cpp"
#endif

#ifndef SHADPS4_DRAW_ADMISSION_RASTERIZER_SOURCE_PATH
#define SHADPS4_DRAW_ADMISSION_RASTERIZER_SOURCE_PATH \
    "src/video_core/renderer_vulkan/vk_rasterizer.cpp"
#endif

[[nodiscard]] std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input.is_open()) << path;
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::string SourceSection(const std::string& source, std::string_view begin_marker,
                                        std::string_view end_marker) {
    const auto begin = source.find(begin_marker);
    EXPECT_NE(begin, std::string::npos) << begin_marker;
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = source.find(end_marker, begin + begin_marker.size());
    EXPECT_NE(end, std::string::npos) << end_marker;
    if (end == std::string::npos) {
        return {};
    }
    return source.substr(begin, end - begin);
}

using DrawId = std::uint64_t;

enum class DrawAdmission {
    Ready,
    Retry,
    Reject,
};

// This is the smallest owner-boundary model needed by the production change. The owner gets one
// attempt per pending draw in a boundary. A retry ends the pass at that draw, preserving FIFO
// ordering and preventing a tight retry loop while the transient dependency is repaired.
class DrawAdmissionModel {
public:
    void Queue(DrawId draw) {
        pending_.push_back(draw);
    }

    template <typename Admit>
    void ProcessOwnerBoundary(Admit&& admit) {
        while (!pending_.empty()) {
            const DrawId draw = pending_.front();
            switch (admit(draw)) {
            case DrawAdmission::Ready:
                emitted_.push_back(draw);
                pending_.pop_front();
                break;
            case DrawAdmission::Retry:
                // Keep the draw at the owner boundary. Do not advance to a later draw or spin
                // until the resource/pipeline owner has had a chance to make progress.
                return;
            case DrawAdmission::Reject:
                // A permanent rejection must be explicit and observable, never a silent miss.
                rejected_.push_back(draw);
                pending_.pop_front();
                break;
            }
        }
    }

    [[nodiscard]] std::size_t PendingCount() const noexcept {
        return pending_.size();
    }

    [[nodiscard]] const std::vector<DrawId>& Emitted() const noexcept {
        return emitted_;
    }

    [[nodiscard]] const std::vector<DrawId>& Rejected() const noexcept {
        return rejected_;
    }

private:
    std::deque<DrawId> pending_;
    std::vector<DrawId> emitted_;
    std::vector<DrawId> rejected_;
};

TEST(DrawAdmissionRetry, TransientMissRemainsPendingUntilNextOwnerBoundary) {
    DrawAdmissionModel model;
    model.Queue(0x42);
    std::vector<DrawId> attempts;

    model.ProcessOwnerBoundary([&](DrawId draw) {
        attempts.push_back(draw);
        return attempts.size() == 1 ? DrawAdmission::Retry : DrawAdmission::Ready;
    });

    EXPECT_EQ(attempts, std::vector<DrawId>{0x42});
    EXPECT_EQ(model.PendingCount(), 1U);
    EXPECT_TRUE(model.Emitted().empty());
    EXPECT_TRUE(model.Rejected().empty());

    model.ProcessOwnerBoundary([&](DrawId draw) {
        attempts.push_back(draw);
        return DrawAdmission::Ready;
    });

    EXPECT_EQ(attempts, (std::vector<DrawId>{0x42, 0x42}));
    EXPECT_EQ(model.PendingCount(), 0U);
    EXPECT_EQ(model.Emitted(), std::vector<DrawId>{0x42});
    EXPECT_TRUE(model.Rejected().empty());
}

TEST(DrawAdmissionRetry, RetryPreservesFifoAndDoesNotAdvanceToLaterDraws) {
    DrawAdmissionModel model;
    model.Queue(1);
    model.Queue(2);
    std::vector<DrawId> attempts;

    model.ProcessOwnerBoundary([&](DrawId draw) {
        attempts.push_back(draw);
        return draw == 1 ? DrawAdmission::Retry : DrawAdmission::Ready;
    });

    EXPECT_EQ(attempts, std::vector<DrawId>{1});
    EXPECT_EQ(model.PendingCount(), 2U);
    EXPECT_TRUE(model.Emitted().empty());

    model.ProcessOwnerBoundary([&](DrawId draw) {
        attempts.push_back(draw);
        return DrawAdmission::Ready;
    });

    EXPECT_EQ(attempts, (std::vector<DrawId>{1, 1, 2}));
    EXPECT_EQ(model.PendingCount(), 0U);
    EXPECT_EQ(model.Emitted(), (std::vector<DrawId>{1, 2}));
}

TEST(DrawAdmissionRetry, RetryStopsTheBoundaryInsteadOfBusySpinning) {
    DrawAdmissionModel model;
    model.Queue(7);
    unsigned attempts{};

    model.ProcessOwnerBoundary([&](DrawId) {
        ++attempts;
        return DrawAdmission::Retry;
    });

    EXPECT_EQ(attempts, 1U);
    EXPECT_EQ(model.PendingCount(), 1U);
    EXPECT_TRUE(model.Emitted().empty());
}

TEST(DrawAdmissionRetry, ReadyDrawIsCommittedExactlyOnce) {
    DrawAdmissionModel model;
    model.Queue(9);
    unsigned attempts{};
    const auto admit = [&](DrawId) {
        ++attempts;
        return DrawAdmission::Ready;
    };

    model.ProcessOwnerBoundary(admit);
    model.ProcessOwnerBoundary(admit);

    EXPECT_EQ(attempts, 1U);
    EXPECT_EQ(model.PendingCount(), 0U);
    EXPECT_EQ(model.Emitted(), std::vector<DrawId>{9});
}

TEST(DrawAdmissionRetry, PermanentRejectionIsObservableAndNotSilent) {
    DrawAdmissionModel model;
    model.Queue(0xdead);

    model.ProcessOwnerBoundary([](DrawId) { return DrawAdmission::Reject; });

    EXPECT_EQ(model.PendingCount(), 0U);
    EXPECT_TRUE(model.Emitted().empty());
    EXPECT_EQ(model.Rejected(), std::vector<DrawId>{0xdead});
}

// This is the concrete reason the model above is not wired into Liverpool yet.  A retry that
// re-runs a raw PM4 packet repeats the packet's register/diagnostic side effects.  The production
// seam therefore needs an immutable draw snapshot, not a callback retaining the packet header.
TEST(DrawAdmissionRetry, RawPm4ReplayRepeatsPacketSideEffects) {
    unsigned packet_side_effects{};
    unsigned emitted_draws{};
    const auto execute_pm4_packet = [&](bool admit_draw) {
        ++packet_side_effects;
        if (admit_draw) {
            ++emitted_draws;
        }
    };

    execute_pm4_packet(false); // the first admission misses transiently
    execute_pm4_packet(true);  // a naive raw-packet retry admits the draw

    EXPECT_EQ(emitted_draws, 1U);
    EXPECT_EQ(packet_side_effects, 2U);
}

// Characterize the current owner boundary before any production retry seam is attempted.  The
// graphics parser applies the PM4 packet's state changes and calls Rasterizer synchronously; it
// advances dcb only after that call returns.  There is no retained packet/snapshot to replay at a
// later owner boundary, so a retry cannot safely be added by queueing the raw PM4 header.
TEST(DrawAdmissionRetryWiring, LiverpoolHasNoSafeRawPacketReplayBoundary) {
    const auto source = ReadSource(SHADPS4_DRAW_ADMISSION_LIVERPOOL_SOURCE_PATH);
    const auto graphics =
        SourceSection(source, "Liverpool::Task Liverpool::ProcessGraphics(",
                      "template <bool is_indirect>");
    ASSERT_FALSE(graphics.empty());

    EXPECT_NE(graphics.find("(void)owner;"), std::string::npos);

    constexpr std::array draw_cases{
        std::pair{"case PM4ItOpcode::DrawIndex2:", "case PM4ItOpcode::DrawIndexOffset2:"},
        std::pair{"case PM4ItOpcode::DrawIndexOffset2:", "case PM4ItOpcode::DrawIndexAuto:"},
        std::pair{"case PM4ItOpcode::DrawIndexAuto:", "case PM4ItOpcode::DrawIndirect:"},
        std::pair{"case PM4ItOpcode::DrawIndirect:", "case PM4ItOpcode::DrawIndirectMulti:"},
        std::pair{"case PM4ItOpcode::DrawIndirectMulti:",
                   "case PM4ItOpcode::DrawIndexIndirect:"},
        std::pair{"case PM4ItOpcode::DrawIndexIndirect:",
                   "case PM4ItOpcode::DrawIndexIndirectMulti:"},
        std::pair{"case PM4ItOpcode::DrawIndexIndirectMulti:",
                   "case PM4ItOpcode::DrawIndexIndirectCountMulti:"},
        std::pair{"case PM4ItOpcode::DrawIndexIndirectCountMulti:",
                   "case PM4ItOpcode::DispatchDirect:"},
    };
    for (const auto [opcode, next_opcode] : draw_cases) {
        const auto body = SourceSection(graphics, opcode, next_opcode);
        ASSERT_FALSE(body.empty()) << opcode;
        EXPECT_TRUE(body.find("rasterizer->Draw(") != std::string::npos ||
                    body.find("rasterizer->DrawIndirect(") != std::string::npos)
            << opcode;
        EXPECT_EQ(body.find("SendCommand("), std::string::npos) << opcode;
        EXPECT_EQ(body.find("YIELD_GFX"), std::string::npos) << opcode;
        EXPECT_EQ(body.find("RESUME_GFX"), std::string::npos) << opcode;
    }

    const auto packet_advance =
        graphics.find("dcb = NextPacket(dcb, header->type3.NumWords() + 1);");
    ASSERT_NE(packet_advance, std::string::npos);
    EXPECT_GT(packet_advance, graphics.find("case PM4ItOpcode::DrawIndex2:"));
}

TEST(DrawAdmissionRetryWiring, RasterizerDropsAreTerminalAtTheCurrentCallBoundary) {
    const auto source = ReadSource(SHADPS4_DRAW_ADMISSION_RASTERIZER_SOURCE_PATH);
    for (const auto [begin_marker, end_marker] :
         {std::pair{"void Rasterizer::Draw(bool", "void Rasterizer::DrawIndirect("},
          std::pair{"void Rasterizer::DrawIndirect(", "void Rasterizer::DispatchDirect("}}) {
        const auto body = SourceSection(source, begin_marker, end_marker);
        ASSERT_FALSE(body.empty()) << begin_marker;

        const auto null_pipeline = body.find("ObserveDrawDrop(DrawDropReason::NullPipeline");
        const auto resource_binding =
            body.find("ObserveDrawDrop(DrawDropReason::ResourceBinding");
        ASSERT_NE(null_pipeline, std::string::npos) << begin_marker;
        ASSERT_NE(resource_binding, std::string::npos) << begin_marker;
        EXPECT_NE(body.find("return;", null_pipeline), std::string::npos) << begin_marker;
        EXPECT_NE(body.find("return;", resource_binding), std::string::npos) << begin_marker;
        EXPECT_EQ(body.find("RetryDraw"), std::string::npos) << begin_marker;
        EXPECT_EQ(body.find("ReplayDraw"), std::string::npos) << begin_marker;
    }
}

TEST(DrawAdmissionRetry, DirectDrawResourceFailuresClearTrackedBindingsBeforeReturning) {
    const auto source = ReadSource(SHADPS4_DRAW_ADMISSION_RASTERIZER_SOURCE_PATH);
    const auto body = SourceSection(source, "void Rasterizer::Draw(bool",
                                    "void Rasterizer::DrawIndirect(");
    ASSERT_FALSE(body.empty());

    for (const auto condition : {"if (!PrepareRenderState(pipeline))",
                                 "if (!BindResources(pipeline))"}) {
        const auto begin = body.find(condition);
        ASSERT_NE(begin, std::string::npos) << condition;
        const auto observe = body.find("ObserveDrawDrop(DrawDropReason::ResourceBinding", begin);
        const auto cleanup = body.find("ResetBindings()", begin);
        const auto return_statement = body.find("return;", begin);
        ASSERT_NE(observe, std::string::npos) << condition;
        ASSERT_NE(cleanup, std::string::npos) << condition;
        ASSERT_NE(return_statement, std::string::npos) << condition;
        EXPECT_LT(observe, cleanup) << condition;
        EXPECT_LT(cleanup, return_statement) << condition;
    }
}

TEST(DrawAdmissionRetry, IndirectDrawResourceFailuresClearTrackedBindingsBeforeReturning) {
    const auto source = ReadSource(SHADPS4_DRAW_ADMISSION_RASTERIZER_SOURCE_PATH);
    const auto body = SourceSection(source, "void Rasterizer::DrawIndirect(",
                                    "void Rasterizer::DispatchDirect(");
    ASSERT_FALSE(body.empty());

    for (const auto condition : {"if (!PrepareRenderState(pipeline))",
                                 "if (!BindResources(pipeline))"}) {
        const auto begin = body.find(condition);
        ASSERT_NE(begin, std::string::npos) << condition;
        const auto observe = body.find("ObserveDrawDrop(DrawDropReason::ResourceBinding", begin);
        const auto cleanup = body.find("ResetBindings()", begin);
        const auto return_statement = body.find("return;", begin);
        ASSERT_NE(observe, std::string::npos) << condition;
        ASSERT_NE(cleanup, std::string::npos) << condition;
        ASSERT_NE(return_statement, std::string::npos) << condition;
        EXPECT_LT(observe, cleanup) << condition;
        EXPECT_LT(cleanup, return_statement) << condition;
    }
}

TEST(DrawAdmissionRetry, ComputeResourceFailuresClearTrackedBindingsBeforeReturning) {
    const auto source = ReadSource(SHADPS4_DRAW_ADMISSION_RASTERIZER_SOURCE_PATH);
    for (const auto [begin_marker, end_marker] :
         {std::pair{"void Rasterizer::DispatchDirect()", "void Rasterizer::DispatchIndirect("},
          std::pair{"void Rasterizer::DispatchIndirect(", "void Rasterizer::Finish()"}}) {
        const auto body = SourceSection(source, begin_marker, end_marker);
        ASSERT_FALSE(body.empty()) << begin_marker;

        const auto bind_failure = body.find("if (!BindResources(pipeline))");
        ASSERT_NE(bind_failure, std::string::npos) << begin_marker;
        const auto observe =
            body.find("ObserveDrawDrop(DrawDropReason::ResourceBinding", bind_failure);
        const auto cleanup = body.find("ResetBindings()", bind_failure);
        const auto return_statement = body.find("return;", bind_failure);
        ASSERT_NE(observe, std::string::npos) << begin_marker;
        ASSERT_NE(cleanup, std::string::npos) << begin_marker;
        ASSERT_NE(return_statement, std::string::npos) << begin_marker;
        EXPECT_LT(observe, cleanup) << begin_marker;
        EXPECT_LT(cleanup, return_statement) << begin_marker;
    }
}

} // namespace
