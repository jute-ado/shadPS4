// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <sstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "common/scope_exit.h"
#include "video_core/renderer_vulkan/async_compilation_queue.h"
#include "video_core/renderer_vulkan/async_compilation_completion.h"
#include "video_core/renderer_vulkan/compilation_latency_timer.h"

namespace Vulkan {
namespace {

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(AsyncCompilationQueue, CoalescesDuplicateKeysAndPublishesOneResult) {
    std::promise<void> release;
    auto ready = release.get_future().share();
    std::atomic_uint compile_calls{};
    AsyncCompilationQueue<int, std::string, int> queue{[&](int payload) {
        ++compile_calls;
        ready.wait();
        return std::to_string(payload);
    }};

    EXPECT_TRUE(queue.Submit(7, 41));
    EXPECT_FALSE(queue.Submit(7, 99));
    release.set_value();

    auto completion = queue.WaitTake();
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(completion->key, 7);
    ASSERT_TRUE(completion->result.has_value());
    EXPECT_EQ(*completion->result, "41");
    EXPECT_EQ(compile_calls, 1u);
}

TEST(AsyncCompilationQueue, FailedJobDoesNotKillWorkerOrPublishPartialResult) {
    AsyncCompilationQueue<int, int, int> queue{[](int payload) {
        if (payload < 0) {
            throw std::runtime_error{"compile failed"};
        }
        return payload * 2;
    }};

    EXPECT_TRUE(queue.Submit(1, -1));
    auto failed = queue.WaitTake();
    ASSERT_TRUE(failed.has_value());
    EXPECT_EQ(failed->key, 1);
    EXPECT_FALSE(failed->result.has_value());

    EXPECT_TRUE(queue.Submit(2, 6));
    auto succeeded = queue.WaitTake();
    ASSERT_TRUE(succeeded.has_value());
    ASSERT_TRUE(succeeded->result.has_value());
    EXPECT_EQ(*succeeded->result, 12);
}

TEST(AsyncCompilationQueue, ShutdownRejectsNewWorkAndJoinsTheWorker) {
    AsyncCompilationQueue<int, int, int> queue{[](int payload) { return payload; }};
    queue.Shutdown();

    EXPECT_FALSE(queue.Submit(1, 1));
    EXPECT_FALSE(queue.TryTake().has_value());
}

struct ReadyCompletions {
    using Completion = AsyncCompilationQueue<int, int, int>::Completion;
    std::deque<Completion> completed;
    std::optional<Completion> TryTake() {
        if (completed.empty()) {
            return std::nullopt;
        }
        auto result = std::move(completed.front());
        completed.pop_front();
        return result;
    }
};

TEST(AsyncCompilationCompletion, FailureAlreadyReadyCannotBeLostBeforeOwnerWait) {
    const int requested = 7;
    ReadyCompletions queue{{{7, std::nullopt}}};
    unsigned published = 0;
    EXPECT_TRUE(DrainAsyncCompilationResults(queue, &requested, [&](auto completion) {
        ++published;
        EXPECT_EQ(completion.key, requested);
        EXPECT_FALSE(completion.result);
    }));
    EXPECT_EQ(published, 1U);
    EXPECT_TRUE(queue.completed.empty());
}

TEST(AsyncCompilationCompletion, InterveningSuccessCannotEraseRequestedFailure) {
    const int requested = 7;
    ReadyCompletions queue{{{3, 30}, {7, std::nullopt}, {8, 80}}};
    std::vector<int> published;
    EXPECT_TRUE(DrainAsyncCompilationResults(queue, &requested, [&](auto completion) {
        published.push_back(completion.key);
    }));
    EXPECT_EQ(published, (std::vector<int>{3, 7, 8}));
}

TEST(AsyncCompilationCompletion, UnrelatedFailureDoesNotFailRequestedPipeline) {
    const int requested = 7;
    ReadyCompletions queue{{{3, std::nullopt}, {7, 70}}};
    unsigned published = 0;
    EXPECT_FALSE(DrainAsyncCompilationResults(queue, &requested, [&](auto) { ++published; }));
    EXPECT_EQ(published, 2U);
}

TEST(AsyncCompilationCompletion, OrdinaryPublicationHasNoRequestedFailure) {
    ReadyCompletions queue{{{3, std::nullopt}, {7, 70}}};
    unsigned published = 0;
    EXPECT_FALSE(DrainAsyncCompilationResults(queue, static_cast<const int*>(nullptr),
                                               [&](auto) { ++published; }));
    EXPECT_EQ(published, 2U);
}

TEST(AsyncCompilationCompletion, EmptyDrainDoesNotInventAResult) {
    const int requested = 7;
    ReadyCompletions queue{};
    EXPECT_FALSE(DrainAsyncCompilationResults(queue, &requested, [](auto) { FAIL(); }));
}

TEST(AsyncCompilationCompletion, OwnerHandlesFailedDrainBeforeBlockingTake) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("PipelineCache::WaitForAsyncGraphicsPipeline(");
    const auto end = source.find("const GraphicsPipeline* PipelineCache::GetGraphicsPipeline()", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto checked_drain = body.find("if (PublishAsyncGraphicsPipelines(&key))");
    const auto wait = body.find("async_graphics_compiler->WaitTake()");
    ASSERT_NE(checked_drain, std::string::npos);
    ASSERT_NE(wait, std::string::npos);
    EXPECT_LT(checked_drain, wait);
    EXPECT_NE(body.substr(checked_drain, wait - checked_drain).find("outcome = \"failed\""),
              std::string::npos);
    EXPECT_NE(source.find("DrainAsyncCompilationResults("), std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation, IsExplicitlyGatedAndPublishesOnTheOwnerThread) {
    const auto header = ReadText(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(header.empty());
    ASSERT_FALSE(source.empty());
    EXPECT_NE(header.find("AsyncGraphicsPipelineJob"), std::string::npos);
    EXPECT_NE(header.find("AsyncGraphicsPipelineResult"), std::string::npos);
    EXPECT_NE(header.find("info_copies"), std::string::npos);
    EXPECT_NE(source.find("EmulatorSettings.IsAsyncGraphicsPipelineCompilation()"),
              std::string::npos);
    const auto submit = source.find("async_graphics_compiler->Submit(");
    ASSERT_NE(submit, std::string::npos);
    EXPECT_NE(source.find("graphics_key", submit), std::string::npos);
    EXPECT_NE(source.find("PublishAsyncGraphicsPipelines()"), std::string::npos);
    EXPECT_NE(source.find("DrainAsyncCompilationResults("), std::string::npos);
    EXPECT_NE(source.find("WaitForAsyncGraphicsPipeline(graphics_key)"), std::string::npos);
    EXPECT_NE(source.find("result.pipeline->RebindStages(result.retained_infos)"),
              std::string::npos);

    const auto compute_begin = source.find("PipelineCache::GetComputePipeline()");
    const auto compute_end = source.find("PipelineCache::RefreshGraphicsKey()", compute_begin);
    ASSERT_NE(compute_begin, std::string::npos);
    ASSERT_NE(compute_end, std::string::npos);
    EXPECT_EQ(source.substr(compute_begin, compute_end - compute_begin).find("Submit("),
              std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation, HasAStableExperimentalConfigurationKey) {
    const auto settings_header = ReadText(SHADPS4_EMULATOR_SETTINGS_HEADER_PATH);
    const auto settings_source = ReadText(SHADPS4_EMULATOR_SETTINGS_SOURCE_PATH);

    ASSERT_FALSE(settings_header.empty());
    ASSERT_FALSE(settings_source.empty());
    EXPECT_NE(settings_header.find("async_graphics_pipeline_compilation"), std::string::npos);
    EXPECT_NE(settings_source.find("asyncGraphicsPipelineCompilation"), std::string::npos);
}

TEST(AsyncGraphicsPipelineTiming, WorkerDurationIsOptInAndBracketsActualConstruction) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("PipelineCache::CompileGraphicsPipelineAsync(");
    const auto end = source.find("void PipelineCache::PublishAsyncGraphicsPipeline(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto timer = body.find("CompilationLatencyTimer<> timing{IsCompilationTimingEnabled()}");
    const auto start = body.find("async_pipeline_worker_start key={:#x}");
    const auto guard = body.find("SCOPE_EXIT");
    const auto finish = body.find("async_pipeline_worker_finish key={:#x} elapsedMs={:.3f}");
    const auto compile = body.find("std::make_unique<GraphicsPipeline>(");
    ASSERT_NE(timer, std::string::npos);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(guard, std::string::npos);
    ASSERT_NE(finish, std::string::npos);
    ASSERT_NE(compile, std::string::npos);
    EXPECT_LT(timer, start);
    EXPECT_LT(start, guard);
    EXPECT_LT(guard, compile);
    EXPECT_LT(finish, compile); // Scope-exit callback is installed before construction can throw.
    EXPECT_NE(body.find("timing.ElapsedMilliseconds()"), std::string::npos);
    EXPECT_NE(body.find("std::hash<GraphicsPipelineKey>{}(job.key)"), std::string::npos);
    EXPECT_NE(body.find("if (timing.IsEnabled())"), std::string::npos);
    EXPECT_NE(source.find("should_log(spdlog::level::debug)"), std::string::npos);
}

TEST(AsyncGraphicsPipelineTiming, OwnerWaitIncludesAllCompletionsAndEveryExitReason) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto begin = source.find("PipelineCache::WaitForAsyncGraphicsPipeline(");
    const auto end =
        source.find("const GraphicsPipeline* PipelineCache::GetGraphicsPipeline()", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto timer = body.find("CompilationLatencyTimer<> timing{IsCompilationTimingEnabled()}");
    const auto start = body.find("async_pipeline_wait_start key={:#x}");
    const auto finish = body.find("async_pipeline_wait_finish key={:#x} elapsedMs={:.3f}");
    const auto loop = body.find("for (;;)");
    const auto wait = body.find("async_graphics_compiler->WaitTake()");
    ASSERT_NE(timer, std::string::npos);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(finish, std::string::npos);
    ASSERT_NE(loop, std::string::npos);
    ASSERT_NE(wait, std::string::npos);
    EXPECT_LT(timer, start);
    EXPECT_LT(start, finish);
    EXPECT_LT(finish, loop);
    EXPECT_LT(loop, wait);
    EXPECT_NE(body.find("SCOPE_EXIT"), std::string::npos);
    EXPECT_NE(body.find("timing.ElapsedMilliseconds()"), std::string::npos);
    EXPECT_NE(body.find("std::hash<GraphicsPipelineKey>{}(key)"), std::string::npos);
    EXPECT_NE(body.find("outcome = \"ready\""), std::string::npos);
    EXPECT_NE(body.find("outcome = \"stopped\""), std::string::npos);
    EXPECT_NE(body.find("outcome = \"failed\""), std::string::npos);
}

TEST(AsyncGraphicsPipelineTiming, DedicatedFilterDoesNotEnableVerboseRendererLogging) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto src_root = std::filesystem::path{SHADPS4_PIPELINE_CACHE_SOURCE_PATH}
                              .parent_path()
                              .parent_path()
                              .parent_path();
    const auto classes = ReadText(src_root / "common/logging/classes.h");
    const auto logging = ReadText(src_root / "common/logging/log.cpp");
    ASSERT_FALSE(classes.empty());
    ASSERT_FALSE(logging.empty());
    EXPECT_NE(classes.find("Render_Compilation = \"Render.Compilation\""), std::string::npos);
    EXPECT_NE(logging.find("{Class::Render_Compilation, nullptr}"), std::string::npos);

    const auto begin = source.find("static bool IsCompilationTimingEnabled()");
    const auto end =
        source.find("const GraphicsPipeline* PipelineCache::GetGraphicsPipeline()", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    EXPECT_NE(body.find("ALL_LOGGERS.find(Common::Log::Class::Render_Compilation)"),
              std::string::npos);
    for (const auto* marker : {"async_pipeline_worker_start", "async_pipeline_worker_finish",
                               "async_pipeline_wait_start", "async_pipeline_wait_finish"}) {
        const auto message = body.find(marker);
        ASSERT_NE(message, std::string::npos);
        const auto log = body.rfind("LOG_DEBUG(", message);
        ASSERT_NE(log, std::string::npos);
        EXPECT_NE(body.substr(log, message - log).find("LOG_DEBUG(Render_Compilation,"),
                  std::string::npos);
    }
}

struct FakeCompilationClock {
    using duration = std::chrono::microseconds;
    using time_point = std::chrono::time_point<FakeCompilationClock, duration>;
    static constexpr bool is_steady = true;
    static inline time_point current{};
    static inline unsigned reads{};

    static time_point now() {
        ++reads;
        return current;
    }
};

TEST(CompilationLatencyTimer, DisabledDiagnosticsNeverReadTheClock) {
    FakeCompilationClock::reads = 0;
    CompilationLatencyTimer<FakeCompilationClock> timer{false};
    EXPECT_FALSE(timer.IsEnabled());
    EXPECT_DOUBLE_EQ(timer.ElapsedMilliseconds(), 0.0);
    EXPECT_EQ(FakeCompilationClock::reads, 0U);
}

TEST(CompilationLatencyTimer, ReportsFractionalMillisecondsRatherThanTicksOrSeconds) {
    FakeCompilationClock::current = FakeCompilationClock::time_point{std::chrono::hours{9}};
    CompilationLatencyTimer<FakeCompilationClock> timer{true};
    EXPECT_TRUE(timer.IsEnabled());
    EXPECT_DOUBLE_EQ(timer.ElapsedMilliseconds(), 0.0);
    FakeCompilationClock::current += std::chrono::microseconds{2'100'250};
    EXPECT_DOUBLE_EQ(timer.ElapsedMilliseconds(), 2100.25);
}

TEST(CompilationLatencyTimer, WorkerAndOwnerWaitUseIndependentStartTimes) {
    FakeCompilationClock::current = {};
    CompilationLatencyTimer<FakeCompilationClock> owner{true};
    FakeCompilationClock::current += std::chrono::milliseconds{100};
    CompilationLatencyTimer<FakeCompilationClock> worker{true};
    FakeCompilationClock::current += std::chrono::milliseconds{2100};
    EXPECT_DOUBLE_EQ(worker.ElapsedMilliseconds(), 2100.0);
    EXPECT_DOUBLE_EQ(owner.ElapsedMilliseconds(), 2200.0);
}

TEST(CompilationLatencyTimer, ScopeExitRecordsFailureWithoutSwallowingIt) {
    FakeCompilationClock::current = {};
    double elapsed_ms = -1;
    unsigned finished = 0;
    const auto fails = [&] {
        CompilationLatencyTimer<FakeCompilationClock> timer{true};
        SCOPE_EXIT {
            elapsed_ms = timer.ElapsedMilliseconds();
            ++finished;
        };
        FakeCompilationClock::current += std::chrono::milliseconds{650};
        throw std::runtime_error{"compiler failure"};
    };
    EXPECT_THROW(fails(), std::runtime_error);
    EXPECT_EQ(finished, 1U);
    EXPECT_DOUBLE_EQ(elapsed_ms, 650.0);
}

} // namespace
} // namespace Vulkan
