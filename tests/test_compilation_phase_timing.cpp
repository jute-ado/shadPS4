// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/compilation_timing.h"

namespace {
using namespace Common;
struct FakeClock {
    using duration = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<FakeClock, duration>;
    static constexpr bool is_steady = true;
    static inline time_point current{};
    static inline unsigned reads{};
    static time_point now() {
        ++reads;
        return current;
    }
};

TEST(CompilationPhaseTiming, DisabledDoesNotReadClockClaimIdsOrEvaluateMetadata) {
    FakeClock::reads = 0;
    CompilationTraceBudget budget{2};
    unsigned metadata = 0;
    unsigned emitted = 0;
    auto sink = [&](const auto&) { ++emitted; };
    {
        ScopedCompilationSpan<decltype(sink), FakeClock> span{sink, false, budget, [&] {
                                                                  ++metadata;
                                                                  return CompilationContext{};
                                                              }};
        span.SetOutcome(CompilationOutcome::Complete);
    }
    EmitCompilationTerminal<FakeClock>(false, budget, sink, 123);
    EXPECT_EQ(FakeClock::reads, 0u);
    EXPECT_EQ(metadata, 0u);
    EXPECT_EQ(emitted, 0u);
    EXPECT_EQ(budget.Snapshot().admitted, 0u);
    EXPECT_EQ(budget.Snapshot().dropped, 0u);
}

TEST(CompilationPhaseTiming, WorkerOwnerAndTerminalShareProducerClockEpoch) {
    CompilationTraceBudget budget;
    std::vector<CompilationTraceRecord> records;
    auto sink = [&](const auto& record) { records.push_back(record); };
    FakeClock::current = FakeClock::time_point{std::chrono::nanoseconds{1000}};
    {
        ScopedCompilationSpan<decltype(sink), FakeClock> owner{
            sink, true, budget,
            [] { return CompilationContext{.phase = CompilationPhase::GraphicsWait, .key = 42}; }};
        FakeClock::current += std::chrono::nanoseconds{100};
        {
            ScopedCompilationSpan<decltype(sink), FakeClock> worker{
                sink, true, budget, [] {
                    return CompilationContext{.phase = CompilationPhase::GraphicsWorker, .key = 42};
                }};
            FakeClock::current += std::chrono::nanoseconds{250};
            worker.SetOutcome(CompilationOutcome::Complete);
        }
        FakeClock::current += std::chrono::nanoseconds{50};
        owner.SetOutcome(CompilationOutcome::Ready);
    }
    auto delayed_sink = [&](const auto& record) {
        FakeClock::current += std::chrono::seconds{8};
        records.push_back(record);
    };
    EmitCompilationTerminal<FakeClock>(true, budget, delayed_sink, 0xc0000005);
    ASSERT_EQ(records.size(), 5u);
    EXPECT_EQ(records[0].id, 1u);
    EXPECT_EQ(records[1].id, 2u);
    EXPECT_EQ(records[2].id, 2u);
    EXPECT_EQ(records[3].id, 1u);
    EXPECT_EQ(records[0].at_ns, 1000);
    EXPECT_EQ(records[1].at_ns, 1100);
    EXPECT_EQ(records[2].elapsed_ns, 250);
    EXPECT_EQ(records[3].elapsed_ns, 400);
    EXPECT_EQ(records[2].outcome, CompilationOutcome::Complete);
    EXPECT_EQ(records[3].outcome, CompilationOutcome::Ready);
    EXPECT_EQ(records[4].kind, CompilationRecordKind::Terminal);
    EXPECT_EQ(records[4].at_ns, 1400); // Producer time, not delayed sink time.
    EXPECT_EQ(records[4].admitted, 2u);
    EXPECT_EQ(records[4].code, 0xc0000005u);
}

TEST(CompilationPhaseTiming, UnwindingReportsExceptionWithoutSwallowingIt) {
    CompilationTraceBudget budget;
    std::vector<CompilationTraceRecord> records;
    auto sink = [&](const auto& record) { records.push_back(record); };
    FakeClock::current = {};
    const auto operation = [&] {
        ScopedCompilationSpan<decltype(sink), FakeClock> span{
            sink, true, budget,
            [] { return CompilationContext{.phase = CompilationPhase::OwnerModule, .key = 7}; }};
        FakeClock::current += std::chrono::nanoseconds{19};
        throw std::runtime_error{"original"};
    };
    EXPECT_THROW(operation(), std::runtime_error);
    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records.back().outcome, CompilationOutcome::Exception);
    EXPECT_EQ(records.back().elapsed_ns, 19);
}

TEST(CompilationPhaseTiming, DoesNotInventFinishWhileScopeIsStillActive) {
    CompilationTraceBudget budget;
    std::vector<CompilationTraceRecord> records;
    auto sink = [&](const auto& record) { records.push_back(record); };
    {
        ScopedCompilationSpan<decltype(sink), FakeClock> span{
            sink, true, budget,
            [] { return CompilationContext{.phase = CompilationPhase::OwnerTranslate}; }};
        EXPECT_EQ(records.size(), 1u);
        if (!records.empty())
            EXPECT_EQ(records.front().kind, CompilationRecordKind::Start);
        span.SetOutcome(CompilationOutcome::Rejected);
    }
    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records.back().outcome, CompilationOutcome::Rejected);
}

TEST(CompilationPhaseTiming, BudgetReservesPairsAndReportsOverflowAndTerminalOnce) {
    CompilationTraceBudget budget{2};
    std::vector<CompilationTraceRecord> records;
    auto sink = [&](const auto& record) { records.push_back(record); };
    unsigned metadata = 0;
    for (unsigned i = 0; i < 4; ++i) {
        ScopedCompilationSpan<decltype(sink), FakeClock> span{
            sink, true, budget, [&] {
                ++metadata;
                return CompilationContext{.phase = CompilationPhase::OwnerCompute, .key = 3};
            }};
        span.SetOutcome(CompilationOutcome::Complete);
    }
    EmitCompilationTerminal<FakeClock>(true, budget, sink, 1);
    const auto reads = FakeClock::reads;
    EmitCompilationTerminal<FakeClock>(true, budget, sink, 2);
    EXPECT_EQ(FakeClock::reads, reads);
    EXPECT_EQ(metadata, 2u);
    EXPECT_EQ(budget.Snapshot().admitted, 2u);
    EXPECT_EQ(budget.Snapshot().dropped, 2u);
    ASSERT_EQ(records.size(), 6u);
    EXPECT_EQ(records[0].id, 1u);
    EXPECT_EQ(records[2].id, 2u);
    EXPECT_EQ(records[4].kind, CompilationRecordKind::Overflow);
    EXPECT_EQ(records[4].limit, 2u);
    EXPECT_EQ(records[4].dropped, 1u);
    EXPECT_EQ(records[5].kind, CompilationRecordKind::Terminal);
    EXPECT_EQ(records[5].admitted, 2u);
    EXPECT_EQ(records[5].dropped, 2u);
}

TEST(CompilationPhaseTiming, ConcurrentClaimsAreUniqueAndNeverExceedBudget) {
    CompilationTraceBudget budget{32};
    std::mutex mutex;
    std::vector<CompilationTraceRecord> records;
    auto sink = [&](const auto& record) {
        std::scoped_lock lock{mutex};
        records.push_back(record);
    };
    std::vector<std::jthread> workers;
    for (unsigned thread = 0; thread < 8; ++thread) {
        workers.emplace_back([&] {
            for (unsigned i = 0; i < 8; ++i) {
                ScopedCompilationSpan span{sink, true, budget,
                                           [] { return CompilationContext{.key = 7}; }};
                span.SetOutcome(CompilationOutcome::Complete);
            }
        });
    }
    workers.clear();
    std::vector<std::uint64_t> starts;
    unsigned finishes = 0;
    unsigned overflow = 0;
    for (const auto& record : records) {
        if (record.kind == CompilationRecordKind::Start)
            starts.push_back(record.id);
        if (record.kind == CompilationRecordKind::Finish)
            ++finishes;
        if (record.kind == CompilationRecordKind::Overflow)
            ++overflow;
    }
    std::ranges::sort(starts);
    ASSERT_EQ(starts.size(), 32u);
    for (unsigned i = 0; i < starts.size(); ++i)
        EXPECT_EQ(starts[i], i + 1);
    EXPECT_EQ(finishes, 32u);
    EXPECT_EQ(overflow, 1u);
    EXPECT_EQ(budget.Snapshot().admitted, 32u);
    EXPECT_EQ(budget.Snapshot().dropped, 32u);
}

std::string ReadSource(const char* path) {
    std::ifstream input{path};
    return {std::istreambuf_iterator<char>{input}, {}};
}
std::string Section(const std::string& source, std::string_view begin, std::string_view end) {
    const auto first = source.find(begin);
    const auto last = source.find(end, first);
    return first == std::string::npos || last == std::string::npos
               ? std::string{}
               : source.substr(first, last - first);
}

TEST(CompilationPhaseWiring, TranslationAndModuleUseDistinctScopes) {
    const auto source = ReadSource(SHADPS4_PHASE_PIPELINE_SOURCE);
    const auto translate = Section(source, "PipelineCache::TranslateModule(",
                                   "vk::ShaderModule PipelineCache::CompileModule(");
    const auto module = Section(source, "PipelineCache::CompileModule(",
                                "PipelineCache::Result PipelineCache::GetProgram(");
    EXPECT_NE(translate.find("CompilationPhase::OwnerTranslate"), std::string::npos);
    EXPECT_LT(translate.find("TraceCompilation("), translate.find("Shader::TranslateProgram("));
    EXPECT_NE(translate.find("CompilationOutcome::Rejected"), std::string::npos);
    EXPECT_NE(module.find("CompilationPhase::OwnerModule"), std::string::npos);
    EXPECT_LT(module.find("TraceCompilation("), module.find("CompileSPV("));
}

TEST(CompilationPhaseWiring, BothEmissionPathsAndNewComputeConstructionAreTimed) {
    const auto source = ReadSource(SHADPS4_PHASE_PIPELINE_SOURCE);
    const auto program = Section(source, "PipelineCache::Result PipelineCache::GetProgram(",
                                 "PipelineCache::ReplaceShader(");
    const auto first = program.find("CompilationPhase::OwnerSpirv");
    ASSERT_NE(first, std::string::npos);
    const auto second = program.find("CompilationPhase::OwnerSpirv", first + 1);
    ASSERT_NE(second, std::string::npos);
    EXPECT_EQ(program.find("CompilationPhase::OwnerSpirv", second + 1), std::string::npos);
    EXPECT_LT(first, program.find("Shader::Backend::SPIRV::EmitSPIRV("));
    EXPECT_LT(second, program.find("Shader::Backend::SPIRV::EmitSPIRV(", second));
    const auto compute = Section(source, "PipelineCache::GetComputePipeline()",
                                 "bool PipelineCache::RefreshGraphicsKey(");
    const auto timing = compute.find("CompilationPhase::OwnerCompute");
    EXPECT_LT(compute.find("if (is_new)"), timing);
    EXPECT_LT(timing, compute.find("std::make_unique<ComputePipeline>("));
}

TEST(CompilationPhaseWiring, ExistingWorkerAndWaitUseSameTraceAndNoFakeSubmission) {
    const auto source = ReadSource(SHADPS4_PHASE_PIPELINE_SOURCE);
    const auto worker = Section(source, "PipelineCache::CompileGraphicsPipelineAsync(",
                                "void PipelineCache::PublishAsyncGraphicsPipeline(");
    const auto wait = Section(source, "PipelineCache::WaitForAsyncGraphicsPipeline(",
                              "const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(");
    EXPECT_NE(worker.find("CompilationPhase::GraphicsWorker"), std::string::npos);
    EXPECT_LT(worker.find("TraceCompilation("), worker.find("std::make_unique<GraphicsPipeline>("));
    EXPECT_NE(wait.find("CompilationPhase::GraphicsWait"), std::string::npos);
    EXPECT_LT(wait.find("TraceCompilation("), wait.find("for (;;)"));
    for (const auto* outcome : {"Ready", "Failed", "Stopped"}) {
        EXPECT_NE(wait.find(std::string{"CompilationOutcome::"} + outcome), std::string::npos);
    }
    EXPECT_EQ(source.find("IssuedSubmissionSequence()"), std::string::npos);
}

enum class AsyncPipelinePhase {
    OwnerPreparation,
    QueueSubmission,
    WorkerStart,
    WorkerFinish,
    OwnerPublication,
    DrawRelease,
};

struct AsyncPipelinePhaseLedgerModel {
    std::vector<AsyncPipelinePhase> phases;
    bool queued{};
    bool published{};
    unsigned waiters{};

    bool RequestGraphicsPipeline() {
        phases.push_back(AsyncPipelinePhase::OwnerPreparation);
        if (published) {
            phases.push_back(AsyncPipelinePhase::DrawRelease);
            return true;
        }
        ++waiters;
        if (!queued) {
            queued = true;
            phases.push_back(AsyncPipelinePhase::QueueSubmission);
        }
        return false;
    }

    bool CompleteWorker(bool worker_success) {
        if (!queued)
            return published;

        queued = false;
        phases.push_back(AsyncPipelinePhase::WorkerStart);
        phases.push_back(AsyncPipelinePhase::WorkerFinish);
        if (worker_success)
            phases.push_back(AsyncPipelinePhase::OwnerPublication);

        published = worker_success;
        while (waiters != 0) {
            --waiters;
            phases.push_back(AsyncPipelinePhase::DrawRelease);
        }
        return published;
    }
};

TEST(AsyncPipelinePhaseLedger, GraphicsMissPublishesBeforeRequestedDrawRelease) {
    AsyncPipelinePhaseLedgerModel model;
    ASSERT_FALSE(model.RequestGraphicsPipeline());
    ASSERT_TRUE(model.CompleteWorker(true));
    const std::vector<AsyncPipelinePhase> expected{
        AsyncPipelinePhase::OwnerPreparation, AsyncPipelinePhase::QueueSubmission,
        AsyncPipelinePhase::WorkerStart,      AsyncPipelinePhase::WorkerFinish,
        AsyncPipelinePhase::OwnerPublication, AsyncPipelinePhase::DrawRelease,
    };
    EXPECT_EQ(model.phases, expected);
}

TEST(AsyncPipelinePhaseLedger, DuplicateRequestDoesNotSubmitAnotherWorkerJob) {
    AsyncPipelinePhaseLedgerModel model;
    ASSERT_FALSE(model.RequestGraphicsPipeline());
    ASSERT_FALSE(model.RequestGraphicsPipeline());
    ASSERT_TRUE(model.CompleteWorker(true));
    const auto submissions =
        std::count(model.phases.begin(), model.phases.end(), AsyncPipelinePhase::QueueSubmission);
    const auto releases =
        std::count(model.phases.begin(), model.phases.end(), AsyncPipelinePhase::DrawRelease);
    EXPECT_EQ(submissions, 1);
    EXPECT_EQ(releases, 2);
}

TEST(AsyncPipelinePhaseLedger, FailedWorkerCompletionNeverPublishesPipeline) {
    AsyncPipelinePhaseLedgerModel model;
    ASSERT_FALSE(model.RequestGraphicsPipeline());
    ASSERT_FALSE(model.CompleteWorker(false));
    EXPECT_FALSE(model.published);
    const std::vector<AsyncPipelinePhase> expected{
        AsyncPipelinePhase::OwnerPreparation, AsyncPipelinePhase::QueueSubmission,
        AsyncPipelinePhase::WorkerStart,      AsyncPipelinePhase::WorkerFinish,
        AsyncPipelinePhase::DrawRelease,
    };
    EXPECT_EQ(model.phases, expected);
}

TEST(AsyncPipelineWiring, GraphicsPreparationSubmissionAndWaitRemainOrdered) {
    const auto source = ReadSource(SHADPS4_PHASE_PIPELINE_SOURCE);
    const auto graphics =
        Section(source, "const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(",
                "const ComputePipeline* PipelineCache::GetComputePipeline(");
    ASSERT_FALSE(graphics.empty());

    const auto preparation = graphics.find("RefreshGraphicsKey(draw_user_data)");
    const auto submission = graphics.find("async_graphics_compiler->Submit(");
    const auto wait = graphics.find("WaitForAsyncGraphicsPipeline(graphics_key)");
    ASSERT_NE(preparation, std::string::npos);
    ASSERT_NE(submission, std::string::npos);
    ASSERT_NE(wait, std::string::npos);
    EXPECT_LT(preparation, submission);
    EXPECT_LT(submission, wait);
}

TEST(AsyncPipelineWiring, WorkerValidationPrecedesOwnerPublication) {
    const auto source = ReadSource(SHADPS4_PHASE_PIPELINE_SOURCE);
    const auto worker = Section(source, "PipelineCache::CompileGraphicsPipelineAsync(",
                                "void PipelineCache::PublishAsyncGraphicsPipeline(");
    const auto publish = Section(source, "void PipelineCache::PublishAsyncGraphicsPipeline(",
                                 "bool PipelineCache::PublishAsyncGraphicsPipelines(");
    ASSERT_FALSE(worker.empty());
    ASSERT_FALSE(publish.empty());

    const auto worker_compile = worker.find("std::make_unique<GraphicsPipeline>(");
    const auto worker_validation = worker.find("RequireValidCompiledPipeline(pipeline)");
    const auto publish_validation = publish.find("if (!completion.result");
    const auto map_insert = publish.find("graphics_pipelines.try_emplace(");
    const auto rebind = publish.find("result.pipeline->RebindStages(");
    const auto store = publish.find("it.value() = std::move(result.pipeline)");
    ASSERT_NE(worker_compile, std::string::npos);
    ASSERT_NE(worker_validation, std::string::npos);
    ASSERT_NE(publish_validation, std::string::npos);
    ASSERT_NE(map_insert, std::string::npos);
    ASSERT_NE(rebind, std::string::npos);
    ASSERT_NE(store, std::string::npos);
    EXPECT_LT(worker_compile, worker_validation);
    EXPECT_LT(publish_validation, map_insert);
    EXPECT_LT(rebind, store);
}

TEST(AsyncPipelineWiring, ComputeCompilationRemainsOwnerSynchronousInAsyncMode) {
    const auto source = ReadSource(SHADPS4_PHASE_PIPELINE_SOURCE);
    const auto compute = Section(source, "PipelineCache::GetComputePipeline()",
                                 "bool PipelineCache::RefreshGraphicsKey(");
    ASSERT_FALSE(compute.empty());

    const auto refresh = compute.find("RefreshComputeKey()");
    const auto pause = compute.find("ScopedHostCompilationGuestPause");
    const auto phase = compute.find("CompilationPhase::OwnerCompute");
    const auto construct = compute.find("std::make_unique<ComputePipeline>(");
    ASSERT_NE(refresh, std::string::npos);
    ASSERT_NE(pause, std::string::npos);
    ASSERT_NE(phase, std::string::npos);
    ASSERT_NE(construct, std::string::npos);
    EXPECT_LT(refresh, phase);
    EXPECT_LT(phase, construct);
    EXPECT_NE(compute.find("!EmulatorSettings.IsAsyncGraphicsPipelineCompilation()", pause),
              std::string::npos);
    EXPECT_EQ(compute.find("async_graphics_compiler"), std::string::npos);
    EXPECT_EQ(compute.find("WaitForAsyncGraphicsPipeline"), std::string::npos);
    EXPECT_EQ(compute.find("CompileGraphicsPipelineAsync"), std::string::npos);
}

TEST(CompilationPhaseWiring, TerminalSamplePrecedesExistingCriticalAndShutdown) {
    const auto source = ReadSource(SHADPS4_PHASE_SIGNALS_SOURCE);
    const auto terminal =
        Section(source, "if (report_unhandled)", "return EXCEPTION_CONTINUE_SEARCH;");
    ASSERT_NE(terminal.find("ReportCompilationTraceTerminal(code)"), std::string::npos);
    EXPECT_LT(terminal.find("ReportCompilationTraceTerminal(code)"),
              terminal.find("LOG_CRITICAL(Debug,"));
    EXPECT_LT(terminal.find("ReportCompilationTraceTerminal(code)"), terminal.find("->Shutdown()"));
}
} // namespace
