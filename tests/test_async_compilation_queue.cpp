// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <future>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/async_compilation_queue.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace Vulkan {
namespace {

std::string ReadText(const char* path) {
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

TEST(AsyncCompilationQueue, BoundsOutstandingJobsWithoutBlockingTheSubmittingThread) {
    std::promise<void> release;
    auto ready = release.get_future().share();
    AsyncCompilationQueue<int, int, int> queue{[ready](int payload) {
        ready.wait();
        return payload;
    }, 2};

    EXPECT_TRUE(queue.Submit(1, 1));
    EXPECT_TRUE(queue.Submit(2, 2));
    EXPECT_FALSE(queue.CanAccept());
    EXPECT_FALSE(queue.Submit(3, 3));

    release.set_value();
    ASSERT_TRUE(queue.WaitTake().has_value());
    EXPECT_TRUE(queue.CanAccept());
    EXPECT_TRUE(queue.Submit(3, 3));
}

TEST(AsyncCompilationQueue, CompletionNotifierRunsAfterResultIsPublishable) {
    using Queue = AsyncCompilationQueue<int, int, int>;
    Queue* queue_ptr{};
    std::promise<bool> completion_was_publishable;
    auto observed = completion_was_publishable.get_future();

    Queue queue{[](int payload) { return payload * 2; },
                std::numeric_limits<size_t>::max(),
                [&] {
                    completion_was_publishable.set_value(queue_ptr->TryTake().has_value());
                }};
    queue_ptr = &queue;

    ASSERT_TRUE(queue.Submit(9, 21));
    ASSERT_EQ(observed.wait_for(std::chrono::seconds{2}), std::future_status::ready);
    EXPECT_TRUE(observed.get());
}

TEST(AsyncGraphicsPipelineCompilation, IsExplicitlyGatedAndPublishesOnTheOwnerThread) {
    const auto header = ReadText(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(header.empty());
    ASSERT_FALSE(source.empty());
    EXPECT_NE(header.find("AsyncGraphicsPipelineJob"), std::string::npos);
    EXPECT_NE(header.find("AsyncGraphicsPipelineResult"), std::string::npos);
    EXPECT_NE(header.find("info_copies"), std::string::npos);
    EXPECT_NE(header.find("CompileResourceSnapshot"), std::string::npos);
    EXPECT_NE(source.find("EmulatorSettings.IsAsyncGraphicsPipelineCompilation()"),
              std::string::npos);
    const auto submit = source.find("async_graphics_compiler->Submit(");
    ASSERT_NE(submit, std::string::npos);
    EXPECT_NE(source.find("graphics_key", submit), std::string::npos);
    EXPECT_NE(source.find("PublishAsyncGraphicsPipelines()"), std::string::npos);
    EXPECT_NE(source.find("async_graphics_compiler->TryTake()"), std::string::npos);
    EXPECT_NE(source.find("result.pipeline->RebindStages(result.retained_infos)"),
              std::string::npos);
    const auto rebind = header.find("void RebindCompileInfo() noexcept");
    ASSERT_NE(rebind, std::string::npos);
    const auto rebind_end = header.find("};", rebind);
    ASSERT_NE(rebind_end, std::string::npos);
    const auto rebind_body = header.substr(rebind, rebind_end - rebind);
    EXPECT_NE(rebind_body.find("ClearCompileCaptureReader"), std::string::npos);
    EXPECT_NE(rebind_body.find("ClearCompileFetchShaderSnapshot"), std::string::npos);

    const auto compute_begin = source.find("PipelineCache::GetComputePipeline()");
    const auto compute_end = source.find("PipelineCache::RefreshGraphicsKey()", compute_begin);
    ASSERT_NE(compute_begin, std::string::npos);
    ASSERT_NE(compute_end, std::string::npos);
    EXPECT_EQ(source.substr(compute_begin, compute_end - compute_begin).find("Submit("),
              std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation,
     DoesNotOffloadShaderModuleEmissionUntilPendingDrawReplayIsSupported) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    const auto constructor = source.find("PipelineCache::PipelineCache(");
    const auto destructor = source.find("PipelineCache::~PipelineCache()", constructor);
    ASSERT_NE(constructor, std::string::npos);
    ASSERT_NE(destructor, std::string::npos);
    const auto constructor_body = source.substr(constructor, destructor - constructor);

    EXPECT_NE(constructor_body.find("async_graphics_compiler ="), std::string::npos);
    EXPECT_EQ(constructor_body.find("async_shader_compiler ="), std::string::npos);
}

bool PS4_SYSV_ABI UnexpectedGuestRead(void*, VAddr, void*, u64) {
    return false;
}

TEST(AsyncShaderModuleJob, OwnsSpansAndPreservesOriginalGuestProgramBase) {
    std::array<u32, 4> guest_code{1, 2, 3, 4};
    std::array<u32, Shader::NUM_USER_DATA_REGS> guest_user_data{};
    for (u32 index = 0; index < guest_user_data.size(); ++index) {
        guest_user_data[index] = 0x1000 + index;
    }
    std::array<u32, 3> guest_geometry_copy{11, 22, 33};
    Shader::ShaderParams params{
        .user_data = std::span<const u32, Shader::NUM_USER_DATA_REGS>{guest_user_data},
        .code = guest_code,
        .hash = 0xABCDEF,
    };
    Shader::RuntimeInfo runtime{};
    runtime.Initialize(Shader::Stage::Geometry);
    runtime.gs_info.vs_copy = guest_geometry_copy;

    AsyncShaderModuleJob job;
    job.CaptureOwnedInputs(Shader::Stage::Geometry, Shader::LogicalStage::Geometry, params, runtime,
                           {.unified = 7, .buffer = 8, .user_data = 9});

    EXPECT_EQ(job.info.pgm_base, reinterpret_cast<VAddr>(guest_code.data()));
    EXPECT_NE(job.code.data(), guest_code.data());
    EXPECT_EQ(job.info.user_data.data(), job.user_data_copy.data());
    EXPECT_EQ(job.info.user_data.size(), guest_user_data.size());
    EXPECT_TRUE(std::ranges::equal(job.info.user_data, guest_user_data));
    EXPECT_EQ(job.runtime_info.gs_info.vs_copy.data(), job.runtime_vs_copy.data());
    EXPECT_TRUE(std::ranges::equal(job.runtime_info.gs_info.vs_copy, guest_geometry_copy));
    EXPECT_EQ(job.binding,
              (Shader::Backend::Bindings{.unified = 7, .buffer = 8, .user_data = 9}));

    job.info.UseCompileCaptureReader({UnexpectedGuestRead, nullptr});
    job.RebindOwnedViews();
    EXPECT_EQ(job.info.compile_guest_memory_reader.callback, nullptr);
}

TEST(AsyncShaderModuleJob, PublishedModuleOwnsGeometryRuntimeSpan) {
    std::array<u32, 3> guest_geometry_copy{5, 6, 7};
    Shader::Info info{};
    Shader::StageSpecialization spec{};
    spec.info = &info;
    spec.runtime_info.Initialize(Shader::Stage::Geometry);
    spec.runtime_info.gs_info.vs_copy = guest_geometry_copy;

    Program::Module module{vk::ShaderModule{}, std::move(spec), std::move(info), 17};
    guest_geometry_copy.fill(99);

    EXPECT_EQ(module.spec.info, module.info.get());
    EXPECT_EQ(module.spec.runtime_info.gs_info.vs_copy.data(), module.runtime_vs_copy.data());
    EXPECT_TRUE(
        std::ranges::equal(module.spec.runtime_info.gs_info.vs_copy, module.runtime_vs_copy));
    EXPECT_EQ(module.runtime_vs_copy, (std::vector<u32>{5, 6, 7}));
}

TEST(AsyncGraphicsPipelineJob, OwnsGeometryRuntimeSpanAcrossQueueMoves) {
    constexpr u32 GeometryStage = static_cast<u32>(Shader::LogicalStage::Geometry);
    std::array<u32, 3> guest_geometry_copy{13, 21, 34};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    runtime_infos[GeometryStage].Initialize(Shader::Stage::Geometry);
    runtime_infos[GeometryStage].gs_info.vs_copy = guest_geometry_copy;

    AsyncGraphicsPipelineJob job;
    job.CaptureRuntimeInfos(runtime_infos);
    AsyncGraphicsPipelineJob moved{std::move(job)};
    moved.RebindCompileInfo();
    guest_geometry_copy.fill(99);

    EXPECT_EQ(moved.runtime_infos[GeometryStage].gs_info.vs_copy.data(),
              moved.runtime_vs_copies[GeometryStage].data());
    EXPECT_EQ(moved.runtime_vs_copies[GeometryStage], (std::vector<u32>{13, 21, 34}));
    EXPECT_TRUE(std::ranges::equal(moved.runtime_infos[GeometryStage].gs_info.vs_copy,
                                   moved.runtime_vs_copies[GeometryStage]));
}

TEST(AsyncGraphicsPipelineJob, QueueIdentitySeparatesShaderModuleGenerations) {
    GraphicsPipelineKey key{};
    const AsyncGraphicsPipelineKey first{.pipeline = key, .module_generation = 3};
    const AsyncGraphicsPipelineKey replacement{.pipeline = key, .module_generation = 4};

    EXPECT_NE(first, replacement);
    EXPECT_NE(AsyncGraphicsPipelineKeyHash{}(first),
              AsyncGraphicsPipelineKeyHash{}(replacement));
}

TEST(AsyncGraphicsPipelineCompilation, CacheMissDoesNotBlockTheGpuOwnerThread) {
    const auto header = ReadText(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(header.empty());
    ASSERT_FALSE(source.empty());
    EXPECT_EQ(header.find("WaitForAsyncGraphicsPipeline"), std::string::npos);
    const auto get_begin = source.find("PipelineCache::GetGraphicsPipeline()");
    const auto get_end = source.find("PipelineCache::GetComputePipeline()", get_begin);
    ASSERT_NE(get_begin, std::string::npos);
    ASSERT_NE(get_end, std::string::npos);
    const auto get_body = source.substr(get_begin, get_end - get_begin);

    const auto async_begin = get_body.find("if (async_graphics_compiler)");
    ASSERT_NE(async_begin, std::string::npos);
    const auto sync_begin = get_body.find("const auto [it, is_new]", async_begin);
    ASSERT_NE(sync_begin, std::string::npos);
    const auto async_body = get_body.substr(async_begin, sync_begin - async_begin);

    EXPECT_NE(async_body.find("async_graphics_compiler->Submit("), std::string::npos);
    EXPECT_NE(async_body.find("fetch_shader.reset()"), std::string::npos);
    EXPECT_NE(async_body.find("return nullptr;"), std::string::npos);
    EXPECT_EQ(async_body.find("GraphicsPipelineLookupStatus::Pending"), std::string::npos);
    EXPECT_EQ(async_body.find("completion_epoch"), std::string::npos);
    EXPECT_EQ(async_body.find("WaitTake("), std::string::npos);
    EXPECT_EQ(async_body.find("WaitForAsyncGraphicsPipeline("), std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation, PendingDrawDoesNotDelayLaterEopPackets) {
    const auto pipeline_header = ReadText(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto pipeline_source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto rasterizer_header = ReadText(SHADPS4_RASTERIZER_HEADER_PATH);
    const auto rasterizer_source = ReadText(SHADPS4_RASTERIZER_SOURCE_PATH);
    const auto liverpool_header = ReadText(SHADPS4_LIVERPOOL_HEADER_PATH);
    const auto liverpool_source = ReadText(SHADPS4_LIVERPOOL_SOURCE_PATH);

    ASSERT_FALSE(pipeline_header.empty());
    ASSERT_FALSE(pipeline_source.empty());
    ASSERT_FALSE(rasterizer_header.empty());
    ASSERT_FALSE(rasterizer_source.empty());
    ASSERT_FALSE(liverpool_header.empty());
    ASSERT_FALSE(liverpool_source.empty());

    EXPECT_EQ(pipeline_header.find("GraphicsPipelineLookupResult"), std::string::npos);
    EXPECT_NE(pipeline_header.find("failed_graphics_pipelines"), std::string::npos);
    EXPECT_NE(pipeline_source.find("failed_graphics_pipelines.insert(completion.key.pipeline)"),
              std::string::npos);
    EXPECT_NE(pipeline_source.find("failed_graphics_pipelines.clear()"), std::string::npos);
    EXPECT_EQ(pipeline_source.find("weak_wake_state"), std::string::npos);

    const auto publish_begin =
        pipeline_source.find("PipelineCache::PublishAsyncGraphicsPipeline(");
    const auto publish_end =
        pipeline_source.find("PipelineCache::PublishAsyncGraphicsPipelines()", publish_begin);
    ASSERT_NE(publish_begin, std::string::npos);
    ASSERT_NE(publish_end, std::string::npos);
    const auto publish_body = pipeline_source.substr(publish_begin, publish_end - publish_begin);
    const auto reject_stale =
        publish_body.find("completion.key.module_generation != module_generation");
    const auto publish_failure = publish_body.find("if (!completion.result)");
    ASSERT_NE(reject_stale, std::string::npos);
    ASSERT_NE(publish_failure, std::string::npos);
    EXPECT_LT(reject_stale, publish_failure);
    EXPECT_EQ(rasterizer_header.find("struct DrawResult"), std::string::npos);
    EXPECT_NE(rasterizer_header.find("void Draw("), std::string::npos);
    EXPECT_NE(rasterizer_header.find("void DrawIndirect("), std::string::npos);
    EXPECT_NE(rasterizer_source.find(
                  "const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline()"),
              std::string::npos);
    EXPECT_EQ(rasterizer_source.find("GraphicsPipelineLookupStatus::Pending"),
              std::string::npos);
    EXPECT_EQ(liverpool_header.find("AsyncSubmissionWait async_wait"), std::string::npos);
    EXPECT_NE(liverpool_header.find("Task(const Task&) = delete"), std::string::npos);
    EXPECT_NE(liverpool_header.find("Handle Release()"), std::string::npos);
    EXPECT_EQ(liverpool_source.find("YIELD_GFX_PIPELINE"), std::string::npos);
    EXPECT_EQ(liverpool_source.find("RETRY_GFX_DRAW"), std::string::npos);
    EXPECT_EQ(liverpool_source.find("graphics_pipeline_wake_state->Detach()"),
              std::string::npos);
    EXPECT_NE(liverpool_source.find("handle.destroy()"), std::string::npos);
    EXPECT_EQ(liverpool_source.find("HasRunnableSubmissions("), std::string::npos);
    EXPECT_EQ(liverpool_header.find("HasRunnableSubmissions("), std::string::npos);
    const auto submit_gfx = liverpool_source.find("void Liverpool::SubmitGfx(");
    const auto submit_asc = liverpool_source.find("void Liverpool::SubmitAsc(", submit_gfx);
    ASSERT_NE(submit_gfx, std::string::npos);
    ASSERT_NE(submit_asc, std::string::npos);
    const auto submit_body = liverpool_source.substr(submit_gfx, submit_asc - submit_gfx);
    const auto admission_lock = submit_body.find("std::scoped_lock admission_lock{submit_mutex}");
    const auto queue_lock = submit_body.find("std::scoped_lock queue_lock{queue.m_access}");
    const auto queue_publish = submit_body.find("queue.submits.push(");
    const auto count_publish = submit_body.find("++num_submits");
    ASSERT_NE(admission_lock, std::string::npos);
    ASSERT_NE(queue_lock, std::string::npos);
    ASSERT_NE(queue_publish, std::string::npos);
    ASSERT_NE(count_publish, std::string::npos);
    EXPECT_LT(admission_lock, queue_lock);
    EXPECT_LT(queue_publish, count_publish);

    const auto draw_case = liverpool_source.find("case PM4ItOpcode::DrawIndex2:");
    const auto next_case = liverpool_source.find("case PM4ItOpcode::DrawIndexOffset2:", draw_case);
    ASSERT_NE(draw_case, std::string::npos);
    ASSERT_NE(next_case, std::string::npos);
    const auto draw_body = liverpool_source.substr(draw_case, next_case - draw_case);
    EXPECT_NE(draw_body.find("rasterizer->Draw(true)"), std::string::npos);
    EXPECT_EQ(draw_body.find("RETRY_GFX_DRAW"), std::string::npos);

    // A queued pipeline skips only this draw; the later EOP packet remains reachable.
    const auto packet_advance = liverpool_source.find("dcb = NextPacket(", next_case);
    ASSERT_NE(packet_advance, std::string::npos);
    EXPECT_LT(draw_case, packet_advance);
}

TEST(AsyncGraphicsPipelineCompilation, GraphicsOnlySubmissionProbeUsesTheAdmissionLockOrder) {
    const auto liverpool_header = ReadText(SHADPS4_LIVERPOOL_HEADER_PATH);
    const auto liverpool_source = ReadText(SHADPS4_LIVERPOOL_SOURCE_PATH);

    ASSERT_FALSE(liverpool_header.empty());
    ASSERT_FALSE(liverpool_source.empty());
    EXPECT_NE(liverpool_header.find("HasOnlyGraphicsSubmissions()"), std::string::npos);

    const auto helper_begin = liverpool_source.find("bool Liverpool::HasOnlyGraphicsSubmissions()");
    const auto process_begin = liverpool_source.find("void Liverpool::Process(", helper_begin);
    ASSERT_NE(helper_begin, std::string::npos);
    ASSERT_NE(process_begin, std::string::npos);
    const auto helper_body = liverpool_source.substr(helper_begin, process_begin - helper_begin);
    const auto admission_lock = helper_body.find("std::scoped_lock admission_lock{submit_mutex}");
    const auto queue_lock = helper_body.find("std::scoped_lock queue_lock{queue.m_access}");
    const auto queue_size = helper_body.find("queue.submits.size()");
    ASSERT_NE(admission_lock, std::string::npos);
    ASSERT_NE(queue_lock, std::string::npos);
    ASSERT_NE(queue_size, std::string::npos);
    EXPECT_LT(admission_lock, queue_lock);
    EXPECT_LT(queue_lock, queue_size);

    EXPECT_NE(liverpool_source.find("HasOnlyGraphicsSubmissions()"), std::string::npos);
    EXPECT_EQ(liverpool_source.find("mapped_queues[GfxQueueId].submits.size()"),
              std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation, CommandProcessorShutdownBreaksOutOfYieldingSubmissions) {
    const auto liverpool_source = ReadText(SHADPS4_LIVERPOOL_SOURCE_PATH);

    ASSERT_FALSE(liverpool_source.empty());
    EXPECT_NE(liverpool_source.find(
                  "while (!stoken.stop_requested() && (num_submits || num_commands))"),
              std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation,
     ShaderModuleEmissionOwnsItsInputsAndPublishesOnlyOnTheOwnerThread) {
    const auto header = ReadText(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(header.empty());
    ASSERT_FALSE(source.empty());
    EXPECT_NE(header.find("AsyncShaderModuleKey"), std::string::npos);
    EXPECT_NE(header.find("AsyncShaderModuleJob"), std::string::npos);
    EXPECT_NE(header.find("AsyncShaderModuleResult"), std::string::npos);
    EXPECT_NE(header.find("std::unique_ptr<Shader::Pools> pools"), std::string::npos);
    EXPECT_NE(header.find("std::optional<Shader::IR::Program> ir_program"), std::string::npos);
    EXPECT_NE(header.find("std::vector<u32> code"), std::string::npos);
    EXPECT_NE(header.find("user_data_copy"), std::string::npos);
    EXPECT_NE(header.find("AsyncShaderModuleQueue"), std::string::npos);
    EXPECT_NE(header.find("pending_shader_modules"), std::string::npos);

    const auto worker_begin = source.find("PipelineCache::CompileShaderModuleAsync(");
    const auto worker_end = source.find("PipelineCache::PublishAsyncShaderModule(", worker_begin);
    ASSERT_NE(worker_begin, std::string::npos);
    ASSERT_NE(worker_end, std::string::npos);
    const auto worker = source.substr(worker_begin, worker_end - worker_begin);
    EXPECT_NE(worker.find("EmitSPIRV("), std::string::npos);
    EXPECT_NE(worker.find("createShaderModuleUnique("), std::string::npos);
    const auto emit = worker.find("EmitSPIRV(");
    const auto release_ir = worker.find("job->ir_program.reset()", emit);
    const auto release_pools = worker.find("job->pools.reset()", release_ir);
    ASSERT_NE(release_ir, std::string::npos);
    ASSERT_NE(release_pools, std::string::npos);
    EXPECT_LT(emit, release_ir);
    EXPECT_LT(release_ir, release_pools);
    EXPECT_EQ(worker.find("TranslateModule("), std::string::npos);
    EXPECT_EQ(worker.find("GetShaderPatch("), std::string::npos);
    EXPECT_EQ(worker.find("DumpShader("), std::string::npos);
    EXPECT_EQ(worker.find("DebugState"), std::string::npos);
    EXPECT_EQ(worker.find("RegisterShader"), std::string::npos);

    const auto publish_begin = worker_end;
    const auto publish_end = source.find("PipelineCache::GetGraphicsPipeline()", publish_begin);
    ASSERT_NE(publish_end, std::string::npos);
    const auto publish = source.substr(publish_begin, publish_end - publish_begin);
    EXPECT_NE(publish.find("async_shader_compiler->TryTake()"), std::string::npos);
    EXPECT_NE(publish.find("RegisterShaderMeta("), std::string::npos);
    EXPECT_NE(publish.find("RegisterShaderBinary("), std::string::npos);
    EXPECT_NE(publish.find("DebugState.CollectShader("), std::string::npos);

    const auto get_program = source.find("PipelineCache::Result PipelineCache::GetProgram(");
    ASSERT_NE(get_program, std::string::npos);
    EXPECT_NE(source.find("PublishAsyncShaderModules()", get_program), std::string::npos);
    EXPECT_NE(source.find("l_stage != LogicalStage::Compute", get_program), std::string::npos);
    const auto get_attempt = source.find("PipelineCache::Result PipelineCache::GetProgramAttempt(",
                                         get_program);
    ASSERT_NE(get_attempt, std::string::npos);
    const auto attempt_body = source.substr(get_attempt);
    const auto initial_capacity = attempt_body.find("async_shader_compiler->CanAccept()");
    const auto initial_job = attempt_body.find("std::make_unique<AsyncShaderModuleJob>()");
    const auto cached_capacity = attempt_body.find("async_shader_compiler->CanAccept()",
                                                   initial_capacity + 1);
    const auto cached_job = attempt_body.find("std::make_unique<AsyncShaderModuleJob>()",
                                              initial_job + 1);
    ASSERT_NE(initial_capacity, std::string::npos);
    ASSERT_NE(initial_job, std::string::npos);
    ASSERT_NE(cached_capacity, std::string::npos);
    ASSERT_NE(cached_job, std::string::npos);
    EXPECT_LT(initial_capacity, initial_job);
    EXPECT_LT(cached_capacity, cached_job);
}

TEST(AsyncGraphicsPipelineCompilation, AllowsIndependentShaderAndPipelineHostCompilerCallbacks) {
    const auto header = ReadText(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(header.empty());
    ASSERT_FALSE(source.empty());
    EXPECT_EQ(header.find("std::mutex host_compilation_mutex"), std::string::npos);

    const auto shader_begin = source.find("PipelineCache::CompileShaderModuleAsync(");
    const auto shader_end = source.find("PipelineCache::SubmitAsyncShaderModule(", shader_begin);
    const auto pipeline_begin = source.find("PipelineCache::CompileGraphicsPipelineAsync(");
    const auto pipeline_end = source.find("PipelineCache::PublishAsyncGraphicsPipeline(",
                                          pipeline_begin);
    ASSERT_NE(shader_begin, std::string::npos);
    ASSERT_NE(shader_end, std::string::npos);
    ASSERT_NE(pipeline_begin, std::string::npos);
    ASSERT_NE(pipeline_end, std::string::npos);
    const auto lock = "std::scoped_lock host_compile_lock{host_compilation_mutex}";
    EXPECT_EQ(source.substr(shader_begin, shader_end - shader_begin).find(lock),
              std::string::npos);
    EXPECT_EQ(source.substr(pipeline_begin, pipeline_end - pipeline_begin).find(lock),
              std::string::npos);
    EXPECT_NE(source.find("createPipelineCacheUnique({})"), std::string::npos);
}

TEST(AsyncGraphicsPipelineCompilation,
     ShaderReplacementIsMarshaledToGpuOwnerAndInvalidatesPendingPipelines) {
    const auto source = ReadText(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    const auto wrapper = source.find("PipelineCache::ReplaceShader(");
    const auto owner = source.find("PipelineCache::ReplaceShaderOnGpuOwner(", wrapper);
    const auto shader_name = source.find("PipelineCache::GetShaderName(", owner);
    ASSERT_NE(wrapper, std::string::npos);
    ASSERT_NE(owner, std::string::npos);
    ASSERT_NE(shader_name, std::string::npos);
    const auto wrapper_body = source.substr(wrapper, owner - wrapper);
    const auto owner_body = source.substr(owner, shader_name - owner);

    EXPECT_NE(wrapper_body.find("liverpool->SendCommand<true>"), std::string::npos);
    EXPECT_EQ(wrapper_body.find("program_cache"), std::string::npos);
    EXPECT_EQ(wrapper_body.find("graphics_pipelines"), std::string::npos);
    EXPECT_EQ(wrapper_body.find("module_generation"), std::string::npos);
    EXPECT_NE(owner_body.find("program_cache"), std::string::npos);
    EXPECT_NE(owner_body.find("graphics_pipelines.erase"), std::string::npos);
    EXPECT_NE(owner_body.find("++module_generation"), std::string::npos);
    const auto create_replacement = owner_body.find("createShaderModuleUnique");
    const auto publish_replacement = owner_body.find("std::exchange(m.module");
    const auto retire_old_module = owner_body.find("retired_shader_modules.push_back");
    ASSERT_NE(create_replacement, std::string::npos);
    ASSERT_NE(publish_replacement, std::string::npos);
    ASSERT_NE(retire_old_module, std::string::npos);
    EXPECT_LT(create_replacement, publish_replacement);
    EXPECT_LT(publish_replacement, retire_old_module);
    EXPECT_NE(owner_body.find("scheduler.DeferOperation"), std::string::npos);
    EXPECT_NE(owner_body.find("module_related_pipelines.erase"), std::string::npos);
    EXPECT_NE(owner_body.find("module_related_pipelines[new_module]"), std::string::npos);

    const auto publish = source.find("PipelineCache::PublishAsyncGraphicsPipeline(");
    const auto publish_end = source.find("PipelineCache::PublishAsyncGraphicsPipelines(", publish);
    ASSERT_NE(publish, std::string::npos);
    ASSERT_NE(publish_end, std::string::npos);
    const auto publish_body = source.substr(publish, publish_end - publish);
    EXPECT_NE(publish_body.find("result.module_generation != module_generation"),
              std::string::npos);
    EXPECT_NE(publish_body.find("return;"), std::string::npos);
}

TEST(PersistentShaderCache, ShaderPatchesDisableIncompatiblePipelineCachePersistence) {
    const auto source = ReadText(SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH);

    ASSERT_FALSE(source.empty());
    const auto warmup = source.find("void PipelineCache::WarmUp()");
    const auto sync = source.find("void PipelineCache::Sync()", warmup);
    ASSERT_NE(warmup, std::string::npos);
    ASSERT_NE(sync, std::string::npos);
    const auto warmup_body = source.substr(warmup, sync - warmup);
    const auto patch_gate = warmup_body.find("EmulatorSettings.IsPatchShaders()");
    const auto open = warmup_body.find("Storage::DataBase::Instance().Open()");
    ASSERT_NE(patch_gate, std::string::npos);
    ASSERT_NE(open, std::string::npos);
    EXPECT_LT(patch_gate, open);
}

TEST(AsyncGraphicsPipelineCompilation, HasAStableExperimentalConfigurationKey) {
    const auto settings_header = ReadText(SHADPS4_EMULATOR_SETTINGS_HEADER_PATH);
    const auto settings_source = ReadText(SHADPS4_EMULATOR_SETTINGS_SOURCE_PATH);

    ASSERT_FALSE(settings_header.empty());
    ASSERT_FALSE(settings_source.empty());
    EXPECT_NE(settings_header.find("async_graphics_pipeline_compilation"), std::string::npos);
    EXPECT_NE(settings_source.find("asyncGraphicsPipelineCompilation"), std::string::npos);
}

} // namespace
} // namespace Vulkan
