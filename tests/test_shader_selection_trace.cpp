// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "shader_recompiler/specialization_identity.h"
#include "video_core/shader_selection_trace.h"

namespace {

constexpr u64 ProgramHash = 0x73b8c213ULL;

class TempDirectory {
public:
    explicit TempDirectory(std::string_view name)
        : path{std::filesystem::temp_directory_path() /
               (std::string{name} + "-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))} {
        std::filesystem::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

std::vector<nlohmann::json> ReadJsonLines(const std::filesystem::path& path) {
    std::ifstream input{path};
    std::vector<nlohmann::json> records;
    for (std::string line; std::getline(input, line);) {
        records.push_back(nlohmann::json::parse(line));
    }
    return records;
}

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

std::string ReadSource(const char* path) {
    return ReadText(path);
}

void SetTraceEnvironment(const std::string& value) {
#ifdef _WIN32
    ASSERT_EQ(_putenv_s(VideoCore::ShaderSelectionTraceEnvironment, value.c_str()), 0);
#else
    ASSERT_EQ(setenv(VideoCore::ShaderSelectionTraceEnvironment, value.c_str(), 1), 0);
#endif
}

void ClearTraceEnvironment() {
#ifdef _WIN32
    ASSERT_EQ(_putenv_s(VideoCore::ShaderSelectionTraceEnvironment, ""), 0);
#else
    ASSERT_EQ(unsetenv(VideoCore::ShaderSelectionTraceEnvironment), 0);
#endif
}

class ScopedTraceEnvironment {
public:
    explicit ScopedTraceEnvironment(const std::string& value) {
        const char* existing = std::getenv(VideoCore::ShaderSelectionTraceEnvironment);
        if (existing) {
            previous = existing;
        }
        SetTraceEnvironment(value);
    }

    ~ScopedTraceEnvironment() {
        if (previous) {
            SetTraceEnvironment(*previous);
        } else {
            ClearTraceEnvironment();
        }
    }

private:
    std::optional<std::string> previous;
};

AmdGpu::CompMapping RgbaMapping() {
    return {
        .r = AmdGpu::CompSwizzle::Red,
        .g = AmdGpu::CompSwizzle::Green,
        .b = AmdGpu::CompSwizzle::Blue,
        .a = AmdGpu::CompSwizzle::Alpha,
    };
}

AmdGpu::CompMapping RgbZeroMapping() {
    return {
        .r = AmdGpu::CompSwizzle::Red,
        .g = AmdGpu::CompSwizzle::Green,
        .b = AmdGpu::CompSwizzle::Blue,
        .a = AmdGpu::CompSwizzle::Zero,
    };
}

Shader::StageSpecialization MakeImageSpecialization(AmdGpu::CompMapping mapping) {
    Shader::StageSpecialization spec{};
    spec.runtime_info.Initialize(Shader::Stage::Compute);
    spec.runtime_info.num_user_data = 4;
    spec.runtime_info.cs_info.shared_memory_size = 1024;
    spec.runtime_info.cs_info.workgroup_size = {8, 8, 1};
    spec.runtime_info.cs_info.tgid_enable = {true, true, false};
    spec.start = {.unified = 3, .buffer = 1, .user_data = 2};
    spec.images.push_back({
        .type = AmdGpu::ImageType::Color2D,
        .number_class = AmdGpu::NumberClass::Float,
        .is_storage = true,
        .has_supported_dynamic_model = true,
        .dst_select = mapping,
        .num_conversion = AmdGpu::NumberConversion::None,
        .num_bindings = 1,
    });
    return spec;
}

VideoCore::ShaderSelectionTraceRecord MakeRecord(
    u64 semantic_identity, VideoCore::ShaderSelectionCacheAction action,
    u32 module_count, u64 dispatch_sequence = 7) {
    return {
        .dispatch_sequence = dispatch_sequence,
        .physical_stage = Shader::Stage::Compute,
        .logical_stage = Shader::LogicalStage::Compute,
        .program_hash = ProgramHash,
        .srt_source_digest = 0x1111222233334444ULL,
        .srt_content_digest = 0x5555666677778888ULL,
        .semantic_identity = semantic_identity,
        .cache_action = action,
        .module_count = module_count,
        .selected_permutation_id =
            action == VideoCore::ShaderSelectionCacheAction::Reject ? 0 : semantic_identity,
        .compute_pipeline_id =
            action == VideoCore::ShaderSelectionCacheAction::Reject ? 0 : semantic_identity,
    };
}

} // namespace

TEST(ShaderSelectionTrace, EnvironmentGateIsDisabledWhenUnsetAndCreatesExclusivelyWhenSet) {
    TempDirectory root{"shadps4-shader-selection-env"};
    ClearTraceEnvironment();
    VideoCore::ShaderSelectionTrace disabled;
    EXPECT_FALSE(disabled.Enabled());

    const auto path = root.path / "selection.jsonl";
    {
        ScopedTraceEnvironment environment{path.string()};
        VideoCore::ShaderSelectionTrace enabled;
        ASSERT_TRUE(enabled.Enabled());
        enabled.Record(MakeRecord(0x1234, VideoCore::ShaderSelectionCacheAction::New, 1));
    }
    ASSERT_TRUE(std::filesystem::exists(path));

    const auto original = ReadText(path);
    VideoCore::ShaderSelectionTrace refused{path};
    EXPECT_FALSE(refused.Enabled());
    refused.Record(MakeRecord(0x5678, VideoCore::ShaderSelectionCacheAction::New, 2));
    EXPECT_EQ(ReadText(path), original);
}

TEST(ShaderSelectionTrace, RealSemanticImageSelectionsAreDistinctAndBoundnessReusesInOrder) {
    TempDirectory root{"shadps4-shader-selection-semantics"};
    const auto path = root.path / "selection.jsonl";
    VideoCore::ShaderSelectionTrace trace{path};
    ASSERT_TRUE(trace.Enabled());

    const auto rgba = MakeImageSpecialization(RgbaMapping());
    const auto rgb_zero = MakeImageSpecialization(RgbZeroMapping());
    auto rgba_bound = rgba;
    rgba_bound.bitset.set(0);

    const u64 rgba_id = Shader::ComputeShaderPermutationIdentity(
        ProgramHash, Shader::LogicalStage::Compute, rgba);
    const u64 rgb_zero_id = Shader::ComputeShaderPermutationIdentity(
        ProgramHash, Shader::LogicalStage::Compute, rgb_zero);
    const u64 rgba_bound_id = Shader::ComputeShaderPermutationIdentity(
        ProgramHash, Shader::LogicalStage::Compute, rgba_bound);
    ASSERT_NE(rgba_id, rgb_zero_id);
    ASSERT_EQ(rgba_id, rgba_bound_id);

    trace.Record(MakeRecord(rgba_id, VideoCore::ShaderSelectionCacheAction::New, 1, 41));
    trace.Record(MakeRecord(rgb_zero_id, VideoCore::ShaderSelectionCacheAction::New, 2, 42));
    trace.Record(MakeRecord(rgba_bound_id, VideoCore::ShaderSelectionCacheAction::Reuse, 2, 43));

    const auto records = ReadJsonLines(path);
    ASSERT_EQ(records.size(), 3);
    EXPECT_EQ(records[0]["selection_sequence"], 1);
    EXPECT_EQ(records[1]["selection_sequence"], 2);
    EXPECT_EQ(records[2]["selection_sequence"], 3);
    EXPECT_EQ(records[0]["dispatch_sequence"], 41);
    EXPECT_EQ(records[1]["dispatch_sequence"], 42);
    EXPECT_EQ(records[2]["dispatch_sequence"], 43);
    EXPECT_EQ(records[0]["cache_action"], "new");
    EXPECT_EQ(records[1]["cache_action"], "new");
    EXPECT_EQ(records[2]["cache_action"], "reuse");
    EXPECT_NE(records[0]["semantic_identity"], records[1]["semantic_identity"]);
    EXPECT_NE(records[0]["compute_pipeline_id"], records[1]["compute_pipeline_id"]);
    EXPECT_EQ(records[0]["semantic_identity"], records[2]["semantic_identity"]);
    EXPECT_EQ(records[0]["compute_pipeline_id"], records[2]["compute_pipeline_id"]);
}

TEST(ShaderSelectionTrace, SrtDigestsExcludeAddressesMappingsAndWriterHandles) {
    VideoCore::SrtCaptureProvenance first;
    first.Record({
        .address = 0x111122223333ULL,
        .size = 0x4000,
        .exact_gpu_modified = true,
        .coarse_gpu_owned = false,
        .source = VideoCore::ShaderCaptureSource::GpuBuffer,
        .mapping_token = 0xabcdef,
        .content_digest = 0x1234567890abcdefULL,
        .potential_writer = VideoCore::PotentialShaderWriteProvenance{
            .program_hash = 0xdead,
            .stage = 7,
            .descriptor_index = 4,
            .mapping_token = 0xfedcba,
            .address = 0x777788889999ULL,
            .size = 64,
            .sequence = 88,
        },
    });
    auto second = VideoCore::SrtCaptureProvenance{};
    second.Record({
        .address = 0xaaaa0000bbbbULL,
        .size = 0x4000,
        .exact_gpu_modified = true,
        .coarse_gpu_owned = false,
        .source = VideoCore::ShaderCaptureSource::GpuBuffer,
        .mapping_token = 0x123456,
        .content_digest = 0x1234567890abcdefULL,
        .potential_writer = std::nullopt,
    });

    EXPECT_EQ(VideoCore::DigestShaderSelectionSrtSources(first),
              VideoCore::DigestShaderSelectionSrtSources(second));
    EXPECT_EQ(VideoCore::DigestShaderSelectionSrtContents(first),
              VideoCore::DigestShaderSelectionSrtContents(second));
}

TEST(ShaderSelectionTrace, SchemaIsFixedAndContainsNoSensitiveOrArbitraryFields) {
    TempDirectory root{"shadps4-shader-selection-schema"};
    const auto path = root.path / "selection.jsonl";
    VideoCore::ShaderSelectionTrace trace{path};
    trace.Record(MakeRecord(0x1234, VideoCore::ShaderSelectionCacheAction::Reject, 3));

    const auto records = ReadJsonLines(path);
    ASSERT_EQ(records.size(), 1);
    const auto& record = records.front();
    const std::set<std::string> expected{
        "schema", "kind", "selection_sequence", "dispatch_sequence", "physical_stage",
        "logical_stage", "program_hash", "srt_source_digest", "srt_content_digest",
        "semantic_identity", "cache_action", "module_count", "selected_permutation_id",
        "compute_pipeline_id",
    };
    std::set<std::string> actual;
    for (const auto& [key, _] : record.items()) {
        actual.insert(key);
    }
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(record["schema"], 1);
    EXPECT_EQ(record["kind"], "shader_selection");

    std::string lower = ReadText(path);
    std::ranges::transform(lower, lower.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    for (const std::string_view forbidden : {"address", "title", "handle", "guest", "payload",
                                             "string", "bytes"}) {
        EXPECT_EQ(lower.find(forbidden), std::string::npos) << forbidden;
    }
}

TEST(ShaderSelectionTrace, RecordAndByteCapsAreWholeLineFailClosedBounds) {
    TempDirectory root{"shadps4-shader-selection-bounds"};
    const auto record_path = root.path / "records.jsonl";
    {
        VideoCore::ShaderSelectionTrace trace{
            record_path, {.max_records = 3, .max_bytes = 1024 * 1024}};
        for (u64 index = 0; index < 10; ++index) {
            trace.Record(MakeRecord(index + 1, VideoCore::ShaderSelectionCacheAction::New, 1));
        }
        EXPECT_FALSE(trace.Enabled());
    }
    EXPECT_EQ(ReadJsonLines(record_path).size(), 3);

    const auto byte_path = root.path / "bytes.jsonl";
    {
        VideoCore::ShaderSelectionTrace trace{
            byte_path, {.max_records = 4096, .max_bytes = 600}};
        for (u64 index = 0; index < 10; ++index) {
            trace.Record(MakeRecord(index + 1, VideoCore::ShaderSelectionCacheAction::New, 1));
        }
        EXPECT_FALSE(trace.Enabled());
    }
    const auto bytes = std::filesystem::file_size(byte_path);
    EXPECT_LE(bytes, 600);
    EXPECT_FALSE(ReadJsonLines(byte_path).empty());
}

TEST(ShaderSelectionTrace, ConcurrentWritersHaveOneMonotonicSequence) {
    TempDirectory root{"shadps4-shader-selection-concurrency"};
    const auto path = root.path / "selection.jsonl";
    VideoCore::ShaderSelectionTrace trace{path};
    ASSERT_TRUE(trace.Enabled());

    constexpr u64 ThreadCount = 8;
    constexpr u64 RecordsPerThread = 32;
    std::vector<std::thread> threads;
    for (u64 thread = 0; thread < ThreadCount; ++thread) {
        threads.emplace_back([&, thread] {
            for (u64 index = 0; index < RecordsPerThread; ++index) {
                trace.Record(MakeRecord(thread * RecordsPerThread + index + 1,
                                        VideoCore::ShaderSelectionCacheAction::Reuse, 2));
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    const auto records = ReadJsonLines(path);
    ASSERT_EQ(records.size(), ThreadCount * RecordsPerThread);
    for (u64 index = 0; index < records.size(); ++index) {
        EXPECT_EQ(records[index]["selection_sequence"], index + 1);
    }
}

TEST(ShaderSelectionTrace, RuntimeInstrumentationUsesGenericSemanticAndDispatchSeams) {
    const auto pipeline = ReadSource(SHADPS4_PIPELINE_CACHE_SOURCE_PATH);
    const auto header = ReadSource(SHADPS4_PIPELINE_CACHE_HEADER_PATH);
    const auto rasterizer = ReadSource(SHADPS4_RASTERIZER_SOURCE_PATH);

    EXPECT_NE(header.find("VideoCore::ShaderSelectionTrace shader_selection_trace"),
              std::string::npos);
    EXPECT_NE(pipeline.find("shader_selection_trace.Enabled()"), std::string::npos);
    EXPECT_NE(pipeline.find("DigestShaderSelectionSrtSources"), std::string::npos);
    EXPECT_NE(pipeline.find("DigestShaderSelectionSrtContents"), std::string::npos);

    const auto identity = pipeline.find("ComputeShaderPermutationIdentity");
    const auto lookup = pipeline.find("program->FindPermut(spec)", identity);
    ASSERT_NE(identity, std::string::npos);
    ASSERT_NE(lookup, std::string::npos);
    EXPECT_LT(identity, lookup);

    for (const std::string_view action : {
             "ShaderSelectionCacheAction::New", "ShaderSelectionCacheAction::Reuse",
             "ShaderSelectionCacheAction::Reject"}) {
        EXPECT_NE(pipeline.find(action), std::string::npos) << action;
    }
    for (const auto dispatch : {rasterizer.find("void Rasterizer::DispatchDirect()"),
                                rasterizer.find("void Rasterizer::DispatchIndirect(")}) {
        ASSERT_NE(dispatch, std::string::npos);
        const auto begin = rasterizer.find("BeginComputeDispatchTrace()", dispatch);
        const auto select = rasterizer.find("GetComputePipeline()", dispatch);
        ASSERT_NE(begin, std::string::npos);
        ASSERT_NE(select, std::string::npos);
        EXPECT_LT(begin, select);
    }
    EXPECT_EQ(pipeline.find("title_id"), std::string::npos);
    EXPECT_EQ(pipeline.find("CUSA"), std::string::npos);
}
