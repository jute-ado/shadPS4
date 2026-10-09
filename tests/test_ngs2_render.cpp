// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "core/libraries/audio/audio_pipeline_metadata_trace.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/ngs2/ngs2.h"
#include "core/libraries/ngs2/ngs2_atrac9.h"
#include "core/libraries/ngs2/ngs2_error.h"
#include "core/libraries/ngs2/ngs2_guest_io.h"
#include "core/libraries/ngs2/ngs2_mastering.h"
#include "core/libraries/ngs2/ngs2_render.h"
#include "core/libraries/ngs2/ngs2_sampler.h"
#include "core/libraries/ngs2/ngs2_submixer.h"

namespace Libraries::Ngs2 {
namespace {

constexpr u32 VoiceMatrixLevelsControlId = 0x00000001;
constexpr u32 VoicePortVolumeControlId = 0x00000002;
constexpr u32 VoicePortMatrixControlId = 0x00000003;
constexpr u32 VoicePatchControlId = 0x00000005;
constexpr u32 VoiceEventControlId = 0x00000006;
constexpr u32 SamplerSetupControlId = 0x10000000;
constexpr u32 SamplerWaveformBlocksControlId = 0x10000001;
constexpr u32 SubmixerSetupControlId = 0x20000000;
constexpr u32 SubmixerUserFxControlId = 0x20000004;
constexpr u32 MasteringSetupControlId = 0x30000000;
constexpr u32 MasteringMatrixControlId = 0x30000001;
constexpr u32 MasteringGainControlId = 0x30000004;
constexpr u32 MasteringOutputControlId = 0x30000005;
constexpr u32 Float32WaveformType = 24;

s32 PS4_SYSV_ABI DummyUserFxProcess(OrbisNgs2UserFxProcessContext*) {
    return ORBIS_OK;
}

class RenderDecisionTrace {
public:
    RenderDecisionTrace() {
        path = std::filesystem::temp_directory_path() /
               ("ngs2-render-decision-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                ".jsonl");
        if (const char* previous = std::getenv("SHADPS4_AUDIO_PIPELINE_TRACE_PATH")) {
            old_path = previous;
        }
        SetPath(path.string());
        AudioPipelineTrace::Testing::Reset();
    }
    ~RenderDecisionTrace() {
        AudioPipelineTrace::Testing::Reset();
        SetPath(old_path);
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    std::vector<std::string> Reasons() const {
        AudioPipelineTrace::Testing::Finalize();
        std::ifstream input(path, std::ios::binary);
        std::vector<std::string> reasons;
        std::string line;
        while (std::getline(input, line)) {
            const auto value = nlohmann::json::parse(line);
            if (value.at("event") == "render_decision") {
                reasons.push_back(value.at("reason").get<std::string>());
            }
        }
        return reasons;
    }
private:
    static void SetPath(const std::string& value) {
#ifdef _WIN32
        EXPECT_EQ(_putenv_s("SHADPS4_AUDIO_PIPELINE_TRACE_PATH", value.c_str()), 0);
#else
        EXPECT_EQ(value.empty() ? unsetenv("SHADPS4_AUDIO_PIPELINE_TRACE_PATH")
                               : setenv("SHADPS4_AUDIO_PIPELINE_TRACE_PATH", value.c_str(), 1), 0);
#endif
    }
    std::filesystem::path path;
    std::string old_path;
};

struct ReadRegion {
    uintptr_t base{};
    size_t size{};
};

struct ReadMap {
    std::vector<ReadRegion> regions;
    u32 reads{};
    OrbisNgs2Handle inspect_voice{};
    bool observed_registry_unlocked{};
};

bool ReadMappedBytes(uintptr_t address, void* destination, size_t size, void* context) {
    auto& map = *static_cast<ReadMap*>(context);
    ++map.reads;
    if (map.inspect_voice != 0) {
        map.observed_registry_unlocked |= SnapshotVoiceControl(map.inspect_voice).has_value();
    }
    for (const ReadRegion region : map.regions) {
        if (address < region.base) {
            continue;
        }
        const uintptr_t offset = address - region.base;
        if (offset <= region.size && size <= region.size - offset) {
            std::memcpy(destination, reinterpret_cast<const void*>(address), size);
            return true;
        }
    }
    return false;
}

struct FakeDecoderStats {
    u32 factories{};
    u32 clones{};
    u32 decodes{};
};

class PatternDecoder final : public Atrac9FrameDecoder {
public:
    PatternDecoder(std::shared_ptr<FakeDecoderStats> stats, Atrac9FormatInfo format)
        : stats{std::move(stats)}, format{format} {}

    std::unique_ptr<Atrac9FrameDecoder> Clone() const override {
        ++stats->clones;
        return std::make_unique<PatternDecoder>(stats, format);
    }

    Atrac9FrameDecodeResult Decode(std::span<const u8> encoded, std::span<float> output) override {
        ++stats->decodes;
        if (output.size() != static_cast<size_t>(format.channels) * format.frame_samples) {
            return {};
        }
        for (u32 sample = 0; sample < format.frame_samples; ++sample) {
            for (u32 channel = 0; channel < format.channels; ++channel) {
                output[static_cast<size_t>(sample) * format.channels + channel] =
                    static_cast<float>(sample * 10 + channel);
            }
        }
        return {.succeeded = encoded.size() >= format.frame_bytes,
                .bytes_used = format.frame_bytes};
    }

private:
    std::shared_ptr<FakeDecoderStats> stats;
    Atrac9FormatInfo format;
};

Atrac9FrameDecoderFactory PatternFactory(const std::shared_ptr<FakeDecoderStats>& stats) {
    return [stats](u32, const Atrac9FormatInfo& format) {
        ++stats->factories;
        return std::make_unique<PatternDecoder>(stats, format);
    };
}

struct TestSystem {
    std::vector<u8> storage;
    OrbisNgs2ContextBufferInfo info{};
    OrbisNgs2Handle handle{};
};

struct TestRack {
    std::vector<u8> storage;
    OrbisNgs2RackOption option{};
    OrbisNgs2SubmixerRackOption submixer{};
    OrbisNgs2MasteringRackOption mastering{};
    OrbisNgs2ContextBufferInfo info{};
    OrbisNgs2Handle handle{};
};

class Ngs2SystemRenderTest : public testing::Test {
protected:
    void SetUp() override {
        ResetRuntimeForTesting();
    }

    void TearDown() override {
        ResetRuntimeForTesting();
    }

    TestSystem CreateSystem() {
        TestSystem system;
        EXPECT_EQ(SystemSetup(nullptr, &system.info, nullptr, nullptr), ORBIS_OK);
        system.storage.resize(system.info.hostBufferSize);
        system.info.hostBuffer = system.storage.data();
        EXPECT_EQ(SystemSetup(nullptr, &system.info, nullptr, &system.handle), ORBIS_OK);
        return system;
    }

    TestRack CreateRack(OrbisNgs2Handle system, u32 rack_id, u32 max_voices, u32 max_ports,
                        u32 max_matrices) {
        TestRack rack;
        OrbisNgs2RackOption* option = &rack.option;
        if (rack_id == 0x2000) {
            option = &rack.submixer.rackOption;
            option->size = sizeof(rack.submixer);
            rack.submixer.maxChannels = 8;
            rack.submixer.maxInputs = 1;
        } else if (rack_id == 0x3000) {
            option = &rack.mastering.rackOption;
            option->size = sizeof(rack.mastering);
            rack.mastering.maxChannels = 8;
        } else {
            option->size = sizeof(rack.option);
        }
        option->maxGrainSamples = 512;
        option->maxVoices = max_voices;
        option->maxPorts = max_ports;
        option->maxMatrices = max_matrices;
        EXPECT_EQ(RackQueryBufferSize(rack_id, option, &rack.info), ORBIS_OK);
        rack.storage.resize(rack.info.hostBufferSize);
        rack.info.hostBuffer = rack.storage.data();
        EXPECT_EQ(RackCreate(system, rack_id, option, &rack.info, nullptr, &rack.handle), ORBIS_OK);
        return rack;
    }

    OrbisNgs2Handle Voice(OrbisNgs2Handle rack, u32 index = 0) {
        OrbisNgs2Handle voice{};
        EXPECT_EQ(RackGetVoiceHandle(rack, index, &voice), ORBIS_OK);
        return voice;
    }

    template <typename Commands>
    void Apply(OrbisNgs2Handle voice, Commands& commands) {
        ReadMap reads{{{reinterpret_cast<uintptr_t>(&commands), sizeof(commands)}}};
        ASSERT_EQ(VoiceControl(voice, &commands.FirstHeader(), ReadMappedBytes, &reads), ORBIS_OK);
    }
};

TEST_F(Ngs2SystemRenderTest, PreflightsFloat32OutputShapeBeforeDecodeAndPreservesDestination) {
    auto system = CreateSystem();
    auto stats = std::make_shared<FakeDecoderStats>();
    ReadMap reads;
    std::array<float, 256> output;
    output.fill(123.0F);
    OrbisNgs2RenderBufferInfo descriptor{output.data(), output.size() * sizeof(float),
                                         Float32WaveformType, 1};

    descriptor.waveformType = 23;
    EXPECT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &reads, PatternFactory(stats)),
              ORBIS_NGS2_ERROR_INVALID_WAVEFORM_TYPE);
    descriptor.waveformType = Float32WaveformType;
    descriptor.numChannels = 3;
    EXPECT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &reads, PatternFactory(stats)),
              ORBIS_NGS2_ERROR_INVALID_NUM_CHANNELS);
    descriptor.numChannels = 1;
    descriptor.bufferSize -= sizeof(float);
    EXPECT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &reads, PatternFactory(stats)),
              ORBIS_NGS2_ERROR_INVALID_BUFFER_SIZE);
    descriptor.bufferSize += sizeof(float);
    descriptor.buffer = nullptr;
    EXPECT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &reads, PatternFactory(stats)),
              ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS);
    EXPECT_EQ(stats->factories, 0u);
    EXPECT_EQ(reads.reads, 0u);
    EXPECT_TRUE(std::ranges::all_of(output, [](float value) { return value == 123.0F; }));
}

struct RenderGraph {
    std::array<u8, 240> encoded{};

    struct SamplerCommands {
        OrbisNgs2SamplerVoiceSetupParam setup{};
        OrbisNgs2SamplerVoiceWaveformBlocksParam waveform{};
        OrbisNgs2VoiceMatrixLevelsParam matrix{};
        OrbisNgs2VoicePortMatrixParam port_matrix{};
        OrbisNgs2VoicePortVolumeParam volume{};
        OrbisNgs2VoicePatchParam patch{};
        OrbisNgs2VoiceEventParam play{};
        OrbisNgs2WaveformBlock block{};
        std::array<float, 12> levels{};

        OrbisNgs2VoiceParamHeader& FirstHeader() {
            return setup.header;
        }
    } sampler;

    struct SubmixerCommands {
        OrbisNgs2SubmixerVoiceSetupParam setup{};
        OrbisNgs2VoicePatchParam patch{};
        OrbisNgs2VoiceEventParam play{};

        OrbisNgs2VoiceParamHeader& FirstHeader() {
            return setup.header;
        }
    } submixer;

    struct MasteringCommands {
        OrbisNgs2MasteringVoiceSetupParam setup{};
        OrbisNgs2MasteringVoiceMatrixParam matrix{};
        OrbisNgs2MasteringVoiceGainParam gain{};
        OrbisNgs2MasteringVoiceOutputParam output{};
        OrbisNgs2VoiceEventParam play{};
        std::array<float, 16> levels{1.0F, 0.25F};

        OrbisNgs2VoiceParamHeader& FirstHeader() {
            return setup.header;
        }
    } mastering;
};

void ConfigureGraphCommands(RenderGraph& graph, OrbisNgs2Handle submixer_rack,
                            OrbisNgs2Handle mastering_voice) {
    auto& sampler = graph.sampler;
    sampler.setup.header = {sizeof(sampler.setup),
                            static_cast<s16>(offsetof(RenderGraph::SamplerCommands, waveform) -
                                             offsetof(RenderGraph::SamplerCommands, setup)),
                            SamplerSetupControlId};
    sampler.setup.format = {.waveformType = Atrac9WaveformType,
                            .numChannels = 6,
                            .sampleRate = 48000,
                            .configData = 0xFE761DE0};
    sampler.waveform.header = {sizeof(sampler.waveform),
                               static_cast<s16>(offsetof(RenderGraph::SamplerCommands, matrix) -
                                                offsetof(RenderGraph::SamplerCommands, waveform)),
                               SamplerWaveformBlocksControlId};
    sampler.waveform.data = graph.encoded.data();
    sampler.waveform.numBlocks = 1;
    sampler.waveform.aBlock = &sampler.block;
    sampler.matrix.header = {sizeof(sampler.matrix),
                             static_cast<s16>(offsetof(RenderGraph::SamplerCommands, port_matrix) -
                                              offsetof(RenderGraph::SamplerCommands, matrix)),
                             VoiceMatrixLevelsControlId};
    sampler.matrix.matrixId = 0;
    sampler.matrix.numLevels = static_cast<u32>(sampler.levels.size());
    sampler.matrix.aLevel = sampler.levels.data();
    sampler.levels[0] = 1.0F;
    sampler.levels[7] = 1.0F;
    sampler.port_matrix.header = {
        sizeof(sampler.port_matrix),
        static_cast<s16>(offsetof(RenderGraph::SamplerCommands, volume) -
                         offsetof(RenderGraph::SamplerCommands, port_matrix)),
        VoicePortMatrixControlId};
    sampler.port_matrix.port = 0;
    sampler.port_matrix.matrixId = 0;
    sampler.volume.header = {sizeof(sampler.volume),
                             static_cast<s16>(offsetof(RenderGraph::SamplerCommands, patch) -
                                              offsetof(RenderGraph::SamplerCommands, volume)),
                             VoicePortVolumeControlId};
    sampler.volume.port = 0;
    sampler.volume.level = 0.5F;
    sampler.patch.header = {sizeof(sampler.patch),
                            static_cast<s16>(offsetof(RenderGraph::SamplerCommands, play) -
                                             offsetof(RenderGraph::SamplerCommands, patch)),
                            VoicePatchControlId};
    sampler.patch.destHandle = submixer_rack;
    sampler.play.header = {sizeof(sampler.play), 0, VoiceEventControlId};
    sampler.play.eventId = 0;
    sampler.block = {.dataSize = static_cast<u32>(graph.encoded.size()), .numSamples = 256};

    auto& submixer = graph.submixer;
    submixer.setup.header = {sizeof(submixer.setup),
                             static_cast<s16>(offsetof(RenderGraph::SubmixerCommands, patch) -
                                              offsetof(RenderGraph::SubmixerCommands, setup)),
                             SubmixerSetupControlId};
    submixer.setup.numIoChannels = 2;
    submixer.patch.header = {sizeof(submixer.patch),
                             static_cast<s16>(offsetof(RenderGraph::SubmixerCommands, play) -
                                              offsetof(RenderGraph::SubmixerCommands, patch)),
                             VoicePatchControlId};
    submixer.patch.destHandle = mastering_voice;
    submixer.play.header = {sizeof(submixer.play), 0, VoiceEventControlId};
    submixer.play.eventId = 0;

    auto& mastering = graph.mastering;
    mastering.setup.header = {sizeof(mastering.setup),
                              static_cast<s16>(offsetof(RenderGraph::MasteringCommands, matrix) -
                                               offsetof(RenderGraph::MasteringCommands, setup)),
                              MasteringSetupControlId};
    mastering.setup.numInputChannels = 2;
    mastering.matrix.header = {sizeof(mastering.matrix),
                               static_cast<s16>(offsetof(RenderGraph::MasteringCommands, gain) -
                                                offsetof(RenderGraph::MasteringCommands, matrix)),
                               MasteringMatrixControlId};
    mastering.matrix.type = 4;
    mastering.matrix.numLevels = 2;
    mastering.matrix.aLevel = mastering.levels.data();
    mastering.gain.header = {sizeof(mastering.gain),
                             static_cast<s16>(offsetof(RenderGraph::MasteringCommands, output) -
                                              offsetof(RenderGraph::MasteringCommands, gain)),
                             MasteringGainControlId};
    mastering.gain.fbwLevel = 0.8F;
    mastering.gain.lfeLevel = 0.8F;
    mastering.output.header = {sizeof(mastering.output),
                               static_cast<s16>(offsetof(RenderGraph::MasteringCommands, play) -
                                                offsetof(RenderGraph::MasteringCommands, output)),
                               MasteringOutputControlId};
    mastering.output.outputId = 1;
    mastering.play.header = {sizeof(mastering.play), 0, VoiceEventControlId};
    mastering.play.eventId = 0;
}

TEST_F(Ngs2SystemRenderTest,
       ProducesDeterministicFloatBlockThroughTypedRackSubmixerAndMasteringMatrix) {
    auto system = CreateSystem();
    auto sampler_rack = CreateRack(system.handle, 0x1000, 1, 1, 1);
    auto submixer_rack = CreateRack(system.handle, 0x2000, 1, 1, 0);
    auto mastering_rack = CreateRack(system.handle, 0x3000, 1, 0, 0);
    const auto sampler_voice = Voice(sampler_rack.handle);
    const auto submixer_voice = Voice(submixer_rack.handle);
    const auto mastering_voice = Voice(mastering_rack.handle);

    RenderGraph graph;
    ConfigureGraphCommands(graph, submixer_rack.handle, mastering_voice);
    Apply(sampler_voice, graph.sampler);
    Apply(submixer_voice, graph.submixer);
    Apply(mastering_voice, graph.mastering);

    std::array<float, 256 * 8> output0;
    std::array<float, 256> output1;
    std::array<float, 256 * 8> output2;
    output0.fill(91.0F);
    output1.fill(91.0F);
    output2.fill(91.0F);
    std::array<OrbisNgs2RenderBufferInfo, 3> descriptors{{
        {output0.data(), sizeof(output0), Float32WaveformType, 8},
        {output1.data(), sizeof(output1), Float32WaveformType, 1},
        {output2.data(), sizeof(output2), Float32WaveformType, 8},
    }};
    auto stats = std::make_shared<FakeDecoderStats>();
    ReadMap payload{{{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()}},
                    0,
                    sampler_voice};

    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, descriptors.data(),
                                                static_cast<u32>(descriptors.size()),
                                                ReadMappedBytes, &payload, PatternFactory(stats)),
              ORBIS_OK);
    EXPECT_TRUE(payload.observed_registry_unlocked);
    EXPECT_EQ(stats->factories, 1u);
    EXPECT_EQ(stats->decodes, 1u);
    EXPECT_TRUE(std::ranges::all_of(output0, [](float value) { return value == 0.0F; }));
    EXPECT_TRUE(std::ranges::all_of(output2, [](float value) { return value == 0.0F; }));
    EXPECT_FLOAT_EQ(output1[0], 0.1F);
    EXPECT_FLOAT_EQ(output1[1], 5.1F);
    EXPECT_FLOAT_EQ(output1[255], 1275.1F);

    output0.fill(91.0F);
    output1.fill(91.0F);
    output2.fill(91.0F);
    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, descriptors.data(),
                                                static_cast<u32>(descriptors.size()),
                                                ReadMappedBytes, &payload, PatternFactory(stats)),
              ORBIS_OK);
    EXPECT_TRUE(std::ranges::all_of(output0, [](float value) { return value == 0.0F; }));
    EXPECT_TRUE(std::ranges::all_of(output1, [](float value) { return value == 0.0F; }));
    EXPECT_TRUE(std::ranges::all_of(output2, [](float value) { return value == 0.0F; }));
    EXPECT_EQ(stats->factories, 1u);
    EXPECT_EQ(stats->decodes, 1u);
}

TEST_F(Ngs2SystemRenderTest, SubmixerCascadeIsIndependentOfRackCreationOrder) {
    for (const bool rack_endpoint : {false, true}) {
        std::array<std::array<float, 256>, 2> results{};
        for (const bool reverse_creation : {false, true}) {
            SCOPED_TRACE(testing::Message() << "rack_endpoint=" << rack_endpoint
                                            << " reverse_creation=" << reverse_creation);
            ResetRuntimeForTesting();
            auto system = CreateSystem();
            auto sampler = CreateRack(system.handle, 0x1000, 1, 1, 1);
            auto first = CreateRack(system.handle, 0x2000, 1, 1, 0);
            auto second = CreateRack(system.handle, 0x2000, 1, 1, 0);
            const auto upstream = reverse_creation ? second.handle : first.handle;
            const auto downstream = reverse_creation ? first.handle : second.handle;
            auto mastering = CreateRack(system.handle, 0x3000, 1, 0, 0);

            RenderGraph graph;
            ConfigureGraphCommands(graph, upstream, Voice(mastering.handle));
            auto downstream_commands = graph.submixer;
            graph.submixer.patch.destHandle = rack_endpoint ? downstream : Voice(downstream);
            graph.mastering.output.outputId = 0;
            Apply(Voice(sampler.handle), graph.sampler);
            Apply(Voice(upstream), graph.submixer);
            Apply(Voice(downstream), downstream_commands);
            Apply(Voice(mastering.handle), graph.mastering);

            auto stats = std::make_shared<FakeDecoderStats>();
            ReadMap reads{
                {{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()}}};
            auto& output = results[reverse_creation];
            output.fill(77.0F);
            OrbisNgs2RenderBufferInfo descriptor{output.data(), sizeof(output), Float32WaveformType,
                                                 1};
            ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1,
                                                        ReadMappedBytes, &reads,
                                                        PatternFactory(stats)),
                      ORBIS_OK);
            EXPECT_EQ(stats->decodes, 1u);
            EXPECT_FLOAT_EQ(output[0], 0.1F);
            EXPECT_FLOAT_EQ(output[1], 5.1F);
            EXPECT_FLOAT_EQ(output[255], 1275.1F);
        }
        EXPECT_EQ(results[0], results[1]);
    }
}

TEST_F(Ngs2SystemRenderTest, SubmixerFanInWaitsForEveryUpstreamVoice) {
    auto system = CreateSystem();
    auto sampler = CreateRack(system.handle, 0x1000, 2, 1, 1);
    // The receiving node deliberately sorts before both predecessors in the old traversal.
    auto downstream = CreateRack(system.handle, 0x2000, 1, 1, 0);
    auto upstream_a = CreateRack(system.handle, 0x2000, 1, 1, 0);
    auto upstream_b = CreateRack(system.handle, 0x2000, 1, 1, 0);
    auto mastering = CreateRack(system.handle, 0x3000, 1, 0, 0);

    RenderGraph graph_a;
    RenderGraph graph_b;
    ConfigureGraphCommands(graph_a, upstream_a.handle, Voice(mastering.handle));
    ConfigureGraphCommands(graph_b, upstream_b.handle, Voice(mastering.handle));
    auto downstream_commands = graph_a.submixer;
    graph_a.submixer.patch.destHandle = downstream.handle;
    graph_b.submixer.patch.destHandle = Voice(downstream.handle);
    graph_a.mastering.output.outputId = 0;
    Apply(Voice(sampler.handle, 0), graph_a.sampler);
    Apply(Voice(sampler.handle, 1), graph_b.sampler);
    Apply(Voice(upstream_a.handle), graph_a.submixer);
    Apply(Voice(upstream_b.handle), graph_b.submixer);
    Apply(Voice(downstream.handle), downstream_commands);
    Apply(Voice(mastering.handle), graph_a.mastering);

    std::array<float, 256> output;
    output.fill(77.0F);
    OrbisNgs2RenderBufferInfo descriptor{output.data(), sizeof(output), Float32WaveformType, 1};
    auto stats = std::make_shared<FakeDecoderStats>();
    ReadMap reads{{{reinterpret_cast<uintptr_t>(graph_a.encoded.data()), graph_a.encoded.size()},
                   {reinterpret_cast<uintptr_t>(graph_b.encoded.data()), graph_b.encoded.size()}}};
    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &reads, PatternFactory(stats)),
              ORBIS_OK);
    EXPECT_EQ(stats->decodes, 2u);
    EXPECT_FLOAT_EQ(output[0], 0.2F);
    EXPECT_FLOAT_EQ(output[1], 10.2F);
    EXPECT_FLOAT_EQ(output[255], 2550.2F);
}

TEST_F(Ngs2SystemRenderTest, CyclicSubmixersRejectBeforeDecodeOrOutputAndCanBeRepaired) {
    for (const bool self_cycle : {false, true}) {
        SCOPED_TRACE(testing::Message() << "self_cycle=" << self_cycle);
        ResetRuntimeForTesting();
        auto system = CreateSystem();
        auto sampler = CreateRack(system.handle, 0x1000, 1, 1, 1);
        auto upstream = CreateRack(system.handle, 0x2000, 1, 1, 0);
        auto downstream = CreateRack(system.handle, 0x2000, 1, 1, 0);
        auto mastering = CreateRack(system.handle, 0x3000, 1, 0, 0);
        const auto sampler_voice = Voice(sampler.handle);

        RenderGraph graph;
        ConfigureGraphCommands(graph, upstream.handle, Voice(mastering.handle));
        auto downstream_commands = graph.submixer;
        graph.submixer.patch.destHandle = self_cycle ? upstream.handle : downstream.handle;
        if (!self_cycle) {
            downstream_commands.patch.destHandle = Voice(upstream.handle);
        }
        graph.mastering.output.outputId = 0;
        Apply(sampler_voice, graph.sampler);
        Apply(Voice(upstream.handle), graph.submixer);
        Apply(Voice(downstream.handle), downstream_commands);
        Apply(Voice(mastering.handle), graph.mastering);
        // Apply the queued play events separately so this compares decoder progress, not events.
        ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
        const auto before = SnapshotAppliedSamplerVoice(sampler_voice);
        ASSERT_TRUE(before);
        ASSERT_TRUE(before->decode_runtime);
        ASSERT_FALSE(before->decode_runtime->engine);

        std::array<float, 256> output;
        output.fill(77.0F);
        OrbisNgs2RenderBufferInfo descriptor{output.data(), sizeof(output), Float32WaveformType, 1};
        auto stats = std::make_shared<FakeDecoderStats>();
        ReadMap reads{{{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()}}};
        EXPECT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                    &reads, PatternFactory(stats)),
                  ORBIS_NGS2_ERROR_INVALID_PATCH);
        EXPECT_TRUE(std::ranges::all_of(output, [](float value) { return value == 77.0F; }));
        EXPECT_EQ(stats->factories, 0u);
        EXPECT_EQ(stats->decodes, 0u);
        EXPECT_EQ(reads.reads, 0u);
        EXPECT_FALSE(before->decode_runtime->engine);
        const auto after = SnapshotAppliedSamplerVoice(sampler_voice);
        ASSERT_TRUE(after);
        EXPECT_EQ(after->token.render_state_revision, before->token.render_state_revision);

        graph.submixer.patch.destHandle = downstream.handle;
        downstream_commands.patch.destHandle = Voice(mastering.handle);
        Apply(Voice(upstream.handle), graph.submixer);
        Apply(Voice(downstream.handle), downstream_commands);
        ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                    &reads, PatternFactory(stats)),
                  ORBIS_OK);
        EXPECT_FLOAT_EQ(output[0], 0.1F);
        EXPECT_FLOAT_EQ(output[1], 5.1F);
        EXPECT_FLOAT_EQ(output[255], 1275.1F);
        EXPECT_EQ(stats->decodes, 1u);
    }
}

TEST_F(Ngs2SystemRenderTest, RackEndpointBusIsBoundToTheExactDestinationRackGeneration) {
    auto system = CreateSystem();
    auto sampler_rack = CreateRack(system.handle, 0x1000, 1, 1, 1);
    auto intended_submixer_rack = CreateRack(system.handle, 0x2000, 1, 1, 0);
    auto unrelated_submixer_rack = CreateRack(system.handle, 0x2000, 1, 1, 0);
    auto mastering_rack = CreateRack(system.handle, 0x3000, 1, 0, 0);
    const auto sampler_voice = Voice(sampler_rack.handle);
    const auto intended_submixer_voice = Voice(intended_submixer_rack.handle);
    const auto unrelated_submixer_voice = Voice(unrelated_submixer_rack.handle);
    const auto mastering_voice = Voice(mastering_rack.handle);

    RenderGraph graph;
    ConfigureGraphCommands(graph, intended_submixer_rack.handle, mastering_voice);
    graph.mastering.output.outputId = 0;
    RenderGraph unrelated_graph;
    ConfigureGraphCommands(unrelated_graph, unrelated_submixer_rack.handle, mastering_voice);
    Apply(sampler_voice, graph.sampler);
    Apply(intended_submixer_voice, graph.submixer);
    Apply(unrelated_submixer_voice, unrelated_graph.submixer);
    Apply(mastering_voice, graph.mastering);

    std::array<float, 256> output;
    output.fill(91.0F);
    OrbisNgs2RenderBufferInfo descriptor{output.data(), sizeof(output), Float32WaveformType, 1};
    auto stats = std::make_shared<FakeDecoderStats>();
    ReadMap payload{{{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()}}};

    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &payload, PatternFactory(stats)),
              ORBIS_OK);
    EXPECT_FLOAT_EQ(output[0], 0.1F);
    EXPECT_FLOAT_EQ(output[1], 5.1F);
    EXPECT_FLOAT_EQ(output[255], 1275.1F);
    EXPECT_EQ(stats->factories, 1u);
    EXPECT_EQ(stats->decodes, 1u);
}

TEST_F(Ngs2SystemRenderTest, EnabledUserFxFailsClosedWithoutDecodeAndExplicitNullRestoresAudio) {
    RenderDecisionTrace trace;
    auto system = CreateSystem();
    auto sampler_rack = CreateRack(system.handle, 0x1000, 1, 1, 1);
    auto submixer_rack = CreateRack(system.handle, 0x2000, 1, 1, 0);
    auto mastering_rack = CreateRack(system.handle, 0x3000, 1, 0, 0);
    const auto sampler_voice = Voice(sampler_rack.handle);
    const auto submixer_voice = Voice(submixer_rack.handle);
    const auto mastering_voice = Voice(mastering_rack.handle);
    RenderGraph graph;
    ConfigureGraphCommands(graph, submixer_rack.handle, mastering_voice);
    graph.mastering.output.outputId = 0;
    Apply(sampler_voice, graph.sampler);
    Apply(submixer_voice, graph.submixer);
    Apply(mastering_voice, graph.mastering);

    OrbisNgs2SubmixerVoiceUserFxParam user_fx{};
    user_fx.header = {sizeof(user_fx), 0, SubmixerUserFxControlId};
    user_fx.handler = DummyUserFxProcess;
    ReadMap user_fx_read{{{reinterpret_cast<uintptr_t>(&user_fx), sizeof(user_fx)}}};
    ASSERT_EQ(VoiceControl(submixer_voice, &user_fx.header, ReadMappedBytes, &user_fx_read),
              ORBIS_OK);

    std::array<float, 256> output;
    output.fill(91.0F);
    OrbisNgs2RenderBufferInfo descriptor{output.data(), sizeof(output), Float32WaveformType, 1};
    auto stats = std::make_shared<FakeDecoderStats>();
    ReadMap payload{{{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()}}};
    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &payload, PatternFactory(stats)),
              ORBIS_OK);
    EXPECT_TRUE(std::ranges::all_of(output, [](float value) { return value == 0.0F; }));
    EXPECT_EQ(stats->factories, 0u);
    EXPECT_EQ(stats->decodes, 0u);
    EXPECT_EQ(payload.reads, 0u);

    user_fx.handler = nullptr;
    ASSERT_EQ(VoiceControl(submixer_voice, &user_fx.header, ReadMappedBytes, &user_fx_read),
              ORBIS_OK);
    output.fill(91.0F);
    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &payload, PatternFactory(stats)),
              ORBIS_OK);
    EXPECT_FLOAT_EQ(output[0], 0.1F);
    EXPECT_FLOAT_EQ(output[255], 1275.1F);
    EXPECT_EQ(stats->factories, 1u);
    EXPECT_EQ(stats->decodes, 1u);
    EXPECT_EQ(trace.Reasons(), (std::vector<std::string>{"unsupported_user_fx", "decode_attempt"}));
}

TEST_F(Ngs2SystemRenderTest, EmptyGraphReportsNoEligibleSamplerWithoutDecode) {
    RenderDecisionTrace trace;
    auto system = CreateSystem();
    std::array<float, 256> output{};
    OrbisNgs2RenderBufferInfo descriptor{output.data(), sizeof(output), Float32WaveformType, 1};
    auto stats = std::make_shared<FakeDecoderStats>();
    ReadMap payload{};
    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &payload, PatternFactory(stats)), ORBIS_OK);
    EXPECT_EQ(stats->decodes, 0U);
    EXPECT_EQ(trace.Reasons(), (std::vector<std::string>{"no_eligible_sampler"}));
}

TEST_F(Ngs2SystemRenderTest, RackEndpointWithMultipleActiveVoicesDoesNotGuessFanout) {
    auto system = CreateSystem();
    auto sampler_rack = CreateRack(system.handle, 0x1000, 1, 1, 1);
    auto submixer_rack = CreateRack(system.handle, 0x2000, 2, 1, 0);
    auto mastering_rack = CreateRack(system.handle, 0x3000, 1, 0, 0);
    const auto sampler_voice = Voice(sampler_rack.handle);
    const auto first_submixer_voice = Voice(submixer_rack.handle);
    const auto second_submixer_voice = Voice(submixer_rack.handle, 1);
    const auto mastering_voice = Voice(mastering_rack.handle);
    RenderGraph graph;
    ConfigureGraphCommands(graph, submixer_rack.handle, mastering_voice);
    graph.mastering.output.outputId = 0;
    RenderGraph second_graph;
    ConfigureGraphCommands(second_graph, submixer_rack.handle, mastering_voice);
    Apply(sampler_voice, graph.sampler);
    Apply(first_submixer_voice, graph.submixer);
    Apply(second_submixer_voice, second_graph.submixer);
    Apply(mastering_voice, graph.mastering);

    std::array<float, 256> output;
    output.fill(91.0F);
    OrbisNgs2RenderBufferInfo descriptor{output.data(), sizeof(output), Float32WaveformType, 1};
    auto stats = std::make_shared<FakeDecoderStats>();
    ReadMap payload{{{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()}}};
    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &payload, PatternFactory(stats)),
              ORBIS_OK);
    EXPECT_TRUE(std::ranges::all_of(output, [](float value) { return value == 0.0F; }));
    EXPECT_EQ(stats->factories, 1u);
    EXPECT_EQ(stats->decodes, 1u);
}

TEST_F(Ngs2SystemRenderTest, MultichannelMasterGainDoesNotGuessLfeChannelOrdering) {
    auto system = CreateSystem();
    auto sampler_rack = CreateRack(system.handle, 0x1000, 1, 1, 1);
    auto submixer_rack = CreateRack(system.handle, 0x2000, 1, 1, 0);
    auto mastering_rack = CreateRack(system.handle, 0x3000, 1, 0, 0);
    const auto sampler_voice = Voice(sampler_rack.handle);
    const auto submixer_voice = Voice(submixer_rack.handle);
    const auto mastering_voice = Voice(mastering_rack.handle);
    RenderGraph graph;
    ConfigureGraphCommands(graph, submixer_rack.handle, mastering_voice);
    graph.mastering.output.outputId = 0;
    graph.mastering.matrix.type = 7;
    graph.mastering.matrix.numLevels = 16;
    graph.mastering.levels.fill(0.0F);
    graph.mastering.levels[0] = 1.0F;
    graph.mastering.levels[9] = 1.0F;
    graph.mastering.gain.fbwLevel = 1.0F;
    graph.mastering.gain.lfeLevel = 0.5F;
    Apply(sampler_voice, graph.sampler);
    Apply(submixer_voice, graph.submixer);
    Apply(mastering_voice, graph.mastering);

    std::array<float, 256 * 8> output;
    output.fill(91.0F);
    OrbisNgs2RenderBufferInfo descriptor{output.data(), sizeof(output), Float32WaveformType, 8};
    auto stats = std::make_shared<FakeDecoderStats>();
    ReadMap payload{{{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()}}};
    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, &descriptor, 1, ReadMappedBytes,
                                                &payload, PatternFactory(stats)),
              ORBIS_OK);
    EXPECT_TRUE(std::ranges::all_of(output, [](float value) { return value == 0.0F; }));
    EXPECT_EQ(stats->decodes, 1u);
}

struct MutatingRead {
    ReadMap payload;
    OrbisNgs2Handle mastering_voice{};
    OrbisNgs2MasteringVoiceGainParam gain{};
    bool mutated{};
    s32 result{ORBIS_NGS2_ERROR_FAIL};
};

bool ReadAndMutateMaster(uintptr_t address, void* destination, size_t size, void* context) {
    auto& mutation = *static_cast<MutatingRead*>(context);
    if (!mutation.mutated) {
        mutation.mutated = true;
        ReadMap control{{{reinterpret_cast<uintptr_t>(&mutation.gain), sizeof(mutation.gain)}}};
        mutation.result = VoiceControl(mutation.mastering_voice, &mutation.gain.header,
                                       ReadMappedBytes, &control);
    }
    return ReadMappedBytes(address, destination, size, &mutation.payload);
}

TEST_F(Ngs2SystemRenderTest, StaleGraphRejectsEveryCursorAndLeavesEveryOutputUntouched) {
    auto system = CreateSystem();
    auto sampler_rack = CreateRack(system.handle, 0x1000, 2, 1, 1);
    auto submixer_rack = CreateRack(system.handle, 0x2000, 1, 1, 0);
    auto mastering_rack = CreateRack(system.handle, 0x3000, 1, 0, 0);
    const auto sampler_voice = Voice(sampler_rack.handle);
    const auto second_sampler_voice = Voice(sampler_rack.handle, 1);
    const auto submixer_voice = Voice(submixer_rack.handle);
    const auto mastering_voice = Voice(mastering_rack.handle);
    RenderGraph graph;
    ConfigureGraphCommands(graph, submixer_rack.handle, mastering_voice);
    RenderGraph second_graph;
    ConfigureGraphCommands(second_graph, submixer_rack.handle, mastering_voice);
    Apply(sampler_voice, graph.sampler);
    Apply(second_sampler_voice, second_graph.sampler);
    Apply(submixer_voice, graph.submixer);
    Apply(mastering_voice, graph.mastering);

    std::array<float, 256> output0;
    std::array<float, 256> output1;
    output0.fill(77.0F);
    output1.fill(77.0F);
    std::array<OrbisNgs2RenderBufferInfo, 2> descriptors{{
        {output0.data(), sizeof(output0), Float32WaveformType, 1},
        {output1.data(), sizeof(output1), Float32WaveformType, 1},
    }};
    auto stats = std::make_shared<FakeDecoderStats>();
    MutatingRead mutation{
        .payload = {{{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()},
                     {reinterpret_cast<uintptr_t>(second_graph.encoded.data()),
                      second_graph.encoded.size()}},
                    0,
                    sampler_voice},
        .mastering_voice = mastering_voice,
    };
    mutation.gain.header = {sizeof(mutation.gain), 0, MasteringGainControlId};
    mutation.gain.fbwLevel = 0.5F;
    mutation.gain.lfeLevel = 0.5F;

    EXPECT_EQ(SystemRenderWithFactoryForTesting(system.handle, descriptors.data(), 2,
                                                ReadAndMutateMaster, &mutation,
                                                PatternFactory(stats)),
              ORBIS_NGS2_ERROR_FAIL);
    EXPECT_TRUE(mutation.mutated);
    EXPECT_EQ(mutation.result, ORBIS_OK);
    EXPECT_TRUE(mutation.payload.observed_registry_unlocked);
    EXPECT_TRUE(std::ranges::all_of(output0, [](float value) { return value == 77.0F; }));
    EXPECT_TRUE(std::ranges::all_of(output1, [](float value) { return value == 77.0F; }));

    ReadMap retry{
        {{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()},
         {reinterpret_cast<uintptr_t>(second_graph.encoded.data()), second_graph.encoded.size()}}};
    ASSERT_EQ(SystemRenderWithFactoryForTesting(system.handle, descriptors.data(), 2,
                                                ReadMappedBytes, &retry, PatternFactory(stats)),
              ORBIS_OK);
    EXPECT_TRUE(std::ranges::all_of(output0, [](float value) { return value == 0.0F; }));
    EXPECT_FLOAT_EQ(output1[0], 0.125F);
    EXPECT_FLOAT_EQ(output1[1], 6.375F);
    EXPECT_EQ(stats->factories, 4u);
    EXPECT_EQ(stats->decodes, 4u);
}

class Ngs2PublicationTest : public Ngs2SystemRenderTest {
protected:
    void SetUp() override {
        Ngs2SystemRenderTest::SetUp();
        system = CreateSystem();
        sampler = CreateRack(system.handle, 0x1000, 1, 1, 1);
        submixer = CreateRack(system.handle, 0x2000, 1, 1, 0);
        mastering = CreateRack(system.handle, 0x3000, 1, 0, 0);
        sampler_voice = Voice(sampler.handle);
        ConfigureGraphCommands(graph, submixer.handle, Voice(mastering.handle));
        Apply(sampler_voice, graph.sampler);
        Apply(Voice(submixer.handle), graph.submixer);
        Apply(Voice(mastering.handle), graph.mastering);
        ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
        before = SnapshotAppliedSamplerVoice(sampler_voice);
        ASSERT_TRUE(before);
        output0.fill(77.0F);
        output1.fill(77.0F);
        descriptors = {{{output0.data(), sizeof(output0), Float32WaveformType, 1},
                        {output1.data(), sizeof(output1), Float32WaveformType, 1}}};
        ReadMap descriptor_reads{
            {{reinterpret_cast<uintptr_t>(descriptors.data()), sizeof(descriptors)}}};
        ASSERT_EQ(
            GuestIo::SnapshotRenderBuffers(
                descriptors.data(), static_cast<u32>(descriptors.size()),
                [&](const void* source, void* destination, size_t size) {
                    return ReadMappedBytes(reinterpret_cast<uintptr_t>(source), destination, size,
                                           &descriptor_reads);
                },
                [](std::span<const Core::GuestMemoryRange> ranges) { return ranges.size() == 2; },
                &prepared),
            GuestIo::SnapshotError::None);
    }

    template <typename Writer>
    GuestIo::RenderPublicationResult Publish(Writer&& writer,
                                             OrbisNgs2RuntimeReadHandler reader = ReadMappedBytes,
                                             void* read_context = nullptr) {
        ReadMap payload{
            {{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()}}};
        return GuestIo::RenderPreparedBuffers(
            prepared,
            [&](SystemRenderPublishHandler publish, void* context) {
                return SystemRenderWithFactoryForTesting(
                    system.handle, prepared.descriptors.data(),
                    static_cast<u32>(prepared.descriptors.size()), reader,
                    read_context ? read_context : &payload, PatternFactory(stats), publish,
                    context);
            },
            std::forward<Writer>(writer));
    }

    void ExpectUnadvanced() {
        const auto current = SnapshotAppliedSamplerVoice(sampler_voice);
        ASSERT_TRUE(current);
        EXPECT_EQ(current->token.render_state_revision, before->token.render_state_revision);
        ASSERT_TRUE(current->decode_runtime);
        if (current->decode_runtime->engine) {
            EXPECT_EQ(current->decode_runtime->engine->CommittedCursor(), SamplerDecodeCursor{});
            EXPECT_EQ(current->decode_runtime->engine->CommittedRevision(), 0u);
            EXPECT_FALSE(current->decode_runtime->engine->HasStagedProgress());
        }
    }

    static bool WriteAll(std::span<const Core::GuestMemoryWrite> writes) {
        for (const auto& write : writes) {
            std::memcpy(reinterpret_cast<void*>(write.address), write.data, write.size);
        }
        return true;
    }

    TestSystem system;
    TestRack sampler;
    TestRack submixer;
    TestRack mastering;
    OrbisNgs2Handle sampler_voice{};
    RenderGraph graph;
    std::optional<AppliedSamplerVoiceSnapshot> before;
    std::array<float, 256> output0;
    std::array<float, 256> output1;
    std::array<OrbisNgs2RenderBufferInfo, 2> descriptors;
    GuestIo::RenderBufferSnapshot prepared;
    std::shared_ptr<FakeDecoderStats> stats = std::make_shared<FakeDecoderStats>();
};

TEST_F(Ngs2PublicationTest, FailedFinalBatchedWritePreservesProgressAndRetriesSameGrain) {
    u32 calls{};
    const auto rejected = Publish([&](std::span<const Core::GuestMemoryWrite> writes) {
        ++calls;
        EXPECT_EQ(writes.size(), 2u);
        // A registry query must remain possible while the external writer runs.
        const auto current = SnapshotAppliedSamplerVoice(sampler_voice);
        EXPECT_TRUE(current);
        if (current) {
            EXPECT_EQ(current->token.render_state_revision, before->token.render_state_revision);
        }
        return false;
    });
    EXPECT_EQ(rejected.result, ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS);
    EXPECT_TRUE(rejected.attempted);
    EXPECT_FALSE(rejected.succeeded);
    EXPECT_EQ(calls, 1u);
    EXPECT_TRUE(std::ranges::all_of(output0, [](float value) { return value == 77.0F; }));
    EXPECT_TRUE(std::ranges::all_of(output1, [](float value) { return value == 77.0F; }));
    ExpectUnadvanced();

    const auto retry = Publish(WriteAll);
    ASSERT_EQ(retry.result, ORBIS_OK);
    EXPECT_TRUE(retry.succeeded);
    EXPECT_FLOAT_EQ(output1[0], 0.1F);
    EXPECT_FLOAT_EQ(output1[1], 5.1F);
    EXPECT_FLOAT_EQ(output1[255], 1275.1F);
    EXPECT_EQ(stats->decodes, 2u);
}

TEST_F(Ngs2PublicationTest, ThrowingWriterDoesNotCommitDecoderOrEscape) {
    GuestIo::RenderPublicationResult result;
    EXPECT_NO_THROW(result = Publish([](std::span<const Core::GuestMemoryWrite>) -> bool {
                        throw std::runtime_error("synthetic publication failure");
                    }));
    EXPECT_EQ(result.result, ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS);
    ExpectUnadvanced();
    const auto retry = Publish(WriteAll);
    ASSERT_EQ(retry.result, ORBIS_OK);
    EXPECT_FLOAT_EQ(output1[1], 5.1F);
}

TEST_F(Ngs2PublicationTest, FailedWriteAfterSuccessfulGrainPreservesCommittedDecoder) {
    std::array<u8, 480> two_frames{};
    graph.sampler.waveform.data = two_frames.data();
    graph.sampler.block.dataSize = static_cast<u32>(two_frames.size());
    graph.sampler.block.numSamples = 512;
    Apply(sampler_voice, graph.sampler);
    ReadMap payload{{{reinterpret_cast<uintptr_t>(two_frames.data()), two_frames.size()}}};
    ASSERT_EQ(Publish(WriteAll, ReadMappedBytes, &payload).result, ORBIS_OK);
    const auto committed = SnapshotAppliedSamplerVoice(sampler_voice);
    ASSERT_TRUE(committed);
    const auto cursor = committed->decode_runtime->engine->CommittedCursor();
    const auto revision = committed->decode_runtime->engine->CommittedRevision();
    EXPECT_NE(cursor, SamplerDecodeCursor{});
    output1.fill(77.0F);

    const auto rejected = Publish([](std::span<const Core::GuestMemoryWrite>) { return false; },
                                  ReadMappedBytes, &payload);
    EXPECT_EQ(rejected.result, ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS);
    const auto current = SnapshotAppliedSamplerVoice(sampler_voice);
    ASSERT_TRUE(current);
    EXPECT_EQ(current->token.render_state_revision, committed->token.render_state_revision);
    EXPECT_EQ(current->decode_runtime->engine->CommittedCursor(), cursor);
    EXPECT_EQ(current->decode_runtime->engine->CommittedRevision(), revision);
    EXPECT_FALSE(current->decode_runtime->engine->HasStagedProgress());
    EXPECT_TRUE(std::ranges::all_of(output1, [](float value) { return value == 77.0F; }));

    ASSERT_EQ(Publish(WriteAll, ReadMappedBytes, &payload).result, ORBIS_OK);
    EXPECT_FLOAT_EQ(output1[0], 0.1F);
    EXPECT_FLOAT_EQ(output1[1], 5.1F);
    EXPECT_FLOAT_EQ(output1[255], 1275.1F);
    EXPECT_EQ(stats->decodes, 3u);
    EXPECT_EQ(stats->clones, 2u);
}

TEST_F(Ngs2PublicationTest, PartialWriterFailureDoesNotPretendToRollBackGuestBytes) {
    const auto result = Publish([&](std::span<const Core::GuestMemoryWrite> writes) {
        const auto& active = writes[1];
        std::memcpy(reinterpret_cast<void*>(active.address), active.data, active.size);
        return false;
    });
    EXPECT_EQ(result.result, ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS);
    EXPECT_FLOAT_EQ(output1[1], 5.1F);
    EXPECT_TRUE(std::ranges::all_of(output0, [](float value) { return value == 77.0F; }));
    ExpectUnadvanced();
}

TEST_F(Ngs2PublicationTest, GraphChangeDuringSuccessfulPublicationRejectsOnlyDecoderProgress) {
    const auto result = Publish([&](std::span<const Core::GuestMemoryWrite> writes) {
        WriteAll(writes);
        OrbisNgs2MasteringVoiceGainParam gain{};
        gain.header = {sizeof(gain), 0, MasteringGainControlId};
        gain.fbwLevel = gain.lfeLevel = 0.4F;
        ReadMap controls{{{reinterpret_cast<uintptr_t>(&gain), sizeof(gain)}}};
        EXPECT_EQ(VoiceControl(Voice(mastering.handle), &gain.header, ReadMappedBytes, &controls),
                  ORBIS_OK);
        return true;
    });
    EXPECT_EQ(result.result, ORBIS_NGS2_ERROR_FAIL);
    EXPECT_TRUE(result.attempted);
    EXPECT_TRUE(result.succeeded);
    // Guest bytes are already visible. This deliberately does not claim a global transaction.
    EXPECT_FLOAT_EQ(output1[1], 5.1F);
    ExpectUnadvanced();
    const auto retry = Publish(WriteAll);
    ASSERT_EQ(retry.result, ORBIS_OK);
    EXPECT_FLOAT_EQ(output1[0], 0.05F);
    EXPECT_FLOAT_EQ(output1[1], 2.55F);
}

TEST_F(Ngs2PublicationTest, StaleGraphBeforePublicationNeverCallsWriter) {
    MutatingRead mutation{
        .payload = {{{reinterpret_cast<uintptr_t>(graph.encoded.data()), graph.encoded.size()}}},
        .mastering_voice = Voice(mastering.handle),
    };
    mutation.gain.header = {sizeof(mutation.gain), 0, MasteringGainControlId};
    mutation.gain.fbwLevel = mutation.gain.lfeLevel = 0.4F;
    u32 calls{};
    const auto result = Publish(
        [&](std::span<const Core::GuestMemoryWrite> writes) {
            ++calls;
            return WriteAll(writes);
        },
        ReadAndMutateMaster, &mutation);
    EXPECT_EQ(result.result, ORBIS_NGS2_ERROR_FAIL);
    EXPECT_FALSE(result.attempted);
    EXPECT_EQ(calls, 0u);
    EXPECT_TRUE(std::ranges::all_of(output1, [](float value) { return value == 77.0F; }));
    ExpectUnadvanced();
}

TEST_F(Ngs2PublicationTest, EmptyRenderStillPublishesOneEmptyBatchWithoutDecoding) {
    prepared = {};
    u32 calls{};
    const auto result = Publish([&](std::span<const Core::GuestMemoryWrite> writes) {
        ++calls;
        EXPECT_TRUE(writes.empty());
        return true;
    });
    EXPECT_EQ(result.result, ORBIS_OK);
    EXPECT_TRUE(result.succeeded);
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(stats->decodes, 0u);
    ExpectUnadvanced();
}

TEST(Ngs2PublicationWiring, PublicExportUsesPreparedBatchedPublicationHelper) {
    const auto source_path = std::filesystem::path{__FILE__}.parent_path().parent_path() /
                             "src/core/libraries/ngs2/ngs2.cpp";
    std::ifstream input{source_path};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    EXPECT_NE(source.find("GuestIo::RenderPreparedBuffers("), std::string::npos);
    EXPECT_EQ(source.find("memory->TryWriteGuest(snapshot.output_writes)"), std::string::npos);
}

TEST(Ngs2PublicationWiring, FailureTracePreservesActualPublicationOutcome) {
    const auto source_path = std::filesystem::path{__FILE__}.parent_path().parent_path() /
                             "src/core/libraries/ngs2/ngs2.cpp";
    std::ifstream input{source_path};
    ASSERT_TRUE(input.is_open());
    const std::string source{std::istreambuf_iterator<char>{input}, {}};
    const auto render_result = source.find("const s32 result = publication.result;");
    ASSERT_NE(render_result, std::string::npos);
    const auto end_failure = source.find("return result;", render_result);
    ASSERT_NE(end_failure, std::string::npos);
    const auto failure = source.substr(render_result, end_failure - render_result);
    EXPECT_NE(failure.find(".render_succeeded = false"), std::string::npos);
    EXPECT_NE(failure.find(".commit_succeeded = publication.succeeded"), std::string::npos);
}

} // namespace
} // namespace Libraries::Ngs2
