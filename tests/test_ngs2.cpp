// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "core/libraries/audio/audio_pipeline_metadata_trace.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/ngs2/ngs2.h"
#include "core/libraries/ngs2/ngs2_atrac9.h"
#include "core/libraries/ngs2/ngs2_custom.h"
#include "core/libraries/ngs2/ngs2_error.h"
#include "core/libraries/ngs2/ngs2_mastering.h"
#include "core/libraries/ngs2/ngs2_reverb.h"
#include "core/libraries/ngs2/ngs2_sampler.h"
#include "core/libraries/ngs2/ngs2_submixer.h"

namespace Libraries::Ngs2 {
namespace {

static_assert(sizeof(OrbisNgs2VoiceParamHeader) == 8);
static_assert(sizeof(OrbisNgs2WaveformFormat) == 24);
static_assert(sizeof(OrbisNgs2WaveformBlock) == 32);
static_assert(sizeof(OrbisNgs2SamplerVoiceSetupParam) == 40);
static_assert(sizeof(OrbisNgs2SamplerVoiceWaveformBlocksParam) == 32);
static_assert(sizeof(OrbisNgs2SamplerVoiceUserFxParam) == 40);
static_assert(sizeof(OrbisNgs2SubmixerVoiceUserFxParam) == 40);
static_assert(sizeof(OrbisNgs2VoiceMatrixLevelsParam) == 24);
static_assert(sizeof(OrbisNgs2VoicePortVolumeParam) == 16);
static_assert(sizeof(OrbisNgs2VoicePortMatrixParam) == 16);
static_assert(sizeof(OrbisNgs2VoicePatchParam) == 24);
static_assert(sizeof(OrbisNgs2MasteringVoiceOutputParam) == 16);
static_assert(sizeof(OrbisNgs2RenderBufferInfo) == 24);
static_assert(sizeof(OrbisNgs2SamplerVoiceState) == 48);
static_assert(sizeof(OrbisNgs2SubmixerVoiceState) == 16);
static_assert(sizeof(OrbisNgs2ReverbVoiceState) == 4);
static_assert(sizeof(OrbisNgs2CustomSubmixerVoiceState) == 12);
static_assert(sizeof(OrbisNgs2ContextBufferInfo) == 64);
static_assert(sizeof(OrbisNgs2SystemOption) == 64);
static_assert(sizeof(OrbisNgs2RackOption) == 128);

constexpr u32 SamplerRackId = 0x1000;
constexpr u32 VoiceEventControlId = 0x00000006;
constexpr u32 SamplerSetupControlId = 0x10000000;
constexpr u32 SamplerWaveformBlocksControlId = 0x10000001;
constexpr u32 SamplerUserFxControlId = 0x10000008;
constexpr u32 SubmixerUserFxControlId = 0x20000004;
constexpr u32 VoiceMatrixLevelsControlId = 0x00000001;
constexpr u32 VoicePortVolumeControlId = 0x00000002;
constexpr u32 VoicePortMatrixControlId = 0x00000003;
constexpr u32 VoicePatchControlId = 0x00000005;
constexpr u32 MasteringOutputControlId = 0x30000005;
constexpr size_t TestRackBufferHeaderSize = 64;
constexpr size_t TestVoiceTokenStride = 16;

struct TestSystem {
    std::vector<u8> storage;
    OrbisNgs2ContextBufferInfo buffer_info{};
    OrbisNgs2Handle handle{};
};

struct TestRack {
    std::vector<u8> storage;
    OrbisNgs2RackOption option{};
    OrbisNgs2ContextBufferInfo buffer_info{};
    OrbisNgs2Handle handle{};
};

struct ReadWindow {
    uintptr_t begin;
    size_t size;
};

struct CountingReadWindow {
    ReadWindow window;
    uintptr_t watched_address;
    u32 watched_reads{};
};

struct OverflowDescriptorReadContext {
    ReadWindow command_window;
    uintptr_t descriptor_address;
    u32 descriptor_reads{};
};

struct MutatingRackAllocatorContext {
    std::array<u8, 512> storage{};
    OrbisNgs2RackOption* option{};
    OrbisNgs2BufferAllocator* allocator{};
    u32 alloc_calls{};
    u32 free_calls{};
};

struct MutatingSystemAllocatorContext {
    alignas(8) std::array<u8, 256> storage{};
    OrbisNgs2SystemOption* option{};
    OrbisNgs2BufferAllocator* allocator{};
    bool return_misaligned_buffer{};
    u32 alloc_calls{};
    u32 free_calls{};
};

struct ReplacingSystemAllocatorContext {
    std::array<u8, 256> rack_storage{};
    OrbisNgs2Handle system_handle{};
    OrbisNgs2Handle replacement_handle{};
    s32 cleanup_result{ORBIS_OK};
    s32 recreate_result{ORBIS_OK};
    u32 alloc_calls{};
    u32 free_calls{};
};

s32 PS4_SYSV_ABI MutatingRackAlloc(OrbisNgs2ContextBufferInfo* info) {
    auto& context = *reinterpret_cast<MutatingRackAllocatorContext*>(info->userData);
    ++context.alloc_calls;
    info->hostBuffer = context.storage.data();
    context.option->maxVoices = std::numeric_limits<u32>::max();
    context.allocator->freeHandler = nullptr;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI CountingRackFree(OrbisNgs2ContextBufferInfo* info) {
    auto& context = *reinterpret_cast<MutatingRackAllocatorContext*>(info->userData);
    ++context.free_calls;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI MutatingSystemAlloc(OrbisNgs2ContextBufferInfo* info) {
    auto& context = *reinterpret_cast<MutatingSystemAllocatorContext*>(info->userData);
    ++context.alloc_calls;
    info->hostBuffer = context.storage.data() + (context.return_misaligned_buffer ? 1 : 0);
    context.option->size = 0;
    context.option->sampleRate = std::numeric_limits<u32>::max();
    context.allocator->allocHandler = nullptr;
    context.allocator->freeHandler = nullptr;
    context.allocator->userData = 0;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI CountingSystemFree(OrbisNgs2ContextBufferInfo* info) {
    auto& context = *reinterpret_cast<MutatingSystemAllocatorContext*>(info->userData);
    ++context.free_calls;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI ReplaceSystemDuringRackAlloc(OrbisNgs2ContextBufferInfo* info) {
    auto& context = *reinterpret_cast<ReplacingSystemAllocatorContext*>(info->userData);
    ++context.alloc_calls;
    OrbisNgs2ContextBufferInfo system_info{};
    context.cleanup_result = SystemCleanup(context.system_handle, &system_info);
    if (context.cleanup_result >= 0) {
        context.recreate_result =
            SystemSetup(nullptr, &system_info, nullptr, &context.replacement_handle);
    }
    info->hostBuffer = context.rack_storage.data();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI CountReplacingSystemRackFree(OrbisNgs2ContextBufferInfo* info) {
    auto& context = *reinterpret_cast<ReplacingSystemAllocatorContext*>(info->userData);
    ++context.free_calls;
    return ORBIS_OK;
}

bool ReadWindowBytes(uintptr_t address, void* destination, size_t size, void* context) {
    const auto& window = *static_cast<const ReadWindow*>(context);
    if (address < window.begin) {
        return false;
    }
    const uintptr_t offset = address - window.begin;
    if (offset > window.size || size > window.size - offset) {
        return false;
    }
    std::memcpy(destination, reinterpret_cast<const void*>(address), size);
    return true;
}

bool CountingReadWindowBytes(uintptr_t address, void* destination, size_t size, void* context) {
    auto& counting = *static_cast<CountingReadWindow*>(context);
    if (address == counting.watched_address) {
        ++counting.watched_reads;
    }
    return ReadWindowBytes(address, destination, size, &counting.window);
}

bool OverflowDescriptorReadBytes(uintptr_t address, void* destination, size_t size, void* context) {
    auto& overflow = *static_cast<OverflowDescriptorReadContext*>(context);
    if (address == overflow.descriptor_address && size == sizeof(OrbisNgs2WaveformBlock)) {
        ++overflow.descriptor_reads;
        std::memset(destination, 0, size);
        return true;
    }
    return ReadWindowBytes(address, destination, size, &overflow.command_window);
}

s32 PS4_SYSV_ABI DummyUserFxProcess(OrbisNgs2UserFxProcessContext*) {
    return ORBIS_OK;
}

void SetAudioTracePath(const std::filesystem::path& path) {
#ifdef _WIN32
    ASSERT_EQ(_putenv_s("SHADPS4_AUDIO_PIPELINE_TRACE_PATH", path.string().c_str()), 0);
#else
    ASSERT_EQ(setenv("SHADPS4_AUDIO_PIPELINE_TRACE_PATH", path.string().c_str(), 1), 0);
#endif
    AudioPipelineTrace::Testing::Reset();
}

void ClearAudioTracePath() {
    AudioPipelineTrace::Testing::Reset();
#ifdef _WIN32
    ASSERT_EQ(_putenv_s("SHADPS4_AUDIO_PIPELINE_TRACE_PATH", ""), 0);
#else
    ASSERT_EQ(unsetenv("SHADPS4_AUDIO_PIPELINE_TRACE_PATH"), 0);
#endif
}

std::vector<nlohmann::json> ReadJsonLines(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::vector<nlohmann::json> lines;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(nlohmann::json::parse(line));
    }
    return lines;
}

u64 CurrentTestProcessId() {
#ifdef _WIN32
    return static_cast<u64>(_getpid());
#else
    return static_cast<u64>(getpid());
#endif
}

std::string UniqueTraceFileToken() {
    static std::atomic<u64> sequence{};
    return std::to_string(CurrentTestProcessId()) + "-" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

class ScopedAudioTrace {
public:
    ScopedAudioTrace()
        : path{std::filesystem::temp_directory_path() /
               ("shadps4-ngs2-sampler-trace-" + UniqueTraceFileToken() + ".jsonl")} {
        std::error_code error;
        std::filesystem::remove(path, error);
        SetAudioTracePath(path);
    }

    ~ScopedAudioTrace() {
        ClearAudioTracePath();
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    void Finalize() const {
        AudioPipelineTrace::Testing::Finalize();
    }

    [[nodiscard]] std::vector<nlohmann::json> ReadLines() const {
        return ReadJsonLines(path);
    }

    [[nodiscard]] std::string ReadContents() const {
        std::ifstream input(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>{input}, {});
    }

private:
    std::filesystem::path path;
};

class Ngs2RuntimeTest : public testing::Test {
protected:
    void SetUp() override {
        ResetRuntimeForTesting();
    }

    void TearDown() override {
        ResetRuntimeForTesting();
    }

    TestSystem CreateSystem() {
        TestSystem system;
        system.buffer_info.userData = 0x12345678;
        EXPECT_EQ(SystemSetup(nullptr, &system.buffer_info, nullptr, nullptr), ORBIS_OK);
        EXPECT_GT(system.buffer_info.hostBufferSize, 0);
        system.storage.resize(system.buffer_info.hostBufferSize);
        system.buffer_info.hostBuffer = system.storage.data();
        EXPECT_EQ(SystemSetup(nullptr, &system.buffer_info, nullptr, &system.handle), ORBIS_OK);
        EXPECT_NE(system.handle, 0);
        return system;
    }

    TestRack CreateRackForId(OrbisNgs2Handle system_handle, u32 rack_id, u32 max_voices = 2) {
        TestRack rack;
        rack.option.size = sizeof(rack.option);
        rack.option.maxGrainSamples = 512;
        rack.option.maxVoices = max_voices;
        EXPECT_EQ(RackQueryBufferSize(rack_id, &rack.option, &rack.buffer_info), ORBIS_OK);
        EXPECT_GT(rack.buffer_info.hostBufferSize, 0);
        rack.storage.resize(rack.buffer_info.hostBufferSize);
        rack.buffer_info.hostBuffer = rack.storage.data();
        EXPECT_EQ(RackCreate(system_handle, rack_id, &rack.option, &rack.buffer_info, nullptr,
                             &rack.handle),
                  ORBIS_OK);
        EXPECT_NE(rack.handle, 0);
        return rack;
    }

    TestRack CreateRack(OrbisNgs2Handle system_handle, u32 max_voices = 2) {
        return CreateRackForId(system_handle, SamplerRackId, max_voices);
    }

    OrbisNgs2Handle GetVoice(OrbisNgs2Handle rack_handle, u32 index) {
        OrbisNgs2Handle voice = 0;
        EXPECT_EQ(RackGetVoiceHandle(rack_handle, index, &voice), ORBIS_OK);
        EXPECT_NE(voice, 0);
        return voice;
    }

    u32 GetVoiceFlags(OrbisNgs2Handle voice_handle) {
        u32 flags = std::numeric_limits<u32>::max();
        EXPECT_EQ(VoiceGetStateFlags(voice_handle, &flags), ORBIS_OK);
        return flags;
    }

    s32 SendEvents(OrbisNgs2Handle voice_handle, OrbisNgs2VoiceEventParam* first, void* begin,
                   size_t size) {
        ReadWindow window{reinterpret_cast<uintptr_t>(begin), size};
        return VoiceControl(voice_handle, &first->header, ReadWindowBytes, &window);
    }
};

TEST(Ngs2Test, RackQueryBufferSizeInitializesSuccessfulOutput) {
    OrbisNgs2ContextBufferInfo info;
    std::memset(&info, 0xA5, sizeof(info));
    const size_t poisoned_size = info.hostBufferSize;
    const uintptr_t poisoned_user_data = info.userData;

    EXPECT_EQ(RackQueryBufferSize(SamplerRackId, nullptr, &info), ORBIS_OK);
    EXPECT_EQ(info.hostBuffer, nullptr);
    EXPECT_GT(info.hostBufferSize, 0);
    EXPECT_NE(info.hostBufferSize, poisoned_size);
    for (const auto value : info.reserved) {
        EXPECT_EQ(value, 0);
    }
    EXPECT_EQ(info.userData, poisoned_user_data);
}

TEST(Ngs2Test, RackQueryBufferSizeRejectsInvalidArguments) {
    EXPECT_EQ(RackQueryBufferSize(SamplerRackId, nullptr, nullptr),
              ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS);

    OrbisNgs2RackOption option{};
    option.size = sizeof(option) - 1;
    OrbisNgs2ContextBufferInfo info{};
    EXPECT_EQ(RackQueryBufferSize(SamplerRackId, &option, &info),
              ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE);
}

TEST(Ngs2Test, RackQueryBufferSizeScalesWithVoiceCapacity) {
    OrbisNgs2RackOption option{};
    option.size = sizeof(option);
    option.maxVoices = 1;
    OrbisNgs2ContextBufferInfo one_voice{};
    ASSERT_EQ(RackQueryBufferSize(SamplerRackId, &option, &one_voice), ORBIS_OK);

    option.maxVoices = 4;
    OrbisNgs2ContextBufferInfo four_voices{};
    ASSERT_EQ(RackQueryBufferSize(SamplerRackId, &option, &four_voices), ORBIS_OK);

    EXPECT_GT(four_voices.hostBufferSize, one_voice.hostBufferSize);
}

TEST_F(Ngs2RuntimeTest, CreatesTypedHierarchyAndReturnsStableIndexedVoices) {
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle, 2);

    const auto voice0 = GetVoice(rack.handle, 0);
    const auto voice0_again = GetVoice(rack.handle, 0);
    const auto voice1 = GetVoice(rack.handle, 1);

    EXPECT_EQ(voice0_again, voice0);
    EXPECT_NE(voice1, voice0);
    OrbisNgs2Handle invalid_voice = 0xA5A5;
    EXPECT_EQ(RackGetVoiceHandle(rack.handle, 2, &invalid_voice),
              ORBIS_NGS2_ERROR_INVALID_VOICE_INDEX);
    EXPECT_EQ(invalid_voice, 0xA5A5);
    EXPECT_EQ(RackGetVoiceHandle(system.handle, 0, &invalid_voice),
              ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE);
}

TEST_F(Ngs2RuntimeTest, VoiceGetStateInitializesRequestedBytesAndPreservesCanary) {
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);

    std::array<u8, 56> state;
    state.fill(0xA5);

    EXPECT_EQ(VoiceGetState(voice, state.data(), sizeof(OrbisNgs2SamplerVoiceState)), ORBIS_OK);
    for (size_t i = 0; i < sizeof(OrbisNgs2SamplerVoiceState); ++i) {
        EXPECT_EQ(state[i], 0);
    }
    for (size_t i = sizeof(OrbisNgs2SamplerVoiceState); i < state.size(); ++i) {
        EXPECT_EQ(state[i], 0xA5);
    }
}

TEST_F(Ngs2RuntimeTest, VoiceGetStateRejectsInvalidArguments) {
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);
    OrbisNgs2VoiceState state{};
    EXPECT_EQ(VoiceGetState(0, &state, sizeof(state)), ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE);
    EXPECT_EQ(VoiceGetState(system.handle, &state, sizeof(state)),
              ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE);
    EXPECT_EQ(VoiceGetState(voice, nullptr, sizeof(state)), ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS);
    EXPECT_EQ(VoiceGetState(voice, &state, sizeof(state) - 1),
              ORBIS_NGS2_ERROR_INVALID_VOICE_STATE_SIZE);
    EXPECT_EQ(VoiceGetState(voice, &state, sizeof(state)),
              ORBIS_NGS2_ERROR_INVALID_VOICE_STATE_SIZE);
}

TEST_F(Ngs2RuntimeTest, VoiceStateSizesFollowStandardRackAbiAndBoundCustomState) {
    auto system = CreateSystem();
    const std::array<std::pair<u32, size_t>, 4> exact_cases{{
        {0x1000, sizeof(OrbisNgs2SamplerVoiceState)},
        {0x2000, sizeof(OrbisNgs2SubmixerVoiceState)},
        {0x2001, sizeof(OrbisNgs2ReverbVoiceState)},
        {0x3000, sizeof(OrbisNgs2MasteringVoiceState)},
    }};
    for (const auto [rack_id, state_size] : exact_cases) {
        auto rack = CreateRackForId(system.handle, rack_id, 1);
        const auto voice = GetVoice(rack.handle, 0);
        std::vector<u8> state(state_size, 0xA5);
        ASSERT_EQ(VoiceGetState(voice, state.data(), state.size()), ORBIS_OK);
        EXPECT_TRUE(std::ranges::all_of(state, [](u8 value) { return value == 0; }));
        ASSERT_GT(state_size, 1u);
        EXPECT_EQ(VoiceGetState(voice, state.data(), state_size - 1),
                  ORBIS_NGS2_ERROR_INVALID_VOICE_STATE_SIZE);
        EXPECT_EQ(RackDestroy(rack.handle, nullptr), ORBIS_OK);
    }

    auto custom_rack = CreateRackForId(system.handle, 0x4002, 1);
    const auto custom_voice = GetVoice(custom_rack.handle, 0);
    std::array<u8, 64> custom_state;
    custom_state.fill(0xA5);
    EXPECT_EQ(VoiceGetState(custom_voice, custom_state.data(), custom_state.size()), ORBIS_OK);
    EXPECT_TRUE(std::ranges::all_of(custom_state, [](u8 value) { return value == 0; }));
    EXPECT_EQ(VoiceGetState(custom_voice, custom_state.data(),
                            sizeof(OrbisNgs2CustomSubmixerVoiceState) - 1),
              ORBIS_NGS2_ERROR_INVALID_VOICE_STATE_SIZE);
    EXPECT_EQ(RackDestroy(custom_rack.handle, nullptr), ORBIS_OK);
}

TEST_F(Ngs2RuntimeTest, RejectsOverlappingOpaqueBackingRanges) {
    std::array<u8, 256> storage{};
    OrbisNgs2ContextBufferInfo system_info{};
    ASSERT_EQ(SystemSetup(nullptr, &system_info, nullptr, nullptr), ORBIS_OK);
    system_info.hostBuffer = storage.data();
    ASSERT_LE(system_info.hostBufferSize, storage.size());
    OrbisNgs2Handle system_handle = 0;
    ASSERT_EQ(SystemSetup(nullptr, &system_info, nullptr, &system_handle), ORBIS_OK);

    OrbisNgs2RackOption option{};
    option.size = sizeof(option);
    option.maxGrainSamples = 512;
    option.maxVoices = 1;
    OrbisNgs2ContextBufferInfo rack_info{};
    ASSERT_EQ(RackQueryBufferSize(SamplerRackId, &option, &rack_info), ORBIS_OK);
    rack_info.hostBuffer = storage.data() + 32;
    ASSERT_LE(32 + rack_info.hostBufferSize, storage.size());

    OrbisNgs2Handle rack_handle = 0xA5A5;
    EXPECT_EQ(RackCreate(system_handle, SamplerRackId, &option, &rack_info, nullptr, &rack_handle),
              ORBIS_NGS2_ERROR_BUFFER_BROKEN);
    EXPECT_EQ(rack_handle, 0xA5A5);
}

TEST_F(Ngs2RuntimeTest, RejectsMisalignedOpaqueBackingBuffers) {
    std::array<u8, 256> system_storage{};
    OrbisNgs2ContextBufferInfo system_info{};
    ASSERT_EQ(SystemSetup(nullptr, &system_info, nullptr, nullptr), ORBIS_OK);
    system_info.hostBuffer = system_storage.data() + 1;
    ASSERT_LE(1 + system_info.hostBufferSize, system_storage.size());
    OrbisNgs2Handle system_handle = 0xA5A5;
    EXPECT_EQ(SystemSetup(nullptr, &system_info, nullptr, &system_handle),
              ORBIS_NGS2_ERROR_INVALID_BUFFER_ALIGN);
    EXPECT_EQ(system_handle, 0xA5A5);

    auto system = CreateSystem();
    OrbisNgs2RackOption option{};
    option.size = sizeof(option);
    option.maxGrainSamples = 512;
    option.maxVoices = 1;
    OrbisNgs2ContextBufferInfo rack_info{};
    ASSERT_EQ(RackQueryBufferSize(SamplerRackId, &option, &rack_info), ORBIS_OK);
    std::array<u8, 256> rack_storage{};
    rack_info.hostBuffer = rack_storage.data() + 1;
    ASSERT_LE(1 + rack_info.hostBufferSize, rack_storage.size());
    OrbisNgs2Handle rack_handle = 0x5A5A;
    EXPECT_EQ(RackCreate(system.handle, SamplerRackId, &option, &rack_info, nullptr, &rack_handle),
              ORBIS_NGS2_ERROR_INVALID_BUFFER_ALIGN);
    EXPECT_EQ(rack_handle, 0x5A5A);
}

TEST_F(Ngs2RuntimeTest, AllocatorRackCreationValidatesFirstAndSnapshotsGuestInputs) {
    OrbisNgs2RackOption option{};
    option.size = sizeof(option);
    option.maxGrainSamples = 512;
    option.maxVoices = 2;
    MutatingRackAllocatorContext context{};
    OrbisNgs2BufferAllocator allocator{MutatingRackAlloc, CountingRackFree,
                                       reinterpret_cast<uintptr_t>(&context)};
    context.option = &option;
    context.allocator = &allocator;

    OrbisNgs2Handle rack_handle = 0xA5A5;
    EXPECT_EQ(RackCreateWithAllocator(0, SamplerRackId, &option, &allocator, &rack_handle),
              ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE);
    EXPECT_EQ(context.alloc_calls, 0u);
    EXPECT_EQ(rack_handle, 0xA5A5);

    auto system = CreateSystem();
    ASSERT_EQ(
        RackCreateWithAllocator(system.handle, SamplerRackId, &option, &allocator, &rack_handle),
        ORBIS_OK);
    EXPECT_EQ(context.alloc_calls, 1u);
    EXPECT_NE(rack_handle, 0);
    EXPECT_EQ(RackDestroy(rack_handle, nullptr), ORBIS_OK);
    EXPECT_EQ(context.free_calls, 1u);
}

TEST_F(Ngs2RuntimeTest, AllocatorSystemCreationValidatesFirstAndSnapshotsGuestInputs) {
    OrbisNgs2SystemOption option{};
    option.size = sizeof(option);
    option.maxGrainSamples = 512;
    option.numGrainSamples = 256;
    option.sampleRate = 48000;
    MutatingSystemAllocatorContext context{};
    OrbisNgs2BufferAllocator allocator{MutatingSystemAlloc, CountingSystemFree,
                                       reinterpret_cast<uintptr_t>(&context)};
    context.option = &option;
    context.allocator = &allocator;

    OrbisNgs2SystemOption invalid_option = option;
    invalid_option.size = sizeof(invalid_option) - 1;
    OrbisNgs2Handle system_handle = 0xA5A5;
    EXPECT_EQ(SystemCreateWithAllocator(&invalid_option, &allocator, &system_handle),
              ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE);
    EXPECT_EQ(context.alloc_calls, 0u);
    EXPECT_EQ(system_handle, 0xA5A5);

    ASSERT_EQ(SystemCreateWithAllocator(&option, &allocator, &system_handle), ORBIS_OK);
    EXPECT_EQ(context.alloc_calls, 1u);
    EXPECT_NE(system_handle, 0);
    EXPECT_EQ(option.size, 0u);
    EXPECT_EQ(option.sampleRate, std::numeric_limits<u32>::max());
    EXPECT_EQ(allocator.allocHandler, nullptr);
    EXPECT_EQ(allocator.freeHandler, nullptr);

    EXPECT_EQ(SystemCleanup(system_handle, nullptr), ORBIS_OK);
    EXPECT_EQ(context.free_calls, 1u);
}

TEST_F(Ngs2RuntimeTest, SystemRuntimeSnapshotRetainsEffectiveOptionsAndRejectsStaleGeneration) {
    OrbisNgs2SystemOption option{};
    option.size = sizeof(option);
    option.maxGrainSamples = 1024;
    option.numGrainSamples = 64;
    option.sampleRate = 44100;

    TestSystem system;
    ASSERT_EQ(SystemSetup(&option, &system.buffer_info, nullptr, nullptr), ORBIS_OK);
    system.storage.resize(system.buffer_info.hostBufferSize);
    system.buffer_info.hostBuffer = system.storage.data();
    ASSERT_EQ(SystemSetup(&option, &system.buffer_info, nullptr, &system.handle), ORBIS_OK);

    const auto configured = SnapshotSystemRuntime(system.handle);
    ASSERT_TRUE(configured.has_value());
    EXPECT_EQ(configured->max_grain_samples, 1024u);
    EXPECT_EQ(configured->num_grain_samples, 64u);
    EXPECT_EQ(configured->sample_rate, 44100u);
    ASSERT_NE(configured->generation, 0u);

    OrbisNgs2ContextBufferInfo reusable_info{};
    ASSERT_EQ(SystemCleanup(system.handle, &reusable_info), ORBIS_OK);
    OrbisNgs2Handle replacement_handle = 0;
    ASSERT_EQ(SystemSetup(nullptr, &reusable_info, nullptr, &replacement_handle), ORBIS_OK);
    ASSERT_EQ(replacement_handle, system.handle);

    EXPECT_FALSE(SnapshotSystemRuntime(replacement_handle, configured->generation).has_value());
    const auto defaults = SnapshotSystemRuntime(replacement_handle);
    ASSERT_TRUE(defaults.has_value());
    EXPECT_EQ(defaults->max_grain_samples, 512u);
    EXPECT_EQ(defaults->num_grain_samples, 256u);
    EXPECT_EQ(defaults->sample_rate, 48000u);
    EXPECT_NE(defaults->generation, configured->generation);
    EXPECT_TRUE(SnapshotSystemRuntime(replacement_handle, defaults->generation).has_value());
}

TEST_F(Ngs2RuntimeTest, AllocatorSystemCreationRollsBackWithSnapshottedFreeHandler) {
    OrbisNgs2SystemOption option{};
    option.size = sizeof(option);
    option.maxGrainSamples = 512;
    option.numGrainSamples = 256;
    option.sampleRate = 48000;
    MutatingSystemAllocatorContext context{};
    context.return_misaligned_buffer = true;
    OrbisNgs2BufferAllocator allocator{MutatingSystemAlloc, CountingSystemFree,
                                       reinterpret_cast<uintptr_t>(&context)};
    context.option = &option;
    context.allocator = &allocator;

    OrbisNgs2Handle system_handle = 0xA5A5;
    EXPECT_EQ(SystemCreateWithAllocator(&option, &allocator, &system_handle),
              ORBIS_NGS2_ERROR_INVALID_BUFFER_ALIGN);
    EXPECT_EQ(context.alloc_calls, 1u);
    EXPECT_EQ(context.free_calls, 1u);
    EXPECT_EQ(system_handle, 0xA5A5);
}

TEST_F(Ngs2RuntimeTest, AllocatorRackCreationRejectsSameAddressSystemReplacement) {
    auto system = CreateSystem();
    ReplacingSystemAllocatorContext context{};
    context.system_handle = system.handle;
    OrbisNgs2BufferAllocator allocator{ReplaceSystemDuringRackAlloc, CountReplacingSystemRackFree,
                                       reinterpret_cast<uintptr_t>(&context)};
    OrbisNgs2RackOption option{};
    option.size = sizeof(option);
    option.maxGrainSamples = 512;
    option.maxVoices = 1;

    OrbisNgs2Handle rack_handle = 0xA5A5;
    EXPECT_EQ(
        RackCreateWithAllocator(system.handle, SamplerRackId, &option, &allocator, &rack_handle),
        ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE);
    EXPECT_EQ(context.cleanup_result, ORBIS_OK);
    EXPECT_EQ(context.recreate_result, ORBIS_OK);
    EXPECT_EQ(context.replacement_handle, system.handle);
    EXPECT_EQ(context.alloc_calls, 1u);
    EXPECT_EQ(context.free_calls, 1u);
    EXPECT_EQ(rack_handle, 0xA5A5);
}

TEST_F(Ngs2RuntimeTest, VoiceEventsTransitionOnRenderAndPersist) {
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);
    OrbisNgs2VoiceEventParam event{};
    event.header.size = sizeof(event);
    event.header.id = VoiceEventControlId;

    event.eventId = 0;
    ASSERT_EQ(SendEvents(voice, &event, &event, sizeof(event)), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0u);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0x3u);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0x3u);

    event.eventId = 4;
    ASSERT_EQ(SendEvents(voice, &event, &event, sizeof(event)), ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0x5u);

    event.eventId = 5;
    ASSERT_EQ(SendEvents(voice, &event, &event, sizeof(event)), ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0x3u);

    event.eventId = 1;
    ASSERT_EQ(SendEvents(voice, &event, &event, sizeof(event)), ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0xBu);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0xBu);

    event.eventId = 3;
    ASSERT_EQ(SendEvents(voice, &event, &event, sizeof(event)), ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0u);
}

TEST_F(Ngs2RuntimeTest, SignedVoiceControlChainIsBoundedAtomicAndCycleChecked) {
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);

    std::array<OrbisNgs2VoiceEventParam, 2> commands{};
    commands[0].header.size = sizeof(commands[0]);
    commands[0].header.id = VoiceEventControlId;
    commands[0].eventId = 4;
    commands[1].header.size = sizeof(commands[1]);
    commands[1].header.next = -static_cast<s16>(sizeof(commands[1]));
    commands[1].header.id = VoiceEventControlId;
    commands[1].eventId = 0;

    ASSERT_EQ(SendEvents(voice, &commands[1], commands.data(), sizeof(commands)), ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0x5u);

    commands[0].header.next = static_cast<s16>(sizeof(commands[0]));
    EXPECT_EQ(SendEvents(voice, &commands[0], commands.data(), sizeof(commands)),
              ORBIS_NGS2_ERROR_DETECTED_CIRCULAR_VOICE_CONTROL);

    commands[0].header.next = static_cast<s16>(sizeof(commands[0]));
    commands[0].eventId = 3;
    commands[1].header.next = 0;
    commands[1].header.size = sizeof(OrbisNgs2VoiceParamHeader) - 1;
    EXPECT_EQ(SendEvents(voice, &commands[0], commands.data(), sizeof(commands)),
              ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0x5u);
}

TEST_F(Ngs2RuntimeTest, VoiceControlTracesBoundedSamplerAndCallbackMetadata) {
    struct Commands {
        OrbisNgs2SamplerVoiceSetupParam setup{};
        OrbisNgs2SamplerVoiceWaveformBlocksParam waveform_blocks{};
        OrbisNgs2SamplerVoiceUserFxParam user_fx{};
        std::array<OrbisNgs2WaveformBlock, 2> blocks{};
    } commands;

    static_assert(offsetof(Commands, waveform_blocks) - offsetof(Commands, setup) <=
                  std::numeric_limits<s16>::max());
    static_assert(offsetof(Commands, user_fx) - offsetof(Commands, waveform_blocks) <=
                  std::numeric_limits<s16>::max());

    commands.setup.header = {
        sizeof(commands.setup),
        static_cast<s16>(offsetof(Commands, waveform_blocks) - offsetof(Commands, setup)),
        SamplerSetupControlId,
    };
    commands.setup.format = {
        .waveformType = 9,
        .numChannels = 6,
        .sampleRate = 48000,
        .configData = 0xDEADBEEF,
        .frameOffset = 1024,
        .frameMargin = 32,
    };
    commands.setup.flags = 5;

    commands.waveform_blocks.header = {
        sizeof(commands.waveform_blocks),
        static_cast<s16>(offsetof(Commands, user_fx) - offsetof(Commands, waveform_blocks)),
        SamplerWaveformBlocksControlId,
    };
    commands.waveform_blocks.data = reinterpret_cast<const void*>(0xDEADC0DE);
    commands.waveform_blocks.flags = 7;
    commands.waveform_blocks.numBlocks = commands.blocks.size();
    commands.waveform_blocks.aBlock = commands.blocks.data();
    commands.blocks[0] = {
        .dataOffset = 32,
        .dataSize = 4096,
        .numRepeats = 0,
        .numSkipSamples = 2,
        .numSamples = 2048,
        .reserved = 0,
        .userData = 0x1111222233334444ULL,
    };
    commands.blocks[1] = {
        .dataOffset = 64,
        .dataSize = 8192,
        .numRepeats = 3,
        .numSkipSamples = 4,
        .numSamples = 4096,
        .reserved = 0,
        .userData = 0x5555666677778888ULL,
    };

    commands.user_fx.header = {sizeof(commands.user_fx), 0, SamplerUserFxControlId};
    commands.user_fx.handler = DummyUserFxProcess;
    commands.user_fx.userData0 = 0xAAAAAAAAAAAAAAAAULL;
    commands.user_fx.userData1 = 0xBBBBBBBBBBBBBBBBULL;
    commands.user_fx.userData2 = 0xCCCCCCCCCCCCCCCCULL;

    ScopedAudioTrace trace;

    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);
    ReadWindow window{reinterpret_cast<uintptr_t>(&commands), sizeof(commands)};
    ASSERT_EQ(VoiceControl(voice, &commands.setup.header, ReadWindowBytes, &window), ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto sampler = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "sampler_setup"; });
    ASSERT_NE(sampler, lines.end());
    EXPECT_EQ(sampler->at("source_rack_id"), SamplerRackId);
    EXPECT_EQ(sampler->at("source_voice_index"), 0);
    EXPECT_EQ(sampler->at("waveform_type"), 9);
    EXPECT_EQ(sampler->at("channels"), 6);
    EXPECT_EQ(sampler->at("sample_rate"), 48000);
    EXPECT_EQ(sampler->at("codec_config"), 0xDEADBEEF);
    EXPECT_EQ(sampler->at("frame_offset"), 1024);
    EXPECT_EQ(sampler->at("frame_margin"), 32);
    EXPECT_EQ(sampler->at("flags"), 5);
    EXPECT_FALSE(sampler->at("atrac9_applicable"));
    EXPECT_FALSE(sampler->at("atrac9_config_valid"));
    EXPECT_FALSE(sampler->at("atrac9_declared_format_matches"));
    EXPECT_EQ(sampler->at("atrac9_superframe_bytes"), 0);

    const auto blocks = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "waveform_blocks"; });
    ASSERT_NE(blocks, lines.end());
    EXPECT_EQ(blocks->at("source_rack_id"), SamplerRackId);
    EXPECT_EQ(blocks->at("source_voice_index"), 0);
    EXPECT_EQ(blocks->at("flags"), 7);
    EXPECT_TRUE(blocks->at("data_present"));
    EXPECT_EQ(blocks->at("block_count"), 2);
    EXPECT_EQ(blocks->at("aggregate_data_bytes"), 12288);
    EXPECT_EQ(blocks->at("aggregate_samples"), 6144);
    EXPECT_EQ(blocks->at("aggregate_skip_samples"), 6);
    EXPECT_EQ(blocks->at("minimum_data_offset"), 32);
    EXPECT_EQ(blocks->at("maximum_data_end"), 8256);
    EXPECT_TRUE(blocks->at("data_range_arithmetic_valid"));
    EXPECT_TRUE(blocks->at("has_repeats"));
    EXPECT_TRUE(blocks->at("metadata_complete"));

    const auto callback =
        std::ranges::find_if(lines, [](const auto& line) { return line.at("event") == "user_fx"; });
    ASSERT_NE(callback, lines.end());
    EXPECT_EQ(callback->at("control_id"), SamplerUserFxControlId);
    EXPECT_TRUE(callback->at("callback_present"));

    const std::string contents = trace.ReadContents();
    EXPECT_EQ(contents.find("DEADC0DE"), std::string::npos);
    EXPECT_EQ(
        contents.find(std::to_string(reinterpret_cast<uintptr_t>(commands.waveform_blocks.data))),
        std::string::npos);
    EXPECT_EQ(contents.find("1111222233334444"), std::string::npos);
    EXPECT_EQ(contents.find("AAAAAAAAAAAAAAAA"), std::string::npos);
}

TEST_F(Ngs2RuntimeTest, VoiceControlTraceValidatesApplicableAtrac9SetupEndToEnd) {
    OrbisNgs2SamplerVoiceSetupParam command{};
    command.header = {sizeof(command), 0, SamplerSetupControlId};
    command.format = {
        .waveformType = Atrac9WaveformType,
        .numChannels = 6,
        .sampleRate = 48000,
        .configData = 0xFE761DF0,
    };

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);
    ReadWindow window{reinterpret_cast<uintptr_t>(&command), sizeof(command)};
    ASSERT_EQ(VoiceControl(voice, &command.header, ReadWindowBytes, &window), ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto sampler = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "sampler_setup"; });
    ASSERT_NE(sampler, lines.end());
    EXPECT_TRUE(sampler->at("atrac9_applicable"));
    EXPECT_TRUE(sampler->at("atrac9_config_valid"));
    EXPECT_TRUE(sampler->at("atrac9_declared_format_matches"));
    EXPECT_EQ(sampler->at("atrac9_frame_bytes"), 240);
    EXPECT_EQ(sampler->at("atrac9_frames_per_superframe"), 4);
    EXPECT_EQ(sampler->at("atrac9_frame_samples"), 256);
    EXPECT_EQ(sampler->at("atrac9_superframe_bytes"), 960);
    EXPECT_EQ(sampler->at("atrac9_superframe_samples"), 1024);
}

TEST_F(Ngs2RuntimeTest, VoiceControlTraceCapsDescriptorReadsWithoutChangingGuestResult) {
    OrbisNgs2SamplerVoiceWaveformBlocksParam command{};
    command.header = {sizeof(command), 0, SamplerWaveformBlocksControlId};
    command.numBlocks = 4097;
    command.aBlock = reinterpret_cast<const OrbisNgs2WaveformBlock*>(0xDEADC0DE);

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);
    CountingReadWindow window{
        .window = {reinterpret_cast<uintptr_t>(&command), sizeof(command)},
        .watched_address = reinterpret_cast<uintptr_t>(command.aBlock),
    };
    EXPECT_EQ(VoiceControl(voice, &command.header, CountingReadWindowBytes, &window), ORBIS_OK);
    EXPECT_EQ(window.watched_reads, 0);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto blocks = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "waveform_blocks"; });
    ASSERT_NE(blocks, lines.end());
    EXPECT_EQ(blocks->at("block_count"), 4097);
    EXPECT_FALSE(blocks->at("metadata_complete"));
    EXPECT_EQ(blocks->at("aggregate_data_bytes"), 0);
    EXPECT_EQ(blocks->at("aggregate_samples"), 0);
}

TEST_F(Ngs2RuntimeTest, VoiceControlWithTracingDisabledDoesNotReadNestedWaveformDescriptors) {
    ClearAudioTracePath();

    OrbisNgs2SamplerVoiceWaveformBlocksParam command{};
    command.header = {sizeof(command), 0, SamplerWaveformBlocksControlId};
    command.numBlocks = 1;
    command.aBlock = reinterpret_cast<const OrbisNgs2WaveformBlock*>(0xDEADC0DE);

    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);
    CountingReadWindow window{
        .window = {reinterpret_cast<uintptr_t>(&command), sizeof(command)},
        .watched_address = reinterpret_cast<uintptr_t>(command.aBlock),
    };
    EXPECT_EQ(VoiceControl(voice, &command.header, CountingReadWindowBytes, &window), ORBIS_OK);
    EXPECT_EQ(window.watched_reads, 0);
}

TEST_F(Ngs2RuntimeTest, VoiceControlTraceSharesDescriptorBudgetAcrossTheWholeCommandChain) {
    struct Commands {
        std::array<OrbisNgs2SamplerVoiceWaveformBlocksParam, 2> params{};
        std::array<OrbisNgs2WaveformBlock, 4097> blocks{};
    };
    auto commands = std::make_unique<Commands>();
    commands->params[0].header = {
        sizeof(commands->params[0]),
        static_cast<s16>(sizeof(commands->params[0])),
        SamplerWaveformBlocksControlId,
    };
    commands->params[0].numBlocks = 4096;
    commands->params[0].aBlock = commands->blocks.data();
    commands->params[1].header = {
        sizeof(commands->params[1]),
        0,
        SamplerWaveformBlocksControlId,
    };
    commands->params[1].numBlocks = 1;
    commands->params[1].aBlock = &commands->blocks.back();

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);
    CountingReadWindow window{
        .window = {reinterpret_cast<uintptr_t>(commands.get()), sizeof(Commands)},
        .watched_address = reinterpret_cast<uintptr_t>(&commands->blocks.back()),
    };
    EXPECT_EQ(VoiceControl(voice, &commands->params[0].header, CountingReadWindowBytes, &window),
              ORBIS_OK);
    EXPECT_EQ(window.watched_reads, 0);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto exhausted = std::ranges::find_if(lines, [](const auto& line) {
        return line.at("event") == "waveform_blocks" && line.at("block_count") == 1;
    });
    ASSERT_NE(exhausted, lines.end());
    EXPECT_FALSE(exhausted->at("metadata_complete"));
    EXPECT_EQ(exhausted->at("aggregate_data_bytes"), 0);
    EXPECT_EQ(exhausted->at("aggregate_samples"), 0);
}

TEST_F(Ngs2RuntimeTest, VoiceControlTraceUnreadableDescriptorIsIncompleteButNonFatal) {
    OrbisNgs2SamplerVoiceWaveformBlocksParam command{};
    command.header = {sizeof(command), 0, SamplerWaveformBlocksControlId};
    command.numBlocks = 1;
    command.aBlock = reinterpret_cast<const OrbisNgs2WaveformBlock*>(0xDEADC0DE);

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);
    CountingReadWindow window{
        .window = {reinterpret_cast<uintptr_t>(&command), sizeof(command)},
        .watched_address = reinterpret_cast<uintptr_t>(command.aBlock),
    };
    EXPECT_EQ(VoiceControl(voice, &command.header, CountingReadWindowBytes, &window), ORBIS_OK);
    EXPECT_EQ(window.watched_reads, 1);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto blocks = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "waveform_blocks"; });
    ASSERT_NE(blocks, lines.end());
    EXPECT_FALSE(blocks->at("metadata_complete"));
    EXPECT_EQ(blocks->at("aggregate_data_bytes"), 0);
    EXPECT_EQ(blocks->at("aggregate_samples"), 0);
}

TEST_F(Ngs2RuntimeTest, VoiceControlTraceDescriptorAddressOverflowIsIncompleteButNonFatal) {
    OrbisNgs2SamplerVoiceWaveformBlocksParam command{};
    command.header = {sizeof(command), 0, SamplerWaveformBlocksControlId};
    command.numBlocks = 2;
    constexpr uintptr_t DescriptorAddress = std::numeric_limits<uintptr_t>::max() - 15;
    command.aBlock = reinterpret_cast<const OrbisNgs2WaveformBlock*>(DescriptorAddress);

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);
    OverflowDescriptorReadContext context{
        .command_window = {reinterpret_cast<uintptr_t>(&command), sizeof(command)},
        .descriptor_address = DescriptorAddress,
    };
    EXPECT_EQ(VoiceControl(voice, &command.header, OverflowDescriptorReadBytes, &context),
              ORBIS_OK);
    EXPECT_EQ(context.descriptor_reads, 0);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto blocks = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "waveform_blocks"; });
    ASSERT_NE(blocks, lines.end());
    EXPECT_FALSE(blocks->at("metadata_complete"));
}

TEST_F(Ngs2RuntimeTest, VoiceControlTraceRecognizesSubmixerUserFxAbi) {
    OrbisNgs2SubmixerVoiceUserFxParam command{};
    command.header = {sizeof(command), 0, SubmixerUserFxControlId};
    command.handler = DummyUserFxProcess;
    command.userData0 = 0xAAAAAAAAAAAAAAAAULL;

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto rack = CreateRackForId(system.handle, 0x2000, 1);
    const auto voice = GetVoice(rack.handle, 0);
    ReadWindow window{reinterpret_cast<uintptr_t>(&command), sizeof(command)};
    EXPECT_EQ(VoiceControl(voice, &command.header, ReadWindowBytes, &window), ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto callback =
        std::ranges::find_if(lines, [](const auto& line) { return line.at("event") == "user_fx"; });
    ASSERT_NE(callback, lines.end());
    EXPECT_EQ(callback->at("control_id"), SubmixerUserFxControlId);
    EXPECT_TRUE(callback->at("callback_present"));
    EXPECT_EQ(trace.ReadContents().find("AAAAAAAAAAAAAAAA"), std::string::npos);
}

TEST_F(Ngs2RuntimeTest, VoiceControlTraceResolvesPrivacySafeRoutingTopology) {
    struct RoutingCommands {
        OrbisNgs2VoiceMatrixLevelsParam matrix_levels{};
        OrbisNgs2VoicePortVolumeParam port_volume{};
        OrbisNgs2VoicePortMatrixParam port_matrix{};
        OrbisNgs2VoicePatchParam patch{};
    } commands;

    static_assert(offsetof(RoutingCommands, port_volume) -
                      offsetof(RoutingCommands, matrix_levels) <=
                  std::numeric_limits<s16>::max());
    static_assert(offsetof(RoutingCommands, port_matrix) - offsetof(RoutingCommands, port_volume) <=
                  std::numeric_limits<s16>::max());
    static_assert(offsetof(RoutingCommands, patch) - offsetof(RoutingCommands, port_matrix) <=
                  std::numeric_limits<s16>::max());

    constexpr uintptr_t LevelsAddress = 0xDEADC0DE;
    commands.matrix_levels.header = {
        sizeof(commands.matrix_levels),
        static_cast<s16>(offsetof(RoutingCommands, port_volume) -
                         offsetof(RoutingCommands, matrix_levels)),
        VoiceMatrixLevelsControlId,
    };
    commands.matrix_levels.matrixId = 9;
    commands.matrix_levels.numLevels = ORBIS_NGS2_MAX_MATRIX_LEVELS;
    commands.matrix_levels.aLevel = reinterpret_cast<const float*>(LevelsAddress);
    commands.port_volume.header = {
        sizeof(commands.port_volume),
        static_cast<s16>(offsetof(RoutingCommands, port_matrix) -
                         offsetof(RoutingCommands, port_volume)),
        VoicePortVolumeControlId,
    };
    commands.port_volume.port = 6;
    commands.port_volume.level = 1.0F;
    commands.port_matrix.header = {
        sizeof(commands.port_matrix),
        static_cast<s16>(offsetof(RoutingCommands, patch) - offsetof(RoutingCommands, port_matrix)),
        VoicePortMatrixControlId,
    };
    commands.port_matrix.port = 6;
    commands.port_matrix.matrixId = -1;
    commands.patch.header = {sizeof(commands.patch), 0, VoicePatchControlId};
    commands.patch.port = 6;
    commands.patch.destInputId = 4;

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto sampler = CreateRackForId(system.handle, 0x1000, 2);
    auto submixer = CreateRackForId(system.handle, 0x2000, 2);
    auto mastering = CreateRackForId(system.handle, 0x3000, 3);
    const auto source_voice = GetVoice(sampler.handle, 1);
    const auto destination_voice = GetVoice(submixer.handle, 0);
    const auto mastering_voice = GetVoice(mastering.handle, 2);
    commands.patch.destHandle = destination_voice;

    CountingReadWindow window{
        .window = {reinterpret_cast<uintptr_t>(&commands), sizeof(commands)},
        .watched_address = LevelsAddress,
    };
    ASSERT_EQ(VoiceControl(source_voice, &commands.matrix_levels.header, CountingReadWindowBytes,
                           &window),
              ORBIS_OK);
    EXPECT_EQ(window.watched_reads, 0);

    OrbisNgs2MasteringVoiceOutputParam output{};
    output.header = {sizeof(output), 0, MasteringOutputControlId};
    output.outputId = 2;
    ReadWindow output_window{reinterpret_cast<uintptr_t>(&output), sizeof(output)};
    ASSERT_EQ(VoiceControl(mastering_voice, &output.header, ReadWindowBytes, &output_window),
              ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto levels = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "voice_matrix_levels"; });
    ASSERT_NE(levels, lines.end());
    EXPECT_EQ(levels->at("source_rack_id"), 0x1000);
    EXPECT_EQ(levels->at("source_voice_index"), 1);
    EXPECT_EQ(levels->at("matrix_id"), 9);
    EXPECT_EQ(levels->at("level_count"), ORBIS_NGS2_MAX_MATRIX_LEVELS);
    EXPECT_TRUE(levels->at("levels_present"));

    const auto volume = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "voice_port_volume"; });
    ASSERT_NE(volume, lines.end());
    EXPECT_EQ(volume->at("source_rack_id"), 0x1000);
    EXPECT_EQ(volume->at("source_voice_index"), 1);
    EXPECT_EQ(volume->at("port"), 6);
    EXPECT_EQ(volume->at("level_shape"), "unity");

    const auto matrix = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "voice_port_matrix"; });
    ASSERT_NE(matrix, lines.end());
    EXPECT_EQ(matrix->at("port"), 6);
    EXPECT_EQ(matrix->at("matrix_id"), -1);

    const auto patch = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "voice_patch"; });
    ASSERT_NE(patch, lines.end());
    EXPECT_EQ(patch->at("source_rack_id"), 0x1000);
    EXPECT_EQ(patch->at("source_voice_index"), 1);
    EXPECT_EQ(patch->at("source_port"), 6);
    EXPECT_EQ(patch->at("destination_input_id"), 4);
    EXPECT_EQ(patch->at("destination_rack_id"), 0x2000);
    EXPECT_EQ(patch->at("destination_voice_index"), 0);
    EXPECT_TRUE(patch->at("destination_resolved"));
    EXPECT_TRUE(patch->at("destination_present"));

    const auto mastering_output = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "mastering_output"; });
    ASSERT_NE(mastering_output, lines.end());
    EXPECT_EQ(mastering_output->at("source_rack_id"), 0x3000);
    EXPECT_EQ(mastering_output->at("source_voice_index"), 2);
    EXPECT_EQ(mastering_output->at("output_id"), 2);

    const std::string contents = trace.ReadContents();
    EXPECT_EQ(contents.find(std::to_string(source_voice)), std::string::npos);
    EXPECT_EQ(contents.find(std::to_string(destination_voice)), std::string::npos);
    EXPECT_EQ(contents.find(std::to_string(mastering_voice)), std::string::npos);
    EXPECT_EQ(contents.find(std::to_string(LevelsAddress)), std::string::npos);
}

TEST_F(Ngs2RuntimeTest, VoiceControlTraceLeavesInvalidPatchDestinationUnresolvedAndAccepted) {
    constexpr OrbisNgs2Handle InvalidDestination = 0x1111222233334444ULL;
    OrbisNgs2VoicePatchParam command{};
    command.header = {sizeof(command), 0, VoicePatchControlId};
    command.port = 7;
    command.destInputId = 5;
    command.destHandle = InvalidDestination;

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto sampler = CreateRackForId(system.handle, 0x1000, 2);
    const auto source_voice = GetVoice(sampler.handle, 1);
    ReadWindow window{reinterpret_cast<uintptr_t>(&command), sizeof(command)};
    EXPECT_EQ(VoiceControl(source_voice, &command.header, ReadWindowBytes, &window), ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto patch = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "voice_patch"; });
    ASSERT_NE(patch, lines.end());
    EXPECT_EQ(patch->at("source_rack_id"), 0x1000);
    EXPECT_EQ(patch->at("source_voice_index"), 1);
    EXPECT_EQ(patch->at("source_port"), 7);
    EXPECT_EQ(patch->at("destination_input_id"), 5);
    EXPECT_EQ(patch->at("destination_rack_id"), 0);
    EXPECT_EQ(patch->at("destination_voice_index"), 0);
    EXPECT_FALSE(patch->at("destination_resolved"));
    EXPECT_TRUE(patch->at("destination_present"));
    EXPECT_EQ(trace.ReadContents().find(std::to_string(InvalidDestination)), std::string::npos);
}

TEST_F(Ngs2RuntimeTest, VoiceControlTraceDistinguishesDisconnectFromUnknownDestination) {
    OrbisNgs2VoicePatchParam command{};
    command.header = {sizeof(command), 0, VoicePatchControlId};
    command.port = 0;
    command.destInputId = 0;
    command.destHandle = 0;

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto sampler = CreateRackForId(system.handle, 0x1000, 1);
    const auto source_voice = GetVoice(sampler.handle, 0);
    ReadWindow window{reinterpret_cast<uintptr_t>(&command), sizeof(command)};
    EXPECT_EQ(VoiceControl(source_voice, &command.header, ReadWindowBytes, &window), ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto patch = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "voice_patch"; });
    ASSERT_NE(patch, lines.end());
    EXPECT_FALSE(patch->at("destination_present"));
    EXPECT_FALSE(patch->at("destination_resolved"));
    EXPECT_EQ(patch->at("destination_rack_id"), 0);
    EXPECT_EQ(patch->at("destination_voice_index"), 0);
}

TEST_F(Ngs2RuntimeTest, VoicePatchStateClassifiesLiveDestinationLayoutWithoutRawIdentity) {
    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto sampler = CreateRackForId(system.handle, SamplerRackId, 1);
    auto destination = CreateRackForId(system.handle, 0x2000, 2);
    const auto source_voice = GetVoice(sampler.handle, 0);
    const OrbisNgs2Handle destination_voice =
        destination.handle + TestRackBufferHeaderSize + TestVoiceTokenStride;

    const auto send_patch = [&](u32 port, OrbisNgs2Handle destination_handle) {
        OrbisNgs2VoicePatchParam patch{};
        patch.header = {sizeof(patch), 0, VoicePatchControlId};
        patch.port = port;
        patch.destHandle = destination_handle;
        ReadWindow window{reinterpret_cast<uintptr_t>(&patch), sizeof(patch)};
        ASSERT_EQ(VoiceControl(source_voice, &patch.header, ReadWindowBytes, &window), ORBIS_OK);
    };

    send_patch(0, system.handle);
    send_patch(1, destination.handle);
    send_patch(2, destination_voice);
    send_patch(3, destination_voice + 1);
    send_patch(4, destination.handle + 8);
    send_patch(5, 0x1111222233334444ULL);

    OrbisNgs2Handle returned_voice = 0;
    ASSERT_EQ(RackGetVoiceHandle(destination.handle, 1, &returned_voice), ORBIS_OK);
    ASSERT_EQ(returned_voice, destination_voice);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto state_for = [&](u32 port, const char* phase) -> const nlohmann::json& {
        const auto state = std::ranges::find_if(lines, [&](const auto& line) {
            return line.at("event") == "voice_patch_state" && line.at("source_port") == port &&
                   line.at("phase") == phase;
        });
        EXPECT_NE(state, lines.end()) << "port=" << port << " phase=" << phase;
        return *state;
    };

    EXPECT_EQ(state_for(0, "committed").at("destination_exact_match"), "system");
    const auto& rack = state_for(1, "committed");
    EXPECT_EQ(rack.at("destination_exact_match"), "rack");
    EXPECT_TRUE(rack.at("destination_inside_live_rack"));
    EXPECT_EQ(rack.at("destination_offset_class"), "rack_base");

    const auto& voice = state_for(2, "committed");
    EXPECT_EQ(voice.at("destination_exact_match"), "voice");
    EXPECT_TRUE(voice.at("destination_inside_live_rack"));
    EXPECT_EQ(voice.at("destination_offset_class"), "voice_slot_aligned");
    EXPECT_TRUE(voice.at("destination_candidate_voice_index_valid"));
    EXPECT_EQ(voice.at("destination_candidate_voice_index"), 1);
    EXPECT_FALSE(voice.at("destination_previously_returned"));
    EXPECT_GT(voice.at("destination_live_generation"), 0);

    const auto& interior = state_for(3, "committed");
    EXPECT_EQ(interior.at("destination_exact_match"), "none");
    EXPECT_TRUE(interior.at("destination_inside_live_rack"));
    EXPECT_EQ(interior.at("destination_offset_class"), "voice_slot_interior");
    EXPECT_TRUE(interior.at("destination_candidate_voice_index_valid"));
    EXPECT_EQ(interior.at("destination_candidate_voice_index"), 1);

    const auto& header = state_for(4, "committed");
    EXPECT_EQ(header.at("destination_offset_class"), "rack_header");
    EXPECT_FALSE(header.at("destination_candidate_voice_index_valid"));

    const auto& outside = state_for(5, "committed");
    EXPECT_FALSE(outside.at("destination_inside_live_rack"));
    EXPECT_EQ(outside.at("destination_offset_class"), "outside_live_rack");
    EXPECT_EQ(outside.at("destination_tombstone_kind"), "none");

    const auto& returned = state_for(2, "render_transition");
    EXPECT_TRUE(returned.at("destination_previously_returned"));
    EXPECT_EQ(trace.ReadContents().find(std::to_string(destination.handle)), std::string::npos);
    EXPECT_EQ(trace.ReadContents().find(std::to_string(destination_voice)), std::string::npos);
}

TEST_F(Ngs2RuntimeTest, VoicePatchStateTracksDestroyedAndRecreatedDestinationGeneration) {
    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto sampler = CreateRackForId(system.handle, SamplerRackId, 1);
    auto destination = CreateRackForId(system.handle, 0x2000, 1);
    const auto source_voice = GetVoice(sampler.handle, 0);
    const auto destination_voice = GetVoice(destination.handle, 0);

    OrbisNgs2VoicePatchParam patch{};
    patch.header = {sizeof(patch), 0, VoicePatchControlId};
    patch.port = 0;
    patch.destHandle = destination_voice;
    ReadWindow window{reinterpret_cast<uintptr_t>(&patch), sizeof(patch)};
    ASSERT_EQ(VoiceControl(source_voice, &patch.header, ReadWindowBytes, &window), ORBIS_OK);
    ASSERT_EQ(RackDestroy(destination.handle, nullptr), ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    ASSERT_EQ(RackCreate(system.handle, 0x2000, &destination.option, &destination.buffer_info,
                         nullptr, &destination.handle),
              ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    std::vector<nlohmann::json> states;
    std::ranges::copy_if(lines, std::back_inserter(states), [](const auto& line) {
        return line.at("event") == "voice_patch_state" && line.at("source_port") == 0;
    });
    ASSERT_EQ(states.size(), 3);
    EXPECT_EQ(states[0].at("destination_exact_match"), "voice");
    EXPECT_TRUE(states[0].at("destination_previously_returned"));
    const u64 original_voice_generation = states[0].at("destination_live_generation");
    ASSERT_NE(original_voice_generation, 0u);

    EXPECT_EQ(states[1].at("destination_exact_match"), "none");
    EXPECT_EQ(states[1].at("destination_tombstone_kind"), "voice");
    EXPECT_EQ(states[1].at("destination_tombstone_generation"), original_voice_generation);
    EXPECT_TRUE(states[1].at("destination_generation_changed_since_commit"));

    EXPECT_EQ(states[2].at("destination_exact_match"), "voice");
    EXPECT_NE(states[2].at("destination_live_generation"), original_voice_generation);
    EXPECT_EQ(states[2].at("destination_tombstone_generation"), original_voice_generation);
    EXPECT_TRUE(states[2].at("destination_generation_changed_since_commit"));
    EXPECT_TRUE(states[2].at("destination_previously_returned"));

    std::vector<nlohmann::json> lifecycle;
    std::ranges::copy_if(lines, std::back_inserter(lifecycle), [](const auto& line) {
        return line.at("event") == "rack_lifecycle" && line.at("rack_id") == 0x2000;
    });
    ASSERT_EQ(lifecycle.size(), 3);
    EXPECT_EQ(lifecycle[0].at("phase"), "created");
    EXPECT_FALSE(lifecycle[0].at("reused_identity"));
    EXPECT_EQ(lifecycle[1].at("phase"), "destroyed");
    EXPECT_EQ(lifecycle[1].at("generation"), lifecycle[0].at("generation"));
    EXPECT_EQ(lifecycle[2].at("phase"), "created");
    EXPECT_TRUE(lifecycle[2].at("reused_identity"));
    EXPECT_EQ(lifecycle[2].at("previous_generation"), lifecycle[0].at("generation"));
    EXPECT_NE(lifecycle[2].at("generation"), lifecycle[0].at("generation"));
}

TEST_F(Ngs2RuntimeTest, DestroyedDestinationTombstonesAreBoundedAndEvictOldest) {
    struct alignas(8) SystemStorage {
        std::array<u8, 64> bytes{};
    };

    ScopedAudioTrace trace;
    auto source_system = CreateSystem();
    auto sampler = CreateRackForId(source_system.handle, SamplerRackId, 1);
    const auto source_voice = GetVoice(sampler.handle, 0);
    std::vector<SystemStorage> storage(MaxDestroyedHandleTombstones + 1);
    std::vector<OrbisNgs2Handle> destroyed_handles;
    destroyed_handles.reserve(storage.size());
    for (auto& slot : storage) {
        OrbisNgs2ContextBufferInfo info{};
        info.hostBuffer = slot.bytes.data();
        info.hostBufferSize = slot.bytes.size();
        OrbisNgs2Handle handle = 0;
        ASSERT_EQ(SystemSetup(nullptr, &info, nullptr, &handle), ORBIS_OK);
        destroyed_handles.push_back(handle);
        ASSERT_EQ(SystemCleanup(handle, nullptr), ORBIS_OK);
    }

    const auto send_patch = [&](u32 port, OrbisNgs2Handle destination_handle) {
        OrbisNgs2VoicePatchParam patch{};
        patch.header = {sizeof(patch), 0, VoicePatchControlId};
        patch.port = port;
        patch.destHandle = destination_handle;
        ReadWindow window{reinterpret_cast<uintptr_t>(&patch), sizeof(patch)};
        ASSERT_EQ(VoiceControl(source_voice, &patch.header, ReadWindowBytes, &window), ORBIS_OK);
    };
    send_patch(0, destroyed_handles.front());
    send_patch(1, destroyed_handles.back());
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto state_for = [&](u32 port) -> const nlohmann::json& {
        const auto state = std::ranges::find_if(lines, [&](const auto& line) {
            return line.at("event") == "voice_patch_state" && line.at("source_port") == port;
        });
        EXPECT_NE(state, lines.end());
        return *state;
    };
    EXPECT_EQ(state_for(0).at("destination_tombstone_kind"), "none");
    EXPECT_EQ(state_for(0).at("destination_tombstone_generation"), 0);
    EXPECT_EQ(state_for(1).at("destination_tombstone_kind"), "system");
    EXPECT_GT(state_for(1).at("destination_tombstone_generation"), 0);
}

TEST_F(Ngs2RuntimeTest, VoicePatchStateTracksLateResolutionAndDestinationDestruction) {
    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto sampler = CreateRackForId(system.handle, SamplerRackId, 1);
    const auto source_voice = GetVoice(sampler.handle, 0);

    auto destination = CreateRackForId(system.handle, 0x2000, 1);
    const auto future_destination_voice = GetVoice(destination.handle, 0);
    ASSERT_EQ(RackDestroy(destination.handle, nullptr), ORBIS_OK);

    OrbisNgs2VoicePatchParam patch{};
    patch.header = {sizeof(patch), 0, VoicePatchControlId};
    patch.port = 0;
    patch.destInputId = 0;
    patch.destHandle = future_destination_voice;
    ReadWindow patch_window{reinterpret_cast<uintptr_t>(&patch), sizeof(patch)};
    ASSERT_EQ(VoiceControl(source_voice, &patch.header, ReadWindowBytes, &patch_window), ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);

    ASSERT_EQ(RackCreate(system.handle, 0x2000, &destination.option, &destination.buffer_info,
                         nullptr, &destination.handle),
              ORBIS_OK);
    EXPECT_EQ(GetVoice(destination.handle, 0), future_destination_voice);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    ASSERT_EQ(RackDestroy(destination.handle, nullptr), ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    std::vector<nlohmann::json> states;
    std::ranges::copy_if(lines, std::back_inserter(states),
                         [](const auto& line) { return line.at("event") == "voice_patch_state"; });
    ASSERT_EQ(states.size(), 3);
    EXPECT_EQ(states[0].at("phase"), "committed");
    EXPECT_TRUE(states[0].at("destination_present"));
    EXPECT_FALSE(states[0].at("destination_resolved"));
    EXPECT_EQ(states[1].at("phase"), "render_transition");
    EXPECT_TRUE(states[1].at("destination_resolved"));
    EXPECT_TRUE(states[1].at("destination_same_system"));
    EXPECT_EQ(states[1].at("destination_rack_id"), 0x2000);
    EXPECT_EQ(states[1].at("destination_voice_index"), 0);
    EXPECT_EQ(states[2].at("phase"), "render_transition");
    EXPECT_FALSE(states[2].at("destination_resolved"));
}

TEST_F(Ngs2RuntimeTest, InvalidVoiceControlChainDoesNotCommitOrTracePatch) {
    struct Commands {
        OrbisNgs2SamplerVoiceSetupParam setup{};
        OrbisNgs2SamplerVoiceWaveformBlocksParam blocks{};
        OrbisNgs2VoicePatchParam patch{};
        OrbisNgs2VoiceParamHeader invalid{};
        OrbisNgs2WaveformBlock block{};
    } commands;
    commands.setup.header = {
        sizeof(commands.setup),
        static_cast<s16>(offsetof(Commands, blocks) - offsetof(Commands, setup)),
        SamplerSetupControlId,
    };
    commands.setup.format = {
        .waveformType = Atrac9WaveformType,
        .numChannels = 6,
        .sampleRate = 48000,
        .configData = 0xFE761DF0,
    };
    commands.blocks.header = {
        sizeof(commands.blocks),
        static_cast<s16>(offsetof(Commands, patch) - offsetof(Commands, blocks)),
        SamplerWaveformBlocksControlId,
    };
    commands.blocks.data = reinterpret_cast<const void*>(0xDEADC0DE);
    commands.blocks.numBlocks = 1;
    commands.blocks.aBlock = &commands.block;
    commands.block = {.dataSize = 960, .numSamples = 1024};
    commands.patch.header = {
        sizeof(commands.patch),
        static_cast<s16>(offsetof(Commands, invalid) - offsetof(Commands, patch)),
        VoicePatchControlId,
    };
    commands.patch.port = 0;
    commands.patch.destInputId = 0;
    commands.patch.destHandle = 0x1111222233334444ULL;
    commands.invalid = {sizeof(OrbisNgs2VoiceParamHeader) - 1, 0, VoicePortVolumeControlId};

    ScopedAudioTrace trace;
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle, 1);
    const auto voice = GetVoice(rack.handle, 0);
    ReadWindow window{reinterpret_cast<uintptr_t>(&commands), sizeof(commands)};
    EXPECT_EQ(VoiceControl(voice, &commands.setup.header, ReadWindowBytes, &window),
              ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    EXPECT_EQ(std::ranges::count_if(lines,
                                    [](const auto& line) {
                                        return line.at("event") == "voice_control" ||
                                               line.at("event") == "sampler_setup" ||
                                               line.at("event") == "waveform_blocks" ||
                                               line.at("event") == "voice_patch" ||
                                               line.at("event") == "voice_patch_state";
                                    }),
              0);
}

TEST_F(Ngs2RuntimeTest, SamplerPlayAndStateRemainVisibleAfterGenericTraceSaturation) {
    ScopedAudioTrace trace;
    for (u32 index = 0; index < 100; ++index) {
        AudioPipelineTrace::RecordVoiceControl({index, 12, 0, false, 0, true, 0x2000, index, true});
    }

    auto system = CreateSystem();
    auto rack = CreateRack(system.handle, 1);
    const auto voice = GetVoice(rack.handle, 0);
    for (u32 index = 0; index < 100; ++index) {
        EXPECT_EQ(GetVoiceFlags(voice), 0u);
    }
    OrbisNgs2VoiceEventParam play{{sizeof(play), 0, VoiceEventControlId}, 0};
    ReadWindow window{reinterpret_cast<uintptr_t>(&play), sizeof(play)};
    ASSERT_EQ(VoiceControl(voice, &play.header, ReadWindowBytes, &window), ORBIS_OK);
    ASSERT_EQ(SystemRender(system.handle, nullptr, 0), ORBIS_OK);
    EXPECT_EQ(GetVoiceFlags(voice), 0x3u);
    trace.Finalize();

    const auto lines = trace.ReadLines();
    const auto sampler_event = std::ranges::find_if(
        lines, [](const auto& line) { return line.at("event") == "sampler_event"; });
    ASSERT_NE(sampler_event, lines.end());
    EXPECT_EQ(sampler_event->at("source_voice_index"), 0);
    EXPECT_EQ(sampler_event->at("event_id"), 0);
    EXPECT_TRUE(sampler_event->at("accepted"));

    const auto transition = std::ranges::find_if(lines, [](const auto& line) {
        return line.at("event") == "sampler_voice_state" &&
               line.at("observation") == "render_transition";
    });
    ASSERT_NE(transition, lines.end());
    EXPECT_EQ(transition->at("previous_state"), "empty");
    EXPECT_EQ(transition->at("current_state"), "playing");

    const auto poll = std::ranges::find_if(lines, [](const auto& line) {
        return line.at("event") == "sampler_voice_state" && line.at("observation") == "poll" &&
               line.at("current_state") == "playing";
    });
    ASSERT_NE(poll, lines.end());
    EXPECT_EQ(poll->at("current_state"), "playing");
}

TEST_F(Ngs2RuntimeTest, SystemRenderClearsCompleteOutputBuffers) {
    auto system = CreateSystem();
    std::array<float, 32> output;
    output.fill(1.0f);
    OrbisNgs2RenderBufferInfo info{output.data(), sizeof(output), 0, 2};

    ASSERT_EQ(SystemRender(system.handle, &info, 1), ORBIS_OK);
    for (const float sample : output) {
        EXPECT_EQ(sample, 0.0f);
    }
    EXPECT_EQ(SystemRender(0, &info, 1), ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE);
    EXPECT_EQ(SystemRender(system.handle, nullptr, 1), ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO);
}

TEST_F(Ngs2RuntimeTest, DestroyInvalidatesDescendantHandles) {
    auto system = CreateSystem();
    auto rack = CreateRack(system.handle);
    const auto voice = GetVoice(rack.handle, 0);

    EXPECT_EQ(RackDestroy(rack.handle, nullptr), ORBIS_OK);
    u32 flags = 0;
    EXPECT_EQ(VoiceGetStateFlags(voice, &flags), ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE);
    OrbisNgs2Handle ignored = 0;
    EXPECT_EQ(RackGetVoiceHandle(rack.handle, 0, &ignored), ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE);

    auto second_rack = CreateRack(system.handle);
    const auto second_voice = GetVoice(second_rack.handle, 0);
    EXPECT_EQ(SystemCleanup(system.handle, nullptr), ORBIS_OK);
    EXPECT_EQ(VoiceGetStateFlags(second_voice, &flags), ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE);
    EXPECT_EQ(RackGetVoiceHandle(second_rack.handle, 0, &ignored),
              ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE);
}

} // namespace
} // namespace Libraries::Ngs2
