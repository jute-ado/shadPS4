// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <compare>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "core/libraries/kernel/threads/pthread.h"

namespace Libraries::Ngs2 {

struct OrbisNgs2RackOption;
struct OrbisNgs2BufferAllocator;
struct OrbisNgs2RenderBufferInfo;
struct OrbisNgs2VoiceParamHeader;
struct VoiceDecodeRuntime;

static const int ORBIS_NGS2_SYSTEM_NAME_LENGTH = 16;
static const int ORBIS_NGS2_RACK_NAME_LENGTH = 16;

typedef uintptr_t OrbisNgs2Handle;

struct OrbisNgs2ContextBufferInfo {
    void* hostBuffer;
    size_t hostBufferSize;
    uintptr_t reserved[5];
    uintptr_t userData;
};

struct OrbisNgs2SystemOption {
    size_t size;
    char name[ORBIS_NGS2_SYSTEM_NAME_LENGTH];

    u32 flags;
    u32 maxGrainSamples;
    u32 numGrainSamples;
    u32 sampleRate;
    u32 aReserved[6];
};

constexpr u32 OrbisNgs2DefaultMaxGrainSamples = 512;
constexpr u32 OrbisNgs2DefaultNumGrainSamples = 256;
constexpr u32 OrbisNgs2DefaultSampleRate = 48000;
constexpr size_t MaxDestroyedHandleTombstones = 256;

struct SystemRuntimeSnapshot {
    u32 max_grain_samples{};
    u32 num_grain_samples{};
    u32 sample_rate{};
    u64 generation{};
};

struct WaveformFormatSnapshot {
    u32 waveform_type{};
    u32 num_channels{};
    u32 sample_rate{};
    u32 config_data{};
    u32 frame_offset{};
    u32 frame_margin{};

    auto operator<=>(const WaveformFormatSnapshot&) const = default;
};

struct WaveformBlockSnapshot {
    u32 data_offset{};
    u32 data_size{};
    u32 num_repeats{};
    u32 num_skip_samples{};
    u32 num_samples{};

    auto operator<=>(const WaveformBlockSnapshot&) const = default;
};

struct SamplerControlSnapshot {
    std::optional<WaveformFormatSnapshot> format;
    u32 setup_flags{};
    bool waveform_configured{};
    uintptr_t guest_data_token{};
    u32 waveform_flags{};
    std::vector<WaveformBlockSnapshot> waveform_blocks;
    bool user_fx_configured{};
    bool user_fx_enabled{};

    bool operator==(const SamplerControlSnapshot&) const = default;
};

struct VoiceMatrixSnapshot {
    u32 matrix_id{};
    std::vector<float> levels;

    bool operator==(const VoiceMatrixSnapshot&) const = default;
};

struct VoicePortSnapshot {
    u32 port{};
    bool volume_configured{};
    float volume{1.0F};
    bool matrix_configured{};
    s32 matrix_id{-1};

    auto operator<=>(const VoicePortSnapshot&) const = default;
};

enum class VoiceRouteEndpointKind : u8 {
    None,
    Rack,
    Voice,
};

struct VoiceRouteSnapshot {
    u32 source_port{};
    u32 destination_input_id{};
    VoiceRouteEndpointKind endpoint_kind{};
    u32 destination_rack_id{};
    bool destination_voice_index_valid{};
    u32 destination_voice_index{};
    u64 destination_generation{};
    bool connected{};

    auto operator<=>(const VoiceRouteSnapshot&) const = default;
};

struct SubmixerControlSnapshot {
    bool setup_configured{};
    u32 num_io_channels{};
    u32 flags{};
    bool user_fx_configured{};
    bool user_fx_enabled{};

    auto operator<=>(const SubmixerControlSnapshot&) const = default;
};

struct MasteringControlSnapshot {
    bool setup_configured{};
    u32 num_input_channels{};
    u32 setup_flags{};
    bool matrix_configured{};
    u32 matrix_type{};
    u32 matrix_input_channels{};
    u32 matrix_output_channels{};
    std::vector<float> matrix_levels;
    bool gain_configured{};
    float fbw_level{1.0F};
    float lfe_level{1.0F};
    bool output_configured{};
    u32 output_id{};

    auto operator<=>(const MasteringControlSnapshot&) const = default;
};

struct VoiceControlSnapshot {
    u32 rack_id{};
    u32 voice_index{};
    u64 system_generation{};
    u64 rack_generation{};
    u64 voice_generation{};
    u64 render_state_revision{};
    u32 max_inputs{};
    u32 max_matrices{};
    u32 max_ports{};
    std::optional<SamplerControlSnapshot> sampler;
    std::vector<VoiceMatrixSnapshot> matrices;
    std::vector<VoicePortSnapshot> ports;
    std::vector<VoiceRouteSnapshot> routes;
    std::optional<SubmixerControlSnapshot> submixer;
    std::optional<MasteringControlSnapshot> mastering;

    bool operator==(const VoiceControlSnapshot&) const = default;
};

enum class VoiceRenderPlaybackState : u8 {
    Empty,
    Playing,
    Paused,
    Stopped,
};

struct VoiceRenderCommitToken {
    OrbisNgs2Handle system_handle{};
    OrbisNgs2Handle rack_handle{};
    OrbisNgs2Handle voice_handle{};
    u32 rack_id{};
    u32 voice_index{};
    u64 system_generation{};
    u64 rack_generation{};
    u64 voice_generation{};
    u64 render_state_revision{};

    auto operator<=>(const VoiceRenderCommitToken&) const = default;
};

struct AppliedSamplerVoiceSnapshot {
    VoiceRenderCommitToken token{};
    VoiceRenderPlaybackState playback_state{};
    u32 max_grain_samples{};
    u32 num_grain_samples{};
    u32 system_sample_rate{};
    u64 sampler_source_epoch{};
    SamplerControlSnapshot sampler{};
    std::shared_ptr<VoiceDecodeRuntime> decode_runtime;
};

using VoiceRenderProgressCommitHandler = void (*)(u64 committed_revision, void* context) noexcept;

struct SystemRenderVoiceSnapshot {
    VoiceRenderCommitToken token{};
    VoiceRenderPlaybackState playback_state{};
    u64 sampler_source_epoch{};
    VoiceControlSnapshot control{};
    std::shared_ptr<VoiceDecodeRuntime> decode_runtime;
};

struct SystemRenderGraphSnapshot {
    OrbisNgs2Handle system_handle{};
    SystemRuntimeSnapshot system{};
    std::vector<SystemRenderVoiceSnapshot> voices;
};

struct VoiceRenderProgressCommit {
    VoiceRenderCommitToken token{};
    std::shared_ptr<VoiceDecodeRuntime> decode_runtime;
    VoiceRenderProgressCommitHandler handler{};
    void* context{};
};

using OrbisNgs2BufferAllocHandler =
    s32 PS4_SYSV_ABI (*)(OrbisNgs2ContextBufferInfo* io_buffer_info);
using OrbisNgs2BufferFreeHandler = s32 PS4_SYSV_ABI (*)(OrbisNgs2ContextBufferInfo* io_buffer_info);
using OrbisNgs2RuntimeReadHandler = bool (*)(uintptr_t address, void* destination, size_t size,
                                             void* context);
// Synchronous host-side final writer, called without the registry mutex. Decoder locks remain
// held: this callback must not re-enter rendering. Returning false must not accept staged state.
using SystemRenderPublishHandler = bool (*)(void* context);

struct OrbisNgs2SystemInfo {
    char name[ORBIS_NGS2_SYSTEM_NAME_LENGTH]; // 0

    OrbisNgs2Handle systemHandle;          // 16
    OrbisNgs2ContextBufferInfo bufferInfo; // 24

    u32 uid;             // 88
    u32 minGrainSamples; // 92
    u32 maxGrainSamples; // 96

    u32 stateFlags;        // 100
    u32 rackCount;         // 104
    float lastRenderRatio; // 108
    s64 lastRenderTick;    // 112
    s64 renderCount;       // 120
    u32 sampleRate;        // 128
    u32 numGrainSamples;   // 132
};

struct OrbisNgs2RackInfo {
    char name[ORBIS_NGS2_RACK_NAME_LENGTH]; // 0

    OrbisNgs2Handle rackHandle;            // 16
    OrbisNgs2ContextBufferInfo bufferInfo; // 24

    OrbisNgs2Handle ownerSystemHandle; // 88

    u32 type;            // 96
    u32 rackId;          // 100
    u32 uid;             // 104
    u32 minGrainSamples; // 108
    u32 maxGrainSamples; // 112
    u32 maxVoices;       // 116
    u32 maxChannelWorks; // 120
    u32 maxInputs;       // 124
    u32 maxMatrices;     // 128
    u32 maxPorts;        // 132

    u32 stateFlags;             // 136
    float lastProcessRatio;     // 140
    u64 lastProcessTick;        // 144
    u64 renderCount;            // 152
    u32 activeVoiceCount;       // 160
    u32 activeChannelWorkCount; // 164
};

struct StackBuffer {
    void** top;
    void* base;
    size_t size;
    size_t currentOffset;
    size_t usedSize;
    size_t totalSize;
    size_t alignment;
    u8 flags;
    char padding[7];
};

struct SystemInternal {
    // setup init
    char name[ORBIS_NGS2_SYSTEM_NAME_LENGTH]; // 0
    OrbisNgs2ContextBufferInfo bufferInfo;    // 16
    OrbisNgs2BufferFreeHandler hostFree;      // 80
    OrbisNgs2Handle systemHandle;             // 88
    void* unknown1;                           // 96
    void* unknown2;                           // 104
    OrbisNgs2Handle rackHandle;               // 112
    uintptr_t* userData;                      // 120
    SystemInternal* systemList;               // 128
    StackBuffer* stackBuffer;                 // 136
    OrbisNgs2SystemInfo ownerSystemInfo;      // 144

    struct rackList {
        void* prev;
        void* next;
        void* unknown;
    };

    rackList rackListPreset; // 152
    rackList rackListNormal; // 176
    rackList rackListMaster; // 200

    void* unknown3;       // 208
    void* systemListPrev; // 216
    void* unknown4;       // 224
    void* systemListNext; // 232
    void* rackFunction;   // 240

    Kernel::PthreadMutex processLock; // 248
    u32 hasProcessMutex;              // 256
    u32 unknown5;                     // 260
    Kernel::PthreadMutex flushLock;   // 264
    u32 hasFlushMutex;                // 272
    u32 unknown6;                     // 276

    // info
    u64 lastRenderTick;         // 280
    u64 renderCount;            // 288
    u32 isActive;               // 296
    std::atomic<int> lockCount; // 300
    u32 uid;                    // 304
    u32 systemType;             // 308

    struct {
        u8 isBufferValid : 1;
        u8 isRendering : 1;
        u8 isSorted : 1;
        u8 isFlushReady : 1;
    } flags; // 312

    u16 currentMaxGrainSamples; // 316
    u16 minGrainSamples;        // 318
    u16 maxGrainSamples;        // 320
    u16 numGrainSamples;        // 322
    u32 currentNumGrainSamples; // 324
    u32 sampleRate;             // 328
    u32 currentSampleRate;      // 332
    u32 rackCount;              // 336
    float lastRenderRatio;      // 340
    float cpuLoad;              // 344
};

struct HandleInternal {
    HandleInternal* selfPtr;    // 0
    SystemInternal* systemData; // 8
    std::atomic<int> refCount;  // 16
    u32 handleType;             // 24
    u32 handleID;               // 28
};

s32 StackBufferClose(StackBuffer* stackBuffer, size_t* outTotalSize);
s32 StackBufferOpen(StackBuffer* stackBuffer, void* buffer, size_t bufferSize, void** outBuffer,
                    u8 flags);
s32 SystemSetupCore(StackBuffer* stackBuffer, const OrbisNgs2SystemOption* option,
                    SystemInternal* outSystem);

s32 HandleReportInvalid(OrbisNgs2Handle handle, u32 handleType);
void* MemoryClear(void* buffer, size_t size);
s32 SystemCleanup(OrbisNgs2Handle systemHandle, OrbisNgs2ContextBufferInfo* outInfo);
s32 SystemSetup(const OrbisNgs2SystemOption* option, OrbisNgs2ContextBufferInfo* hostBufferInfo,
                OrbisNgs2BufferFreeHandler hostFree, OrbisNgs2Handle* outHandle);
s32 SystemCreateWithAllocator(const OrbisNgs2SystemOption* option,
                              const OrbisNgs2BufferAllocator* allocator,
                              OrbisNgs2Handle* outHandle);
s32 RackCreate(OrbisNgs2Handle systemHandle, u32 rackId, const OrbisNgs2RackOption* option,
               const OrbisNgs2ContextBufferInfo* bufferInfo, OrbisNgs2BufferFreeHandler hostFree,
               OrbisNgs2Handle* outHandle);
s32 RackCreateWithAllocator(OrbisNgs2Handle systemHandle, u32 rackId,
                            const OrbisNgs2RackOption* option,
                            const OrbisNgs2BufferAllocator* allocator, OrbisNgs2Handle* outHandle);
s32 RackDestroy(OrbisNgs2Handle rackHandle, OrbisNgs2ContextBufferInfo* outBufferInfo);
s32 RackGetVoiceHandle(OrbisNgs2Handle rackHandle, u32 voiceIndex, OrbisNgs2Handle* outHandle);
s32 RackQueryBufferSize(u32 rackId, const OrbisNgs2RackOption* option,
                        OrbisNgs2ContextBufferInfo* outBufferInfo);
s32 RackQueryBufferSize(const OrbisNgs2RackOption* option,
                        OrbisNgs2ContextBufferInfo* outBufferInfo);
s32 SystemRender(OrbisNgs2Handle systemHandle, const OrbisNgs2RenderBufferInfo* bufferInfo,
                 u32 numBufferInfo);
s32 SystemRender(OrbisNgs2Handle systemHandle, const OrbisNgs2RenderBufferInfo* bufferInfo,
                 u32 numBufferInfo, OrbisNgs2RuntimeReadHandler readHandler, void* readContext,
                 SystemRenderPublishHandler publishHandler = nullptr,
                 void* publishContext = nullptr);
s32 VoiceControl(OrbisNgs2Handle voiceHandle, const OrbisNgs2VoiceParamHeader* paramList,
                 OrbisNgs2RuntimeReadHandler readHandler, void* readContext);
s32 VoiceGetState(OrbisNgs2Handle voiceHandle, void* outState, size_t stateSize);
s32 VoiceGetStateFlags(OrbisNgs2Handle voiceHandle, u32* outStateFlags);
[[nodiscard]] std::optional<SystemRuntimeSnapshot> SnapshotSystemRuntime(
    OrbisNgs2Handle systemHandle);
[[nodiscard]] std::optional<SystemRuntimeSnapshot> SnapshotSystemRuntime(
    OrbisNgs2Handle systemHandle, u64 expectedGeneration);
[[nodiscard]] std::optional<VoiceControlSnapshot> SnapshotVoiceControl(OrbisNgs2Handle voiceHandle);
[[nodiscard]] std::optional<AppliedSamplerVoiceSnapshot> SnapshotAppliedSamplerVoice(
    OrbisNgs2Handle voiceHandle);
[[nodiscard]] std::optional<SystemRenderGraphSnapshot> SnapshotSystemRenderGraph(
    OrbisNgs2Handle systemHandle);
// Atomically validates the complete applied-graph identity and installs staged render progress.
// The handler must only move already-owned state and must not invoke guest callbacks.
[[nodiscard]] std::optional<u64> CommitVoiceRenderProgress(
    const VoiceRenderCommitToken& token, const std::shared_ptr<VoiceDecodeRuntime>& decodeRuntime,
    VoiceRenderProgressCommitHandler commitHandler, void* commitContext);
// Validates the complete system graph before accepting any staged decoder state. Callers must
// hold each requested decode runtime's mutex in deterministic voice order.
[[nodiscard]] bool ValidateSystemRenderProgress(
    const SystemRenderGraphSnapshot& snapshot, std::span<const VoiceRenderProgressCommit> progress);
[[nodiscard]] bool CommitSystemRenderProgress(const SystemRenderGraphSnapshot& snapshot,
                                              std::span<const VoiceRenderProgressCommit> progress);

void ResetRuntimeForTesting();

} // namespace Libraries::Ngs2
