// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "core/libraries/error_codes.h"
#include "core/libraries/ngs2/ngs2.h"
#include "core/libraries/ngs2/ngs2_error.h"

namespace Libraries::Ngs2 {
namespace {

std::array<u8, 64> allocator_storage;
size_t allocator_requested_size;
uintptr_t allocator_user_data;

s32 PS4_SYSV_ABI AllocateRackBuffer(OrbisNgs2ContextBufferInfo* info) {
    allocator_requested_size = info->hostBufferSize;
    allocator_user_data = info->userData;
    info->hostBuffer = allocator_storage.data();
    return ORBIS_OK;
}

TEST(Ngs2Test, RackQueryBufferSizeInitializesSuccessfulOutput) {
    OrbisNgs2ContextBufferInfo info;
    std::memset(&info, 0xA5, sizeof(info));
    const size_t poisoned_size = info.hostBufferSize;
    const uintptr_t poisoned_user_data = info.userData;

    EXPECT_EQ(RackQueryBufferSize(nullptr, &info), ORBIS_OK);
    EXPECT_EQ(info.hostBuffer, nullptr);
    EXPECT_GT(info.hostBufferSize, 0);
    EXPECT_NE(info.hostBufferSize, poisoned_size);
    for (const auto value : info.reserved) {
        EXPECT_EQ(value, 0);
    }
    EXPECT_EQ(info.userData, poisoned_user_data);
}

TEST(Ngs2Test, RackQueryBufferSizeRejectsInvalidArguments) {
    EXPECT_EQ(RackQueryBufferSize(nullptr, nullptr), ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS);

    OrbisNgs2RackOption option{};
    option.size = sizeof(option) - 1;
    OrbisNgs2ContextBufferInfo info{};
    EXPECT_EQ(RackQueryBufferSize(&option, &info), ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE);
}

TEST(Ngs2Test, RackCreateReturnsCallerBufferAsStableHandle) {
    std::array<u8, 64> storage{};
    OrbisNgs2ContextBufferInfo info{};
    ASSERT_EQ(RackQueryBufferSize(nullptr, &info), ORBIS_OK);
    ASSERT_LE(info.hostBufferSize, storage.size());
    info.hostBuffer = storage.data();
    OrbisNgs2Handle handle = 0;

    EXPECT_EQ(RackCreate(1, nullptr, &info, &handle), ORBIS_OK);
    EXPECT_EQ(handle, reinterpret_cast<OrbisNgs2Handle>(storage.data()));
}

TEST(Ngs2Test, RackCreateRejectsInvalidArguments) {
    std::array<u8, 64> storage{};
    OrbisNgs2ContextBufferInfo info{};
    ASSERT_EQ(RackQueryBufferSize(nullptr, &info), ORBIS_OK);
    ASSERT_LE(info.hostBufferSize, storage.size());
    info.hostBuffer = storage.data();
    OrbisNgs2Handle handle = 0;

    EXPECT_EQ(RackCreate(0, nullptr, &info, &handle), ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE);
    EXPECT_EQ(RackCreate(1, nullptr, nullptr, &handle), ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO);
    EXPECT_EQ(RackCreate(1, nullptr, &info, nullptr), ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS);
    info.hostBuffer = nullptr;
    EXPECT_EQ(RackCreate(1, nullptr, &info, &handle), ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS);
    info.hostBuffer = storage.data();
    --info.hostBufferSize;
    EXPECT_EQ(RackCreate(1, nullptr, &info, &handle), ORBIS_NGS2_ERROR_INVALID_BUFFER_SIZE);
}

TEST(Ngs2Test, RackCreateWithAllocatorRequestsAndReturnsStableBufferHandle) {
    allocator_requested_size = 0;
    allocator_user_data = 0;
    OrbisNgs2BufferAllocator allocator{};
    allocator.allocHandler = AllocateRackBuffer;
    allocator.userData = 0x1234;
    OrbisNgs2Handle handle = 0;

    EXPECT_EQ(RackCreateWithAllocator(1, nullptr, &allocator, &handle), ORBIS_OK);
    EXPECT_GT(allocator_requested_size, 0);
    EXPECT_LE(allocator_requested_size, allocator_storage.size());
    EXPECT_EQ(allocator_user_data, allocator.userData);
    EXPECT_EQ(handle, reinterpret_cast<OrbisNgs2Handle>(allocator_storage.data()));
}

TEST(Ngs2Test, RackCreateWithAllocatorRejectsInvalidArguments) {
    OrbisNgs2BufferAllocator allocator{};
    allocator.allocHandler = AllocateRackBuffer;
    OrbisNgs2Handle handle = 0;

    EXPECT_EQ(RackCreateWithAllocator(0, nullptr, &allocator, &handle),
              ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE);
    EXPECT_EQ(RackCreateWithAllocator(1, nullptr, nullptr, &handle),
              ORBIS_NGS2_ERROR_INVALID_BUFFER_ALLOCATOR);
    EXPECT_EQ(RackCreateWithAllocator(1, nullptr, &allocator, nullptr),
              ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS);
    allocator.allocHandler = nullptr;
    EXPECT_EQ(RackCreateWithAllocator(1, nullptr, &allocator, &handle),
              ORBIS_NGS2_ERROR_INVALID_BUFFER_ALLOCATOR);
}

TEST(Ngs2Test, VoiceGetStateInitializesEntireCallerBuffer) {
    std::array<u8, 48> state;
    state.fill(0xA5);

    EXPECT_EQ(VoiceGetState(1, state.data(), state.size()), ORBIS_OK);
    for (const auto value : state) {
        EXPECT_EQ(value, 0);
    }
}

TEST(Ngs2Test, VoiceGetStateRejectsInvalidArguments) {
    OrbisNgs2VoiceState state{};
    EXPECT_EQ(VoiceGetState(0, &state, sizeof(state)), ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE);
    EXPECT_EQ(VoiceGetState(1, nullptr, sizeof(state)), ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS);
    EXPECT_EQ(VoiceGetState(1, &state, sizeof(state) - 1),
              ORBIS_NGS2_ERROR_INVALID_VOICE_STATE_SIZE);
}

TEST(Ngs2Test, VoiceControlAcceptsWellFormedSingleParameter) {
    OrbisNgs2VoiceParamHeader param{};
    param.size = sizeof(param);

    EXPECT_EQ(ValidateVoiceControlRequest(1, &param), ORBIS_OK);
}

TEST(Ngs2Test, VoiceControlRejectsInvalidArguments) {
    OrbisNgs2VoiceParamHeader param{};
    param.size = sizeof(param);

    EXPECT_EQ(ValidateVoiceControlRequest(0, &param), ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE);
    EXPECT_EQ(ValidateVoiceControlRequest(1, nullptr),
              ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS);
    param.size = sizeof(param) - 1;
    EXPECT_EQ(ValidateVoiceControlRequest(1, &param), ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE);
}

} // namespace
} // namespace Libraries::Ngs2
