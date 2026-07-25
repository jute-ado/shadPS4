// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/system_gesture/system_gesture_behavior.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::SystemGesture {

int PS4_SYSV_ABI sceSystemGestureAppendTouchRecognizer(s32 handle, TouchRecognizer* recognizer);
int PS4_SYSV_ABI sceSystemGestureClose(s32 handle);
int PS4_SYSV_ABI sceSystemGestureCreateTouchRecognizer(s32 handle, TouchRecognizer* recognizer,
                                                       s32 gesture_type, const Rectangle* rectangle,
                                                       const void* parameters);
int PS4_SYSV_ABI sceSystemGestureFinalizePrimitiveTouchRecognizer();
int PS4_SYSV_ABI sceSystemGestureGetPrimitiveTouchEventByIndex(s32 handle, u32 index,
                                                               PrimitiveTouchEvent* event);
int PS4_SYSV_ABI sceSystemGestureGetPrimitiveTouchEventByPrimitiveID(
    s32 handle, u16 primitive_id, PrimitiveTouchEvent* event);
int PS4_SYSV_ABI sceSystemGestureGetPrimitiveTouchEvents(s32 handle,
                                                         PrimitiveTouchEvent* events, u32 capacity,
                                                         u32* event_count);
int PS4_SYSV_ABI sceSystemGestureGetPrimitiveTouchEventsCount(s32 handle);
int PS4_SYSV_ABI sceSystemGestureGetTouchEventByEventID(s32 handle,
                                                        const TouchRecognizer* recognizer,
                                                        u32 event_id, TouchEvent* event);
int PS4_SYSV_ABI sceSystemGestureGetTouchEventByIndex(s32 handle,
                                                      const TouchRecognizer* recognizer, u32 index,
                                                      TouchEvent* event);
int PS4_SYSV_ABI sceSystemGestureGetTouchEvents(s32 handle, const TouchRecognizer* recognizer,
                                                TouchEvent* events, u32 capacity, u32* event_count);
int PS4_SYSV_ABI sceSystemGestureGetTouchEventsCount(s32 handle,
                                                     const TouchRecognizer* recognizer);
int PS4_SYSV_ABI sceSystemGestureGetTouchRecognizerInformation(
    s32 handle, const TouchRecognizer* recognizer, TouchRecognizerInformation* information);
int PS4_SYSV_ABI sceSystemGestureInitializePrimitiveTouchRecognizer(const void* parameters);
int PS4_SYSV_ABI sceSystemGestureOpen(s32 input_type, const void* parameters);
int PS4_SYSV_ABI sceSystemGestureRemoveTouchRecognizer(s32 handle, TouchRecognizer* recognizer);
int PS4_SYSV_ABI sceSystemGestureResetPrimitiveTouchRecognizer(s32 handle);
int PS4_SYSV_ABI sceSystemGestureResetTouchRecognizer(s32 handle, TouchRecognizer* recognizer);
int PS4_SYSV_ABI sceSystemGestureUpdateAllTouchRecognizer(s32 handle);
int PS4_SYSV_ABI sceSystemGestureUpdatePrimitiveTouchRecognizer(s32 handle,
                                                                 const void* touch_data);
int PS4_SYSV_ABI sceSystemGestureUpdateTouchRecognizer(s32 handle, TouchRecognizer* recognizer);
int PS4_SYSV_ABI sceSystemGestureUpdateTouchRecognizerRectangle(
    s32 handle, TouchRecognizer* recognizer, const Rectangle* rectangle);

void RegisterLib(Core::Loader::SymbolsResolver* sym);
} // namespace Libraries::SystemGesture
