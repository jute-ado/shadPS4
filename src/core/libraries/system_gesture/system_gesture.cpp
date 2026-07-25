// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/libs.h"
#include "system_gesture.h"

namespace Libraries::SystemGesture {

int PS4_SYSV_ABI sceSystemGestureAppendTouchRecognizer(s32 handle,
                                                       TouchRecognizer* recognizer) {
    return Behavior::ValidateRecognizer(handle, recognizer);
}

int PS4_SYSV_ABI sceSystemGestureClose(s32 handle) {
    return Behavior::Close(handle);
}

int PS4_SYSV_ABI sceSystemGestureCreateTouchRecognizer(s32 handle, TouchRecognizer* recognizer,
                                                       s32, const Rectangle*, const void*) {
    return Behavior::CreateRecognizer(handle, recognizer);
}

int PS4_SYSV_ABI sceSystemGestureFinalizePrimitiveTouchRecognizer() {
    return Ok;
}

int PS4_SYSV_ABI sceSystemGestureGetPrimitiveTouchEventByIndex(s32 handle, u32 index,
                                                               PrimitiveTouchEvent* event) {
    return Behavior::GetPrimitiveEventByIndex(handle, index, event);
}

int PS4_SYSV_ABI sceSystemGestureGetPrimitiveTouchEventByPrimitiveID(
    s32 handle, u16 primitive_id, PrimitiveTouchEvent* event) {
    return Behavior::GetPrimitiveEventById(handle, primitive_id, event);
}

int PS4_SYSV_ABI sceSystemGestureGetPrimitiveTouchEvents(s32 handle,
                                                         PrimitiveTouchEvent* events, u32 capacity,
                                                         u32* event_count) {
    return Behavior::GetPrimitiveEvents(handle, events, capacity, event_count);
}

int PS4_SYSV_ABI sceSystemGestureGetPrimitiveTouchEventsCount(s32 handle) {
    return Behavior::GetPrimitiveEventsCount(handle);
}

int PS4_SYSV_ABI sceSystemGestureGetTouchEventByEventID(s32 handle,
                                                        const TouchRecognizer* recognizer,
                                                        u32 event_id, TouchEvent* event) {
    return Behavior::GetTouchEventById(handle, recognizer, event_id, event);
}

int PS4_SYSV_ABI sceSystemGestureGetTouchEventByIndex(s32 handle,
                                                      const TouchRecognizer* recognizer, u32 index,
                                                      TouchEvent* event) {
    return Behavior::GetTouchEventByIndex(handle, recognizer, index, event);
}

int PS4_SYSV_ABI sceSystemGestureGetTouchEvents(s32 handle, const TouchRecognizer* recognizer,
                                                TouchEvent* events, u32 capacity,
                                                u32* event_count) {
    return Behavior::GetTouchEvents(handle, recognizer, events, capacity, event_count);
}

int PS4_SYSV_ABI sceSystemGestureGetTouchEventsCount(s32 handle,
                                                     const TouchRecognizer* recognizer) {
    return Behavior::GetTouchEventsCount(handle, recognizer);
}

int PS4_SYSV_ABI sceSystemGestureGetTouchRecognizerInformation(
    s32 handle, const TouchRecognizer* recognizer, TouchRecognizerInformation* information) {
    return Behavior::GetRecognizerInformation(handle, recognizer, information);
}

int PS4_SYSV_ABI sceSystemGestureInitializePrimitiveTouchRecognizer(const void*) {
    return Ok;
}

int PS4_SYSV_ABI sceSystemGestureOpen(s32 input_type, const void*) {
    return Behavior::Open(static_cast<InputType>(input_type));
}

int PS4_SYSV_ABI sceSystemGestureRemoveTouchRecognizer(s32 handle, TouchRecognizer* recognizer) {
    return Behavior::ValidateRecognizer(handle, recognizer);
}

int PS4_SYSV_ABI sceSystemGestureResetPrimitiveTouchRecognizer(s32 handle) {
    return Behavior::Close(handle);
}

int PS4_SYSV_ABI sceSystemGestureResetTouchRecognizer(s32 handle, TouchRecognizer* recognizer) {
    return Behavior::ResetRecognizer(handle, recognizer);
}

int PS4_SYSV_ABI sceSystemGestureUpdateAllTouchRecognizer(s32 handle) {
    return Behavior::UpdateAllRecognizers(handle);
}

int PS4_SYSV_ABI sceSystemGestureUpdatePrimitiveTouchRecognizer(s32 handle,
                                                                 const void*) {
    return Behavior::UpdatePrimitive(handle);
}

int PS4_SYSV_ABI sceSystemGestureUpdateTouchRecognizer(s32 handle,
                                                       TouchRecognizer* recognizer) {
    return Behavior::ValidateRecognizer(handle, recognizer);
}

int PS4_SYSV_ABI sceSystemGestureUpdateTouchRecognizerRectangle(
    s32 handle, TouchRecognizer* recognizer, const Rectangle* rectangle) {
    return Behavior::UpdateRecognizerRectangle(handle, recognizer, rectangle);
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("1MMK0W-kMgA", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureAppendTouchRecognizer);
    LIB_FUNCTION("j4yXIA2jJ68", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureClose);
    LIB_FUNCTION("FWF8zkhr854", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureCreateTouchRecognizer);
    LIB_FUNCTION("3QYCmMlOlCY", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureFinalizePrimitiveTouchRecognizer);
    LIB_FUNCTION("KAeP0+cQPVU", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureGetPrimitiveTouchEventByIndex);
    LIB_FUNCTION("yBaQ0h9m1NM", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureGetPrimitiveTouchEventByPrimitiveID);
    LIB_FUNCTION("L8YmemOeSNY", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureGetPrimitiveTouchEvents);
    LIB_FUNCTION("JhwByySf9FY", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureGetPrimitiveTouchEventsCount);
    LIB_FUNCTION("lpsXm7tzeoc", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureGetTouchEventByEventID);
    LIB_FUNCTION("TSKvgSz5ChU", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureGetTouchEventByIndex);
    LIB_FUNCTION("fLTseA7XiWY", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureGetTouchEvents);
    LIB_FUNCTION("h8uongcBNVs", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureGetTouchEventsCount);
    LIB_FUNCTION("0KrW5eMnrwY", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureGetTouchRecognizerInformation);
    LIB_FUNCTION("3pcAvmwKCvM", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureInitializePrimitiveTouchRecognizer);
    LIB_FUNCTION("qpo-mEOwje0", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureOpen);
    LIB_FUNCTION("ELvBVG-LKT0", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureRemoveTouchRecognizer);
    LIB_FUNCTION("o11J529VaAE", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureResetPrimitiveTouchRecognizer);
    LIB_FUNCTION("oBuH3zFWYIg", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureResetTouchRecognizer);
    LIB_FUNCTION("wPJGwI2RM2I", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureUpdateAllTouchRecognizer);
    LIB_FUNCTION("GgFMb22sbbI", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureUpdatePrimitiveTouchRecognizer);
    LIB_FUNCTION("j4h82CQWENo", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureUpdateTouchRecognizer);
    LIB_FUNCTION("4WOA1eTx3V8", "libSceSystemGesture", 1, "libSceSystemGesture",
                 sceSystemGestureUpdateTouchRecognizerRectangle);
}

} // namespace Libraries::SystemGesture
