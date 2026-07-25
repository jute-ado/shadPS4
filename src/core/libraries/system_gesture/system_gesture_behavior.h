// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>

#include "common/types.h"

namespace Libraries::SystemGesture {

constexpr s32 Ok = 0;
constexpr s32 ErrorInvalidArgument = static_cast<s32>(0x80890001);
constexpr s32 ErrorEventDataNotFound = static_cast<s32>(0x80890005);
constexpr s32 ErrorInvalidHandle = static_cast<s32>(0x80890006);
constexpr s32 ValidHandle = 1;

enum class InputType : s32 {
    TouchPad = 0,
};

struct Vector2 {
    float x;
    float y;
};

struct PrimitiveTouchEvent {
    s32 event_state;
    u16 primitive_id;
    u8 is_updated;
    u8 reserved0;
    Vector2 pressed_position;
    Vector2 current_position;
    Vector2 delta_vector;
    u64 delta_time;
    u64 elapsed_time;
    u8 reserved[32];
};
static_assert(sizeof(PrimitiveTouchEvent) == 80);

struct Rectangle {
    float x;
    float y;
    float width;
    float height;
    u8 reserved[8];
};
static_assert(sizeof(Rectangle) == 24);

struct TouchRecognizer {
    u64 reserved[361];
};
static_assert(sizeof(TouchRecognizer) == 2888);

struct TouchRecognizerInformation {
    s32 gesture_type;
    Rectangle rectangle;
    u64 updated_time;
    u8 reserved[256];
};
static_assert(sizeof(TouchRecognizerInformation) == 296);

struct TouchEvent {
    u8 reserved[168];
};
static_assert(sizeof(TouchEvent) == 168);

namespace Behavior {

inline bool IsValidHandle(s32 handle) {
    return handle == ValidHandle;
}

inline s32 Open(InputType input_type) {
    return input_type == InputType::TouchPad ? ValidHandle : ErrorInvalidArgument;
}

inline s32 Close(s32 handle) {
    return IsValidHandle(handle) ? Ok : ErrorInvalidHandle;
}

inline s32 UpdatePrimitive(s32 handle) {
    return IsValidHandle(handle) ? Ok : ErrorInvalidHandle;
}

inline s32 GetPrimitiveEvents(s32 handle, PrimitiveTouchEvent* events, u32 capacity,
                              u32* event_count) {
    if (!IsValidHandle(handle)) {
        return ErrorInvalidHandle;
    }
    if (event_count == nullptr) {
        return ErrorInvalidArgument;
    }
    *event_count = 0;
    if (events != nullptr && capacity != 0) {
        std::memset(events, 0, sizeof(*events));
    }
    return Ok;
}

inline s32 GetPrimitiveEventsCount(s32 handle) {
    return IsValidHandle(handle) ? 0 : ErrorInvalidHandle;
}

inline s32 GetPrimitiveEvent(s32 handle, PrimitiveTouchEvent* event) {
    if (!IsValidHandle(handle)) {
        return ErrorInvalidHandle;
    }
    if (event != nullptr) {
        std::memset(event, 0, sizeof(*event));
    }
    return ErrorEventDataNotFound;
}

inline s32 GetPrimitiveEventByIndex(s32 handle, u32, PrimitiveTouchEvent* event) {
    return GetPrimitiveEvent(handle, event);
}

inline s32 GetPrimitiveEventById(s32 handle, u16, PrimitiveTouchEvent* event) {
    return GetPrimitiveEvent(handle, event);
}

inline s32 CreateRecognizer(s32 handle, TouchRecognizer* recognizer) {
    if (!IsValidHandle(handle)) {
        return ErrorInvalidHandle;
    }
    if (recognizer == nullptr) {
        return ErrorInvalidArgument;
    }
    std::memset(recognizer, 0, sizeof(*recognizer));
    return Ok;
}

inline s32 ValidateRecognizer(s32 handle, const TouchRecognizer* recognizer) {
    if (!IsValidHandle(handle)) {
        return ErrorInvalidHandle;
    }
    return recognizer != nullptr ? Ok : ErrorInvalidArgument;
}

inline s32 ResetRecognizer(s32 handle, TouchRecognizer* recognizer) {
    const s32 result = ValidateRecognizer(handle, recognizer);
    if (result == Ok) {
        std::memset(recognizer, 0, sizeof(*recognizer));
    }
    return result;
}

inline s32 GetRecognizerInformation(s32 handle, const TouchRecognizer* recognizer,
                                    TouchRecognizerInformation* information) {
    if (!IsValidHandle(handle)) {
        return ErrorInvalidHandle;
    }
    if (recognizer == nullptr || information == nullptr) {
        return ErrorInvalidArgument;
    }
    std::memset(information, 0, sizeof(*information));
    return Ok;
}

inline s32 UpdateAllRecognizers(s32 handle) {
    return IsValidHandle(handle) ? Ok : ErrorInvalidHandle;
}

inline s32 UpdateRecognizerRectangle(s32 handle, const TouchRecognizer* recognizer,
                                     const Rectangle* rectangle) {
    if (!IsValidHandle(handle)) {
        return ErrorInvalidHandle;
    }
    return recognizer != nullptr && rectangle != nullptr ? Ok : ErrorInvalidArgument;
}

inline s32 GetTouchEvents(s32 handle, const TouchRecognizer* recognizer, TouchEvent* events,
                          u32 capacity, u32* event_count) {
    if (!IsValidHandle(handle)) {
        return ErrorInvalidHandle;
    }
    if (recognizer == nullptr || event_count == nullptr) {
        return ErrorInvalidArgument;
    }
    *event_count = 0;
    if (events != nullptr && capacity != 0) {
        std::memset(events, 0, sizeof(*events));
    }
    return Ok;
}

inline s32 GetTouchEventsCount(s32 handle, const TouchRecognizer* recognizer) {
    return ValidateRecognizer(handle, recognizer);
}

inline s32 GetTouchEvent(s32 handle, const TouchRecognizer* recognizer, TouchEvent* event) {
    const s32 result = ValidateRecognizer(handle, recognizer);
    if (result != Ok) {
        return result;
    }
    if (event != nullptr) {
        std::memset(event, 0, sizeof(*event));
    }
    return ErrorEventDataNotFound;
}

inline s32 GetTouchEventByIndex(s32 handle, const TouchRecognizer* recognizer, u32,
                                TouchEvent* event) {
    return GetTouchEvent(handle, recognizer, event);
}

inline s32 GetTouchEventById(s32 handle, const TouchRecognizer* recognizer, u32,
                             TouchEvent* event) {
    return GetTouchEvent(handle, recognizer, event);
}

} // namespace Behavior
} // namespace Libraries::SystemGesture
