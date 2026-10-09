// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderdoc_capture_state.h"

namespace VideoCore {

// A small injectable API boundary: native tests need neither RenderDoc nor a GPU.
struct RenderDocCaptureApi {
    void* context{};
    bool (*is_capturing)(void*){};
    void (*start)(void*){};
    bool (*end)(void*){};
    bool (*discard)(void*){};

    [[nodiscard]] bool Available() const {
        return is_capturing && start && end && discard;
    }
};

enum class GuestCaptureResult {
    NotRequested,
    Active,
    Captured,
    ApiUnavailable,
    ExternalCapture,
    StartRefused,
    SubmissionFailed,
    EndFailed,
    ApiException,
};

// Owned by the submitting thread, only across one actual queue submission.
// The caller must report a queue error before invoking a non-unwinding fatal check.
class GuestSubmissionCapture {
public:
    GuestSubmissionCapture(RenderDocCaptureState& state, RenderDocCaptureApi api,
                           const bool has_guest_work) noexcept
        : api{api} {
        if (!state.TryBeginGuestSubmission(has_guest_work)) {
            return;
        }
        request_state = &state;
        if (!api.Available()) {
            result = GuestCaptureResult::ApiUnavailable;
            FinishRequest();
            return;
        }
        try {
            if (api.is_capturing(api.context)) {
                result = GuestCaptureResult::ExternalCapture;
                FinishRequest();
                return;
            }
            // If the start callback throws after starting, cleanup still owns that attempt.
            owns_capture = true;
            api.start(api.context);
            if (!api.is_capturing(api.context)) {
                owns_capture = false;
                result = GuestCaptureResult::StartRefused;
                FinishRequest();
                return;
            }
            result = GuestCaptureResult::Active;
        } catch (...) {
            result = GuestCaptureResult::ApiException;
            DiscardOwnedCapture();
            FinishRequest();
        }
    }

    GuestSubmissionCapture(const GuestSubmissionCapture&) = delete;
    GuestSubmissionCapture& operator=(const GuestSubmissionCapture&) = delete;

    ~GuestSubmissionCapture() {
        DiscardOwnedCapture();
        FinishRequest();
    }

    [[nodiscard]] GuestCaptureResult CompleteSubmission(const bool succeeded,
                                                        bool& pending_guest_work) noexcept {
        if (submission_reported) {
            return result;
        }
        submission_reported = true;
        // Submission success, not diagnostic success, transfers the recorded work to the GPU.
        if (succeeded) {
            pending_guest_work = false;
        }
        if (owns_capture) {
            if (!succeeded) {
                result = GuestCaptureResult::SubmissionFailed;
                DiscardOwnedCapture();
            } else {
                try {
                    if (api.end(api.context)) {
                        owns_capture = false;
                        result = GuestCaptureResult::Captured;
                    } else {
                        result = GuestCaptureResult::EndFailed;
                        DiscardOwnedCapture();
                    }
                } catch (...) {
                    result = GuestCaptureResult::ApiException;
                    DiscardOwnedCapture();
                }
            }
        }
        FinishRequest();
        return result;
    }

    [[nodiscard]] GuestCaptureResult Result() const noexcept {
        return result;
    }

    [[nodiscard]] bool CleanupFailed() const noexcept {
        return cleanup_failed;
    }

private:
    void DiscardOwnedCapture() noexcept {
        if (!owns_capture) {
            return;
        }
        owns_capture = false;
        try {
            cleanup_failed |= !api.discard(api.context);
        } catch (...) {
            cleanup_failed = true;
        }
    }

    void FinishRequest() noexcept {
        if (request_state) {
            (void)request_state->FinishGuestSubmission();
            request_state = nullptr;
        }
    }

    RenderDocCaptureApi api;
    RenderDocCaptureState* request_state{};
    GuestCaptureResult result{GuestCaptureResult::NotRequested};
    bool owns_capture{};
    bool submission_reported{};
    bool cleanup_failed{};
};

} // namespace VideoCore
