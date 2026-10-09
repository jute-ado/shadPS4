// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/logging/formatter.h"
#include "core/emulator_settings.h"
#include "video_core/renderdoc.h"
#include "video_core/renderdoc_capture_state.h"
#include "video_core/renderdoc_shader_selector.h"
#include "video_core/renderdoc_path.h"

#include <cstdlib>
#include <renderdoc_app.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>

namespace VideoCore {

static RenderDocCaptureState capture_state;
static ScreenshotRequestQueue screenshot_requests;

RENDERDOC_API_1_6_0* rdoc_api{};

static const RenderDocShaderSelector& GetShaderCaptureSelector() noexcept {
    // Read the launch environment once. Selection never changes the command stream or submission
    // order, and must not change midway through recording an eligible command buffer.
    static const auto selector = [] {
        const char* compute = std::getenv("SHADPS4_RENDERDOC_COMPUTE_HASH");
        const char* vertex = std::getenv("SHADPS4_RENDERDOC_VERTEX_HASH");
        const char* color = std::getenv("SHADPS4_RENDERDOC_COLOR_DRAW");
        return RenderDocShaderSelector::Parse(
            compute ? std::optional<std::string_view>{compute} : std::nullopt,
            vertex ? std::optional<std::string_view>{vertex} : std::nullopt,
            color ? std::optional<std::string_view>{color} : std::nullopt);
    }();
    return selector;
}

bool IsGuestWorkCaptureEligible(std::optional<u64> compute_shader_hash,
                                std::optional<u64> vertex_shader_hash,
                                bool has_color_attachment,
                                bool has_presentable_color_attachment) noexcept {
    const auto proof = RenderDocColorDrawProof{has_color_attachment,
                                               has_presentable_color_attachment};
    return GetShaderCaptureSelector().Allows(compute_shader_hash, vertex_shader_hash, proof);
}

bool GuestWorkCaptureRequiresColorAttachment() noexcept {
    return GetShaderCaptureSelector().RequiresColorAttachment();
}

void LoadRenderDoc(const bool allow_offline_loading) {
    if (rdoc_api) {
        return;
    }

    const char* configured_path = std::getenv("SHADPS4_RENDERDOC_PATH");
#ifdef _WIN32

    // Check if we are running by RDoc GUI
    HMODULE mod = GetModuleHandleA("renderdoc.dll");
    if (!mod && configured_path) {
        if (const auto path = ResolveRenderDocModulePath(configured_path)) {
            mod = LoadLibraryW(path->c_str());
        } else {
            LOG_ERROR(Render, "SHADPS4_RENDERDOC_PATH does not contain a RenderDoc library");
        }
    }
    if (!mod && allow_offline_loading && EmulatorSettings.IsRenderdocEnabled()) {
        // If enabled in config, try to load RDoc runtime in offline mode
        HKEY h_reg_key;
        LONG result = RegOpenKeyExW(HKEY_LOCAL_MACHINE,
                                    L"SOFTWARE\\Classes\\RenderDoc.RDCCapture.1\\DefaultIcon\\", 0,
                                    KEY_READ, &h_reg_key);
        if (result != ERROR_SUCCESS) {
            return;
        }
        std::array<wchar_t, MAX_PATH> key_str{};
        DWORD str_sz_out{key_str.size()};
        result = RegQueryValueExW(h_reg_key, L"", 0, NULL, (LPBYTE)key_str.data(), &str_sz_out);
        if (result != ERROR_SUCCESS) {
            return;
        }

        std::filesystem::path path{key_str.cbegin(), key_str.cend()};
        path = path.parent_path().append("renderdoc.dll");
        const auto path_to_lib = path.generic_string();
        mod = LoadLibraryA(path_to_lib.c_str());
    }

    if (mod) {
        const auto RENDERDOC_GetAPI =
            reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
        if (!RENDERDOC_GetAPI) {
            LOG_ERROR(Render, "Loaded RenderDoc library does not export RENDERDOC_GetAPI");
            return;
        }
        const s32 ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
        ASSERT(ret == 1);
    }
#else
#ifdef ANDROID
    static constexpr const char RENDERDOC_LIB[] = "libVkLayer_GLES_RenderDoc.so";
#else
    static constexpr const char RENDERDOC_LIB[] = "librenderdoc.so";
#endif
    // Check if we are running by RDoc GUI
    void* mod = dlopen(RENDERDOC_LIB, RTLD_NOW | RTLD_NOLOAD);
    if (!mod && configured_path) {
        if (const auto path = ResolveRenderDocModulePath(configured_path)) {
            mod = dlopen(path->c_str(), RTLD_NOW);
            if (!mod) {
                LOG_ERROR(Render, "Cannot load RenderDoc from SHADPS4_RENDERDOC_PATH: {}",
                          dlerror());
            }
        } else {
            LOG_ERROR(Render, "SHADPS4_RENDERDOC_PATH does not contain a RenderDoc library");
        }
    }
    if (!mod && allow_offline_loading && EmulatorSettings.IsRenderdocEnabled()) {
        // If enabled in config, try to load RDoc runtime in offline mode
        if (!(mod = dlopen(RENDERDOC_LIB, RTLD_NOW))) {
            LOG_ERROR(Render, "Cannot load RenderDoc: {}", dlerror());
        }
    }
    if (mod) {
        const auto RENDERDOC_GetAPI =
            reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
        if (!RENDERDOC_GetAPI) {
            LOG_ERROR(Render, "Loaded RenderDoc library does not export RENDERDOC_GetAPI");
            return;
        }
        const s32 ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
        ASSERT(ret == 1);
    }
#endif
    if (rdoc_api) {
        // Disable default capture keys as they suppose to trigger present-to-present capturing
        // and it is not what we want
        rdoc_api->SetCaptureKeys(nullptr, 0);

        // Also remove rdoc crash handler
        rdoc_api->UnloadCrashHandler();
    }
}

void TriggerCapture() {
    if (!GetShaderCaptureSelector().Valid()) {
        static std::atomic_flag reported;
        if (!reported.test_and_set(std::memory_order_relaxed)) {
            LOG_ERROR(Common,
                      "Invalid or ambiguous SHADPS4_RENDERDOC_COMPUTE_HASH/"
                      "SHADPS4_RENDERDOC_VERTEX_HASH/SHADPS4_RENDERDOC_COLOR_DRAW; "
                      "capture request refused");
        }
        return;
    }
    (void)capture_state.Trigger();
}

GuestSubmissionCapture BeginGuestWorkCapture(void* vulkan_instance, const bool has_guest_work) {
    RenderDocCaptureApi api{};
    if (rdoc_api) {
        api = {
            .context = vulkan_instance,
            .is_capturing = [](void*) { return rdoc_api->IsFrameCapturing() != 0; },
            .start =
                [](void* instance) {
                    const auto device = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance);
                    // An app-controlled Vulkan capture need not contain a window presentation.
                    rdoc_api->StartFrameCapture(device, nullptr);
                },
            .end =
                [](void* instance) {
                    const auto device = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance);
                    return rdoc_api->EndFrameCapture(device, nullptr) != 0;
                },
            .discard =
                [](void* instance) {
                    const auto device = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance);
                    return rdoc_api->DiscardFrameCapture(device, nullptr) != 0;
                },
        };
    }
    return GuestSubmissionCapture{capture_state, api, has_guest_work};
}

void CompleteGuestWorkCapture(GuestSubmissionCapture& capture, const bool submission_succeeded,
                              bool& pending_guest_work) {
    const auto result = capture.CompleteSubmission(submission_succeeded, pending_guest_work);
    if (result == GuestCaptureResult::Captured) {
        LOG_WARNING(Common, "RenderDoc guest-work submission capture completed");
    } else if (result != GuestCaptureResult::NotRequested) {
        LOG_ERROR(Common, "RenderDoc guest-work capture failed: reason={}, cleanup_failed={}",
                  static_cast<u32>(result), capture.CleanupFailed());
    }
}

void SetOutputDir(const std::filesystem::path& path, const std::string& prefix) {
    if (!rdoc_api) {
        return;
    }
    LOG_WARNING(Common, "RenderDoc capture path: {}", (path / prefix).string());
    rdoc_api->SetCaptureFilePathTemplate(fmt::UTF((path / prefix).u8string()).data.data());
}

bool IsRenderDocLoaded() {
    return rdoc_api != nullptr;
}

void RequestScreenshot(const ScreenshotRequest request, const ScreenshotRequestOrigin origin) {
    screenshot_requests.Push(request, origin);
}

ScreenshotRequestBatch ConsumeGameOnlyScreenshotRequests() {
    return screenshot_requests.ConsumeGameOnly();
}

ScreenshotRequestBatch ConsumeWithOverlaysScreenshotRequests() {
    return screenshot_requests.ConsumeWithOverlays();
}

} // namespace VideoCore
