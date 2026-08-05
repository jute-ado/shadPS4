// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/logging/formatter.h"
#include "core/emulator_settings.h"
#include "video_core/renderdoc.h"
#include "video_core/renderdoc_capture_state.h"
#include "video_core/renderdoc_path.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <renderdoc_app.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>

namespace VideoCore {

namespace {

constexpr u32 MaximumDiagnosticCaptureFrames = 8;

u32 ConfiguredCaptureFrameCount() {
    const char* const value = std::getenv("SHADPS4_RENDERDOC_CAPTURE_FRAMES");
    if (!value) {
        return 1;
    }

    u32 count{};
    const auto* const end = value + std::strlen(value);
    const auto [parsed_end, error] = std::from_chars(value, end, count);
    if (error != std::errc{} || parsed_end != end || count == 0) {
        return 1;
    }
    return std::min(count, MaximumDiagnosticCaptureFrames);
}

} // namespace

static RenderDocCaptureState capture_state{ConfiguredCaptureFrameCount()};
static ScreenshotRequestQueue screenshot_requests;

RENDERDOC_API_1_6_0* rdoc_api{};

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
    (void)capture_state.Trigger();
}

bool BeginNextPresentedFrameCapture(void* vulkan_instance, void* window_handle) {
    if (!rdoc_api || !capture_state.ConsumePresentedFrameTrigger()) {
        return false;
    }

    const auto device = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(vulkan_instance);
    rdoc_api->StartFrameCapture(device, window_handle);
    LOG_WARNING(Common, "RenderDoc capture started before guest-frame submission");
    return true;
}

bool IsPresentedFrameCaptureActive() {
    return capture_state.IsCapturing();
}

void EndPresentedFrameCapture(void* vulkan_instance, void* window_handle) {
    if (!rdoc_api || !capture_state.FinishPresentedFrameCapture()) {
        return;
    }
    const auto device = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(vulkan_instance);
    const u32 result = rdoc_api->EndFrameCapture(device, window_handle);
    LOG_WARNING(Common, "RenderDoc presented-frame capture end result: {}", result);
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
