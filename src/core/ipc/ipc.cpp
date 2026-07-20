//  SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
//  SPDX-License-Identifier: GPL-2.0-or-later

#include "ipc.h"

#include <iostream>
#include <string>

#include <SDL3/SDL.h>

#include "common/memory_patcher.h"
#include "common/thread.h"
#include "common/types.h"
#include "core/debug_state.h"
#include "core/debugger.h"
#include "core/emulator_settings.h"
#include "core/emulator_state.h"
#include "core/libraries/audio/audioout.h"
#include "input/controller_axis.h"
#include "input/controller_button.h"
#include "input/controller_touch.h"
#include "input/input_handler.h"
#include "sdl_window.h"
#include "src/core/libraries/usbd/usbd.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;

/**
 * Protocol summary:
 * - IPC is enabled by setting the SHADPS4_ENABLE_IPC environment variable to "true"
 * - Input will be stdin & output stderr
 * - Strings are sent as UTF8
 * - Each communication line is terminated by a newline character ('\n')
 * - Each command parameter will be separated by a newline character ('\n'),
 *   variadic commands will start sending the number of parameters after the cmd word.
 *   Any ('\n') in the parameter must be escaped by a backslash ('\\n')
 * - Numbers can be sent with any base. Must prefix the number with '0x' for hex,
 *   '0b' for binary, or '0' for octal. Decimal numbers
 *   will be sent without any prefix.
 * - Output will be started by (';')
 * - The IPC server(this) will send a block started by
 *   #IPC_ENABLED
 *   and ended by
 *   #IPC_END
 *   In between, it will send the current capabilities and commands before the emulator start
 * - The IPC client(e.g., launcher) will send RUN then START to continue the execution
 **/

/**
 * Command list:
 * - CAPABILITIES:
 *   - ENABLE_MEMORY_PATCH: enables PATCH_MEMORY command
 *   - ENABLE_EMU_CONTROL: enables emulator control commands
 *   - ENABLE_SCREENSHOT: enables SCREENSHOT command
 *   - ENABLE_RENDERDOC_CAPTURE: enables the RenderDoc capture command
 *   - ENABLE_GAMEPAD
 * - INPUT CMD:
 *   - RUN: start the emulator execution
 *   - START: start the game execution
 *   - PATCH_MEMORY(
 *       modName: str, offset: str, value: str,
 *       target: str, size: str, isOffset: number, littleEndian: number,
 *       patchMask: number, maskOffset: number
 *     ): add a memory patch, check @ref MemoryPatcher::PatchMemory for details
 *   - PAUSE: pause the game execution
 *   - RESUME: resume the game execution
 *   - STOP: stop and quit the emulator
 *   - TOGGLE_FULLSCREEN: enable / disable fullscreen
 *   - SCREENSHOT: capture the next game-only frame
 *   - RENDERDOC_CAPTURE: capture the next frame with RenderDoc when loaded
 *   - GAMEPAD_BUTTON
 *     - button: player-one button name
 *     - pressed: 1 to press, 0 to
 *release
 *   - GAMEPAD_AXIS
 *     - axis: player-one axis name
 *     - value: exact PS4 byte (0-255)
 *  -
 * GAMEPAD_TOUCH
 *     - finger: touch index, 0 or 1
 *     - down: touch state (0 or 1)
 *     -
 * x: native PS4 coordinate (0-1919)
 *     - y: native PS4 coordinate (0-941)
 * - OUTPUT CMD:
 *   - RESTART(argn: number, argv: ...string): Request restart of the emulator, must call STOP
 **/

void IPC::Init() {
    const char* enabledEnv = std::getenv("SHADPS4_ENABLE_IPC");
    enabled = enabledEnv && strcmp(enabledEnv, "true") == 0;
    if (!enabled) {
        return;
    }

    EmulatorState::GetInstance()->SetAutoPatchesLoadEnabled(false);

    input_thread = std::jthread([this] {
        Common::SetCurrentThreadName("IPC Read thread");
        this->InputLoop();
    });

    std::cerr << ";#IPC_ENABLED\n";
    std::cerr << ";ENABLE_MEMORY_PATCH\n";
    std::cerr << ";ENABLE_EMU_CONTROL\n";
    std::cerr << ";ENABLE_SCREENSHOT\n";
    std::cerr << ";ENABLE_RENDERDOC_CAPTURE\n";
    std::cerr << ";ENABLE_GAMEPAD\n";
    std::cerr << ";#IPC_END\n";
    std::cerr.flush();

    const auto ok = run_semaphore.try_acquire_for(std::chrono::seconds(5));
    if (!ok) {
        std::cerr << "IPC: Failed to acquire run semaphore, closing process.\n";
        exit(1);
    }
}

void IPC::SendRestart(const std::vector<std::string>& args) {
    std::cerr << ";RESTART\n";
    std::cerr << ";" << args.size() << "\n";
    for (const auto& arg : args) {
        std::cerr << ";" << arg << "\n";
    }
    std::cerr.flush();
}

void IPC::InputLoop() {
    auto next_str = [&] -> const std::string& {
        static std::string line_buffer;
        do {
            std::getline(std::cin, line_buffer, '\n');
        } while (!line_buffer.empty() && line_buffer.back() == '\\');
        return line_buffer;
    };
    auto next_u64 = [&] -> u64 {
        auto& str = next_str();
        return std::stoull(str, nullptr, 0);
    };

    while (true) {
        auto& cmd = next_str();
        if (cmd.empty()) {
            continue;
        }
        if (cmd == "RUN") {
            run_semaphore.release();
        } else if (cmd == "START") {
            start_semaphore.release();
        } else if (cmd == "PATCH_MEMORY") {
            const MemoryPatcher::patchInfo entry = {
                .gameSerial = "*",
                .modNameStr = next_str(),
                .offsetStr = next_str(),
                .valueStr = next_str(),
                .targetStr = next_str(),
                .sizeStr = next_str(),
                .isOffset = next_u64() != 0,
                .littleEndian = next_u64() != 0,
                .patchMask = static_cast<MemoryPatcher::PatchMask>(next_u64()),
                .maskOffset = static_cast<int>(next_u64()),
            };
            MemoryPatcher::AddPatchToQueue(entry);
        } else if (cmd == "PAUSE") {
            DebugState.PauseGuestThreads();
        } else if (cmd == "RESUME") {
            DebugState.ResumeGuestThreads();
        } else if (cmd == "STOP") {
            SDL_Event event;
            SDL_memset(&event, 0, sizeof(event));
            event.type = SDL_EVENT_QUIT;
            SDL_PushEvent(&event);
        } else if (cmd == "TOGGLE_FULLSCREEN") {
            SDL_Event event;
            SDL_memset(&event, 0, sizeof(event));
            event.type = SDL_EVENT_TOGGLE_FULLSCREEN;
            SDL_PushEvent(&event);
        } else if (cmd == "SCREENSHOT") {
            VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);
        } else if (cmd == "RENDERDOC_CAPTURE") {
            VideoCore::TriggerCapture();
        } else if (cmd == "GAMEPAD_BUTTON") {
            const std::string name = next_str();
            const auto button = Input::ParseControllerButton(name);
            const bool pressed = next_u64() != 0;
            if (!button) {
                std::cerr << ";INVALID GAMEPAD BUTTON: " << name << '\n';
                std::cerr.flush();
                continue;
            }
            SDL_Event event;
            SDL_memset(&event, 0, sizeof(event));
            event.type = SDL_EVENT_INJECT_GAMEPAD_BUTTON;
            event.user.code = static_cast<Sint32>(*button);
            event.user.data1 = reinterpret_cast<void*>(static_cast<uintptr_t>(pressed));
            SDL_PushEvent(&event);
        } else if (cmd == "GAMEPAD_AXIS") {
            const std::string name = next_str();
            const auto axis = Input::ParseControllerAxis(name);
            const u64 value = next_u64();
            if (!axis || value > 255) {
                std::cerr << ";INVALID GAMEPAD AXIS: " << name << ' ' << value << '\n';
                std::cerr.flush();
                continue;
            }
            SDL_Event event;
            SDL_memset(&event, 0, sizeof(event));
            event.type = SDL_EVENT_INJECT_GAMEPAD_AXIS;
            event.user.code = static_cast<Sint32>(*axis);
            event.user.data1 = reinterpret_cast<void*>(static_cast<uintptr_t>(value));
            SDL_PushEvent(&event);
        } else if (cmd == "GAMEPAD_TOUCH") {
            const u64 finger = next_u64();
            const u64 down = next_u64();
            const u64 x = next_u64();
            const u64 y = next_u64();
            if (down > 1 || !Input::IsValidControllerTouch(finger, x, y)) {
                std::cerr << ";INVALID GAMEPAD TOUCH: " << finger << ' ' << down << ' ' << x << ' '
                          << y << '\n';
                std::cerr.flush();
                continue;
            }
            SDL_Event event;
            SDL_memset(&event, 0, sizeof(event));
            event.type = SDL_EVENT_INJECT_GAMEPAD_TOUCH;
            event.user.code = static_cast<Sint32>(finger);
            event.user.data1 = reinterpret_cast<void*>(static_cast<uintptr_t>(
                Input::PackControllerTouch(static_cast<u16>(x), static_cast<u16>(y), down != 0)));
            SDL_PushEvent(&event);
        } else if (cmd == "ADJUST_VOLUME") {
            int value = static_cast<int>(next_u64());
            bool is_game_specific = next_u64() != 0;
            EmulatorSettings.SetVolumeSlider(value, is_game_specific);
            Libraries::AudioOut::AdjustVol();
        } else if (cmd == "SET_FSR") {
            bool use_fsr = next_u64() != 0;
            if (presenter) {
                presenter->GetFsrSettingsRef().enable = use_fsr;
            }
        } else if (cmd == "SET_RCAS") {
            bool use_rcas = next_u64() != 0;
            if (presenter) {
                presenter->GetFsrSettingsRef().use_rcas = use_rcas;
            }
        } else if (cmd == "SET_RCAS_ATTENUATION") {
            int value = static_cast<int>(next_u64());
            if (presenter) {
                presenter->GetFsrSettingsRef().rcas_attenuation =
                    static_cast<float>(value / 1000.0f);
            }
        } else if (cmd == "USB_LOAD_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                std::string file_name = next_str();
                const u8 pad = next_u64();
                const u8 slot = next_u64();
                ref->LoadFigure(file_name, pad, slot);
            }
        } else if (cmd == "USB_REMOVE_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                const u8 pad = next_u64();
                const u8 slot = next_u64();
                bool full_remove = next_u64() != 0;
                ref->RemoveFigure(pad, slot, full_remove);
            }
        } else if (cmd == "USB_MOVE_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                const u8 new_pad = next_u64();
                const u8 new_index = next_u64();
                const u8 old_pad = next_u64();
                const u8 old_index = next_u64();
                ref->MoveFigure(new_pad, new_index, old_pad, old_index);
            }
        } else if (cmd == "USB_TEMP_REMOVE_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                const u8 index = next_u64();
                ref->TempRemoveFigure(index);
            }
        } else if (cmd == "USB_CANCEL_REMOVE_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                const u8 index = next_u64();
                ref->CancelRemoveFigure(index);
            }
        } else if (cmd == "RELOAD_INPUTS") {
            std::string config = next_str();
            Input::ParseInputConfig(config);
        } else {
            std::cerr << ";UNKNOWN CMD: " << cmd << std::endl;
        }
    }
}
