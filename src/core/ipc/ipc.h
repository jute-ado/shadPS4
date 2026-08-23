//  SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
//  SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/singleton.h"
#include "core/ipc/presented_frame_input_queue.h"

#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

class IPC {
    bool enabled{false};
    std::jthread input_thread{};

    std::binary_semaphore run_semaphore{0};
    std::binary_semaphore start_semaphore{0};

public:
    static IPC& Instance() {
        return *Common::Singleton<IPC>::Instance();
    }

    void Init();

    operator bool() const {
        return enabled;
    }

    [[nodiscard]] bool IsEnabled() const {
        return enabled;
    }

    void WaitForStart() {
        start_semaphore.acquire();
    }

    void SendRestart(const std::vector<std::string>& args);

    void DispatchPresentedFrameInputs(u64 presented_frame);

private:
    bool SchedulePresentedFrameInput(const Core::Ipc::PresentedFrameInputEvent& event);

    [[noreturn]] void InputLoop();

    std::mutex presented_frame_input_mutex;
    Core::Ipc::PresentedFrameInputQueue presented_frame_input_queue;
};
