// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/debug.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/eop_flip_presentation.h"
#include "core/libraries/videoout/videoout_error.h"
#include "imgui/renderer/imgui_core.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Libraries::VideoOut {

constexpr static bool Is32BppPixelFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8R8G8B8Srgb:
    case PixelFormat::A8B8G8R8Srgb:
    case PixelFormat::A2R10G10B10:
    case PixelFormat::A2R10G10B10Srgb:
    case PixelFormat::A2R10G10B10Bt2020Pq:
        return true;
    default:
        return false;
    }
}

constexpr u32 PixelFormatBpp(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::A16R16G16B16Float:
        return 8;
    default:
        return 4;
    }
}

VideoOutDriver::VideoOutDriver(u32 width, u32 height) {
    main_port.resolution.full_width = width;
    main_port.resolution.full_height = height;
    main_port.resolution.pane_width = width;
    main_port.resolution.pane_height = height;
    present_thread = std::jthread([&](std::stop_token token) { PresentThread(token); });
}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::Open(const ServiceThreadParams* params) {
    std::scoped_lock port_lock{main_port.port_mutex};
    std::scoped_lock lock{mutex};
    if (main_port.lifecycle.IsOpen()) {
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }
    main_port.lifecycle.Open();
    liverpool->SetVoPort(&main_port);
    return 1;
}

void VideoOutDriver::Close(s32 handle) {
    std::scoped_lock port_lock{main_port.port_mutex};
    std::scoped_lock lock{mutex};

    // Mark as closed
    main_port.lifecycle.Close();
    main_port.flip_rate = 0;
    main_port.prev_index = -1;

    // Requests are tied to the open lifetime that created them. Discard queued requests while the
    // queue lock is held so a close/reopen cannot present a stale frame on the new lifetime.
    DrainVideoOutRequests(requests);

    // Clear port information. VO labels are polled by the graphics owner while the presenter
    // thread may be resetting them, so publish the reset under the waiter's mutex and wake any
    // owner that was waiting for a label transition.
    {
        std::scoped_lock lock{main_port.vo_mutex};
        std::memset(main_port.buffer_labels.data(), 0, sizeof(main_port.buffer_labels));
        main_port.vo_cv.notify_all();

        for (auto event : main_port.vblank_events) {
            auto equeue = Kernel::GetEqueue(event);
            if (equeue != nullptr) {
                equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                    Kernel::OrbisKernelEvent::Filter::VideoOut);
            }
        }
        main_port.vblank_events.clear();
        std::memset(&main_port.vblank_status, 0, sizeof(main_port.vblank_status));
        main_port.vblank_cv.notify_all();
    }
    std::memset(main_port.groups.data(), 0, sizeof(main_port.groups));
    main_port.flip_status = FlipStatus{};

    // Re-initialize buffers
    std::memset(main_port.buffer_slots.data(), 0, sizeof(main_port.buffer_slots));
    for (auto& buffer : main_port.buffer_slots) {
        buffer.group_index = -1;
    }

    // Clear events
    for (auto event : main_port.flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.flip_events.clear();
}

VideoOutPort* VideoOutDriver::GetPort(int handle) {
    if (handle != 1) [[unlikely]] {
        return nullptr;
    }
    return &main_port;
}

int VideoOutDriver::RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses,
                                    s32 bufferNum, const BufferAttribute* attribute) {
    std::scoped_lock port_lock{port->port_mutex};
    if (!port->lifecycle.IsOpen()) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    const s32 group_index = port->FindFreeGroup();
    if (group_index >= MaxDisplayBufferGroups) {
        return ORBIS_VIDEO_OUT_ERROR_NO_EMPTY_SLOT;
    }

    if (startIndex + bufferNum > MaxDisplayBuffers || startIndex > MaxDisplayBuffers ||
        bufferNum > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut,
                  "Attempted to register too many buffers startIndex = {}, bufferNum = {}",
                  startIndex, bufferNum);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    const s32 end_index = startIndex + bufferNum;
    if (bufferNum > 0 &&
        std::any_of(port->buffer_slots.begin() + startIndex, port->buffer_slots.begin() + end_index,
                    [](auto& buffer) { return buffer.group_index != -1; })) {
        return ORBIS_VIDEO_OUT_ERROR_SLOT_OCCUPIED;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "startIndex = {}, bufferNum = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             startIndex, bufferNum, GetPixelFormatString(attribute->pixel_format),
             attribute->aspect_ratio, static_cast<u32>(attribute->tiling_mode), attribute->width,
             attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    std::memcpy(&group.attrib, attribute, sizeof(BufferAttribute));
    group.is_occupied = true;

    for (u32 i = 0; i < bufferNum; i++) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffer_slots[startIndex + i] = VideoOutBuffer{
            .group_index = group_index,
            .address_left = address,
            .address_right = 0,
        };

        // Reset flip label also when registering buffer. This is synchronized with graphics
        // WaitRegMem polls of the host-owned label array.
        {
            std::scoped_lock lock{port->vo_mutex};
            port->buffer_labels[startIndex + i] = 0;
        }
        port->SignalVoLabel();

        presenter->RegisterVideoOutSurface(group, address);
        LOG_INFO(Lib_VideoOut, "buffers[{}] = {:#x}", i + startIndex, address);
    }

    return group_index;
}

int VideoOutDriver::UnregisterBuffers(VideoOutPort* port, s32 attributeIndex) {
    std::scoped_lock port_lock{port->port_mutex};
    if (!port->lifecycle.IsOpen()) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    if (attributeIndex < 0 || attributeIndex >= MaxDisplayBufferGroups ||
        !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto& group = port->groups[attributeIndex];
    group.is_occupied = false;

    for (auto& buffer : port->buffer_slots) {
        if (buffer.group_index != attributeIndex) {
            continue;
        }
        buffer.group_index = -1;
    }

    return ORBIS_OK;
}

int VideoOutDriver::ChangeBufferAttribute(VideoOutPort* port, s32 attributeIndex,
                                          const BufferAttribute* attribute) {
    std::scoped_lock port_lock{port->port_mutex};
    if (!port->lifecycle.IsOpen()) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    if (attributeIndex < 0 || attributeIndex >= MaxDisplayBufferGroups ||
        !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "attributeIndex = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             attributeIndex, GetPixelFormatString(attribute->pixel_format), attribute->aspect_ratio,
             static_cast<u32>(attribute->tiling_mode), attribute->width, attribute->height,
             attribute->pitch_in_pixel, attribute->option);

    std::memcpy(&port->groups[attributeIndex].attrib, attribute, sizeof(BufferAttribute));
    return 0;
}

void VideoOutDriver::Flip(const Request& req) {
    auto* port = req.port;
    std::unique_lock port_lock{port->port_mutex};
    if (!port->lifecycle.IsCurrent(req.lifecycle_epoch)) {
        return;
    }

    // Update HDR status before presenting.
    presenter->SetHDR(port->is_hdr);

    // Present the frame.
    presenter->Present(req.frame);

    // Update flip status.
    auto& flip_status = port->flip_status;
    flip_status.count++;
    flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
    flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
    flip_status.flip_arg = req.flip_arg;
    flip_status.current_buffer = req.index;
    if (req.eop) {
        --flip_status.gc_queue_num;
    }
    --flip_status.flip_pending_num;

    // Trigger flip events for the port.
    for (auto event : port->flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->TriggerEvent(
                static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                        (req.flip_arg << 16)));
        }
    }

    // Reset the previous flip label only when no newer request has reserved this buffer. A
    // queued request can publish its label before an older request reaches the presenter.
    bool released_label = false;
    bool release_was_eligible = false;
    s32 previous_index = -1;
    previous_index = port->prev_index;
    release_was_eligible =
        previous_index != -1 && port->flip_label_generations[previous_index].CanReleaseDisplayed();
    if (release_was_eligible) {
        // Keep the port lock while taking vo_mutex so a newer reservation cannot slip between the
        // generation check and the label clear, and so Close cannot reset the port concurrently.
        std::scoped_lock vo_lock{port->vo_mutex};
        port->buffer_labels[previous_index] = 0;
        released_label = true;
    }
    if (req.index != -1) {
        port->flip_label_generations[req.index].Display(req.generation);
    }
    port->prev_index = req.index;
    port_lock.unlock();
    if (released_label) {
        port->SignalVoLabel();
    }
}

void VideoOutDriver::DrawBlankFrame() {
    const auto empty_frame = presenter->PrepareBlankFrame(true);
    presenter->Present(empty_frame);
}

void VideoOutDriver::DrawLastFrame() {
    const auto frame = presenter->PrepareLastFrame();
    if (frame != nullptr) {
        presenter->Present(frame, true);
    }
}

u64 VideoOutDriver::ReserveFlipLabel(VideoOutPort* port, s32 index) {
    if (index == -1) {
        return 0;
    }
    std::scoped_lock lock{port->port_mutex};
    return port->flip_label_generations[index].Reserve();
}

bool VideoOutDriver::SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop,
                                u64 generation, u64 lifecycle_epoch) {
    u64 request_epoch;
    {
        std::unique_lock lock{port->port_mutex};
        if (!port->lifecycle.IsOpen() ||
            (lifecycle_epoch != 0 && port->lifecycle.CurrentEpoch() != lifecycle_epoch)) {
            return false;
        }
        request_epoch = port->lifecycle.CurrentEpoch();
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "Flip queue is full");
            return false;
        }

        if (is_eop) {
            ++port->flip_status.gc_queue_num;
        }
        if (is_eop && index != -1 && generation == 0) {
            generation = port->flip_label_generations[index].Reserve();
        }
        // EOP registration reserves a generation before the GPU reaches its GfxFlip IRQ. Mark it
        // complete only at that IRQ/SubmitFlip boundary; a newer reservation must not postpone
        // release of the older presentation indefinitely.
        if (is_eop && index != -1) {
            port->flip_label_generations[index].Complete(generation);
        }
        ++port->flip_status.flip_pending_num; // integral GPU and CPU pending flips counter
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
    }

    if (!is_eop) {
        // Non EOP flips can arrive from any thread so ask GPU thread to perform them
        liverpool->SendCommand([=, this]() {
            SubmitFlipInternal(port, index, flip_arg, is_eop, generation, request_epoch);
        });
    } else {
        SubmitFlipInternal(port, index, flip_arg, is_eop, generation, request_epoch);
    }

    return true;
}

void VideoOutDriver::SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop,
                                        u64 generation, u64 lifecycle_epoch) {
    const auto discard_pending = [&] {
        // A callback can outlive its port lifetime. Only roll back accounting when the request
        // still belongs to the current open lifetime; Close resets the counters, and a reopened
        // port may already have admitted unrelated flips.
        if (!port->lifecycle.IsCurrent(lifecycle_epoch)) {
            return;
        }
        if (port->flip_status.flip_pending_num > 0) {
            --port->flip_status.flip_pending_num;
        }
        if (is_eop && port->flip_status.gc_queue_num > 0) {
            --port->flip_status.gc_queue_num;
        }
    };

    Vulkan::Frame* frame;
    u64 submitted_vblank;
    {
        std::scoped_lock port_lock{port->port_mutex};
        if (!port->lifecycle.IsCurrent(lifecycle_epoch)) {
            discard_pending();
            return;
        }
        if (index == -1) {
            frame = presenter->PrepareBlankFrame(false);
        } else {
            const auto& buffer = port->buffer_slots[index];
            if (buffer.group_index < 0) {
                discard_pending();
                return;
            }
            ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
            const auto& group = port->groups[buffer.group_index];
            frame = presenter->PrepareFrame(group, buffer.address_left);
        }

        std::scoped_lock vo_lock{port->vo_mutex};
        submitted_vblank = port->vblank_status.count;
    }

    std::unique_lock lock{mutex};
    if (!port->lifecycle.IsCurrent(lifecycle_epoch)) {
        // Close takes port_mutex before mutex. Release the queue lock before touching the port so
        // this stale callback cannot invert that order while rolling back its own accounting.
        lock.unlock();
        {
            std::scoped_lock port_lock{port->port_mutex};
            discard_pending();
        }
        return;
    }
    requests.push({
        .frame = frame,
        .port = port,
        .flip_arg = flip_arg,
        .index = index,
        .eop = is_eop,
        .generation = generation,
        .lifecycle_epoch = lifecycle_epoch,
        .submitted_vblank = submitted_vblank,
    });
}

void VideoOutDriver::PresentThread(std::stop_token token) {
    const std::chrono::nanoseconds vblank_period(1000000000 /
                                                 EmulatorSettings.GetVblankFrequency());

    Common::SetCurrentThreadName("shadPS4:PresentThread");
    Common::SetCurrentThreadRealtime(vblank_period);

    Common::AccurateTimer timer{vblank_period};

    const auto receive_request = [this](u64 current_vblank) -> Request {
        std::scoped_lock lk{mutex};
        if (!requests.empty() &&
            IsFlipPresentationEligible(requests.front().eop, requests.front().submitted_vblank,
                                       current_vblank)) {
            const auto request = requests.front();
            requests.pop();
            return request;
        }
        return {};
    };

    while (!token.stop_requested()) {
        timer.Start();

        if (DebugState.IsGuestThreadsPaused()) {
            DrawLastFrame();
            timer.End();
            continue;
        }

        // Check if it's time to take a request.
        u64 current_vblank;
        s32 flip_rate;
        {
            // Keep the lock order identical to Close/Flip: port state first, then vblank state.
            // The presenter must not race a guest SetFlipRate or a close/reopen transition.
            std::scoped_lock port_lock{main_port.port_mutex};
            std::scoped_lock vo_lock{main_port.vo_mutex};
            current_vblank = main_port.vblank_status.count;
            flip_rate = main_port.flip_rate;
        }
        if (current_vblank % (flip_rate + 1) == 0) {
            const auto request = receive_request(current_vblank);
            if (!request) {
                if (timer.GetTotalWait().count() < 0) { // Dont draw too fast
                    if (!main_port.lifecycle.IsOpen()) {
                        DrawBlankFrame();
                    } else if (ImGui::Core::MustKeepDrawing()) {
                        DrawLastFrame();
                    }
                }
            } else {
                Flip(request);
                FRAME_END;
            }
        }

        {
            // Needs lock here as can be concurrently read by `sceVideoOutGetVblankStatus`
            std::scoped_lock lock{main_port.vo_mutex};

            // Trigger flip events for the port
            for (auto event : main_port.vblank_events) {
                auto equeue = Kernel::GetEqueue(event);
                if (equeue != nullptr) {
                    equeue->TriggerEvent(
                        static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                        Kernel::OrbisKernelEvent::Filter::VideoOut,
                        reinterpret_cast<void*>(
                            static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                            (main_port.vblank_status.count << 16)));
                }
            }

            // Update vblank status
            main_port.vblank_status.count++;
            main_port.vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            main_port.vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            main_port.vblank_cv.notify_all();
        }

        timer.End();
    }
}

} // namespace Libraries::VideoOut
