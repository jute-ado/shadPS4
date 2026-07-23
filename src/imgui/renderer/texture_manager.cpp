// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <deque>
#include <utility>

#include <imgui.h>
#include "common/assert.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "imgui_impl_vulkan.h"
#include "png_decode.h"
#include "texture_job_queue.h"
#include "texture_manager.h"

namespace ImGui {

namespace Core::TextureManager {
struct Inner {
    std::atomic_int count = 0;
    ImTextureID texture_id = nullptr;
    u32 width = 0;
    u32 height = 0;

    Vulkan::UploadTextureData upload_data;

    ~Inner();
};
} // namespace Core::TextureManager

using namespace Core::TextureManager;

RefCountedTexture::RefCountedTexture(Inner* inner) : inner(inner) {
    ++inner->count;
}

RefCountedTexture RefCountedTexture::DecodePngTexture(std::vector<u8> data) {
    const auto core = new Inner;
    Core::TextureManager::DecodePngTexture(std::move(data), core);
    return RefCountedTexture(core);
}

RefCountedTexture RefCountedTexture::DecodePngFile(std::filesystem::path path) {
    const auto core = new Inner;
    Core::TextureManager::DecodePngFile(std::move(path), core);
    return RefCountedTexture(core);
}

RefCountedTexture::RefCountedTexture() : inner(nullptr) {}

RefCountedTexture::RefCountedTexture(const RefCountedTexture& other) : inner(other.inner) {
    if (inner != nullptr) {
        ++inner->count;
    }
}

RefCountedTexture::RefCountedTexture(RefCountedTexture&& other) noexcept : inner(other.inner) {
    other.inner = nullptr;
}

RefCountedTexture& RefCountedTexture::operator=(const RefCountedTexture& other) {
    if (this == &other)
        return *this;
    inner = other.inner;
    if (inner != nullptr) {
        ++inner->count;
    }
    return *this;
}

RefCountedTexture& RefCountedTexture::operator=(RefCountedTexture&& other) noexcept {
    if (this == &other)
        return *this;
    std::swap(inner, other.inner);
    return *this;
}

RefCountedTexture::~RefCountedTexture() {
    if (inner != nullptr) {
        if (inner->count.fetch_sub(1) == 1) {
            delete inner;
        }
    }
}

RefCountedTexture::Image RefCountedTexture::GetTexture() const {
    if (inner == nullptr) {
        return {};
    }
    return Image{
        .im_id = inner->texture_id,
        .width = inner->width,
        .height = inner->height,
    };
}

RefCountedTexture::operator bool() const {
    return inner != nullptr && inner->texture_id != nullptr;
}

struct Job {
    Inner* core;
    std::vector<u8> data;
    std::filesystem::path path;
};

struct UploadJob {
    Inner* core = nullptr;
    Vulkan::UploadTextureData data;
    int tick = 0; // Used to skip the first frame when destroying to await the current frame to draw
};

static std::jthread g_worker_thread;
static JobQueue<Job> g_job_queue;

static std::mutex g_upload_mtx;
static std::deque<UploadJob> g_upload_list;

namespace Core::TextureManager {

void ReleaseWorkerReference(Inner* core) {
    if (core->count.fetch_sub(1) == 1) {
        delete core;
    }
}

Inner::~Inner() {
    if (upload_data.im_texture != nullptr) {
        std::unique_lock lk{g_upload_mtx};
        g_upload_list.emplace_back(UploadJob{
            .data = this->upload_data,
            .tick = 2,
        });
    }
}

void WorkerLoop() {
    Common::SetCurrentThreadName("shadPS4:ImGuiTextureManager");
    while (auto job = g_job_queue.WaitPop()) {
        auto [core, png_raw, path] = std::move(*job);

        if (EmulatorSettings.IsVkCrashDiagnosticEnabled()) {
            // FIXME: Crash diagnostic hangs when building the command buffer here
            ReleaseWorkerReference(core);
            continue;
        }

        if (!path.empty()) { // Decode PNG from file
            Common::FS::IOFile file(path, Common::FS::FileAccessMode::Read);
            if (!file.IsOpen()) {
                LOG_ERROR(ImGui, "Failed to open PNG file: {}", path.string());
                ReleaseWorkerReference(core);
                continue;
            }
            png_raw.resize(file.GetSize());
            file.Seek(0);
            file.ReadRaw<u8>(png_raw.data(), png_raw.size());
            file.Close();
        }

        auto decoded = DecodePngRgba(png_raw);
        if (!decoded) {
            LOG_ERROR(ImGui, "Failed to decode PNG texture{}",
                      path.empty() ? "" : ": " + path.string());
            ReleaseWorkerReference(core);
            continue;
        }

        auto texture = Vulkan::UploadTexture(decoded->pixels.get(), vk::Format::eR8G8B8A8Unorm,
                                             decoded->width, decoded->height, decoded->SizeBytes());

        core->upload_data = texture;
        core->width = decoded->width;
        core->height = decoded->height;

        std::unique_lock upload_lk{g_upload_mtx};
        g_upload_list.emplace_back(UploadJob{
            .core = core,
        });
    }
}

void StartWorker() {
    ASSERT(!g_worker_thread.joinable());
    g_job_queue.Start();
    g_worker_thread = std::jthread(WorkerLoop);
}

void StopWorker() {
    ASSERT(g_worker_thread.joinable());
    g_job_queue.Stop();
    g_worker_thread.join();
}

void DecodePngTexture(std::vector<u8> data, Inner* core) {
    ++core->count;
    Job job{
        .core = core,
        .data = std::move(data),
    };
    ASSERT(g_job_queue.Push(std::move(job)));
}

void DecodePngFile(std::filesystem::path path, Inner* core) {
    ++core->count;
    Job job{
        .core = core,
        .path = std::move(path),
    };
    ASSERT(g_job_queue.Push(std::move(job)));
}

void Submit() {
    UploadJob upload;
    {
        std::unique_lock lk{g_upload_mtx};
        if (g_upload_list.empty()) {
            return;
        }
        // Upload one texture at a time to avoid slow down
        upload = g_upload_list.front();
        g_upload_list.pop_front();
        if (upload.tick > 0) {
            --upload.tick;
            g_upload_list.emplace_back(upload);
            return;
        }
    }
    if (upload.core != nullptr) {
        upload.core->upload_data.Upload();
        upload.core->texture_id = upload.core->upload_data.im_texture;
        ReleaseWorkerReference(upload.core);
    } else {
        upload.data.Destroy();
    }
}
} // namespace Core::TextureManager

} // namespace ImGui
