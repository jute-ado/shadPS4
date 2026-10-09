// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/path_util.h"
#include "common/singleton.h"
#include "common/types.h"

#include <functional>
#include <thread>
#include <vector>

namespace Storage {

enum class BlobType : u32 {
    ShaderMeta,
    ShaderBinary,
    PipelineKey,
    ShaderProfile,
};

[[nodiscard]] constexpr size_t MaxCacheBlobBytes(BlobType type) noexcept {
    switch (type) {
    case BlobType::ShaderBinary:
        return 4U << 20;
    case BlobType::ShaderMeta:
    case BlobType::PipelineKey:
        return 16U << 20;
    case BlobType::ShaderProfile:
        return 1U << 20;
    }
    return 0;
}

[[nodiscard]] constexpr bool IsCacheBlobSizeValid(BlobType type, u64 byte_size,
                                                  size_t element_size) noexcept {
    return element_size != 0 && byte_size <= MaxCacheBlobBytes(type) &&
           byte_size % element_size == 0;
}

template <typename T>
[[nodiscard]] bool PrepareCacheBlobBuffer(BlobType type, u64 byte_size,
                                          std::vector<T>& output) noexcept {
    output.clear();
    if (!IsCacheBlobSizeValid(type, byte_size, sizeof(T))) {
        return false;
    }
    try {
        output.resize(static_cast<size_t>(byte_size / sizeof(T)));
    } catch (...) {
        output.clear();
        return false;
    }
    return true;
}

class DataBase {
public:
    static DataBase& Instance() {
        return *Common::Singleton<DataBase>::Instance();
    }

    void Open();
    void Close();
    [[nodiscard]] bool IsOpened() const {
        return opened;
    }
    void FinishPreload();

    bool Save(BlobType type, const std::string& name, std::vector<u8>&& data);
    bool Save(BlobType type, const std::string& name, std::vector<u32>&& data);

    void Load(BlobType type, const std::string& name, std::vector<u8>& data);
    void Load(BlobType type, const std::string& name, std::vector<u32>& data);

    void ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func);

private:
    std::jthread io_worker{};
    std::filesystem::path cache_path{};
    bool opened{};
};

} // namespace Storage
