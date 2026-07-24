// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "playgo_chunk.h"

#include <algorithm>
#include <cstring>

bool PlaygoFile::Open(const std::filesystem::path& filepath) {
    Common::FS::IOFile file(filepath, Common::FS::FileAccessMode::Read);
    if (!file.IsOpen() || file.Read(playgoHeader) != 1 ||
        !ValidatePlaygoHeaderLayout(playgoHeader, file.GetSize())) {
        playgoHeader = {};
        chunks.clear();
        return false;
    }
    if (!LoadChunks(file)) {
        playgoHeader = {};
        chunks.clear();
        return false;
    }
    return true;
}

bool PlaygoFile::LoadChunks(const Common::FS::IOFile& file) {
    if (file.IsOpen()) {
        if (playgoHeader.magic == PLAYGO_MAGIC) {
            bool ret = true;

            std::string chunk_attrs_data, chunk_mchunks_data, chunk_labels_data, mchunk_attrs_data;
            ret = ret && load_chunk_data(file, playgoHeader.chunk_attrs, chunk_attrs_data);
            ret = ret && load_chunk_data(file, playgoHeader.chunk_mchunks, chunk_mchunks_data);
            ret = ret && load_chunk_data(file, playgoHeader.chunk_labels, chunk_labels_data);
            ret = ret && load_chunk_data(file, playgoHeader.mchunk_attrs, mchunk_attrs_data);

            if (ret) {
                chunks.resize(playgoHeader.chunk_count);

                for (u16 i = 0; i < playgoHeader.chunk_count; i++) {
                    playgo_chunk_attr_entry_t chunk_attr{};
                    std::memcpy(&chunk_attr,
                                chunk_attrs_data.data() +
                                    static_cast<size_t>(i) * sizeof(chunk_attr),
                                sizeof(chunk_attr));

                    if (chunk_attr.label_offset >= chunk_labels_data.size()) {
                        return false;
                    }
                    const auto label_begin =
                        chunk_labels_data.begin() + chunk_attr.label_offset;
                    const auto label_end =
                        std::find(label_begin, chunk_labels_data.end(), '\0');
                    if (label_end == chunk_labels_data.end()) {
                        return false;
                    }

                    chunks[i].req_locus = chunk_attr.req_locus;
                    chunks[i].language_mask = chunk_attr.language_mask;
                    chunks[i].label_name.assign(label_begin, label_end);

                    u64 total_size = 0;
                    const u16 mchunk_count = chunk_attr.mchunk_count;
                    if (mchunk_count != 0) {
                        if (!IsPlaygoSubrangeWithinSection(
                                chunk_attr.mchunks_offset, mchunk_count, sizeof(u16),
                                chunk_mchunks_data.size())) {
                            return false;
                        }

                        for (u16 j = 0; j < mchunk_count; j++) {
                            u16 mchunk_id{};
                            std::memcpy(&mchunk_id,
                                        chunk_mchunks_data.data() + chunk_attr.mchunks_offset +
                                            static_cast<size_t>(j) * sizeof(mchunk_id),
                                        sizeof(mchunk_id));
                            if (mchunk_id >= playgoHeader.mchunk_count) {
                                return false;
                            }

                            playgo_mchunk_attr_entry_t mchunk_attr{};
                            std::memcpy(&mchunk_attr,
                                        mchunk_attrs_data.data() +
                                            static_cast<size_t>(mchunk_id) * sizeof(mchunk_attr),
                                        sizeof(mchunk_attr));
                            total_size += mchunk_attr.size.size;
                        }
                    }
                    chunks[i].total_size = total_size;
                }
            }

            return ret;
        }
    }
    return false;
}

bool PlaygoFile::load_chunk_data(const Common::FS::IOFile& file, const chunk_t chunk,
                                 std::string& data) {
    if (file.IsOpen()) {
        if (file.Seek(chunk.offset)) {
            data.resize(chunk.length);
            return data.empty() ||
                   file.ReadRaw<char>(data.data(), data.size()) == data.size();
        }
    }
    return false;
}
