// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cerrno>
#include <limits>
#include <string>
#include "core/libraries/avplayer/avplayer_file_streamer.h"
#include "core/libraries/avplayer/avplayer_stream_policy.h"

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
#include <libavutil/mem.h>
}

constexpr u32 AVPLAYER_AVIO_BUFFER_SIZE = 4096;

namespace Libraries::AvPlayer {

AvPlayerFileStreamer::AvPlayerFileStreamer(const AvPlayerFileReplacement& file_replacement)
    : m_file_replacement(file_replacement) {}

AvPlayerFileStreamer::~AvPlayerFileStreamer() {
    if (m_avio_context != nullptr) {
        avio_context_free(&m_avio_context);
    }
    if (m_file_replacement.close != nullptr && m_fd >= 0) {
        const auto close = m_file_replacement.close;
        const auto ptr = m_file_replacement.object_ptr;
        close(ptr);
    }
}

bool AvPlayerFileStreamer::Init(std::string_view path) {
    if (m_fd >= 0 || m_avio_context != nullptr) {
        return false;
    }
    const auto ptr = m_file_replacement.object_ptr;
    const std::string null_terminated_path{path};
    m_fd = m_file_replacement.open(ptr, null_terminated_path.c_str());
    if (m_fd < 0) {
        return false;
    }
    m_file_size = m_file_replacement.size(ptr);
    // avio_buffer is deallocated in `avio_context_free`
    const auto avio_buffer = reinterpret_cast<u8*>(av_malloc(AVPLAYER_AVIO_BUFFER_SIZE));
    m_avio_context =
        avio_alloc_context(avio_buffer, AVPLAYER_AVIO_BUFFER_SIZE, 0, this,
                           &AvPlayerFileStreamer::ReadPacket, nullptr, &AvPlayerFileStreamer::Seek);
    return true;
}

void AvPlayerFileStreamer::Reset() {
    m_position = 0;
}

s32 AvPlayerFileStreamer::ReadPacket(void* opaque, u8* buffer, s32 size) {
    const auto self = reinterpret_cast<AvPlayerFileStreamer*>(opaque);
    if (self->m_position >= self->m_file_size) {
        return AVERROR_EOF;
    }
    const auto request_size = ResolveFileReadRequest(self->m_position, self->m_file_size, size);
    if (!request_size) {
        return AVERROR(EINVAL);
    }
    const auto read_offset = self->m_file_replacement.read_offset;
    const auto ptr = self->m_file_replacement.object_ptr;
    const auto bytes_read = read_offset(ptr, buffer, self->m_position, *request_size);
    const auto result = ResolveFileReadResult(self->m_position, *request_size, bytes_read);
    if (result.return_value == 0 && *request_size != 0) {
        return AVERROR_EOF;
    }
    self->m_position = result.next_position;
    return result.return_value;
}

s64 AvPlayerFileStreamer::Seek(void* opaque, s64 offset, int whence) {
    const auto self = reinterpret_cast<AvPlayerFileStreamer*>(opaque);
    const auto seek_limit =
        std::min(self->m_file_size, static_cast<u64>(std::numeric_limits<s64>::max()));
    if (whence & AVSEEK_SIZE) {
        return static_cast<s64>(seek_limit);
    }

    if (whence == SEEK_CUR) {
        self->m_position = ResolveFileSeek(self->m_position, seek_limit, offset);
        return static_cast<s64>(self->m_position);
    } else if (whence == SEEK_SET) {
        self->m_position = ResolveFileSeek(0, seek_limit, offset);
        return static_cast<s64>(self->m_position);
    } else if (whence == SEEK_END) {
        self->m_position = ResolveFileSeek(seek_limit, seek_limit, offset);
        return static_cast<s64>(self->m_position);
    }

    return -1;
}

} // namespace Libraries::AvPlayer
