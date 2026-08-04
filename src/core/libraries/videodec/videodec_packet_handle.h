// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

extern "C" {
#include <libavcodec/packet.h>
}

namespace Libraries::Videodec {

struct PacketDeleter {
    void operator()(AVPacket* packet) const {
        av_packet_free(&packet);
    }
};

using PacketHandle = std::unique_ptr<AVPacket, PacketDeleter>;

inline PacketHandle AdoptPacket(AVPacket* packet) {
    return PacketHandle{packet};
}

} // namespace Libraries::Videodec
