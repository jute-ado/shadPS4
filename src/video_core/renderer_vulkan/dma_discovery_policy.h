// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace VideoCore {

enum class DmaAttachmentMode {
    Discovery,
    Publication,
};

struct DmaAttachmentPolicy {
    bool load_clear;
    bool consume_metadata;
};

constexpr DmaAttachmentPolicy ResolveDmaAttachmentPolicy(bool source_is_clear,
                                                         DmaAttachmentMode mode) {
    if (mode == DmaAttachmentMode::Discovery) {
        return {};
    }
    return {.load_clear = source_is_clear, .consume_metadata = true};
}

} // namespace VideoCore
