// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace VideoCore {

class ImageWriteHistory {
public:
    // Conservative lifetime provenance, independent of refresh hashes and readback/GC flags.
    // Uploading guest bytes later does not prove every copied/rendered subresource was replaced.
    void RecordImageOnlyWrite() {
        image_only_write_seen = true;
    }

    [[nodiscard]] bool MayHaveImageOnlyContents() const {
        return image_only_write_seen;
    }

private:
    bool image_only_write_seen{};
};

} // namespace VideoCore
