// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "common/types.h"

namespace AmdGpu {

class OwnedCommandBuffers final {
public:
    [[nodiscard]] static std::shared_ptr<const OwnedCommandBuffers> Copy(std::span<const u32> dcb,
                                                                         std::span<const u32> ccb) {
        return std::shared_ptr<const OwnedCommandBuffers>{new OwnedCommandBuffers{dcb, ccb}};
    }

    [[nodiscard]] std::span<const u32> Dcb() const noexcept {
        return dcb;
    }

    [[nodiscard]] std::span<const u32> Ccb() const noexcept {
        return ccb;
    }

private:
    OwnedCommandBuffers(std::span<const u32> dcb_, std::span<const u32> ccb_)
        : dcb{dcb_.begin(), dcb_.end()}, ccb{ccb_.begin(), ccb_.end()} {}

    std::vector<u32> dcb;
    std::vector<u32> ccb;
};

struct OwnedGraphicsSubmission {
    std::shared_ptr<const OwnedCommandBuffers> owner;
    std::span<const u32> dcb;
    std::span<const u32> ccb;
};

[[nodiscard]] inline OwnedGraphicsSubmission PrepareGraphicsSubmission(std::span<const u32> dcb,
                                                                       std::span<const u32> ccb) {
    auto owner = OwnedCommandBuffers::Copy(dcb, ccb);
    return {
        .owner = owner,
        .dcb = owner->Dcb(),
        .ccb = owner->Ccb(),
    };
}

} // namespace AmdGpu
