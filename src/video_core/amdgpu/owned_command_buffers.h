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
        auto owner = std::shared_ptr<OwnedCommandBuffers>{new OwnedCommandBuffers{dcb, ccb}};
        owner->CaptureIndirectBuffers(owner->dcb);
        owner->CaptureIndirectBuffers(owner->ccb);
        return owner;
    }

    [[nodiscard]] std::span<const u32> Dcb() const noexcept {
        return dcb;
    }

    [[nodiscard]] std::span<const u32> Ccb() const noexcept {
        return ccb;
    }

    [[nodiscard]] std::span<const u32> ResolveIndirect(const u32* address,
                                                       u32 size) const noexcept {
        for (const auto& indirect : indirect_buffers) {
            if (indirect.address == address && indirect.words.size() == size) {
                return indirect.words;
            }
        }
        return {address, size};
    }

private:
    OwnedCommandBuffers(std::span<const u32> dcb_, std::span<const u32> ccb_)
        : dcb{dcb_.begin(), dcb_.end()}, ccb{ccb_.begin(), ccb_.end()} {}

    void CaptureIndirectBuffers(std::span<const u32> commands) {
        constexpr u32 Type3 = 3;
        constexpr u32 IndirectBufferConst = 0x33;
        constexpr u32 IndirectBuffer = 0x3f;

        while (!commands.empty()) {
            const u32 header = commands.front();
            const u32 type = header >> 30;
            u32 packet_words{};
            if (type == Type3) {
                packet_words = ((header >> 16) & 0x3fff) + 2;
            } else if (type == 0) {
                packet_words = ((header >> 16) & 0x3fff) + 2;
            } else if (type == 2) {
                packet_words = 1;
            } else {
                return;
            }
            if (packet_words > commands.size()) {
                return;
            }

            const u32 opcode = (header >> 8) & 0xff;
            if (type == Type3 && packet_words >= 4 &&
                (opcode == IndirectBuffer || opcode == IndirectBufferConst)) {
                const uintptr_t raw_address = static_cast<uintptr_t>(commands[1]) |
                                              (static_cast<uintptr_t>(commands[2] & 0xffff) << 32);
                const auto* address = reinterpret_cast<const u32*>(raw_address);
                const u32 size = commands[3] & 0xfffff;
                CaptureIndirectBuffer(address, size);
            }
            commands = commands.subspan(packet_words);
        }
    }

    void CaptureIndirectBuffer(const u32* address, u32 size) {
        if (address == nullptr || size == 0) {
            return;
        }
        for (const auto& indirect : indirect_buffers) {
            if (indirect.address == address && indirect.words.size() == size) {
                return;
            }
        }
        indirect_buffers.push_back({address, {address, address + size}});
        CaptureIndirectBuffers(indirect_buffers.back().words);
    }

    struct IndirectBufferSnapshot {
        const u32* address{};
        std::vector<u32> words;
    };

    std::vector<u32> dcb;
    std::vector<u32> ccb;
    std::vector<IndirectBufferSnapshot> indirect_buffers;
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
