// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "common/types.h"

namespace AmdGpu {

// Production command-buffer submissions originate in the guest address space.  The reader is
// supplied by the VMM owner so snapshots never dereference an unvalidated guest pointer.  Tests
// and built-in initialization packets can continue to use the direct in-process Copy overload.
using GuestCommandBufferReader = std::function<bool(VAddr, std::span<u32>)>;

class OwnedCommandBuffers final {
public:
    [[nodiscard]] static std::shared_ptr<const OwnedCommandBuffers> Copy(std::span<const u32> dcb,
                                                                         std::span<const u32> ccb) {
        auto owner = std::shared_ptr<OwnedCommandBuffers>{new OwnedCommandBuffers{dcb, ccb}};
        owner->CaptureIndirectBuffers(owner->dcb);
        owner->CaptureIndirectBuffers(owner->ccb);
        return owner;
    }

    // Copy command buffers through a VMM-backed reader.  A failed root read returns no owner so a
    // caller cannot accidentally enqueue a span that still aliases an invalid guest mapping.
    [[nodiscard]] static std::shared_ptr<const OwnedCommandBuffers> CopyFromGuest(
        std::span<const u32> dcb, std::span<const u32> ccb,
        const GuestCommandBufferReader& reader) {
        if (!reader) {
            return Copy(dcb, ccb);
        }

        auto owner = std::shared_ptr<OwnedCommandBuffers>{new OwnedCommandBuffers{}};
        owner->reader = reader;
        if (!owner->CopyRoot(dcb, owner->dcb) || !owner->CopyRoot(ccb, owner->ccb)) {
            return {};
        }
        owner->CaptureIndirectBuffers(owner->dcb, &owner->reader);
        owner->CaptureIndirectBuffers(owner->ccb, &owner->reader);
        return owner;
    }

    // Build an owner for a ring submission whose root packet was reconstructed from two
    // submissions. The additional packet is only inspected while this call is synchronous; the
    // resulting owner contains its own root copies and recursively-owned indirect buffers.
    [[nodiscard]] static std::shared_ptr<const OwnedCommandBuffers> CopyWithIndirectPacket(
        std::span<const u32> dcb, std::span<const u32> ccb, std::span<const u32> packet,
        std::span<const u32> suffix = {}, GuestCommandBufferReader reader = {}) {
        auto owner = std::shared_ptr<OwnedCommandBuffers>{new OwnedCommandBuffers{dcb, ccb}};
        owner->reader = std::move(reader);
        if (owner->reader) {
            owner->CaptureIndirectBuffers(owner->dcb, &owner->reader);
            owner->CaptureIndirectBuffers(owner->ccb, &owner->reader);
            owner->CaptureIndirectBuffers(packet, &owner->reader);
            owner->CaptureIndirectBuffers(suffix, &owner->reader);
        } else {
            owner->CaptureIndirectBuffers(owner->dcb);
            owner->CaptureIndirectBuffers(owner->ccb);
            owner->CaptureIndirectBuffers(packet);
            owner->CaptureIndirectBuffers(suffix);
        }
        return owner;
    }

    [[nodiscard]] std::span<const u32> Dcb() const noexcept {
        return dcb;
    }

    [[nodiscard]] std::span<const u32> Ccb() const noexcept {
        return ccb;
    }

    [[nodiscard]] const GuestCommandBufferReader& Reader() const noexcept {
        return reader;
    }

    [[nodiscard]] std::span<const u32> ResolveIndirect(const u32* address,
                                                       u32 size) const noexcept {
        for (const auto& indirect : indirect_buffers) {
            if (indirect.address == address && indirect.words.size() == size) {
                return indirect.words;
            }
        }
        // The submission owns immutable snapshots of command buffers. Borrowing an unresolved
        // guest pointer here would let a deferred GPU task read freed, unmapped, or repurposed
        // guest memory after the producer returned. An unresolved IB is malformed or outside the
        // captured submission; fail closed so callers process no commands from it.
        return {};
    }

private:
    OwnedCommandBuffers() = default;

    OwnedCommandBuffers(std::span<const u32> dcb_, std::span<const u32> ccb_)
        : dcb{dcb_.begin(), dcb_.end()}, ccb{ccb_.begin(), ccb_.end()} {}

    bool CopyRoot(std::span<const u32> source, std::vector<u32>& destination) {
        destination.resize(source.size());
        if (source.empty()) {
            return true;
        }
        return reader(static_cast<VAddr>(reinterpret_cast<uintptr_t>(source.data())),
                      std::span<u32>{destination});
    }

    void CaptureIndirectBuffers(
        std::span<const u32> commands,
        const GuestCommandBufferReader* guest_reader = nullptr) {
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
                CaptureIndirectBuffer(address, size, guest_reader);
            }
            commands = commands.subspan(packet_words);
        }
    }

    void CaptureIndirectBuffer(const u32* address, u32 size,
                               const GuestCommandBufferReader* guest_reader = nullptr) {
        if (address == nullptr || size == 0) {
            return;
        }
        for (const auto& indirect : indirect_buffers) {
            if (indirect.address == address && indirect.words.size() == size) {
                return;
            }
        }
        if (guest_reader != nullptr) {
            std::vector<u32> words(size);
            if (!(*guest_reader)(static_cast<VAddr>(reinterpret_cast<uintptr_t>(address)),
                                 std::span<u32>{words})) {
                return;
            }
            indirect_buffers.push_back({address, std::move(words)});
        } else {
            indirect_buffers.push_back({address, {address, address + size}});
        }
        CaptureIndirectBuffers(indirect_buffers.back().words, guest_reader);
    }

    struct IndirectBufferSnapshot {
        const u32* address{};
        std::vector<u32> words;
    };

    std::vector<u32> dcb;
    std::vector<u32> ccb;
    std::vector<IndirectBufferSnapshot> indirect_buffers;
    GuestCommandBufferReader reader;
};

struct OwnedGraphicsSubmission {
    std::shared_ptr<const OwnedCommandBuffers> owner;
    std::span<const u32> dcb;
    std::span<const u32> ccb;
};

struct OwnedComputeSubmission {
    std::shared_ptr<const OwnedCommandBuffers> owner;
    std::span<const u32> acb;
};

[[nodiscard]] inline OwnedGraphicsSubmission PrepareGraphicsSubmission(std::span<const u32> dcb,
                                                                       std::span<const u32> ccb,
                                                                       GuestCommandBufferReader reader = {}) {
    auto owner = reader ? OwnedCommandBuffers::CopyFromGuest(dcb, ccb, reader)
                        : OwnedCommandBuffers::Copy(dcb, ccb);
    if (!owner) {
        return {};
    }
    return {
        .owner = owner,
        .dcb = owner->Dcb(),
        .ccb = owner->Ccb(),
    };
}

[[nodiscard]] inline OwnedComputeSubmission PrepareComputeSubmission(
    std::span<const u32> acb, GuestCommandBufferReader reader = {}) {
    auto owner = reader ? OwnedCommandBuffers::CopyFromGuest(acb, {}, reader)
                        : OwnedCommandBuffers::Copy(acb, {});
    if (!owner) {
        return {};
    }
    return {
        .owner = owner,
        .acb = owner->Dcb(),
    };
}

} // namespace AmdGpu
