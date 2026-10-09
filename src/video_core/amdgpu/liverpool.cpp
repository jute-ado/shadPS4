// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>

#include <optional>

#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/eop_completion.h"
#include "video_core/amdgpu/index_draw_state.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/pm4_lod_stats.h"
#include "video_core/amdgpu/pm4_predication.h"
#include "video_core/indirect_draw_register_provenance.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    // Refuse new memory/cache users, then let admitted operations finish while this owner
    // can still service their cache work. The closed lifetime may outlive this object.
    if (gpu_owner_lifetime) {
        gpu_owner_lifetime->CloseAndWait();
    }
    eop_submission_batch.Cancel();
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        callback();
    }
}

void Liverpool::FlushPendingEops() {
    eop_submission_batch.Flush(
        [this] { rasterizer->ProcessDownloadImages(); },
        [this](Common::UniqueFunction<void, u64>&& on_submitted,
               Common::UniqueFunction<void>&& wake_owner) {
            rasterizer->FlushWithGpuCompletion(std::move(on_submitted), std::move(wake_owner));
        },
        [this] { SendCommand([this] { DrainCompletedEops(); }); });
}

void Liverpool::DrainCompletedEops() {
    eop_submission_batch.DrainCompleted(
        [this](u64 tick) { return rasterizer->GetScheduler().IsFree(tick); });
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken, [this] {
                return num_commands || num_submits || submission_completions.HasPendingBoundaries();
            });
        }
        if (stoken.stop_requested()) {
            break;
        }

        curr_qid = -1;
        submission_completions.DrainReadyBoundaries();

        while (!stoken.stop_requested() && (num_submits || num_commands)) {
            ProcessCommands();

            const auto mapped_queue_count =
                num_mapped_queues.load(std::memory_order_acquire);
            ASSERT_MSG(mapped_queue_count > 0 && mapped_queue_count <= NumTotalQueues,
                       "Invalid mapped queue count {}", mapped_queue_count);
            curr_qid = (curr_qid + 1) % mapped_queue_count;

            auto& queue = mapped_queues[curr_qid];

            GpuQueue::Submission submission{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                submission = queue.submits.front();
            }
            const auto task = submission.handle;
            task.resume();

            if (task.done()) {
                task.destroy();

                {
                    std::scoped_lock lock{queue.m_access};
                    queue.submits.pop();
                }

                --num_submits;
                {
                    std::scoped_lock lock2{submit_mutex};
                    submit_cv.notify_all();
                }

                submission_completions.CompleteSubmission(submission.sequence);
            }
        }

        if (!stoken.stop_requested()) {
            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
        }
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb, CmdBufferOwner owner) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        // CE runs later on the GPU worker from an immutable CCB snapshot. Validate the complete
        // packet before reading its body; in particular, a truncated WAIT/WRITE packet must not
        // turn its header count into an out-of-bounds host read or advance past the snapshot.
        const auto packet_words = static_cast<size_t>(header->type3.NumWords()) + 1;
        if (packet_words > ccb.size()) {
            LOG_WARNING(Lib_GnmDriver,
                        "Ignoring truncated CE PM4 packet: packet_words={}, available_words={}",
                        packet_words, ccb.size());
            break;
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            const u32 offset = write_const->Offset();
            const u32 size = write_const->Size();
            if (offset > cblock.constants_heap.size() ||
                size > cblock.constants_heap.size() - offset) {
                LOG_WARNING(Lib_GnmDriver,
                            "Ignoring malformed CE write outside constant RAM: offset={:#x}, "
                            "size={:#x}, capacity={:#x}",
                            offset, size, cblock.constants_heap.size());
                break;
            }
            memcpy(cblock.constants_heap.data() + offset, write_const->data, size);
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            const u32 offset = dump_const->Offset();
            const u32 size = dump_const->Size();
            if (offset > cblock.constants_heap.size() ||
                size > cblock.constants_heap.size() - offset) {
                LOG_WARNING(Lib_GnmDriver,
                            "Ignoring malformed CE dump outside constant RAM: offset={:#x}, "
                            "size={:#x}, capacity={:#x}",
                            offset, size, cblock.constants_heap.size());
                break;
            }
            const Core::GuestMemoryWrite destination{
                dump_const->Address<VAddr>(), cblock.constants_heap.data() + offset, size};
            if (auto* memory = Core::Memory::Instance();
                !memory ||
                !memory->TryWriteGuest(std::span<const Core::GuestMemoryWrite>{&destination, 1})) {
                LOG_WARNING(Lib_GnmDriver,
                            "Ignoring CE dump to unmapped or non-writable guest address {:#x}",
                            destination.address);
            }
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            const auto indirect_commands = owner->ResolveIndirect(
                indirect_buffer->Address<const u32>(), indirect_buffer->ib_size);
            auto task = ProcessCeUpdate(indirect_commands, owner);
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            if (task.handle) {
                task.handle.destroy();
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, packet_words);
    }

    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb,
                                           SubmissionCompletionQueue::Sequence submission_sequence,
                                           CmdBufferOwner owner) {
    FIBER_ENTER(dcb_task_name);
    (void)owner;

    auto* memory = Core::Memory::Instance();
    const auto is_vo_label_range = [this](VAddr address, u64 size) {
        if (size == 0) {
            return true;
        }
        if (!vo_port) {
            return false;
        }
        const auto begin = reinterpret_cast<uintptr_t>(vo_port->buffer_labels.data());
        const auto end = begin + sizeof(vo_port->buffer_labels);
        const auto target = static_cast<uintptr_t>(address);
        return target >= begin && target <= end && size <= end - target;
    };
    const auto try_read_guest = [memory, this, is_vo_label_range](VAddr address, void* data,
                                                                   u64 size) {
        if (size == 0) {
            return true;
        }
        if (is_vo_label_range(address, size)) {
            // The presenter resets labels on its own thread. Keep checked PM4 polls on the same
            // mutex as those resets so a label cannot be observed while it is being updated.
            std::scoped_lock lock{vo_port->vo_mutex};
            std::memcpy(data, reinterpret_cast<const void*>(address), size);
            return true;
        }
        return memory && memory->TryReadBacking(std::bit_cast<const void*>(address), data, size);
    };
    const auto try_write_guest = [memory, this, is_vo_label_range](VAddr address, const void* data,
                                                                     u64 size) {
        if (size == 0) {
            return true;
        }
        if (is_vo_label_range(address, size)) {
            // PM4 WriteData is the producer side of the VO-label wait protocol. Publish the
            // value while holding the same mutex and wake all possible waiters afterwards.
            std::scoped_lock lock{vo_port->vo_mutex};
            std::memcpy(reinterpret_cast<void*>(address), data, size);
            vo_port->vo_cv.notify_all();
            return true;
        }
        if (!memory) {
            return false;
        }
        const Core::GuestMemoryWrite write{address, data, size};
        if (memory->TryWriteGuest(std::span<const Core::GuestMemoryWrite>{&write, 1})) {
            return true;
        }
        // PM4 writes can target physically backed GPU labels that are not exposed as writable
        // guest mappings. Keep the access checked against the VMA/physical backing rather than
        // falling back to an unchecked host-pointer dereference.
        return memory->TryWriteBacking(std::bit_cast<void*>(address), data, size);
    };
    // Fence destinations normally refer to guest-backed memory, but video-out labels are a
    // host-owned array intentionally exposed to the guest by address.  They cannot be represented
    // by MemoryManager's VMA/identity ledger; publish them through the same checked label path as
    // IT_WRITE_DATA so an EOP flip can observe the value and wake its waiter.
    const auto publish_fence = [memory, this, is_vo_label_range](
                                   VAddr address,
                                   const std::optional<Core::DeferredBackingWrite>& fence_token,
                                   u64 data, u32 num_bytes) {
        if (num_bytes == 0) {
            return true;
        }
        if (is_vo_label_range(address, num_bytes)) {
            std::scoped_lock lock{vo_port->vo_mutex};
            std::memcpy(reinterpret_cast<void*>(address), &data, num_bytes);
            vo_port->vo_cv.notify_all();
            return true;
        }
        if (!fence_token || fence_token->source_size != num_bytes || !memory) {
            return false;
        }
        const auto* bytes = reinterpret_cast<const u8*>(&data);
        return memory->TryPublishFence(
            *fence_token, std::span<const u8>{bytes, static_cast<size_t>(num_bytes)});
    };
    const auto test_wait_reg_mem = [&](const PM4CmdWaitRegMem& wait) -> std::optional<bool> {
        u32 value{};
        if (wait.mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
            if (!try_read_guest(wait.Address<VAddr>(), &value, sizeof(value))) {
                return std::nullopt;
            }
        } else {
            const auto reg = static_cast<size_t>(wait.Reg());
            if (reg >= regs.reg_array.size()) {
                return std::nullopt;
            }
            value = regs.reg_array[reg];
        }

        switch (wait.function.Value()) {
        case PM4CmdWaitRegMem::Function::Always:
            return true;
        case PM4CmdWaitRegMem::Function::LessThan:
            return (value & wait.mask) < wait.ref;
        case PM4CmdWaitRegMem::Function::LessThanEqual:
            return (value & wait.mask) <= wait.ref;
        case PM4CmdWaitRegMem::Function::Equal:
            return (value & wait.mask) == wait.ref;
        case PM4CmdWaitRegMem::Function::NotEqual:
            return (value & wait.mask) != wait.ref;
        case PM4CmdWaitRegMem::Function::GreaterThanEqual:
            return (value & wait.mask) >= wait.ref;
        case PM4CmdWaitRegMem::Function::GreaterThan:
            return (value & wait.mask) > wait.ref;
        case PM4CmdWaitRegMem::Function::Reserved:
        default:
            return std::nullopt;
        }
    };

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb, owner);
        RESUME_GFX(ce_task);
    }
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();
    const bool guest_markers_enabled = rasterizer && EmulatorSettings.IsVkGuestMarkersEnabled();

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (!dcb.empty()) {
        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;
        const u32 packet_words = type == 3 ? header->type3.NumWords() + 1 : 1;
        if (type == 3 && packet_words > dcb.size()) {
            LOG_WARNING(Render,
                        "Ignoring truncated graphics PM4 packet: packet_words={}, "
                        "available_words={}", packet_words, dcb.size());
            break;
        }
        const bool is_eop = type == 3 && header->type3.opcode == PM4ItOpcode::EventWriteEop;
        const bool continues_eop_batch = is_eop && eop_submission_batch.HasPending();
        if (!is_eop && rasterizer) {
            FlushPendingEops();
        }
        if (!continues_eop_batch) {
            ProcessCommands();
        }

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = packet_words - 1;
            const PM4ItOpcode opcode = header->type3.opcode;
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                const auto nop_count = nop->header.count.Value();
                if (nop_count == 0 || (nop_count == 0x3fffu && nop->header.NumWords() == 0)) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    const auto eop_position = DecodeFlipEopPosition(nop->header.count.Value());
                    ASSERT_MSG(
                        eop_flip_tracker.QueueFlip(eop_position,
                                                   [this] {
                                                       SendCommand([] {
                                                           Platform::IrqC::Instance()->Signal(
                                                               Platform::InterruptId::GfxFlip);
                                                       });
                                                   }),
                        "A patched flip could not be associated with its EOP packet");
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto packet_words = static_cast<size_t>(header->type3.NumWords());
                if (packet_words == 0) {
                    LOG_WARNING(Lib_GnmDriver, "Ignoring graphics SetShReg packet with zero count");
                    break;
                }
                const auto set_size = (packet_words - 1) * sizeof(u32);
                const auto reg_offset = static_cast<size_t>(set_data->reg_offset);
                constexpr size_t compute_reg_base = 0x200;
                constexpr size_t compute_state_words = sizeof(ComputeProgram) / sizeof(u32);

                if (reg_offset >= compute_reg_base &&
                    reg_offset < compute_reg_base + compute_state_words) {
                    const auto state_offset = (reg_offset - compute_reg_base) * sizeof(u32);
                    if (set_size > sizeof(ComputeProgram) - state_offset) {
                        LOG_WARNING(Lib_GnmDriver,
                                    "Ignoring graphics SetShReg write past compute state: "
                                    "offset={}, size={}, state_size={}",
                                    state_offset, set_size, sizeof(ComputeProgram));
                        break;
                    }
                    auto* addr = reinterpret_cast<u8*>(&mapped_queues[GfxQueueId].cs_state) +
                                 state_offset;
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    const auto reg_index = static_cast<size_t>(Regs::ShRegWordOffset) + reg_offset;
                    if (reg_index > regs.reg_array.size() ||
                        set_size > (regs.reg_array.size() - reg_index) * sizeof(u32)) {
                        LOG_WARNING(Lib_GnmDriver,
                                    "Ignoring graphics SetShReg write past register state: "
                                    "offset={}, size={}, register_count={}",
                                    reg_index, set_size, regs.reg_array.size());
                        break;
                    }
                    std::memcpy(&regs.reg_array[reg_index], header + 2, set_size);
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetPredication: {
                if (count < 2 || dcb.size() < static_cast<size_t>(count) + 1) {
                    // DecodeLiverpoolSetPredication reads two body dwords. Keep malformed PM4
                    // input from turning the span construction into an out-of-bounds host read.
                    LOG_WARNING(Render, "Ignoring truncated IT_SET_PREDICATION packet");
                    break;
                }
                const auto packet = DecodeLiverpoolSetPredication(dcb.subspan(1, count));
                if (!packet) {
                    LOG_WARNING(Render, "Ignoring truncated IT_SET_PREDICATION packet");
                    break;
                }
                LOG_WARNING(Render,
                            "Unimplemented IT_SET_PREDICATION address={:#x} operation={} "
                            "draw_if_visible={} wait={} continue={}",
                            packet->address, static_cast<u32>(packet->operation),
                            packet->draw_if_visible, packet->wait_for_results,
                            packet->continue_predicate);
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                ApplyDirectIndexDrawSize(regs, draw_index->max_size, draw_index->index_count);
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DrawIndex2", cmd_address));
                        rasterizer->Draw(true);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                ApplyDirectIndexDrawSize(regs, draw_index_off->max_size,
                                         draw_index_off->index_count);
                regs.draw_initiator = draw_index_off->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexOffset2", cmd_address));
                        rasterizer->Draw(true, draw_index_off->index_offset);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true, draw_index_off->index_offset);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                ApplyAutoIndexDrawSize(regs, draw_index->index_count);
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexAuto", cmd_address));
                        rasterizer->Draw(false);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(false);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto register_locations =
                    VideoCore::CaptureIndirectDrawRegisterLocations(*draw_indirect);
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirect", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0,
                                                 register_locations);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0,
                                                 register_locations);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto register_locations =
                    VideoCore::CaptureIndirectDrawRegisterLocations(*draw_indirect);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0,
                                                 register_locations);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0,
                                                 register_locations);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto register_locations =
                    VideoCore::CaptureIndirectDrawRegisterLocations(*draw_index_indirect);
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirect", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0,
                                                 register_locations);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0,
                                                 register_locations);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto register_locations =
                    VideoCore::CaptureIndirectDrawRegisterLocations(*draw_index_indirect);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0, register_locations);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0, register_locations);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto register_locations =
                    VideoCore::CaptureIndirectDrawRegisterLocations(*draw_index_indirect);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectCountMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0,
                                                 register_locations);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0,
                                                 register_locations);
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchDirect", cmd_address));
                        rasterizer->DispatchDirect();
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchDirect();
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dispatch_initiator = dispatch_indirect->dispatch_initiator;
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchIndirect", cmd_address));
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                    }
                }
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                ApplyIndexBufferSize(regs, index_size->num_indices);
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        const VAddr results_address = event->Address<VAddr>();
                        bool write_succeeded = true;
                        for (s32 i = 0; i < num_counter_pairs; ++i) {
                            const u64 value = pixel_counter | OcclusionCounterValidMask;
                            const VAddr address = results_address +
                                                  static_cast<u64>(i) * 2 * sizeof(u64);
                            write_succeeded &= try_write_guest(address, &value, sizeof(value));
                        }
                        if (!write_succeeded) {
                            LOG_WARNING(Render,
                                        "Ignoring IT_EVENT_WRITE pixel-stat write to unmapped "
                                        "guest address");
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (rasterizer) {
                    rasterizer->ProcessDownloadImages();
                }
                const auto fence_token = [&]() -> std::optional<Core::DeferredBackingWrite> {
                    if (event_eos->command != PM4CmdEventWriteEos::Command::SignalFence) {
                        return std::nullopt;
                    }
                    const Core::ReadbackWriteRange destination{
                        reinterpret_cast<VAddr>(event_eos->Address()), sizeof(u32), 0};
                    return Core::Memory::Instance()->CaptureReadbackDestinations(
                        std::span<const Core::ReadbackWriteRange>{&destination, 1}, sizeof(u32));
                }();
                event_eos->SignalFence(
                    [publish_fence, fence_token = std::move(fence_token)](void* address, u64 data,
                                                                           u32 num_bytes) {
                        (void)publish_fence(reinterpret_cast<VAddr>(address), fence_token, data,
                                            num_bytes);
                    });
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        rasterizer->Finish();
                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        const Core::GuestMemoryWrite destination{
                            reinterpret_cast<VAddr>(event_eos->Address()), &value, sizeof(value)};
                        (void)try_write_guest(destination.address, destination.data,
                                              destination.size);
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                auto complete_eop_flip = eop_flip_tracker.BeginEop();
                const auto fence_size = [&] {
                    switch (event_eop->data_sel.Value()) {
                    case DataSelect::Data32Low:
                        return u64{sizeof(u32)};
                    case DataSelect::Data64:
                    case DataSelect::GpuClock64:
                    case DataSelect::PerfCounter:
                        return u64{sizeof(u64)};
                    case DataSelect::None:
                    default:
                        return u64{0};
                    }
                }();
                const auto fence_token = [&]() -> std::optional<Core::DeferredBackingWrite> {
                    if (fence_size == 0) {
                        return std::nullopt;
                    }
                    const Core::ReadbackWriteRange destination{
                        reinterpret_cast<VAddr>(event_eop->Address<u8*>()), fence_size, 0};
                    return Core::Memory::Instance()->CaptureReadbackDestinations(
                        std::span<const Core::ReadbackWriteRange>{&destination, 1}, fence_size);
                }();
                const VAddr fence_address = reinterpret_cast<VAddr>(event_eop->Address<u8*>());
                auto write_memory = [publish_fence, fence_address,
                                     fence_token = std::move(fence_token)](void* /*address*/, u64 data,
                                                                            u32 num_bytes) {
                    (void)publish_fence(fence_address, fence_token, data, num_bytes);
                };
                auto signal_interrupt = [] {
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop);
                };
                if (rasterizer) {
                    eop_submission_batch.Add(*event_eop, std::move(write_memory),
                                             std::move(signal_interrupt),
                                             std::move(complete_eop_flip));
                } else {
                    PublishEop(*event_eop, std::move(write_memory), std::move(signal_interrupt),
                               std::move(complete_eop_flip));
                }
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                    u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const auto body_words = static_cast<size_t>(header->type3.NumWords());
                // WRITE_DATA has one control dword and a two-dword destination before its
                // optional payload. Keep the size calculation in body-word units; using the raw
                // COUNT here obscures the header's N - 1 encoding and made the ASC path drift by
                // one dword.
                if (body_words < 3) {
                    LOG_WARNING(Render, "Ignoring malformed IT_WRITE_DATA packet");
                    break;
                }
                const auto data_size = (body_words - 3) * sizeof(u32);
                const VAddr address = write_data->Address<VAddr>();
                if (!write_data->wr_one_addr.Value()) {
                    if (!try_write_guest(address, write_data->data, data_size)) {
                        LOG_WARNING(Render,
                                    "Ignoring IT_WRITE_DATA write to unmapped guest address");
                    }
                } else {
                    for (u32 offset = 0; offset < data_size; offset += sizeof(u32)) {
                        if (!try_write_guest(address,
                                             reinterpret_cast<const u8*>(write_data->data) + offset,
                                             sizeof(u32))) {
                            LOG_WARNING(Render,
                                        "Ignoring IT_WRITE_DATA write to unmapped guest "
                                        "address");
                            break;
                        }
                    }
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                const VAddr address = mem_semaphore->Address<VAddr>();
                if (mem_semaphore->IsSignaling()) {
                    u64 value{};
                    if (!try_read_guest(address, &value, sizeof(value))) {
                        LOG_WARNING(Render,
                                    "Ignoring IT_MEM_SEMAPHORE signal for unmapped guest "
                                    "address");
                        break;
                    }
                    switch (mem_semaphore->signal_type.Value()) {
                    case PM4CmdMemSemaphore::SignalType::Increment:
                        ++value;
                        break;
                    case PM4CmdMemSemaphore::SignalType::Write:
                        value = 1;
                        break;
                    default:
                        UNREACHABLE_MSG("Unknown semaphore signal type {}",
                                        static_cast<u32>(mem_semaphore->signal_type.Value()));
                    }
                    if (!try_write_guest(address, &value, sizeof(value))) {
                        LOG_WARNING(Render,
                                    "Ignoring IT_MEM_SEMAPHORE signal for unmapped guest "
                                    "address");
                    }
                } else {
                    while (true) {
                        u64 value{};
                        if (!try_read_guest(address, &value, sizeof(value))) {
                            LOG_WARNING(Render,
                                        "Abandoning IT_MEM_SEMAPHORE wait for unmapped guest "
                                        "address");
                            break;
                        }
                        if (value > 0) {
                            --value;
                            if (!try_write_guest(address, &value, sizeof(value))) {
                                LOG_WARNING(Render,
                                            "Abandoning IT_MEM_SEMAPHORE decrement for unmapped "
                                            "guest address");
                            }
                            break;
                        }
                        YIELD_GFX();
                    }
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // EOP completion queues flips back to this owner. A VO-label
                // wait must yield too: sleeping the owner would prevent the
                // queued flip which eventually releases that label.
                const auto wait_address = wait_reg_mem->Address<VAddr>();
                const auto is_last_graphics_submission = [&] {
                    // SubmitGfx may append to this queue while the current owner is polling. Do
                    // not inspect std::queue::size() without the queue's admission mutex.
                    std::scoped_lock lock{mapped_queues[GfxQueueId].m_access};
                    return num_submits.load(std::memory_order_acquire) ==
                           mapped_queues[GfxQueueId].submits.size();
                };
                const bool wait_is_memory =
                    wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory;
                const bool wait_is_vo_label =
                    wait_is_memory && is_vo_label_range(wait_address, sizeof(u64));
                // A VO-label poll belongs to the VideoOut port lifetime in which it was emitted.
                // Closing and reopening the port must invalidate a stale generic poll as well as
                // the final-submission path below; otherwise an old label can spin forever.
                const auto wait_lifecycle_epoch =
                    wait_is_vo_label ? vo_port->lifecycle.CurrentEpoch() : 0;
                const bool wait_is_last_submission =
                    wait_is_vo_label && is_last_graphics_submission();
                if (wait_is_vo_label && wait_is_last_submission) {
                    // Do not sleep the graphics owner here. EOP completion queues the flip that
                    // releases this label back to the same owner, so a condition-variable wait
                    // would deadlock the final submission. Keep the close escape while polling
                    // through the checked path used by ordinary WAIT_REG_MEM packets.
                    while (vo_port->lifecycle.IsCurrent(wait_lifecycle_epoch)) {
                        const auto result = test_wait_reg_mem(*wait_reg_mem);
                        if (!result || *result) {
                            break;
                        }
                        YIELD_GFX();
                    }
                    break;
                }
                while (true) {
                    if (wait_is_vo_label &&
                        !vo_port->lifecycle.IsCurrent(wait_lifecycle_epoch)) {
                        break;
                    }
                    const auto result = test_wait_reg_mem(*wait_reg_mem);
                    if (!result) {
                        break;
                    }
                    if (*result) {
                        break;
                    }
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                const auto indirect_commands = owner->ResolveIndirect(
                    indirect_buffer->Address<const u32>(), indirect_buffer->ib_size);
                if (indirect_commands.empty() && indirect_buffer->ib_size != 0) {
                    LOG_WARNING(Render,
                                "Dropping unresolved graphics indirect buffer: address={:#x}, "
                                "size={}",
                                reinterpret_cast<uintptr_t>(indirect_buffer->Address<const u32>()),
                                indirect_buffer->ib_size.Value());
                }
                auto task = ProcessGraphics(indirect_commands, {}, submission_sequence, owner);
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                if (rasterizer) {
                    rasterizer->CpSync();
                }
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                // The report schema and completion ordering are not established yet. Validate
                // the packet syntax without touching the guest report destination; in
                // particular, never reinterpret the guest VAddr as a host pointer here.
                const auto packet_words = static_cast<size_t>(header->type3.NumWords());
                if (packet_words != LodStatsPayloadDwords || dcb.size() < packet_words + 1) {
                    LOG_WARNING(Render_Vulkan,
                                "Ignoring malformed IT_GET_LOD_STATS packet: payload_words={}, "
                                "available_dwords={}",
                                packet_words, dcb.size() > 0 ? dcb.size() - 1 : 0);
                    break;
                }
                const auto packet = DecodeLodStatsPacket(dcb.subspan(1, packet_words));
                if (!packet) {
                    LOG_WARNING(Render_Vulkan,
                                "Ignoring invalid IT_GET_LOD_STATS packet (syntax-only)");
                    break;
                }
                LOG_DEBUG(Render_Vulkan,
                          "IT_GET_LOD_STATS accepted syntax-only packet: address={:#x}, size={:#x}, "
                          "control={:#x}; report execution deferred",
                          packet->address, packet->size, packet->control);
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                const VAddr condition_address = std::bit_cast<VAddr>(cond_exec->Address());
                u8 condition{};
                if (!try_read_guest(condition_address, &condition, sizeof(condition))) {
                    LOG_WARNING(Render,
                                "Ignoring IT_COND_EXEC condition from unmapped guest address");
                    condition = 0;
                }
                const auto skip = condition == 0;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (rasterizer) {
        FlushPendingEops();
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(
    std::span<const u32> acb, u32 vqid, SubmissionCompletionQueue::Sequence submission_sequence,
    CmdBufferOwner owner, std::shared_ptr<AscQueueInfo> queue_state, VAddr source_base_addr) {
    FIBER_ENTER(acb_task_name[vqid]);
    if (!owner || !queue_state || !queue_state->active.load(std::memory_order_acquire)) {
        FIBER_EXIT;
        co_return;
    }

    auto& queue = *queue_state;
    // `acb` may still point into a prior immutable owner after a split packet is reassembled.
    // Keep every replaced owner alive for the lifetime of this coroutine before installing the
    // augmented owner used for indirect-buffer resolution.
    std::vector<CmdBufferOwner> retained_submission_owners;
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();
    // Rasterizer's public dispatch entry points still read the mapped-queue mirror. Keep that
    // narrow handoff synchronous with the deferred queue-owned
    // state; parsing and register writes themselves remain isolated per ASC generation.
    const auto publish_compute_state_for_dispatch = [&] {
        mapped_queues[vqid + 1].cs_state = queue.cs_state;
    };
    auto* memory = Core::Memory::Instance();
    const auto try_read_guest = [memory](VAddr address, void* data, u64 size) {
        return memory && memory->TryReadBacking(std::bit_cast<const void*>(address), data, size);
    };
    const auto try_write_guest = [memory](VAddr address, const void* data, u64 size) {
        if (!memory) {
            return false;
        }
        const Core::GuestMemoryWrite write{address, data, size};
        return memory->TryWriteGuest(std::span<const Core::GuestMemoryWrite>{&write, 1});
    };
    const auto test_wait_reg_mem = [&](const PM4CmdWaitRegMem& wait) -> std::optional<bool> {
        u32 value{};
        if (wait.mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
            if (!try_read_guest(wait.Address<VAddr>(), &value, sizeof(value))) {
                return std::nullopt;
            }
        } else {
            const auto compute_value = queue.TryReadComputeRegister(wait.Reg());
            if (!compute_value) {
                return std::nullopt;
            }
            value = *compute_value;
        }

        switch (wait.function.Value()) {
        case PM4CmdWaitRegMem::Function::Always:
            return true;
        case PM4CmdWaitRegMem::Function::LessThan:
            return (value & wait.mask) < wait.ref;
        case PM4CmdWaitRegMem::Function::LessThanEqual:
            return (value & wait.mask) <= wait.ref;
        case PM4CmdWaitRegMem::Function::Equal:
            return (value & wait.mask) == wait.ref;
        case PM4CmdWaitRegMem::Function::NotEqual:
            return (value & wait.mask) != wait.ref;
        case PM4CmdWaitRegMem::Function::GreaterThanEqual:
            return (value & wait.mask) >= wait.ref;
        case PM4CmdWaitRegMem::Function::GreaterThan:
            return (value & wait.mask) > wait.ref;
        case PM4CmdWaitRegMem::Function::Reserved:
        default:
            return std::nullopt;
        }
    };

    struct IndirectPatch {
        VAddr header_address;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    const auto base_addr = reinterpret_cast<VAddr>(acb.data());
    VAddr acb_guest_base_addr = source_base_addr;
    const auto snapshot_header_at_guest_address = [](std::span<const u32> snapshot,
                                                     VAddr snapshot_base_addr,
                                                     VAddr guest_address, size_t copy_bytes) {
        const size_t snapshot_size = snapshot.size_bytes();
        if (snapshot_size < sizeof(PM4Header) || guest_address < snapshot_base_addr) {
            return static_cast<const PM4Header*>(nullptr);
        }

        const VAddr destination_offset = guest_address - snapshot_base_addr;
        if (destination_offset < sizeof(PM4Header) || destination_offset >= snapshot_size ||
            copy_bytes > snapshot_size - destination_offset ||
            destination_offset % sizeof(u32) != 0) {
            return static_cast<const PM4Header*>(nullptr);
        }

        const auto header_offset = static_cast<size_t>(destination_offset - sizeof(PM4Header));
        if (header_offset > snapshot_size - sizeof(PM4Header)) {
            return static_cast<const PM4Header*>(nullptr);
        }

        return reinterpret_cast<const PM4Header*>(
            reinterpret_cast<const u8*>(snapshot.data()) + header_offset);
    };
    while (!acb.empty() && queue_state->active.load(std::memory_order_acquire)) {
        ProcessCommands();
        if (!queue_state->active.load(std::memory_order_acquire)) {
            break;
        }

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;
        VAddr packet_guest_base_addr = acb_guest_base_addr;
        std::span<const u32> packet_snapshot = acb;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            if (queue.tmp_dwords > queue.tmp_packet.size()) {
                LOG_WARNING(Lib_GnmDriver,
                            "Discarding ASC packet continuation with invalid buffered size: "
                            "buffered_words={}, capacity_words={}",
                            queue.tmp_dwords, queue.tmp_packet.size());
                queue.tmp_dwords = 0;
                queue_state->active.store(false, std::memory_order_release);
                break;
            }
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            packet_guest_base_addr = queue.tmp_source_base_addr;
            const auto packet_words = header->type3.NumWords() + 1;
            if (packet_words < queue.tmp_dwords ||
                packet_words - queue.tmp_dwords > acb.size() ||
                packet_words - queue.tmp_dwords > queue.tmp_packet.size() - queue.tmp_dwords) {
                LOG_WARNING(Lib_GnmDriver,
                            "Discarding invalid ASC packet continuation: packet_words={}, "
                            "buffered_words={}, available_words={}, capacity_words={}",
                            packet_words, queue.tmp_dwords, acb.size(), queue.tmp_packet.size());
                queue.tmp_dwords = 0;
                queue_state->active.store(false, std::memory_order_release);
                break;
            }
            next_dw_off = packet_words - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            packet_snapshot = {queue.tmp_packet.data(), packet_words};
            // The root packet was split across two ring submissions. The current submission's
            // owner only saw the continuation, so it could not discover an indirect buffer whose
            // header lived in the previous tail. Rebuild an owner with the complete packet before
            // resolving it; otherwise ResolveIndirect would fall back to the raw guest pointer.
            retained_submission_owners.push_back(owner);
            owner = OwnedCommandBuffers::CopyWithIndirectPacket(
                owner->Dcb(), owner->Ccb(),
                std::span<const u32>{queue.tmp_packet.data(), packet_words},
                acb.subspan(next_dw_off), owner->Reader());
            queue.tmp_dwords = 0;
            queue.tmp_source_base_addr = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            if (next_dw_off > queue.tmp_packet.size() || acb.size() > queue.tmp_packet.size()) {
                LOG_WARNING(Lib_GnmDriver,
                            "Discarding oversize ASC packet continuation: packet_words={}, "
                            "available_words={}, capacity_words={}",
                            next_dw_off, acb.size(), queue.tmp_packet.size());
                queue.tmp_dwords = 0;
                queue_state->active.store(false, std::memory_order_release);
                break;
            }
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            queue.tmp_source_base_addr = acb_guest_base_addr;
            if constexpr (!is_indirect) {
                AdvanceAscReadPointer(queue_state, acb.size());
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            acb_guest_base_addr += sizeof(u32);
            if constexpr (!is_indirect) {
                AdvanceAscReadPointer(queue_state, next_dw_off);
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            const auto indirect_commands = owner->ResolveIndirect(
                indirect_buffer->Address<const u32>(), indirect_buffer->ib_size);
            if (indirect_commands.empty() && indirect_buffer->ib_size != 0) {
                LOG_WARNING(Lib_GnmDriver,
                            "Dropping unresolved ASC indirect buffer: address={:#x}, size={}",
                            reinterpret_cast<uintptr_t>(indirect_buffer->Address<const u32>()),
                            indirect_buffer->ib_size.Value());
            }
            auto task = ProcessCompute<true>(
                indirect_commands,
                vqid, submission_sequence, owner, queue_state,
                reinterpret_cast<VAddr>(indirect_buffer->Address<const u32>()));
            RESUME_ASC(task, vqid);

            while (!task.handle.done() &&
                   queue_state->active.load(std::memory_order_acquire)) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            // Give a suspended child one final resume so it can unwind any nested child after the
            // queue is retired, then release the coroutine frame owned by this case.
            if (!task.handle.done()) {
                RESUME_ASC(task, vqid);
            }
            if (task.handle) {
                task.handle.destroy();
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* snapshot_header =
                    snapshot_header_at_guest_address(packet_snapshot, packet_guest_base_addr,
                                                      dst_addr, num_bytes);
                if (snapshot_header &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    snapshot_header->type == 3 &&
                    snapshot_header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    if (dst_addr < sizeof(PM4Header)) {
                        break;
                    }
                    indirect_patches.emplace_back(dst_addr - sizeof(PM4Header), src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::CondExec: {
            const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
            if (cond_exec->command.Value() != 0) {
                LOG_WARNING(Lib_GnmDriver, "ASC IT_COND_EXEC used a reserved command");
            }

            // The condition address belongs to the guest command stream. ProcessCompute runs
            // after submission, so never reinterpret it as a host pointer. Invalid conditions
            // fail closed and skip the guarded block rather than executing stale commands.
            const VAddr condition_address = std::bit_cast<VAddr>(cond_exec->Address());
            u8 condition{};
            if (!try_read_guest(condition_address, &condition, sizeof(condition))) {
                LOG_WARNING(Lib_GnmDriver,
                            "Ignoring ASC IT_COND_EXEC condition from unmapped guest address");
                condition = 0;
            }

            if (condition == 0) {
                const u64 skip_dwords = static_cast<u64>(next_dw_off) +
                                        cond_exec->exec_count.Value();
                if (skip_dwords > acb.size() ||
                    skip_dwords > (std::numeric_limits<VAddr>::max() - acb_guest_base_addr) /
                                        sizeof(u32)) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Stopping ASC queue on out-of-bounds IT_COND_EXEC skip: "
                                "skip_dwords={}, available_dwords={}",
                                skip_dwords, acb.size());
                    queue_state->active.store(false, std::memory_order_release);
                    break;
                }

                acb = NextPacket(acb, static_cast<size_t>(skip_dwords));
                acb_guest_base_addr += skip_dwords * sizeof(u32);
                if constexpr (!is_indirect) {
                    AdvanceAscReadPointer(queue_state, static_cast<u32>(skip_dwords));
                }
                continue;
            }
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid() &&
                   queue_state->active.load(std::memory_order_acquire) &&
                   !queue_state->retired.load(std::memory_order_acquire)) {
                YIELD_ASC(vqid);
            }
            if (!queue_state->active.load(std::memory_order_acquire) ||
                queue_state->retired.load(std::memory_order_acquire)) {
                break;
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto packet_words = static_cast<size_t>(header->type3.NumWords());
            if (packet_words == 0) {
                LOG_WARNING(Lib_GnmDriver, "Ignoring ASC SetShReg packet with zero count");
                break;
            }
            const auto set_size = (packet_words - 1) * sizeof(u32);
            const auto reg_offset = static_cast<size_t>(set_data->reg_offset);
            constexpr size_t compute_reg_base = 0x200;
            constexpr size_t compute_state_words = sizeof(ComputeProgram) / sizeof(u32);

            if (reg_offset >= compute_reg_base &&
                reg_offset < compute_reg_base + compute_state_words) {
                const auto state_offset = (reg_offset - compute_reg_base) * sizeof(u32);
                if (set_size > sizeof(ComputeProgram) - state_offset) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Ignoring ASC SetShReg write past compute state: offset={}, "
                                "size={}, state_size={}",
                                state_offset, set_size, sizeof(ComputeProgram));
                    break;
                }
                auto* addr = reinterpret_cast<u8*>(&queue.cs_state) + state_offset;
                std::memcpy(addr, header + 2, set_size);
            } else {
                // ASC owns only the compute shader register window above. Never mutate the
                // shared graphics register image while a graphics coroutine may be suspended on
                // another queue; unsupported non-compute writes are deliberately fail-closed.
                LOG_WARNING(Lib_GnmDriver,
                            "Ignoring ASC SetShReg targeting non-compute register: offset={}, "
                            "size={}",
                            reg_offset, set_size);
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, packet_guest_base_addr,
                                            &IndirectPatch::header_address);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                publish_compute_state_for_dispatch();
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = queue.cs_state;
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                publish_compute_state_for_dispatch();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchDirect", vqid, cmd_address));
                    rasterizer->DispatchDirect();
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchDirect();
                }
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = queue.cs_state;
            cs_program.dispatch_initiator = dispatch_indirect->dispatch_initiator;
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                publish_compute_state_for_dispatch();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchIndirect", vqid, cmd_address));
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                }
            }
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const auto body_words = static_cast<size_t>(header->type3.NumWords());
            // The body is control + address low/high followed by optional data. NumWords() is
            // the body length, unlike the raw COUNT field used by the legacy graphics code.
            if (body_words < 3) {
                LOG_WARNING(Lib_GnmDriver, "Ignoring malformed ASC IT_WRITE_DATA packet");
                break;
            }
            const auto data_size = (body_words - 3) * sizeof(u32);
            if (!write_data->wr_one_addr.Value()) {
                if (!try_write_guest(write_data->Address<VAddr>(), write_data->data, data_size)) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Ignoring ASC IT_WRITE_DATA write to unmapped guest address");
                }
            } else {
                for (u32 offset = 0; offset < data_size; offset += sizeof(u32)) {
                    if (!try_write_guest(write_data->Address<VAddr>(),
                                         reinterpret_cast<const u8*>(write_data->data) + offset,
                                         sizeof(u32))) {
                        LOG_WARNING(Lib_GnmDriver,
                                    "Ignoring ASC IT_WRITE_DATA write to unmapped guest address");
                        break;
                    }
                }
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            const auto address = mem_semaphore->Address<VAddr>();
            if (mem_semaphore->IsSignaling()) {
                u64 value{};
                if (!try_read_guest(address, &value, sizeof(value))) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Ignoring ASC IT_MEM_SEMAPHORE signal for unmapped guest address");
                    break;
                }
                switch (mem_semaphore->signal_type.Value()) {
                case PM4CmdMemSemaphore::SignalType::Increment:
                    ++value;
                    break;
                case PM4CmdMemSemaphore::SignalType::Write:
                    value = 1;
                    break;
                default:
                    UNREACHABLE_MSG("Unknown semaphore signal type {}",
                                    static_cast<u32>(mem_semaphore->signal_type.Value()));
                }
                if (!try_write_guest(address, &value, sizeof(value))) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Ignoring ASC IT_MEM_SEMAPHORE signal for unmapped guest address");
                }
            } else {
                while (queue_state->active.load(std::memory_order_acquire) &&
                       !queue_state->retired.load(std::memory_order_acquire)) {
                    u64 value{};
                    if (!try_read_guest(address, &value, sizeof(value))) {
                        LOG_WARNING(Lib_GnmDriver,
                                    "Abandoning ASC IT_MEM_SEMAPHORE wait for unmapped guest "
                                    "address");
                        break;
                    }
                    if (value > 0) {
                        --value;
                        if (!try_write_guest(address, &value, sizeof(value))) {
                            LOG_WARNING(Lib_GnmDriver,
                                        "Abandoning ASC IT_MEM_SEMAPHORE decrement for unmapped "
                                        "guest address");
                        }
                        break;
                    }
                    YIELD_ASC(vqid);
                }
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            while (queue_state->active.load(std::memory_order_acquire) &&
                   !queue_state->retired.load(std::memory_order_acquire)) {
                const auto result = test_wait_reg_mem(*wait_reg_mem);
                if (!result) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Abandoning ASC IT_WAIT_REG_MEM for invalid guest/register "
                                "address");
                    break;
                }
                if (*result) {
                    break;
                }
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (rasterizer) {
                rasterizer->ProcessDownloadImages();
            }
            const auto address = release_mem->Address<VAddr>();
            switch (release_mem->data_sel.Value()) {
            case DataSelect::Data32Low: {
                const u32 value = release_mem->DataDWord();
                if (!try_write_guest(address, &value, sizeof(value))) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Ignoring ASC IT_RELEASE_MEM write to unmapped guest address");
                }
                break;
            }
            case DataSelect::Data64: {
                const u64 value = release_mem->DataQWord();
                if (!try_write_guest(address, &value, sizeof(value))) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Ignoring ASC IT_RELEASE_MEM write to unmapped guest address");
                }
                break;
            }
            case DataSelect::GpuClock64: {
                const u64 value = GetGpuClock64();
                if (!try_write_guest(address, &value, sizeof(value))) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Ignoring ASC IT_RELEASE_MEM write to unmapped guest address");
                }
                break;
            }
            case DataSelect::PerfCounter: {
                const u64 value = GetGpuPerfCounter();
                if (!try_write_guest(address, &value, sizeof(value))) {
                    LOG_WARNING(Lib_GnmDriver,
                                "Ignoring ASC IT_RELEASE_MEM write to unmapped guest address");
                }
                break;
            }
            case DataSelect::GdsMemStore:
                if (rasterizer) {
                    rasterizer->CopyBuffer(address, release_mem->gds_index,
                                           release_mem->num_dw * sizeof(u32), false, true);
                }
                break;
            default:
                UNREACHABLE_MSG("Unknown ASC release-memory data select {}",
                                static_cast<u32>(release_mem->data_sel.Value()));
            }
            switch (release_mem->int_sel.Value()) {
            case InterruptSelect::None:
                break;
            case InterruptSelect::IrqUndocumented:
            case InterruptSelect::IrqWhenWriteConfirm:
                Platform::IrqC::Instance()->Signal(
                    static_cast<Platform::InterruptId>(queue.pipe_id));
                break;
            default:
                UNREACHABLE_MSG("Unknown ASC release-memory interrupt select {}",
                                static_cast<u32>(release_mem->int_sel.Value()));
            }
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);
        acb_guest_base_addr += static_cast<VAddr>(next_dw_off) * sizeof(u32);

        if constexpr (!is_indirect) {
            AdvanceAscReadPointer(queue_state, next_dw_off);
        }
    }

    FIBER_EXIT;
}

void Liverpool::AdvanceAscReadPointer(const std::shared_ptr<AscQueueInfo>& queue_state,
                                      u32 delta) {
    if (!queue_state || queue_state->ring_size_dw == 0) {
        return;
    }

    std::scoped_lock lock{asc_queue_mutex};
    if (!queue_state->active.load(std::memory_order_acquire) ||
        queue_state->retired.load(std::memory_order_acquire)) {
        return;
    }

    queue_state->read_offset_dw =
        static_cast<u32>((static_cast<u64>(queue_state->read_offset_dw) + delta) %
                         queue_state->ring_size_dw);
    const u32 value = queue_state->read_offset_dw;
    const Core::GuestMemoryWrite destination{queue_state->read_addr, &value, sizeof(value)};
    if (auto* memory = Core::Memory::Instance()) {
        (void)memory->TryWriteGuest(std::span<const Core::GuestMemoryWrite>{&destination, 1});
    }
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb,
                          GuestCommandBufferReader reader) {
    auto& queue = mapped_queues[GfxQueueId];

    auto submission = PrepareGraphicsSubmission(dcb, ccb, std::move(reader));
    if (!submission.owner) {
        LOG_WARNING(Lib_GnmDriver,
                    "Dropping graphics submission whose command-buffer range is unreadable");
        return;
    }

    const auto sequence = submission_completions.IssueSubmission();
    auto task =
        ProcessGraphics(submission.dcb, submission.ccb, sequence, std::move(submission.owner));
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.push({task.handle, sequence});
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb,
                          std::shared_ptr<AscQueueInfo> queue_state,
                          std::shared_ptr<const OwnedCommandBuffers> owner) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    ASSERT_MSG(queue_state, "ASC submission requires queue state");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    const auto sequence = submission_completions.IssueSubmission();
    auto owned_submission = owner
                                ? OwnedComputeSubmission{owner, owner->Dcb()}
                                : PrepareComputeSubmission(acb);
    const auto task = ProcessCompute(owned_submission.acb, vqid, sequence,
                                     std::move(owned_submission.owner),
                                     std::move(queue_state), reinterpret_cast<VAddr>(acb.data()));
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.push({task.handle, sequence});
    }

    std::scoped_lock lk{submit_mutex};
    const auto mapped_queue_count = num_mapped_queues.load(std::memory_order_relaxed);
    if (mapped_queue_count < gnm_vqid + 1) {
        // SubmitAsc serializes publication with num_submits under submit_mutex. Keep the count
        // atomic as the GPU owner reads it after releasing that mutex while rotating queues.
        num_mapped_queues.store(gnm_vqid + 1, std::memory_order_release);
    }
    ++num_submits;
    submit_cv.notify_one();
}

void Liverpool::SubmitDone(Common::UniqueFunction<void>&& completion) noexcept {
    submission_completions.EnqueueBoundary([this, completion = std::move(completion)]() mutable {
        if (rasterizer) {
            rasterizer->OnSubmit();
            rasterizer->Flush();
        }
        if (completion) {
            completion();
        }
    });
    std::scoped_lock lk{submit_mutex};
    submit_cv.notify_one();
}

SubmissionCompletionQueue::CompletionTicket Liverpool::CaptureRetirementTicket() {
    auto ticket = submission_completions.CaptureCompletionTicket();
    // The caller holds Gnm issuance. Never drain callbacks here: a same-sequence SubmitDone
    // callback can itself need that gate. Wake the normal owner loop even for an idle ticket.
    std::scoped_lock lock{submit_mutex};
    submit_cv.notify_one();
    return ticket;
}

void Liverpool::WaitGpuCompletion() {
    SendCommand<true>([this] {
        if (rasterizer) {
            rasterizer->Finish();
            // Physical retirement alone does not publish deferred host fences.
            // Destructive memory barriers must not return before those writes.
            DrainCompletedEops();
        }
    });
}

} // namespace AmdGpu
