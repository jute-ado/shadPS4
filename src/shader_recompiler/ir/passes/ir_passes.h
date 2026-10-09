// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "shader_recompiler/ir/program.h"

namespace Shader {
struct Profile;
void InjectClipDistanceAttributes(IR::Program& program, RuntimeInfo& runtime_info);
} // namespace Shader

namespace Shader::Optimization {

void SsaRewritePass(IR::Program& program, bool predicate_vector_by_exec = false,
                    bool capture_split_wave64_masks = false);
void SsaDestroyPass(IR::Program& program, bool spill_cross_block_values = false);
void IdentityRemovalPass(IR::BlockList& program);
void DeadCodeEliminationPass(IR::Program& program);
void ConstantPropagationPass(IR::BlockList& program);
void FlattenExtendedUserdataPass(IR::Program& program, SrtMemoryReader reader = {});
void ReadLaneEliminationPass(IR::Program& program);
void ResourceTrackingPass(IR::Program& program, const Profile& profile);
void BufferAccessRangePass(IR::Program& program);
void CollectShaderInfoPass(IR::Program& program, const Profile& profile);
void LowerBufferFormatToRaw(IR::Program& program);
void LowerUserClipPlanes(IR::Program& program, const RuntimeInfo& runtime_info);
void LowerFp64ToFp32(IR::Program& program);
void RingAccessElimination(const IR::Program& program, const RuntimeInfo& runtime_info);
void TessellationPreprocess(IR::Program& program, RuntimeInfo& runtime_info);
void HullShaderTransform(IR::Program& program, const RuntimeInfo& runtime_info);
void DomainShaderTransform(const IR::Program& program, const RuntimeInfo& runtime_info);
void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile);
void Wave64SubgroupLoweringPass(IR::Program& program, RuntimeInfo& runtime_info,
                                const Profile& profile, bool guest_exec_predication = false,
                                bool enable_split_wave64_scalar_control = false,
                                bool enable_wave64_read_first_lane = false);
void Wave64ProtocolCleanupPass(IR::Program& program);
void SharedMemorySimplifyPass(IR::Program& program, const Profile& profile);
void SharedMemoryToStoragePass(IR::Program& program, const RuntimeInfo& runtime_info,
                               const Profile& profile);

} // namespace Shader::Optimization
