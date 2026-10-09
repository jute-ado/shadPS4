// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/ngs2/ngs2_decode.h"

namespace Libraries::Ngs2 {

s32 RenderSystemAudio(OrbisNgs2Handle system_handle, const OrbisNgs2RenderBufferInfo* buffer_info,
                      u32 num_buffer_info, OrbisNgs2RuntimeReadHandler read_handler,
                      void* read_context, Atrac9FrameDecoderFactory decoder_factory,
                      SystemRenderPublishHandler publish_handler = nullptr,
                      void* publish_context = nullptr);

// Deterministic seam for exercising the complete lower SystemRender path without requiring the
// proprietary ATRAC9 decoder to synthesize specific PCM values.
s32 SystemRenderWithFactoryForTesting(OrbisNgs2Handle system_handle,
                                      const OrbisNgs2RenderBufferInfo* buffer_info,
                                      u32 num_buffer_info, OrbisNgs2RuntimeReadHandler read_handler,
                                      void* read_context, Atrac9FrameDecoderFactory decoder_factory,
                                      SystemRenderPublishHandler publish_handler = nullptr,
                                      void* publish_context = nullptr);

} // namespace Libraries::Ngs2
