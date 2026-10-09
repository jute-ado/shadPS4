# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

# Find or download Google Test
include(FetchContent)
    FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang 22 diagnoses a conversion inside GoogleTest 1.17 while this
    # project promotes dependency warnings to errors.
    foreach(googletest_target gtest gtest_main gmock gmock_main)
        target_compile_options(${googletest_target} PRIVATE /clang:-Wno-character-conversion)
    endforeach()
endif()

set(TEST_TARGETS
    shadps4_controller_trace_test
    shadps4_settings_test
    shadps4_ngs2_test
    shadps4_gcn_test
)

add_executable(shadps4_controller_trace_test
    ${CMAKE_SOURCE_DIR}/src/input/controller_axis.cpp
    ${CMAKE_SOURCE_DIR}/src/input/controller_button.cpp
    ${CMAKE_SOURCE_DIR}/src/input/controller_trace.cpp
    test_controller_axis.cpp
    test_controller_button.cpp
    test_controller_connection.cpp
    test_controller_lightbar.cpp
    test_controller_trace.cpp
)
target_include_directories(shadps4_controller_trace_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_controller_trace_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_controller_trace_test PRIVATE
    GTest::gtest_main
    SDL3::SDL3
    spdlog::spdlog
)

add_executable(shadps4_shadnet_client_version_test
    test_shadnet_client_version.cpp
)
target_include_directories(shadps4_shadnet_client_version_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_shadnet_client_version_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_shadnet_client_version_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_shadnet_client_version_test)

add_executable(shadps4_renderdoc_path_test
    ${CMAKE_SOURCE_DIR}/src/video_core/renderdoc_path.cpp
    test_renderdoc_path.cpp
)
target_include_directories(shadps4_renderdoc_path_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_renderdoc_path_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_renderdoc_path_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_renderdoc_path_test)

add_executable(shadps4_renderdoc_capture_state_test
    test_renderdoc_capture_state.cpp
)
target_include_directories(shadps4_renderdoc_capture_state_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_renderdoc_capture_state_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_renderdoc_capture_state_test PRIVATE
    SHADPS4_PRESENTER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_presenter.cpp"
    SHADPS4_SCHEDULER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_scheduler.cpp"
    SHADPS4_RASTERIZER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_rasterizer.cpp"
    SHADPS4_RENDERDOC_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderdoc.cpp"
)
target_link_libraries(shadps4_renderdoc_capture_state_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_renderdoc_capture_state_test)

add_executable(shadps4_ipc_capabilities_test
    test_ipc_capabilities.cpp
)
target_include_directories(shadps4_ipc_capabilities_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_ipc_capabilities_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_ipc_capabilities_test PRIVATE
    SHADPS4_IPC_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/ipc/ipc.cpp"
    SHADPS4_PRESENTER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_presenter.cpp"
    SHADPS4_PRESENTED_FRAME_QUEUE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/ipc/presented_frame_input_queue.h"
)
target_link_libraries(shadps4_ipc_capabilities_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_ipc_capabilities_test)

add_executable(shadps4_screenshot_request_queue_test
    test_screenshot_request_queue.cpp
)
target_include_directories(shadps4_screenshot_request_queue_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_screenshot_request_queue_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_screenshot_request_queue_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_screenshot_request_queue_test)

add_executable(shadps4_present_frame_ownership_test
    test_present_frame_ownership.cpp
)
target_include_directories(shadps4_present_frame_ownership_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_present_frame_ownership_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_present_frame_ownership_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_present_frame_ownership_test)

add_executable(shadps4_present_frame_transition_test
    test_present_frame_transition.cpp
)
target_include_directories(shadps4_present_frame_transition_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_present_frame_transition_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_present_frame_transition_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_present_frame_transition_test)

add_executable(shadps4_descriptor_image_layout_test
    test_descriptor_image_layout.cpp
)
target_include_directories(shadps4_descriptor_image_layout_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_descriptor_image_layout_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_descriptor_image_layout_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_descriptor_image_layout_test)

add_executable(shadps4_vulkan_result_retry_test
    test_vulkan_result_retry.cpp
)
target_include_directories(shadps4_vulkan_result_retry_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_vulkan_result_retry_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_vulkan_result_retry_test PRIVATE
    SHADPS4_MASTER_SEMAPHORE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_master_semaphore.cpp"
    SHADPS4_SCHEDULER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_scheduler.cpp"
)
target_link_libraries(shadps4_vulkan_result_retry_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_vulkan_result_retry_test)

add_executable(shadps4_vulkan_device_features_test
    test_vulkan_device_features.cpp
)
target_compile_features(shadps4_vulkan_device_features_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_vulkan_device_features_test PRIVATE
    SHADPS4_VK_INSTANCE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_instance.cpp"
    SHADPS4_VK_INSTANCE_HEADER_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_instance.h"
    SHADPS4_VK_PIPELINE_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp"
    SHADPS4_PROFILE_HEADER_PATH="${CMAKE_SOURCE_DIR}/src/shader_recompiler/profile.h"
)
target_link_libraries(shadps4_vulkan_device_features_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_vulkan_device_features_test)

add_executable(shadps4_pipeline_bind_tracker_test
    test_pipeline_bind_tracker.cpp
)
target_include_directories(shadps4_pipeline_bind_tracker_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_pipeline_bind_tracker_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_pipeline_bind_tracker_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_pipeline_bind_tracker_test)

add_executable(shadps4_shader_permutation_identity_test
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/specialization_identity.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/specialization_serialization.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_pipeline_cache_serialization.cpp
    stubs/serialization_stub.cpp
    test_pipeline_cache_serialization.cpp
    test_shader_permutation_identity.cpp
)
target_include_directories(shadps4_shader_permutation_identity_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_shader_permutation_identity_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_shader_permutation_identity_test PRIVATE
    GTest::gtest_main
    Boost::headers
    magic_enum::magic_enum
    spdlog::spdlog
    tsl::robin_map
    Vulkan::Headers
    xxHash::xxhash
)
list(APPEND TEST_TARGETS shadps4_shader_permutation_identity_test)

add_executable(shadps4_texture_types_test
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    test_texture_types.cpp
    stubs/common_stub.cpp
    stubs/logging_stub.cpp
    stubs/signals_stub.cpp
)
target_include_directories(shadps4_texture_types_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_texture_types_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_texture_types_test PRIVATE
    SHADPS4_PIPELINE_SERIALIZATION_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/shader_recompiler/specialization_serialization.cpp"
    SHADPS4_RASTERIZER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_rasterizer.cpp"
    SHADPS4_RASTERIZER_HEADER_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_rasterizer.h"
)
target_link_libraries(shadps4_texture_types_test PRIVATE
    Boost::headers
    GTest::gtest_main
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_texture_types_test)

add_executable(shadps4_image_layout_policy_test
    test_image_layout_policy.cpp
)
target_include_directories(shadps4_image_layout_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_image_layout_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_image_layout_policy_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_image_layout_policy_test)

add_executable(shadps4_image_tiling_range_test test_image_tiling_range.cpp)
target_include_directories(shadps4_image_tiling_range_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_image_tiling_range_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_image_tiling_range_test PRIVATE
    SHADPS4_TILE_MANAGER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/texture_cache/tile_manager.cpp"
    SHADPS4_BUFFER_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/buffer_cache.cpp"
    SHADPS4_TILING_SHADER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/host_shaders/tiling.comp"
)
target_link_libraries(shadps4_image_tiling_range_test PRIVATE
    GTest::gtest_main Vulkan::Headers spdlog::spdlog)
list(APPEND TEST_TARGETS shadps4_image_tiling_range_test)

add_executable(shadps4_texture_rejection_diagnostics_test test_texture_rejection_diagnostics.cpp)
target_include_directories(shadps4_texture_rejection_diagnostics_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_texture_rejection_diagnostics_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_texture_rejection_diagnostics_test PRIVATE
    SHADPS4_TEXTURE_REJECTION_RASTERIZER_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_rasterizer.cpp"
    SHADPS4_TEXTURE_REJECTION_IMAGE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/texture_cache/image.cpp"
)
target_link_libraries(shadps4_texture_rejection_diagnostics_test PRIVATE GTest::gtest_main spdlog::spdlog)
list(APPEND TEST_TARGETS shadps4_texture_rejection_diagnostics_test)

add_executable(shadps4_sampler_rejection_diagnostics_test test_sampler_rejection_diagnostics.cpp)
target_include_directories(shadps4_sampler_rejection_diagnostics_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_sampler_rejection_diagnostics_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_sampler_rejection_diagnostics_test PRIVATE
    SHADPS4_SAMPLER_REJECTION_RASTERIZER_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_rasterizer.cpp"
)
target_link_libraries(shadps4_sampler_rejection_diagnostics_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_sampler_rejection_diagnostics_test)

add_executable(shadps4_unique_buffer_test
    ${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/unique_buffer.cpp
    test_unique_buffer.cpp
)
target_include_directories(shadps4_unique_buffer_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_unique_buffer_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_unique_buffer_test PRIVATE
    GTest::gtest_main Boost::headers Vulkan::Headers GPUOpen::VulkanMemoryAllocator spdlog::spdlog)
list(APPEND TEST_TARGETS shadps4_unique_buffer_test)

add_executable(shadps4_deferred_fault_residency_test test_deferred_fault_residency.cpp)
target_include_directories(shadps4_deferred_fault_residency_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_deferred_fault_residency_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_deferred_fault_residency_test PRIVATE
    SHADPS4_FAULT_RESIDENCY_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/fault_manager.cpp"
    SHADPS4_FAULT_RESIDENCY_RASTERIZER_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_rasterizer.cpp"
    SHADPS4_FAULT_RESIDENCY_CACHE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/buffer_cache.cpp")
target_link_libraries(shadps4_deferred_fault_residency_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_deferred_fault_residency_test)

add_executable(shadps4_download_readback_policy_test
    test_download_readback_policy.cpp
)
target_include_directories(shadps4_download_readback_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_download_readback_policy_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_download_readback_policy_test PRIVATE
    SHADPS4_BUFFER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/buffer.cpp"
    SHADPS4_BUFFER_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/buffer_cache.cpp"
)
target_link_libraries(shadps4_download_readback_policy_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_download_readback_policy_test)

add_executable(shadps4_srt_memory_sync_test
    test_srt_memory_sync.cpp
    test_srt_preparation_retry.cpp
    test_srt_pipeline_sync_source.cpp
    test_srt_memory_sync_source.cpp
    test_srt_snapshot_epoch_coordinator.cpp
)
target_include_directories(shadps4_srt_memory_sync_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_srt_memory_sync_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_srt_memory_sync_test PRIVATE
    SHADPS4_BUFFER_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/buffer_cache.cpp"
    SHADPS4_PIPELINE_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp"
    SHADPS4_GRAPHICS_PIPELINE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp"
    SHADPS4_RASTERIZER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_rasterizer.cpp"
)
target_link_libraries(shadps4_srt_memory_sync_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_srt_memory_sync_test)

add_executable(shadps4_image_usage_policy_test
    test_image_usage_policy.cpp
)
target_include_directories(shadps4_image_usage_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_image_usage_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_image_usage_policy_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_image_usage_policy_test)

add_executable(shadps4_host_image_scaling_test
    test_host_image_scaling.cpp
)
target_include_directories(shadps4_host_image_scaling_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_host_image_scaling_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_host_image_scaling_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_host_image_scaling_test)

add_executable(shadps4_fault_range_test
    test_fault_range.cpp
)
target_include_directories(shadps4_fault_range_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_fault_range_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_fault_range_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_fault_range_test)

add_executable(shadps4_address_space_operation_gate_test
    test_address_space_operation_gate.cpp
)
target_include_directories(shadps4_address_space_operation_gate_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_address_space_operation_gate_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_address_space_operation_gate_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_address_space_operation_gate_test)

add_executable(shadps4_gnm_graphics_event_test
    test_gnm_graphics_event.cpp
)
target_include_directories(shadps4_gnm_graphics_event_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_gnm_graphics_event_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_gnm_graphics_event_test PRIVATE
    GTest::gtest_main
    Boost::headers
)
list(APPEND TEST_TARGETS shadps4_gnm_graphics_event_test)

add_executable(shadps4_videodec2_picture_info_store_test
    test_videodec2_picture_info_store.cpp
)
target_include_directories(shadps4_videodec2_picture_info_store_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_videodec2_picture_info_store_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_videodec2_picture_info_store_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_videodec2_picture_info_store_test)

if (WIN32)
    add_executable(shadps4_windows_long_path_test
        ${CMAKE_SOURCE_DIR}/src/dist/windows_long_path_aware.manifest
        test_windows_long_path.cpp
    )
    target_compile_features(shadps4_windows_long_path_test PRIVATE cxx_std_23)
    target_link_libraries(shadps4_windows_long_path_test PRIVATE GTest::gtest_main)
    list(APPEND TEST_TARGETS shadps4_windows_long_path_test)
endif()

add_executable(shadps4_gnm_submission_gate_test
    test_gnm_submission_gate.cpp
)
target_include_directories(shadps4_gnm_submission_gate_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_gnm_submission_gate_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_gnm_submission_gate_test PRIVATE GTest::gtest_main)
target_compile_definitions(shadps4_gnm_submission_gate_test PRIVATE
    LIVERPOOL_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/liverpool.cpp"
)
list(APPEND TEST_TARGETS shadps4_gnm_submission_gate_test)

add_executable(shadps4_memory_retirement_test
    test_memory_retirement.cpp
)
target_include_directories(shadps4_memory_retirement_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_memory_retirement_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_memory_retirement_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_memory_retirement_test)

add_executable(shadps4_sysmodule_requirement_test
    test_sysmodule_requirement.cpp
)
target_include_directories(shadps4_sysmodule_requirement_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_sysmodule_requirement_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_sysmodule_requirement_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_sysmodule_requirement_test)

add_executable(shadps4_host_compilation_pause_test
    test_async_compilation_queue.cpp
    test_host_compilation_pause.cpp
)
target_include_directories(shadps4_host_compilation_pause_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(shadps4_host_compilation_pause_test PRIVATE
    SHADPS4_PIPELINE_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp"
    SHADPS4_PIPELINE_CACHE_HEADER_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_pipeline_cache.h"
    SHADPS4_EMULATOR_SETTINGS_HEADER_PATH="${CMAKE_SOURCE_DIR}/src/core/emulator_settings.h"
    SHADPS4_EMULATOR_SETTINGS_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp"
)
target_compile_features(shadps4_host_compilation_pause_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_host_compilation_pause_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_host_compilation_pause_test)

add_executable(shadps4_async_graphics_optimization_diagnostic_test
    test_async_graphics_optimization_diagnostic.cpp
)
target_include_directories(shadps4_async_graphics_optimization_diagnostic_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(shadps4_async_graphics_optimization_diagnostic_test PRIVATE
    SHADPS4_OPTIMIZATION_PIPELINE_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp"
    SHADPS4_OPTIMIZATION_GRAPHICS_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp"
    SHADPS4_OPTIMIZATION_GRAPHICS_HEADER_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_graphics_pipeline.h"
    SHADPS4_OPTIMIZATION_SERIALIZATION_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_pipeline_serialization.cpp"
)
target_compile_features(shadps4_async_graphics_optimization_diagnostic_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_async_graphics_optimization_diagnostic_test PRIVATE
    GTest::gtest_main Vulkan::Headers)
list(APPEND TEST_TARGETS shadps4_async_graphics_optimization_diagnostic_test)

add_executable(shadps4_guest_thread_priority_test
    test_guest_thread_priority.cpp
)
target_include_directories(shadps4_guest_thread_priority_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_guest_thread_priority_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_guest_thread_priority_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_guest_thread_priority_test)

add_executable(shadps4_event_write_eop_test
    test_event_write_eop.cpp
)
target_include_directories(shadps4_event_write_eop_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_event_write_eop_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_event_write_eop_test PRIVATE
    SHADPS4_EOP_SOURCE_ROOT="${CMAKE_SOURCE_DIR}/src"
)
target_link_libraries(shadps4_event_write_eop_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_event_write_eop_test)

add_executable(shadps4_videodec_packet_handle_test
    test_videodec_packet_handle.cpp
)
target_include_directories(shadps4_videodec_packet_handle_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_videodec_packet_handle_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_videodec_packet_handle_test PRIVATE
    GTest::gtest_main
    FFmpeg::ffmpeg
)
list(APPEND TEST_TARGETS shadps4_videodec_packet_handle_test)

add_executable(shadps4_gpu_mapping_admission_test
    test_gpu_mapping_admission.cpp
)
target_include_directories(shadps4_gpu_mapping_admission_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_gpu_mapping_admission_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_gpu_mapping_admission_test PRIVATE
    SHADPS4_KERNEL_MEMORY_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/libraries/kernel/memory.cpp"
    SHADPS4_MEMORY_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/memory.cpp"
    SHADPS4_GNM_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/libraries/gnmdriver/gnmdriver.cpp"
    SHADPS4_LIVERPOOL_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/liverpool.cpp"
)
target_link_libraries(shadps4_gpu_mapping_admission_test PRIVATE
    GTest::gtest_main
    Boost::headers
)
list(APPEND TEST_TARGETS shadps4_gpu_mapping_admission_test)

add_executable(shadps4_gpu_owner_lifetime_test test_gpu_owner_lifetime.cpp)
target_include_directories(shadps4_gpu_owner_lifetime_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_gpu_owner_lifetime_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_gpu_owner_lifetime_test PRIVATE
    GPU_OWNER_SOURCE_ROOT="${CMAKE_SOURCE_DIR}/src"
)
target_link_libraries(shadps4_gpu_owner_lifetime_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_gpu_owner_lifetime_test)

add_executable(shadps4_buffer_residency_test
    test_buffer_residency.cpp
)
target_include_directories(shadps4_buffer_residency_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_buffer_residency_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_buffer_residency_test PRIVATE
    SHADPS4_RASTERIZER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_rasterizer.cpp"
    SHADPS4_RASTERIZER_HEADER_PATH="${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/vk_rasterizer.h"
    SHADPS4_PAGE_MANAGER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/page_manager.cpp"
    SHADPS4_MEMORY_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/memory.cpp"
)
target_link_libraries(shadps4_buffer_residency_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_buffer_residency_test)

add_executable(shadps4_buffer_registration_index_test test_buffer_registration_index.cpp)
target_include_directories(shadps4_buffer_registration_index_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_buffer_registration_index_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_buffer_registration_index_test PRIVATE
    BUFFER_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/buffer_cache.cpp"
    BUFFER_CACHE_HEADER_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/buffer_cache.h"
)
target_link_libraries(shadps4_buffer_registration_index_test PRIVATE GTest::gtest_main Boost::headers)
list(APPEND TEST_TARGETS shadps4_buffer_registration_index_test)

add_executable(shadps4_mapped_page_protection_test
    test_mapped_page_protection.cpp
)
target_include_directories(shadps4_mapped_page_protection_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_mapped_page_protection_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_mapped_page_protection_test PRIVATE
    GTest::gtest_main
    Boost::headers
)
list(APPEND TEST_TARGETS shadps4_mapped_page_protection_test)

add_executable(shadps4_pixel_number_conversion_test
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    test_pixel_number_conversion.cpp
    stubs/common_stub.cpp
    stubs/logging_stub.cpp
    stubs/signals_stub.cpp
)
target_include_directories(shadps4_pixel_number_conversion_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_pixel_number_conversion_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_pixel_number_conversion_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_pixel_number_conversion_test)

add_executable(shadps4_buffer_access_range_test
    test_buffer_access_range.cpp
)
target_include_directories(shadps4_buffer_access_range_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_buffer_access_range_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_buffer_access_range_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_buffer_access_range_test)

add_executable(shadps4_buffer_copy_endpoints_test test_buffer_copy_endpoints.cpp)
target_include_directories(shadps4_buffer_copy_endpoints_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_buffer_copy_endpoints_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_buffer_copy_endpoints_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_buffer_copy_endpoints_test)

add_executable(shadps4_cpu_page_write_snapshot_test
    test_cpu_page_write_snapshot.cpp
)
target_include_directories(shadps4_cpu_page_write_snapshot_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_cpu_page_write_snapshot_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_cpu_page_write_snapshot_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_cpu_page_write_snapshot_test)

add_executable(shadps4_sparse_memory_test
    test_sparse_memory.cpp
)
target_include_directories(shadps4_sparse_memory_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_sparse_memory_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_sparse_memory_test PRIVATE
    GTest::gtest_main
    Boost::headers
)
list(APPEND TEST_TARGETS shadps4_sparse_memory_test)

add_executable(shadps4_backing_memory_write_test
    test_backing_memory_write.cpp
    test_deferred_readback_publication.cpp
)
target_include_directories(shadps4_backing_memory_write_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_backing_memory_write_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_backing_memory_write_test PRIVATE
    SHADPS4_MEMORY_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/memory.cpp"
    SHADPS4_BUFFER_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/buffer_cache.cpp"
    SHADPS4_TEXTURE_CACHE_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/texture_cache/texture_cache.cpp"
)
target_link_libraries(shadps4_backing_memory_write_test PRIVATE
    GTest::gtest_main
    Boost::headers
)
list(APPEND TEST_TARGETS shadps4_backing_memory_write_test)

if (WIN32)
    add_executable(shadps4_windows_protection_snapshot_test
        ${CMAKE_SOURCE_DIR}/src/core/windows_protection_snapshot.cpp
        test_windows_protection_snapshot.cpp
    )
    target_link_libraries(shadps4_windows_protection_snapshot_test PRIVATE GTest::gtest_main)
    list(APPEND TEST_TARGETS shadps4_windows_protection_snapshot_test)

    add_executable(shadps4_windows_exception_policy_test
        test_windows_exception_policy.cpp
    )
    target_compile_definitions(shadps4_windows_exception_policy_test PRIVATE
        SHADPS4_ASSERT_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/common/assert.cpp"
        SHADPS4_EMULATOR_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/emulator.cpp"
        SHADPS4_SIGNALS_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/signals.cpp"
        SHADPS4_THREAD_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/core/thread.cpp"
    )
    target_link_libraries(shadps4_windows_exception_policy_test PRIVATE GTest::gtest_main)
    list(APPEND TEST_TARGETS shadps4_windows_exception_policy_test)

endif()

if (WIN32 AND ARCHITECTURE STREQUAL "x86_64")
    add_executable(shadps4_windows_exception_stack_test
        ${CMAKE_SOURCE_DIR}/src/core/libraries/kernel/threads/stack_asm.cpp
        ${CMAKE_SOURCE_DIR}/src/core/windows_exception_stack.cpp
        test_windows_exception_stack.cpp
    )
    target_link_libraries(shadps4_windows_exception_stack_test PRIVATE GTest::gtest_main)
    list(APPEND TEST_TARGETS shadps4_windows_exception_stack_test)
endif()

add_executable(shadps4_depth_association_test
    test_depth_association.cpp
)
target_include_directories(shadps4_depth_association_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_depth_association_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_depth_association_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_depth_association_test)

add_executable(shadps4_depth_stencil_policy_test
    test_depth_stencil_policy.cpp
)
target_include_directories(shadps4_depth_stencil_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_depth_stencil_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_depth_stencil_policy_test PRIVATE
    GTest::gtest_main
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_depth_stencil_policy_test)

add_executable(shadps4_module_mapping_test
    test_module_mapping.cpp
)
target_include_directories(shadps4_module_mapping_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_module_mapping_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_module_mapping_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_module_mapping_test)

add_executable(shadps4_legacy_user_data_test
    ${CMAKE_SOURCE_DIR}/src/core/legacy_user_data.cpp
    test_legacy_user_data.cpp
)
target_include_directories(shadps4_legacy_user_data_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_legacy_user_data_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_legacy_user_data_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_legacy_user_data_test)

set(SETTINGS_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/core_stub.cpp
    stubs/signals_stub.cpp
    stubs/file_sys_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp

    # Tests
    test_emulator_settings.cpp
    test_guest_display_resolution.cpp
    # Windows static guest red-zone protection
    test_windows_guest_red_zone_protection_settings.cpp
)

set(NGS2_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/libraries/audio/atrac9_codec_lock.cpp
    ${CMAKE_SOURCE_DIR}/src/core/libraries/audio/audio_pipeline_metadata_trace.cpp
    ${CMAKE_SOURCE_DIR}/src/core/libraries/ngs2/ngs2_atrac9.cpp
    ${CMAKE_SOURCE_DIR}/src/core/libraries/ngs2/ngs2_decode.cpp
    ${CMAKE_SOURCE_DIR}/src/core/libraries/ngs2/ngs2_impl.cpp
    ${CMAKE_SOURCE_DIR}/src/core/libraries/ngs2/ngs2_render.cpp

    # Minimal common and kernel support
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp
    ${CMAKE_SOURCE_DIR}/src/core/libraries/kernel/sync/mutex.cpp
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/core_stub.cpp
    stubs/signals_stub.cpp
    stubs/file_sys_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp

    # Tests
    test_audio_pipeline_metadata_trace.cpp
    test_ngs2_atrac9.cpp
    test_ngs2_decode.cpp
    test_ngs2_guest_io.cpp
    test_ngs2_render.cpp
    test_ngs2.cpp
)
set(GCN_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_atomic.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_barriers.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_bitwise_conversion.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_composite.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_context_get_set.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_convert.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_floating_point.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_image.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_integer.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_logical.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_select.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_shared_memory.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_special.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_undefined.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_warp.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_instructions.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/spirv_emit_context.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/spirv_emit_context.h
    # The production-pipeline fixture uses the actual CFG/structurizer and every pass called by
    # TranslateProgram, rather than the single-instruction helper's resource-tracking stub.
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/recompiler.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/control_flow_graph.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/copy_shader.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/instruction.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/structured_control_flow.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/format.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/decode.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/decode.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/fetch_shader.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/fetch_shader.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/data_share.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/export.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/translate.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/translate.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/scalar_alu.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/scalar_flow.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/scalar_memory.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/vector_alu.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/vector_interpolation.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/vector_memory.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/abstract_syntax_list.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/abstract_syntax_list.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/basic_block.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/basic_block.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/microinstruction.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/attribute.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/attribute.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/ir_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/ir_emitter.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/opcodes.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/opcodes.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/patch.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/patch.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/program.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/program.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/ir_passes.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/buffer_access_range_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/constant_propagation_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/dead_code_elimination_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/flatten_extended_userdata_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/identity_removal_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/inject_clip_distance_attributes.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/hull_shader_transform.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/lower_fp64_to_fp32.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/lower_user_clip_planes.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/ring_access_elimination.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/resource_tracking_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/lower_buffer_format_to_raw.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/lower_lane_id_to_local_invocation_index_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/shared_memory_barrier_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/shared_memory_simplify_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/shared_memory_to_storage_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/readlane_elimination_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/shader_info_collection_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/ssa_rewrite_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/post_order.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/post_order.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/type.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/type.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/value.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/value.h
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.h
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/resource.h

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/decoder.cpp
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/io_file.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp
    ${CMAKE_SOURCE_DIR}/src/common/ntapi.cpp
    ${CMAKE_SOURCE_DIR}/src/common/signal_context.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/resource_tracking_pass_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp
    stubs/core_stub.cpp
    stubs/signals_stub.cpp
    stubs/file_sys_stub.cpp

    gcn/gcn_test_runner.hpp
    gcn/gcn_test_runner.cpp
    gcn/translator.hpp
    gcn/translator.cpp

    # Tests
    gcn/test_bda_pointer_bounds.cpp
    gcn/test_inline_buffer_write_tracking.cpp
    gcn/test_buffer_access_range_pass.cpp
    gcn/test_barycentric_spirv.cpp
    gcn/test_dynamic_image_table_runtime.cpp
    gcn/test_dynamic_image_table_srt.cpp
    gcn/test_descriptor_array_spirv.cpp
    gcn/test_dynamic_image_table_policy.cpp
    gcn/test_fetch_shader_capture.cpp
    gcn/test_gcn_instructions.cpp
    gcn/test_lane_id_lowering.cpp
    gcn/test_lower_buffer_format_to_raw.cpp
    gcn/test_resource_number_conversion.cpp
    gcn/test_shared_memory_barriers.cpp
    gcn/test_srt_walker.cpp
    gcn/test_specialization_number_conversion.cpp
    gcn/test_ssa_rewrite.cpp
    gcn/test_translate_program.cpp
    gcn/test_wave64_conformance.cpp
)

add_executable(shadps4_settings_test ${SETTINGS_TEST_SOURCES})
add_executable(shadps4_ngs2_test ${NGS2_TEST_SOURCES})
add_executable(shadps4_gcn_test ${GCN_TEST_SOURCES})
target_compile_definitions(shadps4_gcn_test PRIVATE
    SHADPS4_FAULT_MANAGER_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/video_core/buffer_cache/fault_manager.cpp"
)
find_program(SHADPS4_SPIRV_VAL_EXECUTABLE
    NAMES spirv-val spirv-val.exe
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/Bin32"
)
if (SHADPS4_SPIRV_VAL_EXECUTABLE)
    file(TO_CMAKE_PATH "${SHADPS4_SPIRV_VAL_EXECUTABLE}" SHADPS4_SPIRV_VAL_PATH)
    target_compile_definitions(shadps4_gcn_test PRIVATE
        SHADPS4_SPIRV_VAL_PATH="${SHADPS4_SPIRV_VAL_PATH}"
    )
endif()

foreach(t ${TEST_TARGETS})
    target_include_directories(${t} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
    )
    target_compile_features(${t} PRIVATE cxx_std_23)
    target_compile_definitions(${t} PRIVATE BOOST_ASIO_STANDALONE)
endforeach()

target_link_libraries(shadps4_settings_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3
    spdlog::spdlog
)
target_link_libraries(shadps4_ngs2_test PRIVATE
    GTest::gtest_main
    Boost::headers
    fmt::fmt
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3
    spdlog::spdlog
    LibAtrac9
)
target_link_libraries(shadps4_gcn_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    nlohmann_json::nlohmann_json
    magic_enum::magic_enum
    toml11::toml11
    Boost::headers
    Vulkan::Headers
    sirit
    SDL3::SDL3
    spdlog::spdlog
    half::half
    xbyak::xbyak
    xxHash::xxhash
    Zydis::Zydis
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
    if (LIBCPP)
        foreach(t ${TEST_TARGETS})
            target_compile_options(${t} PRIVATE -fexperimental-library)
        endforeach()
    endif()
endif()

foreach(t ${TEST_TARGETS})
    if (WIN32)
        target_link_libraries(${t} PRIVATE onecore)
        target_compile_definitions(${t} PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            NTDDI_VERSION=0x0A000006
            _WIN32_WINNT=0x0A00
            WINVER=0x0A00
        )
        if (MSVC)
            target_compile_definitions(${t} PRIVATE
                _CRT_SECURE_NO_WARNINGS
                _CRT_NONSTDC_NO_DEPRECATE
                _SCL_SECURE_NO_WARNINGS
                _TIMESPEC_DEFINED
            )
        endif()
    endif()
endforeach()

include(GoogleTest)

foreach(t ${TEST_TARGETS})
    gtest_discover_tests(${t}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DISCOVERY_TIMEOUT 60
    )
endforeach()

# The capability probe stays self-contained so the upstream adapter does not
# pull emulator runtime dependencies into this contract test.
add_executable(shadps4_test_lab_probe_test
    ${CMAKE_SOURCE_DIR}/src/core/test_lab_probe.cpp
    test_test_lab_probe.cpp
)
target_include_directories(shadps4_test_lab_probe_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_test_lab_probe_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_test_lab_probe_test PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)
gtest_discover_tests(shadps4_test_lab_probe_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DISCOVERY_TIMEOUT 60
)

add_executable(shadps4_presented_frame_timing_trace_test
    test_presented_frame_timing_trace.cpp
)
target_include_directories(shadps4_presented_frame_timing_trace_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_presented_frame_timing_trace_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_presented_frame_timing_trace_test PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)
gtest_discover_tests(shadps4_presented_frame_timing_trace_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DISCOVERY_TIMEOUT 60
)

add_executable(shadps4_test_lab_audio_capture_test
    ${CMAKE_SOURCE_DIR}/src/core/libraries/audio/test_lab_audio_capture.cpp
    test_test_lab_audio_capture.cpp
)
target_include_directories(shadps4_test_lab_audio_capture_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_test_lab_audio_capture_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_test_lab_audio_capture_test PRIVATE
    GTest::gtest_main
)
gtest_discover_tests(shadps4_test_lab_audio_capture_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DISCOVERY_TIMEOUT 60
)

# ===========================================================================
# HTTP tests (libSceHttp - URI helpers and status-line parsing)
# ===========================================================================
# Self-contained block - mirrors the structure of the targets above so the
# new test does not depend on any earlier foreach iterations.

set(HTTP_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/libraries/network/http.cpp
    # Required to link RegisterLib's LIB_FUNCTION calls and the logger's
    # access to EmulatorSettings.
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp
    stubs/loader_stub.cpp
    stubs/core_stub.cpp
    stubs/signals_stub.cpp
    stubs/file_sys_stub.cpp
    stubs/kernel_stub.cpp

    # Tests
    network/test_http_uri.cpp
    network/test_http_status_line.cpp
    network/test_http_parse_response_header.cpp
    network/test_http_lifecycle.cpp
)

add_executable(shadps4_http_test ${HTTP_TEST_SOURCES})

list(APPEND TEST_TARGETS shadps4_http_test)

target_include_directories(shadps4_http_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}
)
target_compile_features(shadps4_http_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_http_test PRIVATE BOOST_ASIO_STANDALONE)

target_link_libraries(shadps4_http_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    magic_enum::magic_enum
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3
    spdlog::spdlog
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
    if (LIBCPP)
        target_compile_options(shadps4_http_test PRIVATE -fexperimental-library)
    endif()
endif()

if (WIN32)
    target_link_libraries(shadps4_http_test PRIVATE onecore)
    target_compile_definitions(shadps4_http_test PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        NTDDI_VERSION=0x0A000006
        _WIN32_WINNT=0x0A00
        WINVER=0x0A00
    )
    if (MSVC)
        target_compile_definitions(shadps4_http_test PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
            _TIMESPEC_DEFINED
        )
    endif()
endif()

gtest_discover_tests(shadps4_http_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DISCOVERY_TIMEOUT 60
)
