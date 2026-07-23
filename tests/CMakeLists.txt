# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

# Find or download Google Test
include(FetchContent)
    FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

# GoogleTest 1.17 enables /WX for its own targets under clang-cl. Clang 22 diagnoses a
# char8_t-to-char32_t conversion in gtest-printers.h, so keep the third-party target buildable
# without weakening warnings for shadPS4's test targets.
if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    foreach(googletest_target gtest gtest_main gmock gmock_main)
        target_compile_options(${googletest_target} PRIVATE -Wno-character-conversion)
    endforeach()
endif()

set(TEST_TARGETS
    shadps4_settings_test
    shadps4_ngs2_test
    shadps4_gcn_test
)

add_executable(shadps4_controller_button_test
    ${CMAKE_SOURCE_DIR}/src/input/controller_axis.cpp
    ${CMAKE_SOURCE_DIR}/src/input/controller_button.cpp
    test_controller_axis.cpp
    test_controller_button.cpp
    test_controller_connection.cpp
    test_controller_touch.cpp
    test_virtual_gamepad.cpp
)
target_include_directories(shadps4_controller_button_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_controller_button_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_controller_button_test PRIVATE
    GTest::gtest_main
    SDL3::SDL3
)
list(APPEND TEST_TARGETS shadps4_controller_button_test)

add_executable(shadps4_savedata_dialog_navigation_test
    test_savedata_dialog_navigation.cpp
)
target_include_directories(shadps4_savedata_dialog_navigation_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_savedata_dialog_navigation_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_savedata_dialog_navigation_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_savedata_dialog_navigation_test)

add_executable(shadps4_notification_state_test
    test_notification_state.cpp
)
target_include_directories(shadps4_notification_state_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_notification_state_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_notification_state_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_notification_state_test)

add_executable(shadps4_texture_job_queue_test
    test_texture_job_queue.cpp
)
target_include_directories(shadps4_texture_job_queue_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_texture_job_queue_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_texture_job_queue_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_texture_job_queue_test)

add_executable(shadps4_png_decode_test
    test_png_decode.cpp
    ../src/common/stb.cpp
    ../src/imgui/renderer/png_decode.cpp
)
target_include_directories(shadps4_png_decode_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_png_decode_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_png_decode_test PRIVATE
    GTest::gtest_main
    stb
)
list(APPEND TEST_TARGETS shadps4_png_decode_test)

add_executable(shadps4_texture_types_test
    test_texture_types.cpp
)
target_include_directories(shadps4_texture_types_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_texture_types_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_texture_types_test PRIVATE
    GTest::gtest_main
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_texture_types_test)

add_executable(shadps4_buffer_resource_test
    test_buffer_resource.cpp
)
target_include_directories(shadps4_buffer_resource_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_buffer_resource_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_buffer_resource_test PRIVATE
    GTest::gtest_main
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_buffer_resource_test)

add_executable(shadps4_vulkan_layer_settings_test
    test_vulkan_layer_settings.cpp
)
target_include_directories(shadps4_vulkan_layer_settings_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_vulkan_layer_settings_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_vulkan_layer_settings_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_vulkan_layer_settings_test)

add_executable(shadps4_pipeline_bind_history_test
    test_pipeline_bind_history.cpp
)
target_include_directories(shadps4_pipeline_bind_history_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_pipeline_bind_history_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_pipeline_bind_history_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_pipeline_bind_history_test)

add_executable(shadps4_event_write_eop_test
    test_event_write_eop.cpp
)
target_include_directories(shadps4_event_write_eop_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_event_write_eop_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_event_write_eop_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_event_write_eop_test)

add_executable(shadps4_pm4_memory_wait_test
    test_pm4_memory_wait.cpp
)
target_include_directories(shadps4_pm4_memory_wait_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_pm4_memory_wait_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_pm4_memory_wait_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_pm4_memory_wait_test)

add_executable(shadps4_gnm_submission_gate_test
    test_gnm_submission_gate.cpp
)
target_include_directories(shadps4_gnm_submission_gate_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_gnm_submission_gate_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_gnm_submission_gate_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_gnm_submission_gate_test)

add_executable(shadps4_guest_thread_priority_test
    test_guest_thread_priority.cpp
)
target_include_directories(shadps4_guest_thread_priority_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_guest_thread_priority_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_guest_thread_priority_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_guest_thread_priority_test)

add_executable(shadps4_irq_controller_test
    test_irq_controller.cpp
)
target_include_directories(shadps4_irq_controller_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_irq_controller_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_irq_controller_test PRIVATE
    GTest::gtest_main
    magic_enum::magic_enum
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_irq_controller_test)

add_executable(shadps4_backing_write_cursor_test
    test_backing_write_cursor.cpp
)
target_include_directories(shadps4_backing_write_cursor_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_backing_write_cursor_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_backing_write_cursor_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_backing_write_cursor_test)

add_executable(shadps4_vulkan_diagnostic_checkpoint_test
    test_vulkan_diagnostic_checkpoint.cpp
)
target_include_directories(shadps4_vulkan_diagnostic_checkpoint_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_vulkan_diagnostic_checkpoint_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_vulkan_diagnostic_checkpoint_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_vulkan_diagnostic_checkpoint_test)

if(WIN32)
    add_executable(shadps4_windows_sparse_backing_test
        test_windows_sparse_backing.cpp
    )
    target_include_directories(shadps4_windows_sparse_backing_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_features(shadps4_windows_sparse_backing_test PRIVATE cxx_std_23)
    target_link_libraries(shadps4_windows_sparse_backing_test PRIVATE GTest::gtest_main)
    list(APPEND TEST_TARGETS shadps4_windows_sparse_backing_test)
endif()

add_executable(shadps4_pm4_lod_stats_test
    test_pm4_lod_stats.cpp
)
target_include_directories(shadps4_pm4_lod_stats_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_pm4_lod_stats_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_pm4_lod_stats_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_pm4_lod_stats_test)

add_executable(shadps4_pm4_predication_test
    test_pm4_predication.cpp
)
target_include_directories(shadps4_pm4_predication_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_pm4_predication_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_pm4_predication_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_pm4_predication_test)

add_executable(shadps4_pm4_cond_exec_test
    test_pm4_cond_exec.cpp
)
target_include_directories(shadps4_pm4_cond_exec_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_pm4_cond_exec_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_pm4_cond_exec_test PRIVATE
    GTest::gtest_main
    Boost::headers
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_pm4_cond_exec_test)

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

add_executable(shadps4_presenter_sync_test
    test_presenter_sync.cpp
)
target_include_directories(shadps4_presenter_sync_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_presenter_sync_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_presenter_sync_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_presenter_sync_test)

add_executable(shadps4_equeue_event_test
    test_equeue_event.cpp
)
target_include_directories(shadps4_equeue_event_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_equeue_event_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_equeue_event_test PRIVATE GTest::gtest_main Boost::headers)
list(APPEND TEST_TARGETS shadps4_equeue_event_test)

add_executable(shadps4_flip_queue_capacity_test
    test_flip_queue_capacity.cpp
)
target_include_directories(shadps4_flip_queue_capacity_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_flip_queue_capacity_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_flip_queue_capacity_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_flip_queue_capacity_test)

add_executable(shadps4_vblank_tick_test
    test_vblank_tick.cpp
)
target_include_directories(shadps4_vblank_tick_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_vblank_tick_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_vblank_tick_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_vblank_tick_test)

add_executable(shadps4_sleepq_waiter_selection_test
    test_sleepq_waiter_selection.cpp
)
target_include_directories(shadps4_sleepq_waiter_selection_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_sleepq_waiter_selection_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_sleepq_waiter_selection_test PRIVATE
    GTest::gtest_main
    Boost::headers
)
list(APPEND TEST_TARGETS shadps4_sleepq_waiter_selection_test)

add_executable(shadps4_condvar_wait_state_test
    test_condvar_wait_state.cpp
)
target_include_directories(shadps4_condvar_wait_state_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_condvar_wait_state_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_condvar_wait_state_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_condvar_wait_state_test)

add_executable(shadps4_kernel_semaphore_policy_test
    test_kernel_semaphore_policy.cpp
)
target_include_directories(shadps4_kernel_semaphore_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_kernel_semaphore_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_kernel_semaphore_policy_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_kernel_semaphore_policy_test)

add_executable(shadps4_event_flag_policy_test
    test_event_flag_policy.cpp
)
target_include_directories(shadps4_event_flag_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_event_flag_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_event_flag_policy_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_event_flag_policy_test)

add_executable(shadps4_libc_time_test
    test_libc_time.cpp
)
target_include_directories(shadps4_libc_time_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_libc_time_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_libc_time_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_libc_time_test)

add_executable(shadps4_shared_slot_registry_test
    test_shared_slot_registry.cpp
)
target_include_directories(shadps4_shared_slot_registry_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_shared_slot_registry_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_shared_slot_registry_test PRIVATE
    GTest::gtest_main
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_shared_slot_registry_test)

add_executable(shadps4_fault_range_test
    test_fault_range.cpp
)
target_include_directories(shadps4_fault_range_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_fault_range_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_fault_range_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_fault_range_test)

add_executable(shadps4_fault_buffer_count_test
    test_fault_buffer_count.cpp
)
target_include_directories(shadps4_fault_buffer_count_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_fault_buffer_count_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_fault_buffer_count_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_fault_buffer_count_test)

add_executable(shadps4_buffer_barrier_policy_test
    test_buffer_barrier_policy.cpp
)
target_include_directories(shadps4_buffer_barrier_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_buffer_barrier_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_buffer_barrier_policy_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_buffer_barrier_policy_test)

add_executable(shadps4_vertex_input_binding_test
    test_vertex_input_binding.cpp
)
target_include_directories(shadps4_vertex_input_binding_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_vertex_input_binding_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_vertex_input_binding_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_vertex_input_binding_test)

add_executable(shadps4_device_feature_policy_test
    test_device_features.cpp
)
target_include_directories(shadps4_device_feature_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_device_feature_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_device_feature_policy_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_device_feature_policy_test)

add_executable(shadps4_image_offset_policy_test
    test_image_offset_policy.cpp
)
target_include_directories(shadps4_image_offset_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_image_offset_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_image_offset_policy_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_image_offset_policy_test)

add_executable(shadps4_read_const_access_policy_test
    test_read_const_access_policy.cpp
)
target_include_directories(shadps4_read_const_access_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_read_const_access_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_read_const_access_policy_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_read_const_access_policy_test)

add_executable(shadps4_descriptor_type_test
    test_descriptor_type.cpp
)
target_include_directories(shadps4_descriptor_type_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_descriptor_type_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_descriptor_type_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_descriptor_type_test)

add_executable(shadps4_image_copy_policy_test
    test_image_copy_policy.cpp
)
target_include_directories(shadps4_image_copy_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_image_copy_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_image_copy_policy_test PRIVATE
    GTest::gtest_main
    Vulkan::Headers
)
list(APPEND TEST_TARGETS shadps4_image_copy_policy_test)

add_executable(shadps4_image_download_policy_test
    test_image_download_policy.cpp
)
target_include_directories(shadps4_image_download_policy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_image_download_policy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_image_download_policy_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_image_download_policy_test)

add_executable(shadps4_stream_buffer_copy_test
    test_stream_buffer_copy.cpp
)
target_include_directories(shadps4_stream_buffer_copy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_stream_buffer_copy_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_stream_buffer_copy_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_stream_buffer_copy_test)

add_executable(shadps4_aux_tess_shader_test
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_quad_rect.cpp
    test_aux_tess_shader.cpp
)
target_include_directories(shadps4_aux_tess_shader_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_aux_tess_shader_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_aux_tess_shader_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    sirit
    spdlog::spdlog
)
list(APPEND TEST_TARGETS shadps4_aux_tess_shader_test)

add_executable(shadps4_playgo_test
    test_playgo.cpp
)
target_include_directories(shadps4_playgo_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_playgo_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_playgo_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_playgo_test)

set(USER_MIGRATION_TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/user_migration.cpp
    test_user_migration.cpp
)

add_executable(shadps4_user_migration_test ${USER_MIGRATION_TEST_SOURCES})
target_include_directories(shadps4_user_migration_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_user_migration_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_user_migration_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_user_migration_test)

set(KERNEL_SELECT_TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/libraries/kernel/select_timing.cpp
    test_kernel_select_timing.cpp
)

add_executable(shadps4_kernel_select_test ${KERNEL_SELECT_TEST_SOURCES})
target_include_directories(shadps4_kernel_select_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_kernel_select_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_kernel_select_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_kernel_select_test)

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
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp

    # Tests
    test_emulator_settings.cpp
    test_gpu_compatibility_settings.cpp
)

set(NGS2_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/libraries/ngs2/ngs2_impl.cpp

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
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp

    # Tests
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
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/constant_propagation_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/dead_code_elimination_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/identity_removal_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/shared_memory_barrier_pass.cpp
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
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/io_file.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp
    ${CMAKE_SOURCE_DIR}/src/common/ntapi.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/resource_tracking_pass_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp
    stubs/core_stub.cpp

    gcn/gcn_test_runner.hpp
    gcn/gcn_test_runner.cpp
    gcn/translator.hpp
    gcn/translator.cpp

    # Tests
    gcn/test_gcn_instructions.cpp
    gcn/test_shared_memory_barriers.cpp
)

add_executable(shadps4_settings_test ${SETTINGS_TEST_SOURCES})
add_executable(shadps4_ngs2_test ${NGS2_TEST_SOURCES})
add_executable(shadps4_gcn_test ${GCN_TEST_SOURCES})

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
    fmt::fmt
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3
    spdlog::spdlog
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

# The game regression harness is intentionally tested with a synthetic emulator.
# Private game dumps are never required for the normal unit-test suite.
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
add_test(
    NAME game_harness_tests
    COMMAND
        ${Python3_EXECUTABLE}
        -W error::ResourceWarning
        -m unittest discover
        -s ${CMAKE_CURRENT_SOURCE_DIR}/game_harness
        -p test_*.py
)
set_tests_properties(game_harness_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    TIMEOUT 60
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
