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
    target_compile_options(gtest PRIVATE /clang:-Wno-character-conversion)
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
target_link_libraries(shadps4_renderdoc_capture_state_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_renderdoc_capture_state_test)

add_executable(shadps4_ipc_capabilities_test
    test_ipc_capabilities.cpp
)
target_include_directories(shadps4_ipc_capabilities_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_ipc_capabilities_test PRIVATE cxx_std_23)
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

add_executable(shadps4_pipeline_bind_tracker_test
    test_pipeline_bind_tracker.cpp
)
target_include_directories(shadps4_pipeline_bind_tracker_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_pipeline_bind_tracker_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_pipeline_bind_tracker_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_pipeline_bind_tracker_test)

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

add_executable(shadps4_event_write_eop_test
    test_event_write_eop.cpp
)
target_include_directories(shadps4_event_write_eop_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_event_write_eop_test PRIVATE cxx_std_23)
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
target_link_libraries(shadps4_gpu_mapping_admission_test PRIVATE
    GTest::gtest_main
    Boost::headers
)
list(APPEND TEST_TARGETS shadps4_gpu_mapping_admission_test)

add_executable(shadps4_buffer_residency_test
    test_buffer_residency.cpp
)
target_include_directories(shadps4_buffer_residency_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_buffer_residency_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_buffer_residency_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_buffer_residency_test)

add_executable(shadps4_buffer_access_range_test
    test_buffer_access_range.cpp
)
target_include_directories(shadps4_buffer_access_range_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_buffer_access_range_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_buffer_access_range_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_buffer_access_range_test)

add_executable(shadps4_external_host_memory_probe_test
    test_external_host_memory_probe.cpp
)
target_include_directories(shadps4_external_host_memory_probe_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(shadps4_external_host_memory_probe_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_external_host_memory_probe_test PRIVATE GTest::gtest_main)
list(APPEND TEST_TARGETS shadps4_external_host_memory_probe_test)

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
    target_link_libraries(shadps4_windows_exception_policy_test PRIVATE GTest::gtest_main)
    list(APPEND TEST_TARGETS shadps4_windows_exception_policy_test)
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
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp

    # Tests
    test_emulator_settings.cpp
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
    ${CMAKE_SOURCE_DIR}/src/core/signals.cpp
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
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/buffer_access_range_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/constant_propagation_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/dead_code_elimination_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/flatten_extended_userdata_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/identity_removal_pass.cpp
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

    gcn/gcn_test_runner.hpp
    gcn/gcn_test_runner.cpp
    gcn/translator.hpp
    gcn/translator.cpp

    # Tests
    gcn/test_buffer_access_range_pass.cpp
    gcn/test_gcn_instructions.cpp
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
    xbyak::xbyak
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
