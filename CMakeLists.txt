cmake_minimum_required(VERSION 3.16)

project(sana_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Vendored for the native Gemma-2 text encoder
include(FetchContent)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG b9966
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama_cpp)

add_library(sana_core
    src/tensor.cpp
    src/scheduler.cpp
    src/weights_io.cpp
    src/common_ops.cpp
    src/transformer_ops.cpp
    src/vae_ops.cpp
    src/transformer.cpp
    src/vae.cpp
    src/torch_rng.cpp
)

target_include_directories(sana_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(sana_core PRIVATE ggml-base)

if(APPLE)
    target_compile_options(sana_core PUBLIC -O3 -mcpu=native)
    target_link_libraries(sana_core PRIVATE "-framework Accelerate")
else()
    target_compile_options(sana_core PUBLIC -O3 -march=native)
endif()

enable_testing()

add_executable(test_tensor_unit
    tests/test_tensor.cpp
)

target_link_libraries(test_tensor_unit PRIVATE sana_core)
target_compile_options(test_tensor_unit PRIVATE -UNDEBUG)

add_test(NAME tensor_unit COMMAND test_tensor_unit)

add_executable(test_transformer
    tests/test_transformer.cpp
    tests/test_transformer_main.cpp
)

target_link_libraries(test_transformer PRIVATE sana_core)
target_compile_options(test_transformer PRIVATE -UNDEBUG)

add_test(NAME transformer COMMAND test_transformer)

add_executable(test_scheduler
    tests/test_scheduler_main.cpp
)

target_link_libraries(test_scheduler PRIVATE sana_core)
target_compile_options(test_scheduler PRIVATE -UNDEBUG)

add_test(NAME scheduler COMMAND test_scheduler)

add_library(sana_gemma_encoder
    src/gemma_encoder.cpp
)

target_include_directories(sana_gemma_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(sana_gemma_encoder PUBLIC sana_core llama)

add_executable(test_gemma_encoder
    tests/test_gemma_encoder.cpp
)

target_link_libraries(test_gemma_encoder PRIVATE sana_gemma_encoder)
target_compile_options(test_gemma_encoder PRIVATE -UNDEBUG)

add_test(NAME gemma_encoder COMMAND test_gemma_encoder)

add_executable(bench_full_pipeline
    tests/bench_full_pipeline.cpp
)

target_link_libraries(bench_full_pipeline PRIVATE sana_gemma_encoder)

add_executable(sana_infer
    src/infer_main.cpp
)

target_link_libraries(sana_infer PRIVATE sana_gemma_encoder)

if(APPLE)
    target_link_libraries(sana_infer PRIVATE
        "-framework CoreFoundation"
        "-framework CoreGraphics"
        "-framework ImageIO"
    )
endif()
