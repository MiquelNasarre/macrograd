
# Add library with source files.
add_library(macrograd STATIC
    source/nn.cpp
    source/tensor.cpp
    source/tensor_ops.cpp
    kernels/cuda_backend.cu
)

# Add header files.
target_sources(macrograd
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES
            include/macrograd.h
            include/macrograd_nn.h
            include/macrograd_error.h
            include/cuda_backend.h
)

# Add it to the view.
add_library(Macrograd::macrograd ALIAS macrograd)

# Set C++20 Standard.
target_compile_features(macrograd
    PUBLIC
        cxx_std_20
)

# Warning level roughly matching VS Level3.
if(MSVC)
    target_compile_options(macrograd PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
    )
else()
    target_compile_options(macrograd PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()

# CUDA-specific settings.
set_target_properties(macrograd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    DEBUG_POSTFIX "_d"
)

# nvcc extra flag to compile 52-compute CUDA.
target_compile_options(macrograd PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# CUDA include/lib wiring.
find_package(CUDAToolkit REQUIRED)

# Link CUDA dependencies.
target_link_libraries(macrograd
    PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cublasLt
)

# Match the old output layout:
set_target_properties(macrograd PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_CURRENT_SOURCE_DIR}/lib/Debug"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_CURRENT_SOURCE_DIR}/lib/Release"
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
)
