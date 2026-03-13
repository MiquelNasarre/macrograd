add_library(macrograd STATIC
    source/tensor.cpp
    source/tensor_ops.cpp
    source/nn.cpp
    kernels/cuda_backend.cu
)

target_include_directories(macrograd
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)