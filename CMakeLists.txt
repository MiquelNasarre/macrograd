
# Version requirements for up to date CUDA CMake.
cmake_minimum_required(VERSION 3.25)

cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0141 NEW)

# Good debug symbols for MSVC ABI toolchains, including Ninja + MSVC
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
    "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>"
)

project(Macrograd
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Add library and demo projects.
add_subdirectory(macrograd)
add_subdirectory(mini_shakespeare)