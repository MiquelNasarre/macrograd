
# Add mini_shakespeare application.
add_executable(mini_shakespeare
    source/main.cpp
)

# Add header files.
target_sources(mini_shakespeare
    PRIVATE
        FILE_SET HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES
            include/transformer.h
            include/mini_shakespeare.h
            include/tokenizer.h
            include/training.h
            include/MNIST.h
)

# Set C++20 standards.
target_compile_features(mini_shakespeare
    PRIVATE
        cxx_std_20
)

# Set compiler warning level.
if(MSVC)
    target_compile_options(mini_shakespeare PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /sdl>
    )
else()
    target_compile_options(mini_shakespeare PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()

# Link to Macrograd library.
target_link_libraries(mini_shakespeare
    PRIVATE
        Macrograd::macrograd
)

# Set build output directories.
set_target_properties(mini_shakespeare PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_CURRENT_SOURCE_DIR}/bin/Release"
)