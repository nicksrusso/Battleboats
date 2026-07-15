# Minimum CMake version. 3.16 is a safe, widely-available floor that supports
# everything below. CMake refuses to configure with an older binary.
cmake_minimum_required(VERSION 3.16)

# Declares the project: names it, versions it, and tells CMake this is C++
# (CXX). LANGUAGES is worth setting explicitly — otherwise CMake also probes
# for a C compiler you don't need.
project(battleboats_engine
        VERSION 0.1.0
        DESCRIPTION "C++ port of the Battleboats game engine"
        LANGUAGES CXX)

# --- language standard -------------------------------------------------------
# C++17 gives us std::variant / std::visit (the Action dispatch), std::optional
# (the nullable winner), and structured bindings. REQUIRED makes it a hard
# error if the compiler can't provide it, rather than silently falling back.
# EXTENSIONS OFF selects -std=c++17 instead of -std=gnu++17 (no GNU-only
# extensions), so the code stays portable.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json in the build dir. This is what clangd / the VS
# Code C++ extension read to give you accurate IntelliSense, jump-to-def, and
# error squiggles. Without it your editor guesses at include paths.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- the engine library ------------------------------------------------------
# Compile the implementation files into a single static library called
# "engine". The tests (and later, the pybind11 module) link against this
# instead of recompiling the sources. Header-only types (Ship, Port, enums,
# actions...) are NOT listed here — they have no .cpp to compile.
add_library(engine
    src/ship_data.cpp
    src/map.cpp
    src/game_engine.cpp)

# Anyone who links "engine" (PUBLIC) also inherits this include path, so they
# can write #include "battleboats/map/map.h". The library itself needs it too,
# which is exactly what PUBLIC means: "for me and for my consumers."
target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Turn on a strong warning set — but only for GCC/Clang, via a generator
# expression so this file stays valid under MSVC too. PRIVATE: these flags
# apply when building the engine, but don't leak onto its consumers.
target_compile_options(engine PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)

# --- tests -------------------------------------------------------------------
# Register the project with CTest, then pull in the tests/ subdirectory, which
# defines the actual test executables. Run them later with `ctest`.
enable_testing()
add_subdirectory(tests)
