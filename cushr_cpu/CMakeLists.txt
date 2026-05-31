cmake_minimum_required(VERSION 3.16)
project(cushr_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# Dependencies (fetched at configure time so the lab machine needs no extra
# system packages besides zlib, which is required by cnpy).
# ---------------------------------------------------------------------------
include(FetchContent)

find_package(ZLIB REQUIRED)

FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(cnpy)

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        v2.4.11
)
FetchContent_MakeAvailable(doctest)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(cushr
  src/lattice.cpp
  src/scorer.cpp
  src/decoder.cpp
  src/metrics.cpp
  src/json_io.cpp
)
target_include_directories(cushr PUBLIC include)
target_link_libraries(cushr PUBLIC cnpy ZLIB::ZLIB)

# ---------------------------------------------------------------------------
# Evaluation harness binary
# ---------------------------------------------------------------------------
add_executable(cushr_evaluate apps/evaluate.cpp)
target_link_libraries(cushr_evaluate PRIVATE cushr)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_executable(cushr_tests
  tests/test_main.cpp
  tests/test_lattice.cpp
  tests/test_decoder.cpp
  tests/test_scorer.cpp
)
target_link_libraries(cushr_tests PRIVATE cushr doctest::doctest)
add_test(NAME cushr_tests COMMAND cushr_tests)
