cmake_minimum_required(VERSION 3.10)
project(Concurrent CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release with -O2 by default: benchmark numbers are meaningless on a debug build.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Needed once parallel.cpp uses std::thread (links pthread on Linux).
find_package(Threads REQUIRED)

# Shared core compiled once, reused by every target.
add_library(core
    src/model.cpp
    src/data.cpp
    src/serial.cpp
    src/parallel.cpp
)
target_include_directories(core PUBLIC src)
target_link_libraries(core PUBLIC Threads::Threads)

# Unit test for the per-observation math.
add_executable(check_model tests/check_model.cpp)
target_link_libraries(check_model PRIVATE core)

# Main program (uncomment when src/main.cpp exists):
add_executable(mle src/main.cpp)
target_link_libraries(mle PRIVATE core)
