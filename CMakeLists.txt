cmake_minimum_required(VERSION 3.10.0)

# --------------------------------------------------------------------------------------------------
# Get build type
# --------------------------------------------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "CMAKE_BUILD_TYPE error. Only 'Debug' and 'Release' build type are supported.")
endif()

# --------------------------------------------------------------------------------------------------
# Get target os and arch
# --------------------------------------------------------------------------------------------------

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
  set(KL_TARGET_OS "linux")
elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
  set(KL_TARGET_OS "macos")
endif()

# --------------------------------------------------------------------------------------------------
# Sanitizers
# --------------------------------------------------------------------------------------------------

# Enable ASAN by default using the Debug build
set(KL_ENABLE_ADDRESS_SANITIZER OFF CACHE BOOL "Enable address sanitizer" FORCE)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(KL_ENABLE_ADDRESS_SANITIZER ON CACHE BOOL "Enable address sanitizer" FORCE)
endif()

# --------------------------------------------------------------------------------------------------
# Print configuration
# --------------------------------------------------------------------------------------------------

message("")
message("Summary")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "KL_TARGET_OS: ${KL_TARGET_OS}")
message(STATUS "KL_ENABLE_ADDRESS_SANITIZER: ${KL_ENABLE_ADDRESS_SANITIZER}")
message("")

# --------------------------------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------------------------------

project(kl_challenge)

# --------------------------------------------------------------------------------------------------
# CMake configuration
# --------------------------------------------------------------------------------------------------

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# --------------------------------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------------------------------

find_package(Threads REQUIRED)

# --------------------------------------------------------------------------------------------------
# Variables definition
# --------------------------------------------------------------------------------------------------

set(MAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(SRC_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)
set(VENDORS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendors)

# --------------------------------------------------------------------------------------------------
# Compile and link options
# --------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)

add_compile_options("$<$<CONFIG:RELEASE>:-W;-Wall;-Wextra;-pedantic;-Werror;-Wno-unused-parameter>")
# ftrapv checks for overflow and crashes the program if it happens.
add_compile_options("$<$<CONFIG:DEBUG>:-W;-Wall;-Wextra;-pedantic;-Werror;-Wno-unused-parameter;-ftrapv>")

if(KL_ENABLE_ADDRESS_SANITIZER)
  set(KL_ASAN_OPTIONS "halt_on_error=1:strict_string_checks=1:check_initialization_order=1:strict_init_order=1")
  set(KL_SANITIZER_FLAGS "-fsanitize=undefined,address")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KL_SANITIZER_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${KL_SANITIZER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${KL_SANITIZER_FLAGS}")

# --------------------------------------------------------------------------------------------------

# Project includes
include_directories(src)

# 3rd_parties includes, SYSTEM makes the compiler ignore warnings
include_directories(SYSTEM
  ${VENDORS_DIR}
  ${VENDORS_DIR}/tflite/include
)

add_subdirectory(${SRC_SDK_DIR})

enable_testing()
add_subdirectory(${TEST_DIR})
