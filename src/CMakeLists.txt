# --------------------------------------------------------------------------------------------------
# Variables definition
# --------------------------------------------------------------------------------------------------

set(CHALLENGE_LIBNAME challenge)

file(GLOB_RECURSE CHALLENGE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(KL_TARGET_OS STREQUAL "linux")
  set(TFLITE_PATH ${VENDORS_DIR}/tflite/lib/linux/x86_64/libtensorflowlite.a)
elseif(KL_TARGET_OS STREQUAL "macos")
  set(TFLITE_PATH ${VENDORS_DIR}/tflite/lib/macos/x86_64/libtensorflowlite.a)
endif()

# --------------------------------------------------------------------------------------------------
# Set dependencies
# --------------------------------------------------------------------------------------------------

if(KL_TARGET_OS STREQUAL "macos")
  add_library(${CHALLENGE_LIBNAME}
    STATIC ${CHALLENGE_SRCS}
  )
else()
  add_library(${CHALLENGE_LIBNAME}
    STATIC ${CHALLENGE_SRCS}
  )
endif()

# --------------------------------------------------------------------------------------------------
# Build rules
# --------------------------------------------------------------------------------------------------

# Some files are generated in the build folder to not pollute the repo
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(KL_TARGET_OS STREQUAL "linux")
  add_custom_command(TARGET ${CHALLENGE_LIBNAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND}
      -DTFLITE_PATH=${TFLITE_PATH}
      -P ${MAKE_DIR}/generate_ar_input_file.cmake
    COMMAND ar -M < ${CMAKE_CURRENT_BINARY_DIR}/ar_input.txt
    COMMAND ranlib libchallenge.a.tmp
    COMMAND mv libchallenge.a.tmp libchallenge.a
    # COMMAND ${CMAKE_STRIP} --strip-unneeded libchallenge.a
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Joining libs into one"
  )

elseif(KL_TARGET_OS STREQUAL "macos")
  add_custom_command(TARGET ${CHALLENGE_LIBNAME} POST_BUILD
    COMMAND libtool -static -o libchallenge.a
      libchallenge.a
      ${TFLITE_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Joining libs into one"
  )
endif()
