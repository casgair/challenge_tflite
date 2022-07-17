# --------------------------------------------------------------------------------------------------
# Variables definition
# --------------------------------------------------------------------------------------------------

set(TEST_BINARY_NAME test_main)

# --------------------------------------------------------------------------------------------------
# Set dependencies
# --------------------------------------------------------------------------------------------------

add_executable(${TEST_BINARY_NAME} test_main.cpp)

add_dependencies(${TEST_BINARY_NAME} challenge)

# --------------------------------------------------------------------------------------------------
# Build rules
# --------------------------------------------------------------------------------------------------

include_directories(
  ${TEST_DIR}
)

if(KL_TARGET_OS STREQUAL "linux")
  target_link_libraries(${TEST_BINARY_NAME} PRIVATE
    challenge
    pthread
    ${CMAKE_DL_LIBS}
  )
elseif(KL_TARGET_OS STREQUAL "macos")
  target_link_libraries(${TEST_BINARY_NAME} PRIVATE
    challenge
  )
endif()

add_test(NAME ${TEST_BINARY_NAME} COMMAND ${TEST_BINARY_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

if(KL_ENABLE_ADDRESS_SANITIZER)
  set_property(TEST ${TEST_BINARY_NAME} PROPERTY ENVIRONMENT
    ASAN_OPTIONS=${KL_ASAN_OPTIONS}
  )
endif()
