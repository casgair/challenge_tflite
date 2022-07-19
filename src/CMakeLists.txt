# --------------------------------------------------------------------------------------------------
# Variables definition
# --------------------------------------------------------------------------------------------------

set(CHALLENGE_LIBNAME challenge)

file(GLOB_RECURSE CHALLENGE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(KL_TARGET_OS STREQUAL "linux")
  set(TFLITE_PATH ${VENDORS_DIR}/tflite/lib/linux/x86_64/libtensorflowlite.a)
  set(OPENCV_PATH_CORE ${VENDORS_DIR}/opencv2/lib/linux/x86_64/libopencv_core.a)
  set(OPENCV_PATH_HIGHGUI ${VENDORS_DIR}/opencv2/lib/linux/x86_64/libopencv_highgui.a)
  set(OPENCV_PATH_IMGPROC ${VENDORS_DIR}/opencv2/lib/linux/x86_64/libopencv_imgproc.a)
  set(OPENCV_PATH_JPEG ${VENDORS_DIR}/opencv2/lib/linux/x86_64/liblibjpeg.a)
  set(OPENCV_PATH_PNG ${VENDORS_DIR}/opencv2/lib/linux/x86_64/liblibpng.a)
  set(OPENCV_PATH_TIFF ${VENDORS_DIR}/opencv2/lib/linux/x86_64/liblibtiff.a)
  set(OPENCV_PATH_ZLIB ${VENDORS_DIR}/opencv2/lib/linux/x86_64/libzlib.a)
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
      -DOPENCV_PATH_CORE=${OPENCV_PATH_CORE}
      -DOPENCV_PATH_HIGHGUI=${OPENCV_PATH_HIGHGUI}
      -DOPENCV_PATH_IMGPROC=${OPENCV_PATH_IMGPROC}
      -DOPENCV_PATH_JPEG=${OPENCV_PATH_JPEG}
      -DOPENCV_PATH_PNG=${OPENCV_PATH_PNG}
      -DOPENCV_PATH_TIFF=${OPENCV_PATH_TIFF}
      -DOPENCV_PATH_ZLIB=${OPENCV_PATH_ZLIB}
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
