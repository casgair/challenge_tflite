file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ar_input.txt "
  CREATE libchallenge.a.tmp
  ADDLIB ${TFLITE_PATH}
  ADDLIB libchallenge.a
  SAVE
  END
")
