include_directories(.)

add_executable(
  SDT

  SDT.cpp
)

target_link_libraries(
  SDT

  snark
)

target_include_directories(
  SDT

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)

add_executable(
  gradient_descent

  gradient_descent.cpp
)

target_link_libraries(
  gradient_descent

  snark
)

target_include_directories(
  gradient_descent

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)


add_executable(
  teacher_model

  teacher_model_checkpoint.cpp
)

target_link_libraries(
  teacher_model

  snark
)

target_include_directories(
  teacher_model

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)

add_executable(
  teacher_model_whole_1

  teacher_model_whole_1.cpp
)

target_link_libraries(
  teacher_model_whole_1

  snark
)

target_include_directories(
  teacher_model_whole_1

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)

add_executable(
  teacher_model_whole_2

  teacher_model_whole_2.cpp
)

target_link_libraries(
  teacher_model_whole_2

  snark
)

target_include_directories(
  teacher_model_whole_2

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)

add_executable(
  teacher_model_whole_3

  teacher_model_whole_3.cpp
)

target_link_libraries(
  teacher_model_whole_3

  snark
)

target_include_directories(
  teacher_model_whole_3

  PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
)