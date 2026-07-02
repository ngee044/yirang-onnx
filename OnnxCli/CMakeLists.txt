set(EXECUTABLE_NAME OnnxCli)

project(${EXECUTABLE_NAME} VERSION 0.1.0 LANGUAGES CXX)

set(HEADER_FILES
	Configurations.h)

set(SOURCE_FILES
	Configurations.cpp
	main.cpp)

add_executable(${EXECUTABLE_NAME} ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${EXECUTABLE_NAME} PROPERTIES OUTPUT_NAME yirang-onnx)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE OnnxParser Utilities)
