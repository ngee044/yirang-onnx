set(EXECUTABLE_NAME OnnxCli)

project(${EXECUTABLE_NAME} VERSION 0.1.0 LANGUAGES CXX)

set(HEADER_FILES
	RunCommand.h
	Configurations.h)

set(SOURCE_FILES
	RunCommand.cpp
	Configurations.cpp
	main.cpp)

add_executable(${EXECUTABLE_NAME} ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${EXECUTABLE_NAME} PROPERTIES OUTPUT_NAME yirang-onnx)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE OnnxParser Utilities OnnxInference)

set(JSON_FILES
	yirang_onnx_configurations.json
)

foreach(JSON_FILE IN LISTS JSON_FILES)
	add_custom_command(
		TARGET OnnxCli POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${CMAKE_CURRENT_SOURCE_DIR}/${JSON_FILE}
			${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${JSON_FILE}
	)
endforeach()