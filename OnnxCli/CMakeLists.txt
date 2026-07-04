set(EXECUTABLE_NAME OnnxCli)

project(${EXECUTABLE_NAME} VERSION 0.1.0 LANGUAGES CXX)

set(CORE_LIBRARY_NAME OnnxCliCore)

set(CORE_HEADER_FILES
	Configurations.h
	TensorConvert.h)

set(CORE_SOURCE_FILES
	Configurations.cpp
	TensorConvert.cpp)

add_library(${CORE_LIBRARY_NAME} STATIC ${CORE_HEADER_FILES} ${CORE_SOURCE_FILES})
target_include_directories(${CORE_LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${CORE_LIBRARY_NAME} PUBLIC OnnxParser OnnxInference Utilities)

set(HEADER_FILES
	RunCommand.h)

set(SOURCE_FILES
	RunCommand.cpp
	main.cpp)

add_executable(${EXECUTABLE_NAME} ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${EXECUTABLE_NAME} PROPERTIES OUTPUT_NAME yirang-onnx)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${CORE_LIBRARY_NAME})

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
