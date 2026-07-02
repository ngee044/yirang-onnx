set(LIBRARY_NAME OnnxParser)

project(${LIBRARY_NAME} VERSION 0.1.0 LANGUAGES CXX)

find_package(protobuf CONFIG REQUIRED)

set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

add_library(onnx_proto OBJECT "${CMAKE_SOURCE_DIR}/proto/onnx.proto")
target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)
target_include_directories(onnx_proto PUBLIC "$<BUILD_INTERFACE:${PROTO_GEN_DIR}>")

protobuf_generate(
	TARGET onnx_proto
	LANGUAGE cpp
	IMPORT_DIRS "${CMAKE_SOURCE_DIR}/proto"
	PROTOC_OUT_DIR "${PROTO_GEN_DIR}")

set(HEADER_FILES
	ModelTypes.h
	OnnxModel.h)

set(SOURCE_FILES
	OnnxModel.cpp)

add_library(${LIBRARY_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${LIBRARY_NAME} PUBLIC onnx_proto protobuf::libprotobuf Utilities)
