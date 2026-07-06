cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(yirang-onnx VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(APPLE)
	list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
endif()

# Utilities only: the toolkit root force-builds ThreadPool, which fails to compile
# with USE_ENCRYPT_LIBS=OFF (Job.cpp calls the encrypt-only Converter::to_base64).
set(USE_ENCRYPT_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(.CppToolkit/Utilities)

option(YIRANG_ONNX_BUILD_TESTS "Build yirang-onnx tests" ON)

add_subdirectory(OnnxParser)
add_subdirectory(OnnxInference)
add_subdirectory(OnnxCli)

if(YIRANG_ONNX_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()
