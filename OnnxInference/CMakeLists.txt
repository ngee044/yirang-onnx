set(LIBRARY_NAME OnnxInference)

project(${LIBRARY_NAME} VERSION 0.2.0 LANGUAGES CXX)

# Prebuilt ONNX Runtime (brew install onnxruntime). No CMake config is shipped,
# so locate headers + lib directly. Override with -DONNXRUNTIME_ROOT=...
set(ONNXRUNTIME_ROOT "/opt/homebrew/opt/onnxruntime" CACHE PATH "ONNX Runtime install prefix")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS "${ONNXRUNTIME_ROOT}/include/onnxruntime" "${ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIB NAMES onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib")

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
	message(FATAL_ERROR "ONNX Runtime not found under ${ONNXRUNTIME_ROOT}. Install it (brew install onnxruntime) or pass -DONNXRUNTIME_ROOT=<prefix>.")
endif()

set(HEADER_FILES
	Tensor.h
	SessionTuning.h
	InferenceEngine.h)

set(SOURCE_FILES
	InferenceEngine.cpp)

add_library(${LIBRARY_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# onnxruntime headers are an implementation detail (PRIVATE); the .dylib must be
# linked and found at runtime by consumers, so link + rpath are PUBLIC.
target_include_directories(${LIBRARY_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${ONNXRUNTIME_LIB})
set_target_properties(${LIBRARY_NAME} PROPERTIES INTERFACE_LINK_OPTIONS "-Wl,-rpath,${ONNXRUNTIME_ROOT}/lib")
