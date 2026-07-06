project(OnnxParserTests VERSION 0.1.0 LANGUAGES CXX)

find_package(GTest CONFIG REQUIRED)

add_executable(onnx_parser_tests
	OnnxModelTests.cpp
	ConfigurationsTests.cpp
	InputProjectTests.cpp
	InputBuilderTests.cpp
	InferenceEngineTests.cpp
	TensorConvertTests.cpp)
target_link_libraries(onnx_parser_tests PRIVATE OnnxParser OnnxCliCore GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(onnx_parser_tests)
