project(OnnxParserTests VERSION 0.2.0 LANGUAGES CXX)

find_package(GTest CONFIG REQUIRED)

add_executable(onnx_parser_tests
	OnnxModelTests.cpp
	ConfigurationsTests.cpp
	InputProjectTests.cpp
	InputBuilderTests.cpp
	InferenceEngineTests.cpp
	RunSupportTests.cpp
	TensorConvertTests.cpp)
target_link_libraries(onnx_parser_tests PRIVATE OnnxParser OnnxCliCore GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(onnx_parser_tests)

set(CLI_SMOKE_CASES
	help
	version
	no_args
	summary
	job_auto
	job_dump_json
	job_benchmark
	unknown_key
	malformed_script
	script_priority
	outdir_supplement
	missing_model
	cli_infer
	unknown_flag
	script_relative_paths)

foreach(SMOKE_CASE IN LISTS CLI_SMOKE_CASES)
	add_test(NAME cli_smoke_${SMOKE_CASE}
		COMMAND ${CMAKE_COMMAND}
			-DCASE=${SMOKE_CASE}
			-DBIN=$<TARGET_FILE:OnnxCli>
			-DSRC=${CMAKE_SOURCE_DIR}/examples
			-DWORK=${CMAKE_BINARY_DIR}/cli-smoke
			-P ${CMAKE_CURRENT_SOURCE_DIR}/cli_smoke.cmake)
endforeach()
