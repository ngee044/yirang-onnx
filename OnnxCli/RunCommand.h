#pragma once

#include <string>
#include <vector>

namespace YirangOnnx
{
	// Reads input tensors (ONNX TensorProto .pb), runs the model via the inference
	// engine, writes each output as <output_dir>/<name>.pb, prints a summary.
	// Returns a process exit code (0 success).
	auto run_inference(const std::string& model_path, const std::vector<std::string>& input_paths, const std::string& output_dir) -> int;
}
