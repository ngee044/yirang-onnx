#pragma once

#include "Tensor.h"

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace YirangOnnx
{
	// Runs ONNX models via ONNX Runtime. Kept free of onnxruntime headers so
	// consumers only depend on Tensor; the backend lives entirely in the .cpp.
	class InferenceEngine
	{
	public:
		InferenceEngine(void) = default;

		auto run(const std::string& model_path, const std::vector<Tensor>& inputs) const -> std::tuple<std::optional<std::vector<Tensor>>, std::optional<std::string>>;
	};
}
