#pragma once

#include "Tensor.h"

#include <expected>
#include <memory>
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
		InferenceEngine(void);
		~InferenceEngine(void);

		InferenceEngine(InferenceEngine&&) noexcept;
		auto operator=(InferenceEngine&&) noexcept -> InferenceEngine&;

		InferenceEngine(const InferenceEngine&) = delete;
		auto operator=(const InferenceEngine&) -> InferenceEngine& = delete;

		auto load(const std::string& model_path) -> std::expected<void, std::string>;
		auto loaded(void) const -> bool;

		auto run(const std::vector<Tensor>& inputs) const -> std::tuple<std::optional<std::vector<Tensor>>, std::optional<std::string>>;

	private:
		struct Backend;
		std::unique_ptr<Backend> backend_;
	};
} // namespace YirangOnnx
