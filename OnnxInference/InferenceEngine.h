#pragma once

#include "SessionTuning.h"
#include "Tensor.h"

#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace YirangOnnx
{
	class InferenceEngine
	{
	public:
		InferenceEngine(void);
		~InferenceEngine(void);

		InferenceEngine(InferenceEngine&&) noexcept;
		auto operator=(InferenceEngine&&) noexcept -> InferenceEngine&;

		InferenceEngine(const InferenceEngine&) = delete;
		auto operator=(const InferenceEngine&) -> InferenceEngine& = delete;

		auto load(const std::string& model_path, const SessionTuning& tuning = {}) -> std::expected<void, std::string>;
		auto loaded(void) const -> bool;

		auto run(const std::vector<Tensor>& inputs) const -> std::tuple<std::optional<std::vector<Tensor>>, std::optional<std::string>>;

	private:
		struct Backend;
		std::unique_ptr<Backend> backend_;
	};
} // namespace YirangOnnx
