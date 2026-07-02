#pragma once

#include "ModelTypes.h"

#include "onnx.pb.h"

#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace YirangOnnx
{
	class OnnxModel
	{
	public:
		OnnxModel(void) = default;

		static auto load(const std::string& path) -> std::tuple<std::optional<OnnxModel>, std::optional<std::string>>;
		static auto parse(const std::vector<uint8_t>& bytes) -> std::tuple<std::optional<OnnxModel>, std::optional<std::string>>;

		auto metadata(void) const -> ModelMetadata;
		auto inputs(void) const -> std::vector<ValueInfo>;
		auto outputs(void) const -> std::vector<ValueInfo>;
		auto nodes(void) const -> std::vector<NodeInfo>;
		auto initializers(void) const -> std::vector<TensorInfo>;
		auto operator_histogram(void) const -> std::vector<std::pair<std::string, size_t>>;

		auto proto(void) const -> const onnx::ModelProto&;

		auto to_json(void) const -> std::string;
		auto to_dot(void) const -> std::string;
		auto to_summary(void) const -> std::string;

	private:
		onnx::ModelProto model_;
	};
}
