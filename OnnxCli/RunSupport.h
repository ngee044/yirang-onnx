#pragma once

#include "InputProject.h"
#include "ModelTypes.h"
#include "OnnxModel.h"
#include "SessionTuning.h"
#include "Tensor.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace YirangOnnx
{
	struct InferenceJob
	{
		std::vector<InputSpec> inputs_;
		std::map<std::string, int64_t> dim_overrides_;
		RunSpec run_;
		OutputSpec outputs_;
		SessionTuning tuning_;
	};

	auto shape_string(const std::vector<int64_t>& shape) -> std::string;
	auto tuning_summary(const SessionTuning& tuning) -> std::string;
	auto safe_file_name(const std::string& name) -> std::string;
	auto write_file_bytes(const std::string& path, const uint8_t* data, size_t size) -> std::optional<std::string>;
	auto load_tensor_file(const std::string& path) -> std::tuple<std::optional<Tensor>, std::optional<std::string>>;
	auto build_random_input(const ValueInfo& graph_input, const InputSpec& spec, const std::map<std::string, int64_t>& dim_overrides)
		-> std::tuple<std::optional<Tensor>, std::optional<std::string>>;
	auto tensor_values(const Tensor& tensor) -> std::optional<std::vector<double>>;
	auto tensor_stats_line(const Tensor& tensor) -> std::optional<std::string>;
	auto tensor_json_dump(const Tensor& tensor) -> std::optional<std::string>;
	auto resolve_job_inputs(const OnnxModel& model, const InferenceJob& job) -> std::tuple<std::optional<std::vector<Tensor>>, std::optional<std::string>>;
	auto process_outputs(const std::vector<Tensor>& outputs, const OutputSpec& spec) -> int;
} // namespace YirangOnnx
