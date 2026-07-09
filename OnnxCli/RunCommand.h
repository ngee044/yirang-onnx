#pragma once

#include "InferenceEngine.h"
#include "InputProject.h"
#include "OnnxModel.h"

#include <cstdint>
#include <map>
#include <string>
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

	auto run_inference(const OnnxModel& model, const std::string& model_path, const InferenceJob& job) -> int;
} // namespace YirangOnnx
