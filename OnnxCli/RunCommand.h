#pragma once

#include "OnnxModel.h"
#include "RunSupport.h"

#include <string>

namespace YirangOnnx
{
	auto run_inference(const OnnxModel& model, const std::string& model_path, const InferenceJob& job) -> int;
} // namespace YirangOnnx
