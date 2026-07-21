#include "RunCommand.h"

#include "InferenceEngine.h"
#include "Logger.h"
#include "Tensor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <string>
#include <utility>
#include <vector>

using namespace Utilities;

namespace YirangOnnx
{
	auto run_inference(const OnnxModel& model, const std::string& model_path, const InferenceJob& job) -> int
	{
		auto [inputs, resolve_error] = resolve_job_inputs(model, job);
		if (!inputs.has_value())
		{
			Logger::handle().write(LogTypes::Error, resolve_error.value_or("cannot resolve job inputs"));
			return 1;
		}

		Logger::handle().write(LogTypes::Information, tuning_summary(job.tuning_));

		InferenceEngine engine;
		if (auto loaded = engine.load(model_path, job.tuning_); !loaded)
		{
			Logger::handle().write(LogTypes::Error, loaded.error());
			return 1;
		}

		for (uint32_t i = 0; i < job.run_.warmup_; ++i)
		{
			auto [warmup_outputs, run_error] = engine.run(inputs.value());
			if (!warmup_outputs.has_value())
			{
				Logger::handle().write(LogTypes::Error, run_error.value_or("inference failed"));
				return 1;
			}
		}

		const uint32_t repeat = std::max<uint32_t>(1, job.run_.repeat_);
		std::vector<Tensor> outputs;
		std::vector<double> durations_ms;
		durations_ms.reserve(repeat);
		for (uint32_t i = 0; i < repeat; ++i)
		{
			const auto begin = std::chrono::steady_clock::now();
			auto [result, run_error] = engine.run(inputs.value());
			const auto end = std::chrono::steady_clock::now();
			if (!result.has_value())
			{
				Logger::handle().write(LogTypes::Error, run_error.value_or("inference failed"));
				return 1;
			}
			durations_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
			outputs = std::move(result.value());
		}

		if (repeat == 1)
		{
			Logger::handle().write(LogTypes::Information, std::format("inference took {:.2f} ms", durations_ms.front()));
		}
		else
		{
			double minimum = durations_ms.front();
			double maximum = durations_ms.front();
			double sum = 0.0;
			for (double value : durations_ms)
			{
				minimum = std::min(minimum, value);
				maximum = std::max(maximum, value);
				sum += value;
			}
			Logger::handle().write(LogTypes::Information, std::format("inference: {} runs (+{} warmup), avg {:.2f} ms, min {:.2f} ms, max {:.2f} ms", repeat,
																	  job.run_.warmup_, sum / static_cast<double>(durations_ms.size()), minimum, maximum));
		}

		return process_outputs(outputs, job.outputs_);
	}
} // namespace YirangOnnx
