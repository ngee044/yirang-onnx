#include "RunCommand.h"

#include "File.h"
#include "InferenceEngine.h"
#include "InputBuilder.h"
#include "JsonTool.h"
#include "Logger.h"
#include "ModelTypes.h"
#include "Tensor.h"
#include "TensorConvert.h"

#include "onnx.pb.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/json.hpp>

using namespace Utilities;

namespace YirangOnnx
{
	namespace
	{
		auto shape_string(const std::vector<int64_t>& shape) -> std::string
		{
			std::string joined;
			for (size_t i = 0; i < shape.size(); ++i)
			{
				joined += std::format("{}{}", (i == 0 ? "" : ", "), shape[i]);
			}
			return std::format("[{}]", joined);
		}

		auto safe_file_name(const std::string& name) -> std::string
		{
			std::string safe = name;
			for (char& ch : safe)
			{
				if (ch == '/' || ch == '\\')
				{
					ch = '_';
				}
			}
			return safe;
		}

		auto write_file_bytes(const std::string& path, const uint8_t* data, size_t size) -> std::optional<std::string>
		{
			File out;
			if (auto opened = out.open(path, std::ios::out | std::ios::binary | std::ios::trunc); !opened)
			{
				return opened.error();
			}
			if (auto written = out.write_bytes(data, size); !written)
			{
				return written.error();
			}
			return std::nullopt;
		}

		auto load_tensor_file(const std::string& path) -> std::tuple<std::optional<Tensor>, std::optional<std::string>>
		{
			File file;
			if (auto opened = file.open(path, std::ios::in | std::ios::binary); !opened)
			{
				return { std::nullopt, std::format("cannot open input tensor '{}': {}", path, opened.error()) };
			}

			auto bytes = file.read_bytes();
			if (!bytes)
			{
				return { std::nullopt, std::format("cannot read input tensor '{}': {}", path, bytes.error()) };
			}

			onnx::TensorProto proto;
			if (!proto.ParseFromArray(bytes.value().data(), static_cast<int>(bytes.value().size())))
			{
				return { std::nullopt, std::format("'{}' is not a valid TensorProto", path) };
			}
			if (proto.name().empty())
			{
				return { std::nullopt, std::format("input tensor '{}' has no name (must match a graph input)", path) };
			}

			auto [tensor, convert_error] = tensor_from_proto(proto);
			if (!tensor.has_value())
			{
				return { std::nullopt, std::format("input tensor '{}': {}", path, convert_error.value_or("conversion failed")) };
			}
			return { std::move(tensor.value()), std::nullopt };
		}

		auto build_random_input(const ValueInfo& graph_input, const InputSpec& spec, const std::map<std::string, int64_t>& dim_overrides)
			-> std::tuple<std::optional<Tensor>, std::optional<std::string>>
		{
			const RandomInputSpec random = spec.random_.value_or(RandomInputSpec{});

			int32_t elem_type = graph_input.elem_type_;
			if (!random.data_type_.empty())
			{
				auto id = data_type_id(random.data_type_);
				if (!id.has_value())
				{
					return { std::nullopt, std::format("input '{}': unknown data_type '{}'", spec.name_, random.data_type_) };
				}
				elem_type = id.value();
			}
			if (elem_type == 0)
			{
				return { std::nullopt, std::format("input '{}' has no tensor type; set random.data_type", spec.name_) };
			}

			std::vector<int64_t> shape = random.shape_;
			if (shape.empty())
			{
				auto [resolved, resolve_error] = resolve_input_shape(graph_input, dim_overrides);
				if (!resolved.has_value())
				{
					return { std::nullopt, resolve_error };
				}
				for (const auto& note : resolved->notes_)
				{
					Logger::handle().write(LogTypes::Information, note);
				}
				shape = resolved->dims_;
			}

			auto [tensor, error] = make_random_tensor(spec.name_, elem_type, shape, random.seed_);
			if (!tensor.has_value())
			{
				return { std::nullopt, error };
			}

			const std::string seed_note = random.seed_.has_value() ? std::format(" (seed {})", random.seed_.value()) : "";
			Logger::handle().write(LogTypes::Information,
								   std::format("generated random input '{}' {} {}{}", spec.name_, data_type_name(elem_type), shape_string(shape), seed_note));
			return { std::move(tensor.value()), std::nullopt };
		}

		auto tensor_values(const Tensor& tensor) -> std::optional<std::vector<double>>
		{
			std::vector<double> values;
			const uint8_t* data = tensor.data_.data();
			switch (tensor.elem_type_)
			{
			case onnx::TensorProto::FLOAT:
			{
				const size_t count = tensor.data_.size() / sizeof(float);
				values.reserve(count);
				for (size_t i = 0; i < count; ++i)
				{
					float value;
					std::memcpy(&value, data + i * sizeof(float), sizeof(float));
					values.push_back(static_cast<double>(value));
				}
				return values;
			}
			case onnx::TensorProto::DOUBLE:
			{
				const size_t count = tensor.data_.size() / sizeof(double);
				values.reserve(count);
				for (size_t i = 0; i < count; ++i)
				{
					double value;
					std::memcpy(&value, data + i * sizeof(double), sizeof(double));
					values.push_back(value);
				}
				return values;
			}
			case onnx::TensorProto::INT32:
			{
				const size_t count = tensor.data_.size() / sizeof(int32_t);
				values.reserve(count);
				for (size_t i = 0; i < count; ++i)
				{
					int32_t value;
					std::memcpy(&value, data + i * sizeof(int32_t), sizeof(int32_t));
					values.push_back(static_cast<double>(value));
				}
				return values;
			}
			case onnx::TensorProto::INT64:
			{
				const size_t count = tensor.data_.size() / sizeof(int64_t);
				values.reserve(count);
				for (size_t i = 0; i < count; ++i)
				{
					int64_t value;
					std::memcpy(&value, data + i * sizeof(int64_t), sizeof(int64_t));
					values.push_back(static_cast<double>(value));
				}
				return values;
			}
			default:
				return std::nullopt;
			}
		}

		auto log_stats(const Tensor& tensor) -> void
		{
			const auto values = tensor_values(tensor);
			if (!values.has_value() || values->empty())
			{
				return;
			}

			double minimum = values->front();
			double maximum = values->front();
			double sum = 0.0;
			for (double value : values.value())
			{
				minimum = std::min(minimum, value);
				maximum = std::max(maximum, value);
				sum += value;
			}
			Logger::handle().write(LogTypes::Information, std::format("output '{}' stats: min {:.6g} max {:.6g} mean {:.6g}", tensor.name_, minimum, maximum,
																	  sum / static_cast<double>(values->size())));
		}

		auto tensor_to_json(const Tensor& tensor) -> std::optional<boost::json::object>
		{
			namespace json = boost::json;
			json::array data;
			const uint8_t* bytes = tensor.data_.data();
			switch (tensor.elem_type_)
			{
			case onnx::TensorProto::FLOAT:
				for (size_t i = 0; i < tensor.data_.size() / sizeof(float); ++i)
				{
					float value;
					std::memcpy(&value, bytes + i * sizeof(float), sizeof(float));
					data.push_back(json::value(static_cast<double>(value)));
				}
				break;
			case onnx::TensorProto::DOUBLE:
				for (size_t i = 0; i < tensor.data_.size() / sizeof(double); ++i)
				{
					double value;
					std::memcpy(&value, bytes + i * sizeof(double), sizeof(double));
					data.push_back(json::value(value));
				}
				break;
			case onnx::TensorProto::INT32:
				for (size_t i = 0; i < tensor.data_.size() / sizeof(int32_t); ++i)
				{
					int32_t value;
					std::memcpy(&value, bytes + i * sizeof(int32_t), sizeof(int32_t));
					data.push_back(json::value(static_cast<int64_t>(value)));
				}
				break;
			case onnx::TensorProto::INT64:
				for (size_t i = 0; i < tensor.data_.size() / sizeof(int64_t); ++i)
				{
					int64_t value;
					std::memcpy(&value, bytes + i * sizeof(int64_t), sizeof(int64_t));
					data.push_back(json::value(value));
				}
				break;
			default:
				return std::nullopt;
			}

			json::object root;
			root["name"] = tensor.name_;
			root["data_type"] = data_type_name(tensor.elem_type_);
			json::array shape;
			for (int64_t dim : tensor.shape_)
			{
				shape.push_back(json::value(dim));
			}
			root["shape"] = std::move(shape);
			root["data"] = std::move(data);
			return root;
		}
	} // namespace

	auto run_inference(const OnnxModel& model, const std::string& model_path, const InferenceJob& job) -> int
	{
		const auto graph_inputs = model.inputs();

		std::unordered_set<std::string> initializer_names;
		for (const auto& tensor : model.initializers())
		{
			initializer_names.insert(tensor.name_);
		}

		std::vector<InputSpec> specs = job.inputs_;
		if (specs.empty())
		{
			for (const auto& input : graph_inputs)
			{
				if (initializer_names.contains(input.name_))
				{
					continue;
				}
				InputSpec spec;
				spec.name_ = input.name_;
				spec.random_ = RandomInputSpec{};
				specs.push_back(std::move(spec));
			}
			if (specs.empty())
			{
				Logger::handle().write(LogTypes::Error, "model has no graph inputs to feed");
				return 1;
			}
			Logger::handle().write(LogTypes::Information, std::format("no inputs specified; generating random tensors for {} graph input(s)", specs.size()));
		}

		std::vector<Tensor> inputs;
		for (const auto& spec : specs)
		{
			if (!spec.path_.empty())
			{
				auto [tensor, error] = load_tensor_file(spec.path_);
				if (!tensor.has_value())
				{
					Logger::handle().write(LogTypes::Error, error.value_or("cannot load input tensor"));
					return 1;
				}
				inputs.push_back(std::move(tensor.value()));
				continue;
			}

			const ValueInfo* graph_input = nullptr;
			for (const auto& candidate : graph_inputs)
			{
				if (candidate.name_ == spec.name_)
				{
					graph_input = &candidate;
					break;
				}
			}
			if (graph_input == nullptr)
			{
				Logger::handle().write(LogTypes::Error, std::format("input '{}' not found among graph inputs", spec.name_));
				return 1;
			}

			auto [tensor, error] = build_random_input(*graph_input, spec, job.dim_overrides_);
			if (!tensor.has_value())
			{
				Logger::handle().write(LogTypes::Error, error.value_or("cannot build random input"));
				return 1;
			}
			inputs.push_back(std::move(tensor.value()));
		}

		InferenceEngine engine;
		if (auto loaded = engine.load(model_path); !loaded)
		{
			Logger::handle().write(LogTypes::Error, loaded.error());
			return 1;
		}

		for (uint32_t i = 0; i < job.run_.warmup_; ++i)
		{
			auto [warmup_outputs, error] = engine.run(inputs);
			if (!warmup_outputs.has_value())
			{
				Logger::handle().write(LogTypes::Error, error.value_or("inference failed"));
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
			auto [result, error] = engine.run(inputs);
			const auto end = std::chrono::steady_clock::now();
			if (!result.has_value())
			{
				Logger::handle().write(LogTypes::Error, error.value_or("inference failed"));
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

		// Utilities::File::open creates the parent directory, so no explicit mkdir.
		const std::filesystem::path dir = job.outputs_.dir_.empty() ? std::filesystem::path(".") : std::filesystem::path(job.outputs_.dir_);

		for (const auto& output : outputs)
		{
			Logger::handle().write(LogTypes::Information, std::format("output '{}' {} {}", output.name_, shape_string(output.shape_), data_type_name(output.elem_type_)));

			if (job.outputs_.stats_)
			{
				log_stats(output);
			}

			const std::string safe_name = safe_file_name(output.name_);

			if (job.outputs_.save_)
			{
				const std::filesystem::path out_path = dir / std::format("{}.pb", safe_name);
				const onnx::TensorProto proto = proto_from_tensor(output);
				std::string serialized;
				proto.SerializeToString(&serialized);

				if (auto error = write_file_bytes(out_path.string(), reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()); error.has_value())
				{
					Logger::handle().write(LogTypes::Error, std::format("cannot write output '{}': {}", out_path.string(), error.value()));
					return 1;
				}
				Logger::handle().write(LogTypes::Information, std::format("saved '{}'", out_path.string()));
			}

			if (job.outputs_.dump_json_)
			{
				auto dumped = tensor_to_json(output);
				if (!dumped.has_value())
				{
					Logger::handle().write(LogTypes::Warning, std::format("output '{}': json dump unsupported for {}", output.name_, data_type_name(output.elem_type_)));
					continue;
				}

				const std::filesystem::path json_path = dir / std::format("{}.json", safe_name);
				const std::string text = JsonTool::pretty_format(dumped.value());
				if (auto error = write_file_bytes(json_path.string(), reinterpret_cast<const uint8_t*>(text.data()), text.size()); error.has_value())
				{
					Logger::handle().write(LogTypes::Error, std::format("cannot write output '{}': {}", json_path.string(), error.value()));
					return 1;
				}
				Logger::handle().write(LogTypes::Information, std::format("saved '{}'", json_path.string()));
			}
		}
		return 0;
	}
} // namespace YirangOnnx
