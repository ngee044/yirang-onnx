#include "RunSupport.h"

#include "File.h"
#include "InputBuilder.h"
#include "JsonTool.h"
#include "Logger.h"
#include "TensorConvert.h"

#include "onnx.pb.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <format>
#include <string_view>
#include <unordered_set>
#include <utility>

#include <boost/json.hpp>

using namespace Utilities;

namespace YirangOnnx
{
	namespace
	{
		constexpr size_t kMaxJsonDumpBytes = static_cast<size_t>(256) << 20;

		auto execution_mode_name(ExecutionMode mode) -> std::string_view
		{
			switch (mode)
			{
			case ExecutionMode::sequential:
				return "sequential";
			case ExecutionMode::parallel:
				return "parallel";
			}
			return "sequential";
		}

		auto graph_optimization_name(GraphOptimization level) -> std::string_view
		{
			switch (level)
			{
			case GraphOptimization::disabled:
				return "disabled";
			case GraphOptimization::basic:
				return "basic";
			case GraphOptimization::extended:
				return "extended";
			case GraphOptimization::all:
				return "all";
			}
			return "all";
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
			case onnx::TensorProto::BOOL:
				for (size_t i = 0; i < tensor.data_.size(); ++i)
				{
					data.push_back(json::value(bytes[i] != 0));
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

	auto shape_string(const std::vector<int64_t>& shape) -> std::string
	{
		std::string joined;
		for (size_t i = 0; i < shape.size(); ++i)
		{
			joined += std::format("{}{}", (i == 0 ? "" : ", "), shape[i]);
		}
		return std::format("[{}]", joined);
	}

	auto tuning_summary(const SessionTuning& tuning) -> std::string
	{
		const std::string intra = tuning.intra_op_threads_.has_value() ? std::to_string(tuning.intra_op_threads_.value()) : "default";
		const std::string inter = tuning.inter_op_threads_.has_value() ? std::to_string(tuning.inter_op_threads_.value()) : "default";
		return std::format("session tuning: intra_op_threads={}, inter_op_threads={}, mem_pattern={}, cpu_mem_arena={}, execution_mode={}, graph_optimization={}", intra,
						   inter, tuning.enable_mem_pattern_, tuning.enable_cpu_mem_arena_, execution_mode_name(tuning.execution_mode_),
						   graph_optimization_name(tuning.graph_optimization_));
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
		try
		{
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
			}

			std::error_code ec;
			const auto written_size = std::filesystem::file_size(path, ec);
			if (ec)
			{
				return std::format("cannot verify written file: {}", ec.message());
			}
			if (written_size != size)
			{
				return std::format("wrote {} bytes but expected {}", written_size, size);
			}
			return std::nullopt;
		}
		catch (const std::exception& e)
		{
			return std::format("cannot write '{}': {}", path, e.what());
		}
	}

	auto load_tensor_file(const std::string& path) -> std::tuple<std::optional<Tensor>, std::optional<std::string>>
	{
		std::vector<uint8_t> raw;
		try
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
			raw = std::move(bytes.value());
		}
		catch (const std::exception& e)
		{
			return { std::nullopt, std::format("cannot open input tensor '{}': {}", path, e.what()) };
		}

		onnx::TensorProto proto;
		if (!proto.ParseFromArray(raw.data(), static_cast<int>(raw.size())))
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
		case onnx::TensorProto::BOOL:
		{
			values.reserve(tensor.data_.size());
			for (size_t i = 0; i < tensor.data_.size(); ++i)
			{
				values.push_back(data[i] != 0 ? 1.0 : 0.0);
			}
			return values;
		}
		default:
			return std::nullopt;
		}
	}

	auto tensor_stats_line(const Tensor& tensor) -> std::optional<std::string>
	{
		const auto values = tensor_values(tensor);
		if (!values.has_value() || values->empty())
		{
			return std::nullopt;
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
		return std::format("output '{}' stats: min {:.6g} max {:.6g} mean {:.6g}", tensor.name_, minimum, maximum, sum / static_cast<double>(values->size()));
	}

	auto tensor_json_dump(const Tensor& tensor) -> std::optional<std::string>
	{
		auto dumped = tensor_to_json(tensor);
		if (!dumped.has_value())
		{
			return std::nullopt;
		}
		return JsonTool::pretty_format(dumped.value());
	}

	auto resolve_job_inputs(const OnnxModel& model, const InferenceJob& job) -> std::tuple<std::optional<std::vector<Tensor>>, std::optional<std::string>>
	{
		const auto graph_inputs = model.inputs();

		std::unordered_set<std::string> initializer_names;
		for (const auto& tensor : model.initializers())
		{
			initializer_names.insert(tensor.name_);
		}

		if (!job.dim_overrides_.empty())
		{
			std::unordered_set<std::string> symbolic_dims;
			for (const auto& input : graph_inputs)
			{
				for (const auto& dim : input.shape_)
				{
					int64_t numeric = 0;
					auto [ptr, ec] = std::from_chars(dim.data(), dim.data() + dim.size(), numeric);
					if (ec != std::errc() || ptr != dim.data() + dim.size())
					{
						symbolic_dims.insert(dim);
					}
				}
			}
			for (const auto& override_entry : job.dim_overrides_)
			{
				if (!symbolic_dims.contains(override_entry.first))
				{
					Logger::handle().write(LogTypes::Warning,
										   std::format("dim_overrides key '{}' matches no symbolic dimension in the model (ignored)", override_entry.first));
				}
			}
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
				return { std::nullopt, "model has no graph inputs to feed" };
			}
			Logger::handle().write(LogTypes::Information, std::format("no inputs specified; generating random tensors for {} graph input(s)", specs.size()));
		}

		std::unordered_set<std::string> graph_input_names;
		for (const auto& input : graph_inputs)
		{
			graph_input_names.insert(input.name_);
		}

		std::vector<Tensor> inputs;
		std::unordered_set<std::string> fed_names;
		for (const auto& spec : specs)
		{
			if (!spec.path_.empty())
			{
				auto [tensor, error] = load_tensor_file(spec.path_);
				if (!tensor.has_value())
				{
					return { std::nullopt, error.value_or("cannot load input tensor") };
				}
				if (!graph_input_names.contains(tensor.value().name_))
				{
					return { std::nullopt, std::format("input tensor '{}' (from '{}') does not match any graph input", tensor.value().name_, spec.path_) };
				}
				if (!spec.name_.empty() && spec.name_ != tensor.value().name_)
				{
					Logger::handle().write(LogTypes::Warning, std::format("input file '{}' declares name '{}' but the tensor is named '{}'; using '{}'", spec.path_,
																		  spec.name_, tensor.value().name_, tensor.value().name_));
				}
				if (!fed_names.insert(tensor.value().name_).second)
				{
					Logger::handle().write(LogTypes::Warning, std::format("input '{}' is provided more than once", tensor.value().name_));
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
				return { std::nullopt, std::format("input '{}' not found among graph inputs", spec.name_) };
			}

			auto [tensor, error] = build_random_input(*graph_input, spec, job.dim_overrides_);
			if (!tensor.has_value())
			{
				return { std::nullopt, error.value_or("cannot build random input") };
			}
			if (!fed_names.insert(tensor.value().name_).second)
			{
				Logger::handle().write(LogTypes::Warning, std::format("input '{}' is provided more than once", tensor.value().name_));
			}
			inputs.push_back(std::move(tensor.value()));
		}

		std::string missing;
		for (const auto& input : graph_inputs)
		{
			if (initializer_names.contains(input.name_) || fed_names.contains(input.name_))
			{
				continue;
			}
			missing += std::format("{}'{}'", (missing.empty() ? "" : ", "), input.name_);
		}
		if (!missing.empty())
		{
			return { std::nullopt, std::format("missing required graph input(s): {}", missing) };
		}
		return { std::move(inputs), std::nullopt };
	}

	auto process_outputs(const std::vector<Tensor>& outputs, const OutputSpec& spec) -> int
	{
		const std::filesystem::path dir = spec.dir_.empty() ? std::filesystem::path(".") : std::filesystem::path(spec.dir_);

		for (const auto& output : outputs)
		{
			Logger::handle().write(LogTypes::Information, std::format("output '{}' {} {}", output.name_, shape_string(output.shape_), data_type_name(output.elem_type_)));

			if (spec.stats_)
			{
				if (auto line = tensor_stats_line(output); line.has_value())
				{
					Logger::handle().write(LogTypes::Information, line.value());
				}
				else
				{
					Logger::handle().write(LogTypes::Warning, std::format("output '{}': stats unsupported for {}", output.name_, data_type_name(output.elem_type_)));
				}
			}

			const std::string safe_name = safe_file_name(output.name_);

			const auto notify_overwrite = [](const std::filesystem::path& target)
			{
				std::error_code ec;
				if (std::filesystem::exists(target, ec))
				{
					Logger::handle().write(LogTypes::Warning, std::format("overwriting existing file '{}'", target.string()));
				}
			};

			if (spec.save_)
			{
				const std::filesystem::path out_path = dir / std::format("{}.pb", safe_name);
				const onnx::TensorProto proto = proto_from_tensor(output);
				std::string serialized;
				if (!proto.SerializeToString(&serialized))
				{
					Logger::handle().write(LogTypes::Error, std::format("cannot serialize output '{}' ({} data bytes; exceeds protobuf limit or out of memory)",
																		output.name_, output.data_.size()));
					return 1;
				}

				notify_overwrite(out_path);
				if (auto error = write_file_bytes(out_path.string(), reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()); error.has_value())
				{
					Logger::handle().write(LogTypes::Error, std::format("cannot write output '{}': {}", out_path.string(), error.value()));
					return 1;
				}
				Logger::handle().write(LogTypes::Information, std::format("saved '{}'", out_path.string()));
			}

			if (spec.dump_json_)
			{
				if (output.data_.size() > kMaxJsonDumpBytes)
				{
					Logger::handle().write(LogTypes::Warning,
										   std::format("output '{}': {} bytes exceeds json dump limit {} bytes (skipped; use outputs.save for the raw .pb)", output.name_,
													   output.data_.size(), kMaxJsonDumpBytes));
					continue;
				}

				auto dumped = tensor_json_dump(output);
				if (!dumped.has_value())
				{
					Logger::handle().write(LogTypes::Warning, std::format("output '{}': json dump unsupported for {}", output.name_, data_type_name(output.elem_type_)));
					continue;
				}

				const std::filesystem::path json_path = dir / std::format("{}.json", safe_name);
				notify_overwrite(json_path);
				if (auto error = write_file_bytes(json_path.string(), reinterpret_cast<const uint8_t*>(dumped.value().data()), dumped.value().size()); error.has_value())
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
