#include "RunCommand.h"

#include "File.h"
#include "InferenceEngine.h"
#include "Logger.h"
#include "ModelTypes.h"
#include "Tensor.h"
#include "TensorConvert.h"

#include "onnx.pb.h"

#include <filesystem>
#include <format>
#include <string>
#include <vector>

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
	} // namespace

	auto run_inference(const std::string& model_path, const std::vector<std::string>& input_paths, const std::string& output_dir) -> int
	{
		std::vector<Tensor> inputs;
		for (const auto& path : input_paths)
		{
			File file;
			if (auto opened = file.open(path, std::ios::in | std::ios::binary); !opened)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot open input tensor '{}': {}", path, opened.error()));
				return 1;
			}

			auto bytes = file.read_bytes();
			if (!bytes)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot read input tensor '{}': {}", path, bytes.error()));
				return 1;
			}

			onnx::TensorProto proto;
			if (!proto.ParseFromArray(bytes.value().data(), static_cast<int>(bytes.value().size())))
			{
				Logger::handle().write(LogTypes::Error, std::format("'{}' is not a valid TensorProto", path));
				return 1;
			}
			if (proto.name().empty())
			{
				Logger::handle().write(LogTypes::Error, std::format("input tensor '{}' has no name (must match a graph input)", path));
				return 1;
			}

			auto [tensor, convert_error] = tensor_from_proto(proto);
			if (!tensor.has_value())
			{
				Logger::handle().write(LogTypes::Error, std::format("input tensor '{}': {}", path, convert_error.value_or("conversion failed")));
				return 1;
			}
			inputs.push_back(std::move(tensor.value()));
		}

		const InferenceEngine engine;
		auto [outputs, error] = engine.run(model_path, inputs);
		if (!outputs.has_value())
		{
			Logger::handle().write(LogTypes::Error, error.value_or("inference failed"));
			return 1;
		}

		// Utilities::File::open creates the parent directory, so no explicit mkdir.
		const std::filesystem::path dir = output_dir.empty() ? std::filesystem::path(".") : std::filesystem::path(output_dir);

		for (const auto& output : outputs.value())
		{
			std::string safe_name = output.name_;
			for (char& ch : safe_name)
			{
				if (ch == '/' || ch == '\\')
				{
					ch = '_';
				}
			}
			const std::filesystem::path out_path = dir / std::format("{}.pb", safe_name);

			const onnx::TensorProto proto = proto_from_tensor(output);
			std::string serialized;
			proto.SerializeToString(&serialized);

			File out;
			if (auto opened = out.open(out_path.string(), std::ios::out | std::ios::binary | std::ios::trunc); !opened)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot write output '{}': {}", out_path.string(), opened.error()));
				return 1;
			}
			if (auto written = out.write_bytes(reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()); !written)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot write output '{}': {}", out_path.string(), written.error()));
				return 1;
			}

			Logger::handle().write(LogTypes::Information,
								   std::format("output {} {} {} -> {}", output.name_, shape_string(output.shape_), data_type_name(output.elem_type_), out_path.string()));
		}
		return 0;
	}
} // namespace YirangOnnx
